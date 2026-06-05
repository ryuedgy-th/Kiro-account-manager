// Kiro API 调用核心模块
import { v4 as uuidv4 } from 'uuid'
import { fetch as undiciFetch, type RequestInit as UndiciRequestInit, type Dispatcher } from 'undici'
import type {
  KiroPayload,
  KiroUserInputMessage,
  KiroHistoryMessage,
  KiroToolWrapper,
  KiroToolResult,
  KiroImage,
  KiroDocument,
  KiroToolUse,
  KiroCachePoint,
  KiroRequestContext,
  KiroUsage,
  ProxyAccount
} from './types'
import { proxyLogger, formatError } from './logger'
import { getKProxyService } from '../kproxy'
import { getSystemProxy, getDirectPoolAgent, getCachedProxyAgent } from './systemProxy'
import { isServerWebTool, executeWebToolStructured, type WebToolConfig } from './webTools'
import {
  countTokens,
  getModelContextLength,
  setModelContextWindow,
  getModelContextWindow,
  estimateBase64DocumentTokens,
  IMAGE_TOKEN_ESTIMATE
} from './tokenCounter'
// 重新导出以保持向后兼容（proxyServer.ts 等模块仍 from './kiroApi' 导入）
export { setModelContextWindow, getModelContextWindow }

// 是否使用 K-Proxy 代理发送 API 请求（从主进程导入）
let useKProxyForApi = false
let logStreamEvents = false

export function setUseKProxyForApiInProxy(enabled: boolean): void {
  useKProxyForApi = enabled
}

export function setLogStreamEvents(enabled: boolean): void {
  logStreamEvents = enabled
}

// Payload 大小限制（KB），用户可在高级设置中调整
let payloadSizeLimitKB = 1536 // 默认 1.5MB
export function setPayloadSizeLimitKB(limitKB: number): void {
  payloadSizeLimitKB = Math.max(256, Math.min(10240, limitKB))
}

// Token buffer reserve 开关（默认 false = 完全跳过 trimHistoryByTokens）
// 关闭时后端不再裁剪任何旧消息，超出 context window 由 Kiro 后端原样返回错误
let enableTokenBufferReserve = false
export function setEnableTokenBufferReserve(enabled: boolean): void {
  enableTokenBufferReserve = !!enabled
}
export function getEnableTokenBufferReserve(): boolean {
  return enableTokenBufferReserve
}

// Token buffer reserve（仅在 enableTokenBufferReserve=true 时生效）
// 为 model context window 预留的余量，覆盖 system + tools + current + output + 估算偏差 + schema 开销
// 默认 20K：开关启用后的合理初始值（200K → effective 180K, 1M → effective 980K）
let tokenBufferReserve = 20000
export function setTokenBufferReserve(tokens: number): void {
  tokenBufferReserve = Math.max(5000, Math.min(150000, tokens))
}
export function getTokenBufferReserve(): number {
  return tokenBufferReserve
}

// 根据 modelId 和 buffer 计算 effective token limit
// 仅在 enableTokenBufferReserve=true 时被调用
// 查不到 model 时 fallback 到 200K context (Claude 默认)
function getEffectiveTokenLimit(modelId?: string): number {
  // 复用 getModelContextLength（支持 cache 命中 → 模糊匹配 → 关键词兜底）
  const ctx = modelId ? getModelContextLength(modelId) : 200000
  return Math.max(8000, ctx - tokenBufferReserve)
}

// Token 估算 (UTF-8 字节数 / 3.5，对中英混合场景做安全偏保守估算)
// 比真实 cl100k_base tokenizer 略偏高 (10-20%), 用于触发裁剪阈值是安全的
function estimateTokensFromString(str: string): number {
  return Math.ceil(Buffer.byteLength(str, 'utf-8') / 3.5)
}

/**
 * 估算整个 payload 的 token 数（binary-aware）。
 *
 * 关键：不能直接 JSON.stringify(payload) 再按长度估算，否则 images/documents 里的
 * base64（一个 PDF 可达数 MB）会被算成上百万 token，污染裁剪阈值与上下文统计。
 * 这里把 base64 附件单独按“解码字节数”折算，其余文本走字符系数。
 */
function estimatePayloadTokens(payload: KiroPayload): number {
  let tokens = 0
  const accountForMessage = (msg?: KiroHistoryMessage): void => {
    if (!msg) return
    const input = msg.userInputMessage
    if (input) {
      // 文本/上下文：把 base64 字段剥离后再按字符估算
      tokens += estimateTokensFromString(stringifyWithoutBinary(input))
      tokens += (input.images?.length ?? 0) * IMAGE_TOKEN_ESTIMATE
      for (const doc of input.documents ?? []) {
        tokens += estimateBase64DocumentTokens(doc.source?.bytes || '', doc.format)
      }
    }
    if (msg.assistantResponseMessage) {
      tokens += estimateTokensFromString(JSON.stringify(msg.assistantResponseMessage))
    }
  }
  accountForMessage(payload.conversationState.currentMessage)
  for (const msg of payload.conversationState.history ?? []) accountForMessage(msg)
  // profileArn / inferenceConfig / additionalModelRequestFields 等其余字段开销很小，忽略不计
  return Math.max(1, tokens)
}

// 序列化消息但剔除 base64 二进制字段（images/documents 的 source.bytes），
// 这些字段单独按解码字节折算，避免 base64 字符串污染按长度的 token 估算。
function stringifyWithoutBinary(input: KiroUserInputMessage): string {
  if (!input.images?.length && !input.documents?.length) {
    return JSON.stringify(input)
  }
  const { images, documents, ...rest } = input
  return JSON.stringify(rest)
}

/**
 * 获取网络代理 agent
 * 优先级（从高到低）：
 *   1. 账号自身绑定的 proxyUrl（实现"N 个号一个 IP"分桶反代）
 *   2. K-Proxy（如果启用）
 *   3. 环境变量代理
 *   4. 系统代理
 *
 * 传入 account 让账号级代理覆盖全局；不传则走全局逻辑。
 */
function getNetworkAgent(account?: ProxyAccount): Dispatcher | undefined {
  // Kiro 服务路径全程走 getCachedProxyAgent：同一 proxyUrl 复用连接池（keep-alive），
  // 既加速又稳定走同一出口 IP（与「N 账号 ↔ 1 IP」分桶一致）。
  // 注：注册流程不经此函数，仍用 safeCreateProxyAgent 每次新建，保证注册期 IP 轮换不受影响。
  // 1. 账号专属代理：实现"N 个账号共用 1 个 IP"的分桶反代
  if (account?.proxyUrl) {
    const agent = getCachedProxyAgent(account.proxyUrl)
    if (agent) {
      proxyLogger.debug('KiroAPI', `Using account-bound proxy for ${account.email || account.id}`)
      return agent
    }
  }
  // 2. K-Proxy
  if (useKProxyForApi) {
    const kproxyService = getKProxyService()
    if (kproxyService?.isRunning()) {
      const config = kproxyService.getConfig()
      const proxyUrl = `http://${config.host}:${config.port}`
      const agent = getCachedProxyAgent(proxyUrl)
      if (agent) return agent
    }
  }
  // 3. 环境变量
  const envProxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy
  const envAgent = getCachedProxyAgent(envProxy)
  if (envAgent) return envAgent
  // 4. 系统代理
  const sysAgent = getCachedProxyAgent(getSystemProxy())
  if (sysAgent) return sysAgent
  // 5. 直连：返回共享的池化 Agent（keep-alive 复用连接），而非 undefined。
  //    undefined 会回退到 undici 全局 dispatcher——其 keepAliveTimeout 默认仅 4s，
  //    稍有请求间隔就要重做 TCP+TLS 握手，正是反代比 kiro-cli 慢的主因。
  //    复用连接是普通 HTTP 客户端的标准行为，不会增加被识别的风险（反而更像真实客户端）。
  return getDirectPoolAgent()
}

/**
 * 使用代理的 fetch 函数
 * 传入 account 时会优先使用账号绑定的代理（账号-代理 N:1 分桶）
 */
async function fetchWithProxy(url: string, options: RequestInit, account?: ProxyAccount): Promise<Response> {
  const agent = getNetworkAgent(account)
  if (agent) {
    proxyLogger.debug('KiroAPI', `Using proxy agent: ${agent.constructor.name}`)
    return await undiciFetch(url, { ...options, dispatcher: agent } as UndiciRequestInit) as unknown as Response
  }
  return await undiciFetch(url, options as unknown as UndiciRequestInit) as unknown as Response
}

// Kiro API 端点配置
// 端点字段：
//   - regional: true  → url 含 {region} 占位符，按账号 region 替换（kiro.dev 新端点）
//   - rpc: true       → AWS JSON-1.0 RPC 协议：path '/' + x-amz-target 头 + content-type application/x-amz-json-1.0
//                       （旧 amazonaws.com 端点用 REST path 风格，不带 x-amz-target）
//   - identity: 'cli' → 使用 aws-sdk-rust + KIRO_CLI 身份；'ide' → KiroIDE 身份（IDE-only 订阅兼容）
//
// runtime.{region}.kiro.dev 为官方 kiro-cli 当前主端点（经 MITM 抓包确认 2026-06）：
//   POST https://runtime.us-east-1.kiro.dev/  x-amz-target: ...GenerateAssistantResponse
//   返回 application/vnd.amazon.eventstream，事件格式与旧端点一致（assistantResponseEvent 等）。
// 放在首位作为 failover 首选；403（如 IdC 订阅不支持 CLI 应用）会自动回退到旧 codewhisperer/q 端点。
const KIRO_ENDPOINTS = [
  {
    url: 'https://runtime.{region}.kiro.dev/',
    origin: 'KIRO_CLI',
    amzTarget: 'AmazonCodeWhispererStreamingService.GenerateAssistantResponse',
    name: 'KiroRuntime',
    protocol: 'generateAssistantResponse' as const,
    regional: true,
    rpc: true,
    identity: 'cli' as const
  },
  {
    url: 'https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse',
    origin: 'AI_EDITOR',
    amzTarget: 'AmazonCodeWhispererStreamingService.GenerateAssistantResponse',
    name: 'CodeWhisperer',
    protocol: 'generateAssistantResponse' as const
  },
  {
    url: 'https://q.us-east-1.amazonaws.com/generateAssistantResponse',
    origin: 'AI_EDITOR',
    amzTarget: 'AmazonCodeWhispererStreamingService.GenerateAssistantResponse',
    name: 'AmazonQ',
    protocol: 'generateAssistantResponse' as const
  },
  {
    url: 'https://q.us-east-1.amazonaws.com/SendMessageStreaming',
    origin: 'CLI',
    amzTarget: 'AmazonQDeveloperStreamingService.SendMessage',
    name: 'AmazonQCLI'
  }
]

type KiroEndpoint = typeof KIRO_ENDPOINTS[number]

// 按账号 region 解析端点 URL（仅 regional 端点替换 {region}）
function resolveEndpointUrl(endpoint: KiroEndpoint, account: ProxyAccount): string {
  if (!('regional' in endpoint) || !endpoint.regional) return endpoint.url
  const region = account.region?.startsWith('eu') ? 'eu-central-1' : 'us-east-1'
  return endpoint.url.replace('{region}', region)
}

// Kiro 版本号（跟随官方 IDE 更新）
const KIRO_VERSION = '0.12.155'
const AWS_SDK_VERSION = '1.0.34'
const AWS_STREAMING_API_VERSION = '1.0.34'

const OS_PLATFORM = process.platform === 'win32' ? 'win32' : process.platform === 'darwin' ? 'macos' : 'linux'
const OS_RELEASE = (() => { try { return require('os').release() } catch { return '10.0.0' } })()
const NODE_VERSION = process.versions.node || '22.22.0'

function getKiroUserAgent(machineId?: string): string {
  const suffix = machineId ? `KiroIDE-${KIRO_VERSION}-${machineId}` : `KiroIDE-${KIRO_VERSION}`
  return `aws-sdk-js/${AWS_SDK_VERSION} ua/2.1 os/${OS_PLATFORM}#${OS_RELEASE} lang/js md/nodejs#${NODE_VERSION} api/codewhispererstreaming#${AWS_STREAMING_API_VERSION} m/E ${suffix}`
}

function getKiroAmzUserAgent(machineId?: string): string {
  const suffix = machineId ? `KiroIDE ${KIRO_VERSION} ${machineId}` : `KiroIDE-${KIRO_VERSION}`
  return `aws-sdk-js/${AWS_SDK_VERSION} ${suffix}`
}

const KIRO_CLI_OS = OS_PLATFORM === 'win32' ? 'windows' : OS_PLATFORM === 'macos' ? 'macos' : 'linux'
// kiro-cli 当前真实 UA（MITM 抓包确认 2026-06，runtime.kiro.dev 请求头）：
//   user-agent:   aws-sdk-rust/1.3.15 ua/2.1 api/codewhispererstreaming/0.1.16551 os/macos lang/rust/1.92.0 md/appVersion-2.5.1 app/AmazonQ-For-CLI
//   x-amz-user-agent: aws-sdk-rust/1.3.15 ua/2.1 api/codewhispererstreaming/0.1.16551 os/macos lang/rust/1.92.0 m/F app/AmazonQ-For-CLI
const KIRO_CLI_SDK_VERSION = '1.3.15'
const KIRO_CLI_API_VERSION = '0.1.16551'
const KIRO_CLI_RUST_VERSION = '1.92.0'
const KIRO_CLI_APP_VERSION = '2.5.1'
const KIRO_CLI_USER_AGENT = `aws-sdk-rust/${KIRO_CLI_SDK_VERSION} ua/2.1 api/codewhispererstreaming/${KIRO_CLI_API_VERSION} os/${KIRO_CLI_OS} lang/rust/${KIRO_CLI_RUST_VERSION} md/appVersion-${KIRO_CLI_APP_VERSION} app/AmazonQ-For-CLI`
// x-amz-user-agent：抓包确认与 user-agent 基本一致，差异为 m/F（metric flag）替代 md/appVersion
const KIRO_CLI_AMZ_USER_AGENT = `aws-sdk-rust/${KIRO_CLI_SDK_VERSION} ua/2.1 api/codewhispererstreaming/${KIRO_CLI_API_VERSION} os/${KIRO_CLI_OS} lang/rust/${KIRO_CLI_RUST_VERSION} m/F app/AmazonQ-For-CLI`

// Agent 模式
const AGENT_MODE_SPEC = 'spec' // IDE 模式
const AGENT_MODE_VIBE = 'vibe' // CLI 模式

// profileArn 决策中心已迁移到 ../kiroAuthSync，反代和账号管理器主进程共用同一份定义，
// 防止多处常量漂移。注意 KIRO_BUILDER_ID_PLACEHOLDER_ARN 仍以本模块为出口 re-export，
// 这样 main/index.ts 等老 import 路径不需要改。
import {
  KIRO_BUILDER_ID_PLACEHOLDER_ARN as _KIRO_BUILDER_ID_PLACEHOLDER_ARN,
  KIRO_SOCIAL_PROFILE_ARN,
  isPlaceholderProfileArn as _isPlaceholderProfileArn
} from '../kiroAuthSync'

export const KIRO_BUILDER_ID_PLACEHOLDER_ARN = _KIRO_BUILDER_ID_PLACEHOLDER_ARN
export const isPlaceholderProfileArn = _isPlaceholderProfileArn

/**
 * 反代调 Kiro API 时使用的 profileArn 决策（流式端点用）。
 * BuilderId 使用占位符 ARN，Social 使用固定 ARN。常量统一来自 ../kiroAuthSync 防漂移。
 * 注意：流式端点（generateAssistantResponse / SendMessageStreaming）对占位符 ARN 会 403，
 * 需在 callKiroApiStream 中额外用 isPlaceholderProfileArn 剥离。
 */
function resolveProfileArn(account: ProxyAccount): string | undefined {
  if (account.profileArn && !isPlaceholderProfileArn(account.profileArn)) {
    return account.profileArn
  }
  if (account.authMethod === 'social' || account.provider === 'Github' || account.provider === 'Google') {
    return KIRO_SOCIAL_PROFILE_ARN
  }
  return KIRO_BUILDER_ID_PLACEHOLDER_ARN
}

// 「非流式/只读端点」应携带的 profileArn：
//   ListAvailableModels / ListAvailableSubscriptions / CreateSubscriptionToken / setUserPreference
// 这些 AWS 端点要求 profileArn 不能缺省，否则返回 400 "Invalid profileArn."。
// 与流式端点相反：没有真实 ARN 的 BuilderId 账号必须回退到占位符 ARN（这些端点接受占位符），
// 否则 BuilderId 账号的模型列表/订阅查询会持续 400 失败（进而 ctx-window 缓存填不上 → opus 误判 200K）。
// resolveProfileArn 对 BuilderId 已返回占位符，这里仅兜底（理论上不会命中 ?? 分支）。
function resolveProfileArnForRead(account: ProxyAccount): string {
  return resolveProfileArn(account) ?? KIRO_BUILDER_ID_PLACEHOLDER_ARN
}

// 兼容 SDK 部分调用仍想知道社交 ARN 的场景（极少；保留 export 不破坏外部 import）
export { KIRO_SOCIAL_PROFILE_ARN }

// Agentic 模式系统提示 - 防止大文件写入超时
const AGENTIC_SYSTEM_PROMPT = `# CRITICAL: CHUNKED WRITE PROTOCOL (MANDATORY)

You MUST follow these rules for ALL file operations. Violation causes server timeouts and task failure.

## ABSOLUTE LIMITS
- **MAXIMUM 350 LINES** per single write/edit operation - NO EXCEPTIONS
- **RECOMMENDED 300 LINES** or less for optimal performance
- **NEVER** write entire files in one operation if >300 lines

## MANDATORY CHUNKED WRITE STRATEGY

### For NEW FILES (>300 lines total):
1. FIRST: Write initial chunk (first 250-300 lines) using write_to_file/fsWrite
2. THEN: Append remaining content in 250-300 line chunks using file append operations
3. REPEAT: Continue appending until complete

### For EDITING EXISTING FILES:
1. Use surgical edits (apply_diff/targeted edits) - change ONLY what's needed
2. NEVER rewrite entire files - use incremental modifications
3. Split large refactors into multiple small, focused edits

REMEMBER: When in doubt, write LESS per operation. Multiple small operations > one large operation.`

// Thinking 模式标签
const THINKING_MODE_PROMPT = `<thinking_mode>enabled</thinking_mode>
<max_thinking_length>200000</max_thinking_length>`

const CODEWHISPERER_DEFAULT_MODEL_ID = 'CLAUDE_SONNET_4_20250514_V1_0'
const CODEWHISPERER_MODEL_CACHE_TTL = 5 * 60 * 1000

const codeWhispererModelCache = new Map<string, { models: KiroModel[]; timestamp: number }>()

// Kiro 支持的非 Claude 模型族 + 路由别名，按前缀透传（向前兼容新版本号，如 deepseek-3.2 → deepseek-4）
// 依据 /v1/models 实际返回：deepseek / minimax / glm / qwen 系列 + auto / simple-task 路由别名
const KIRO_PASSTHROUGH_MODEL_RE = /^(deepseek|minimax|glm|qwen|kimi|auto|simple-task)/

// 模型 ID 映射
const MODEL_ID_MAP: Record<string, string> = {
  // Claude 4.8 / 4.7 / 4.6 系列 (Claude Code 使用 dash，Kiro canonical 使用 dot)
  'claude-opus-4-8': 'claude-opus-4.8',
  'claude-opus-4.8': 'claude-opus-4.8',
  'claude-opus-4-7': 'claude-opus-4.7',
  'claude-opus-4.7': 'claude-opus-4.7',
  'claude-opus-4-6': 'claude-opus-4.6',
  'claude-opus-4.6': 'claude-opus-4.6',
  'claude-sonnet-4-6': 'claude-sonnet-4.6',
  'claude-sonnet-4.6': 'claude-sonnet-4.6',
  // Claude 4.5 系列
  'claude-sonnet-4-5': 'claude-sonnet-4.5',
  'claude-sonnet-4.5': 'claude-sonnet-4.5',
  'claude-haiku-4-5': 'claude-haiku-4.5',
  'claude-haiku-4.5': 'claude-haiku-4.5',
  'claude-opus-4-5': 'claude-opus-4.5',
  'claude-opus-4.5': 'claude-opus-4.5',
  // Claude 4 系列
  'claude-sonnet-4': 'claude-sonnet-4',
  'claude-sonnet-4-20250514': 'claude-sonnet-4',
  // Claude 3.5 系列 (映射到 Sonnet 4.5)
  'claude-3-5-sonnet': 'claude-sonnet-4.5',
  'claude-3-opus': 'claude-sonnet-4.5',
  'claude-3-sonnet': 'claude-sonnet-4',
  'claude-3-haiku': 'claude-haiku-4.5',
  // GPT 兼容映射 (映射到 Sonnet 4.5)
  'gpt-4': 'claude-sonnet-4.5',
  'gpt-4o': 'claude-sonnet-4.5',
  'gpt-4-turbo': 'claude-sonnet-4.5',
  'gpt-3.5-turbo': 'claude-sonnet-4.5',
  'default': 'claude-sonnet-4.5'
}

/**
 * 归一化 Claude 版本号：把版本号里的短横线转成点号。
 *
 * 背景：部分客户端（如 Claude Code）不允许模型名里出现 "."，会把 "claude-opus-4.6"
 * 写成 "claude-opus-4-6"，若原样透传给 Kiro 会被解析成 "claude-opus-4"（丢掉 minor），
 * 导致 1M 上下文等特性设置失败。这里把 claude-{family}-{major}-{minor} 的最后一段
 * 版本短横转成点号，兼容未来任意新版本（4.6 / 4.7 / 5.0 ...）。
 *
 * 仅当 minor 是 1~2 位数字且其后不是更多数字时才转换，避免误伤日期快照后缀
 * （如 claude-sonnet-4-20250514 不会被改）。
 */
function normalizeClaudeVersion(modelId: string): string {
  return modelId.replace(
    /^(claude-(?:sonnet|haiku|opus))-(\d+)-(\d{1,2})(?=$|[^\d])/i,
    '$1-$2.$3'
  )
}

/**
 * 规范化模型 ID，用于白名单/费率表比对：剥离客户端能力后缀（如 [1m]）+ 版本短横转点号 +
 * 剥离日期快照后缀（-YYYYMMDD）+ 转小写。
 * 与 mapModelId 不同：不做 alias 映射、也不 fallback 到 default，仅把「同一模型的不同写法」归一，
 * 这样 Claude Code 发来的 "claude-opus-4-8[1m]" 能匹配白名单里的规范名 "claude-opus-4.8"。
 *
 * 关键：必须与 mapModelId 的归一口径一致（含日期后缀剥离，见 mapModelId 1b）。否则 Claude Code
 * 子代理/workflow 默认携带的日期后缀模型（如 "claude-haiku-4-5-20251001"）会 canonicalize 成
 * 带日期的串，与 allowlist 里勾选的规范名 "claude-haiku-4.5" 比不上 → isModelAllowed 误判 403，
 * 表现为「勾了 allowlist 后主请求正常、但 subagent/workflow 调模型就被拦」。
 */
export function canonicalizeModelId(model: string): string {
  const stripped = (model || '').trim().replace(/\[[^\]]*\]\s*$/, '').trim()
  let id = normalizeClaudeVersion(stripped).toLowerCase()
  // 剥离 Claude 日期快照后缀（-YYYYMMDD），仅对 Claude 家族，避免误伤其他模型命名
  if (/^claude-(sonnet|haiku|opus)-/.test(id)) {
    id = id.replace(/-\d{8}$/, '')
  }
  return id
}

export function mapModelId(model: string): string {
  // 去掉 Claude Code 等客户端附加的能力后缀，如 "claude-opus-4-8[1m]" → "claude-opus-4-8"
  let modelId = model.trim().replace(/\[[^\]]*\]\s*$/, '').trim()
  if (!modelId) return MODEL_ID_MAP.default
  if (isCodeWhispererModelId(modelId)) return modelId
  // 0) 归一化版本号短横 → 点号（claude-opus-4-6 → claude-opus-4.6），兼容不支持 "." 的客户端
  modelId = normalizeClaudeVersion(modelId)
  const lower = modelId.toLowerCase()
  // 1) 显式 alias 映射优先
  if (MODEL_ID_MAP[lower]) return MODEL_ID_MAP[lower]
  // 1b) 带日期快照后缀的 Claude 模型（如 claude-haiku-4.5-20251001）：
  //     Kiro 只认不带日期的规范名（claude-haiku-4.5），原样透传会被后端拒为 INVALID_MODEL_ID。
  //     先剥掉 -YYYYMMDD 再查一次映射 / 透传规范名。这同时修复 Claude Code 子代理
  //     默认携带日期后缀模型（claude-haiku-4-5-20251001）导致的 400。
  const dateStripped = lower.replace(/-\d{8}$/, '')
  if (dateStripped !== lower) {
    if (MODEL_ID_MAP[dateStripped]) return MODEL_ID_MAP[dateStripped]
    if (/^claude-(sonnet|haiku|opus)-/.test(dateStripped)) return dateStripped
  }
  // 2) 看似 Kiro 支持的 Claude 模型格式 (claude-{sonnet|haiku|opus}-{ver})，原样透传
  //    用于向前兼容尚未加入 MODEL_ID_MAP 的新发布模型
  if (/^claude-(sonnet|haiku|opus)-/.test(lower)) return modelId
  // 3) Kiro 支持的非 Claude 模型（deepseek/minimax/glm/qwen 等）及路由别名（auto/simple-task）原样透传
  //    关键：这些不能 fallback 到 Claude，否则 (a) 用户选的模型被静默替换，
  //    (b) translator 的能力注册表会误判为 Claude 而注入 thinking/effort 导致 400
  if (KIRO_PASSTHROUGH_MODEL_RE.test(lower)) return modelId
  // 4) 完全未知的 model（用户拼错/不存在），兜底到 default 避免直接 400
  console.warn(`[Kiro API] Unknown model "${modelId}" → fallback to "${MODEL_ID_MAP.default}"`)
  return MODEL_ID_MAP.default
}

// payload 深拷贝：每次请求/重试前隔离一份，避免改动污染调用方原对象。
// structuredClone（Node 17+，结构化克隆算法）比 JSON.parse(JSON.stringify()) 快得多，
// 且不会把 base64 附件再 serialize+parse 一遍（payload 可能含数 MB 附件，旧做法在此明显阻塞）。
// 兜底：万一未来 payload 混入不可结构化克隆的字段（函数等），回退到 JSON 深拷贝保证不崩。
function clonePayload(payload: KiroPayload): KiroPayload {
  try {
    return structuredClone(payload)
  } catch {
    return JSON.parse(JSON.stringify(payload)) as KiroPayload
  }
}

function normalizeModelKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}

function modelTokens(value: string): string[] {
  return value.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)
}

function matchesRequestedModel(model: KiroModel, requestedModelId: string): boolean {
  // 1. modelId 级精确匹配（去除符号后比较）
  const requestedKey = normalizeModelKey(requestedModelId)
  const modelIdKey = normalizeModelKey(model.modelId)
  if (modelIdKey === requestedKey || modelIdKey.includes(requestedKey)) return true
  // 2. modelName 精确匹配
  if (model.modelName && normalizeModelKey(model.modelName).includes(requestedKey)) return true
  // 3. token 匹配（所有请求 token 必须在 modelId+modelName 中命中，不搜索 description 避免误匹配）
  const tokens = modelTokens(requestedModelId).filter(token => token !== 'latest' && token !== 'model')
  if (tokens.length === 0) return false
  const candidateTokens = new Set(modelTokens(`${model.modelId} ${model.modelName || ''}`))
  // 必须全部 token 命中
  if (!tokens.every(token => candidateTokens.has(token))) return false
  // 防止模型家族冲突：如果请求包含 opus/sonnet/haiku，候选必须也包含对应的
  const families = ['opus', 'sonnet', 'haiku']
  for (const family of families) {
    if (tokens.includes(family) && !candidateTokens.has(family)) return false
    if (!tokens.includes(family) && candidateTokens.has(family)) return false
  }
  return true
}

function isCodeWhispererModelId(modelId: string): boolean {
  return /^[A-Z0-9_]+$/.test(modelId) && modelId.includes('_')
}

function getModelCacheKey(account: ProxyAccount): string {
  return `${account.id}:${account.region || 'us-east-1'}:${resolveProfileArn(account) ?? 'no-arn'}`
}

async function getCachedCodeWhispererModels(account: ProxyAccount, signal?: AbortSignal): Promise<KiroModel[]> {
  const key = getModelCacheKey(account)
  const cached = codeWhispererModelCache.get(key)
  if (cached && Date.now() - cached.timestamp < CODEWHISPERER_MODEL_CACHE_TTL) return cached.models
  const models = await fetchKiroModels(account, signal)
  codeWhispererModelCache.set(key, { models, timestamp: Date.now() })
  return models
}

async function resolveCodeWhispererModelId(account: ProxyAccount, requestedModelId?: string, signal?: AbortSignal): Promise<string> {
  const modelId = requestedModelId?.trim()
  if (!modelId) return CODEWHISPERER_DEFAULT_MODEL_ID
  if (isCodeWhispererModelId(modelId)) return modelId
  const models = await getCachedCodeWhispererModels(account, signal)
  return models.find(model => matchesRequestedModel(model, modelId))?.modelId || CODEWHISPERER_DEFAULT_MODEL_ID
}

function getPayloadModelId(payload: KiroPayload): string | undefined {
  const currentModelId = payload.conversationState.currentMessage.userInputMessage.modelId
  if (currentModelId) return currentModelId
  return payload.conversationState.history?.find(message => message.userInputMessage?.modelId)?.userInputMessage?.modelId
}

function applyPayloadModelId(payload: KiroPayload, modelId: string): void {
  payload.conversationState.currentMessage.userInputMessage.modelId = modelId
  for (const message of payload.conversationState.history ?? []) {
    if (message.userInputMessage) message.userInputMessage.modelId = modelId
  }
}

function applyPayloadOrigin(payload: KiroPayload, origin: string): void {
  payload.conversationState.currentMessage.userInputMessage.origin = origin
  for (const message of payload.conversationState.history ?? []) {
    if (message.userInputMessage) message.userInputMessage.origin = origin
  }
}

// 检测是否为 Agentic 模式请求
export function isAgenticRequest(model: string, tools?: unknown[]): boolean {
  const lower = model.toLowerCase()
  // 模型名称包含 -agentic 或有工具调用
  return lower.includes('-agentic') || lower.includes('agentic') || Boolean(tools && tools.length > 0)
}

// 检测是否启用 Thinking 模式
export function isThinkingEnabled(headers?: Record<string, string>): boolean {
  if (!headers) return false
  // 检查 Anthropic-Beta 头是否包含 thinking
  const betaHeader = headers['anthropic-beta'] || headers['Anthropic-Beta'] || ''
  return betaHeader.toLowerCase().includes('thinking')
}

// 注入系统提示
export function injectSystemPrompts(
  content: string,
  isAgentic: boolean,
  thinkingEnabled: boolean
): string {
  let result = content
  
  // 注入时间戳
  const timestamp = new Date().toISOString()
  const timestampPrompt = `Current time: ${timestamp}`
  
  // 注入 Thinking 模式（必须在最前面）
  if (thinkingEnabled) {
    result = THINKING_MODE_PROMPT + '\n\n' + result
  }
  
  // 注入 Agentic 模式提示
  if (isAgentic) {
    result = result + '\n\n' + AGENTIC_SYSTEM_PROMPT
  }
  
  // 注入时间戳
  result = timestampPrompt + '\n\n' + result
  
  return result
}

// ============= 消息清理逻辑（参考 Kiro 官方实现）=============

// 占位消息
const HELLO_MESSAGE: KiroHistoryMessage = {
  userInputMessage: { content: 'Hello', origin: 'AI_EDITOR' }
}

const CONTINUE_MESSAGE: KiroHistoryMessage = {
  userInputMessage: { content: 'Continue', origin: 'AI_EDITOR' }
}

const UNDERSTOOD_MESSAGE: KiroHistoryMessage = {
  assistantResponseMessage: { content: 'understood' }
}

// 创建失败的工具结果消息
function createFailedToolUseMessage(toolUseIds: string[]): KiroHistoryMessage {
  return {
    userInputMessage: {
      content: '',
      origin: 'AI_EDITOR',
      userInputMessageContext: {
        toolResults: toolUseIds.map(createFailedToolResult)
      }
    }
  }
}

// 类型检查函数
function isUserInputMessage(message: KiroHistoryMessage): boolean {
  return message != null && 'userInputMessage' in message && message.userInputMessage != null
}

function isAssistantResponseMessage(message: KiroHistoryMessage): boolean {
  return message != null && 'assistantResponseMessage' in message && message.assistantResponseMessage != null
}

function hasToolResults(message: KiroHistoryMessage): boolean {
  return !!(message.userInputMessage?.userInputMessageContext?.toolResults?.length)
}

function hasToolUses(message: KiroHistoryMessage): boolean {
  return !!(message.assistantResponseMessage?.toolUses?.length)
}

function hasMatchingToolResults(
  toolUses: KiroToolUse[] | undefined,
  toolResults: KiroToolResult[] | undefined
): boolean {
  if (!toolUses || !toolUses.length) return true
  if (!toolResults || !toolResults.length) return false
  
  const allToolUsesHaveResults = toolUses.every(
    toolUse => toolResults.some(result => result.toolUseId === toolUse.toolUseId)
  )
  const allToolResultsHaveUses = toolResults.every(
    result => toolUses.some(toolUse => result.toolUseId === toolUse.toolUseId)
  )
  return allToolUsesHaveResults && allToolResultsHaveUses
}

function createFailedToolResult(toolUseId: string): KiroToolResult {
  return {
    toolUseId,
    content: [{ text: 'Tool execution failed' }],
    status: 'error'
  }
}

function stripInvalidToolResults(message: KiroHistoryMessage): KiroHistoryMessage | null {
  if (message.userInputMessage?.content?.trim()) {
    return {
      userInputMessage: {
        ...message.userInputMessage,
        userInputMessageContext: undefined
      }
    }
  }
  return null
}

// 确保以 user 消息开始
function ensureStartsWithUserMessage(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  if (messages.length === 0 || isUserInputMessage(messages[0])) {
    return messages
  }
  return [HELLO_MESSAGE, ...messages]
}

// 确保以 user 消息结束
function ensureEndsWithUserMessage(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  if (messages.length === 0) return [HELLO_MESSAGE]
  if (isUserInputMessage(messages[messages.length - 1])) return messages
  return [...messages, CONTINUE_MESSAGE]
}

// 确保消息交替
function ensureAlternatingMessages(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  if (messages.length <= 1) return messages
  
  const result: KiroHistoryMessage[] = [messages[0]]
  for (let i = 1; i < messages.length; i++) {
    const prevMessage = result[result.length - 1]
    const currentMessage = messages[i]
    
    if (isUserInputMessage(prevMessage) && isUserInputMessage(currentMessage)) {
      result.push(UNDERSTOOD_MESSAGE)
    } else if (isAssistantResponseMessage(prevMessage) && isAssistantResponseMessage(currentMessage)) {
      result.push(CONTINUE_MESSAGE)
    }
    result.push(currentMessage)
  }
  return result
}

function relocateToolResultMessages(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  const assistantToolUseIndexes: number[] = []
  const toolResultIndexById = new Map<string, number>()
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]
    if (isAssistantResponseMessage(message) && hasToolUses(message)) {
      assistantToolUseIndexes.push(i)
    } else if (isUserInputMessage(message) && hasToolResults(message)) {
      for (const toolResult of message.userInputMessage?.userInputMessageContext?.toolResults ?? []) {
        if (toolResult.toolUseId && !toolResultIndexById.has(toolResult.toolUseId)) {
          toolResultIndexById.set(toolResult.toolUseId, i)
        }
      }
    }
  }

  if (assistantToolUseIndexes.length === 0) return messages

  const result: KiroHistoryMessage[] = []
  const usedIndexes = new Set<number>()
  for (let i = 0; i < messages.length; i++) {
    if (usedIndexes.has(i)) continue
    const message = messages[i]
    result.push(message)
    usedIndexes.add(i)

    if (isAssistantResponseMessage(message) && hasToolUses(message)) {
      for (const toolUse of message.assistantResponseMessage?.toolUses ?? []) {
        const toolResultIndex = toolResultIndexById.get(toolUse.toolUseId)
        if (toolResultIndex !== undefined && toolResultIndex !== i + 1 && !usedIndexes.has(toolResultIndex)) {
          const toolResultMessage = messages[toolResultIndex]
          if (toolResultMessage) {
            result.push(toolResultMessage)
            usedIndexes.add(toolResultIndex)
          }
        }
      }
    }
  }
  return result
}

function removeInvalidToolResultMessages(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  const result: KiroHistoryMessage[] = []
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]
    const previousMessage = i > 0 ? messages[i - 1] : null
    if (!isUserInputMessage(message) || !hasToolResults(message)) {
      result.push(message)
      continue
    }
    if (!previousMessage || !isAssistantResponseMessage(previousMessage) || !hasToolUses(previousMessage)) {
      const stripped = stripInvalidToolResults(message)
      if (stripped) result.push(stripped)
      continue
    }

    const validToolUseIds = new Set((previousMessage.assistantResponseMessage?.toolUses ?? []).map(toolUse => toolUse.toolUseId).filter(Boolean))
    const seenToolUseIds = new Set<string>()
    const toolResults = message.userInputMessage?.userInputMessageContext?.toolResults ?? []
    const filteredToolResults = toolResults.filter(toolResult => {
      if (!toolResult.toolUseId || !validToolUseIds.has(toolResult.toolUseId) || seenToolUseIds.has(toolResult.toolUseId)) return false
      seenToolUseIds.add(toolResult.toolUseId)
      return true
    })

    if (filteredToolResults.length === toolResults.length) {
      result.push(message)
    } else if (filteredToolResults.length > 0) {
      result.push({
        userInputMessage: {
          ...message.userInputMessage!,
          userInputMessageContext: {
            ...message.userInputMessage!.userInputMessageContext,
            toolResults: filteredToolResults
          }
        }
      })
    } else {
      const stripped = stripInvalidToolResults(message)
      if (stripped) result.push(stripped)
    }
  }
  return result
}

// 确保工具调用有对应结果
function ensureValidToolUsesAndResults(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  const result: KiroHistoryMessage[] = []
  
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]
    result.push(message)
    
    if (isAssistantResponseMessage(message) && hasToolUses(message)) {
      const nextMessage = i + 1 < messages.length ? messages[i + 1] : null
      const toolUses = message.assistantResponseMessage?.toolUses ?? []
      const toolUseIds = toolUses.map((tu, idx) => tu.toolUseId ?? `toolUse_${idx + 1}`)
      
      if (!nextMessage || !isUserInputMessage(nextMessage) || !hasToolResults(nextMessage)) {
        // 没有对应的工具结果，添加失败消息
        result.push(createFailedToolUseMessage(toolUseIds))
      } else if (!hasMatchingToolResults(
        message.assistantResponseMessage?.toolUses,
        nextMessage.userInputMessage?.userInputMessageContext?.toolResults
      ) && !messages.some((candidate, index) => (
        index !== i
        && isAssistantResponseMessage(candidate)
        && hasToolUses(candidate)
        && hasMatchingToolResults(candidate.assistantResponseMessage?.toolUses, nextMessage.userInputMessage?.userInputMessageContext?.toolResults)
      ))) {
        // 工具结果不匹配，添加失败消息
        const existingToolResults = nextMessage.userInputMessage?.userInputMessageContext?.toolResults ?? []
        const validToolUseIds = new Set(toolUseIds)
        const usedToolUseIds = new Set<string>()
        const completedToolResults = existingToolResults.filter(toolResult => {
          if (!toolResult.toolUseId || !validToolUseIds.has(toolResult.toolUseId) || usedToolUseIds.has(toolResult.toolUseId)) return false
          usedToolUseIds.add(toolResult.toolUseId)
          return true
        })
        for (const toolUseId of toolUseIds) {
          if (!usedToolUseIds.has(toolUseId)) completedToolResults.push(createFailedToolResult(toolUseId))
        }
        result.push({
          userInputMessage: {
            ...nextMessage.userInputMessage!,
            userInputMessageContext: {
              ...nextMessage.userInputMessage!.userInputMessageContext,
              toolResults: completedToolResults
            }
          }
        })
        i++
      }
    }
  }
  return result
}

// 移除空的 user 消息
function removeEmptyUserMessages(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  if (messages.length <= 1) return messages
  
  const firstUserMessageIndex = messages.findIndex(isUserInputMessage)
  return messages.filter((message, index) => {
    if (isAssistantResponseMessage(message)) return true
    if (isUserInputMessage(message) && index === firstUserMessageIndex) return true
    if (isUserInputMessage(message)) {
      const hasContent = message.userInputMessage?.content?.trim() !== ''
      return hasContent || hasToolResults(message)
    }
    return true
  })
}

function validateConversation(messages: KiroHistoryMessage[]): string[] {
  const errors: string[] = []
  if (messages.length === 0 || !isUserInputMessage(messages[0])) {
    errors.push('STARTS_WITH_USER_MESSAGE:index=0')
  }
  if (messages.length === 0 || !isUserInputMessage(messages[messages.length - 1])) {
    errors.push(`ENDS_WITH_USER_MESSAGE:index=${Math.max(messages.length - 1, 0)}`)
  }
  for (let i = 1; i < messages.length; i++) {
    const previousMessage = messages[i - 1]
    const currentMessage = messages[i]
    if (isUserInputMessage(previousMessage) && isUserInputMessage(currentMessage)) {
      errors.push(`ALTERNATING_MESSAGES:index=${i}`)
      break
    }
    if (isAssistantResponseMessage(previousMessage) && isAssistantResponseMessage(currentMessage)) {
      errors.push(`ALTERNATING_MESSAGES:index=${i}`)
      break
    }
  }
  for (let i = 0; i < messages.length - 1; i++) {
    const message = messages[i]
    const nextMessage = messages[i + 1]
    if (isAssistantResponseMessage(message) && hasToolUses(message) && (!isUserInputMessage(nextMessage) || !hasMatchingToolResults(message.assistantResponseMessage?.toolUses, nextMessage?.userInputMessage?.userInputMessageContext?.toolResults))) {
      errors.push(`TOOL_USES_AND_RESULTS:index=${i + 1}`)
      break
    }
    if (isAssistantResponseMessage(message) && !hasToolUses(message) && isUserInputMessage(nextMessage) && hasToolResults(nextMessage)) {
      errors.push(`TOOL_RESULTS_AND_NO_USES:index=${i}`)
      break
    }
  }
  for (let i = 1; i < messages.length; i++) {
    const previousMessage = messages[i - 1]
    const currentMessage = messages[i]
    if (!isAssistantResponseMessage(previousMessage) || !hasToolUses(previousMessage) || !isUserInputMessage(currentMessage) || !hasToolResults(currentMessage)) continue
    const toolUseIds = new Set((previousMessage.assistantResponseMessage?.toolUses ?? []).map(toolUse => toolUse.toolUseId).filter(Boolean))
    const seenToolUseIds = new Set<string>()
    const hasInvalidToolResult = (currentMessage.userInputMessage?.userInputMessageContext?.toolResults ?? []).some(toolResult => {
      if (!toolResult.toolUseId || !toolUseIds.has(toolResult.toolUseId) || seenToolUseIds.has(toolResult.toolUseId)) return true
      seenToolUseIds.add(toolResult.toolUseId)
      return false
    })
    if (hasInvalidToolResult) {
      errors.push(`TOOL_RESULTS_ORPHAN_IDS:index=${i}`)
      break
    }
  }
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]
    if (isUserInputMessage(message) && !message.userInputMessage?.content?.trim() && !hasToolResults(message)) {
      errors.push(`NON_EMPTY_USER_MESSAGE:index=${i}`)
      break
    }
  }
  return errors
}

function getToolNames(tools: KiroToolWrapper[]): Set<string> {
  return new Set(tools.flatMap(tool => 'toolSpecification' in tool ? [tool.toolSpecification.name] : []))
}

function stringifyToolInput(input: unknown): string {
  if (input === undefined) return ''
  if (typeof input === 'string') return input
  try {
    return JSON.stringify(input)
  } catch {
    return String(input)
  }
}

function flattenContent(content: string, extra: string): string {
  const trimmedContent = content.trim()
  if (!trimmedContent) return extra
  if (!extra) return trimmedContent
  return `${trimmedContent}\n\n${extra}`
}

function formatToolUses(toolUses: KiroToolUse[]): string {
  return toolUses.map(toolUse => [
    `<tool_use id="${toolUse.toolUseId}" name="${toolUse.name}">`,
    stringifyToolInput(toolUse.input),
    '</tool_use>'
  ].filter(Boolean).join('\n')).join('\n\n')
}

function formatToolResults(toolResults: KiroToolResult[]): string {
  return toolResults.map(toolResult => [
    `<tool_result id="${toolResult.toolUseId}" status="${toolResult.status}">`,
    toolResult.content.map(content => content.text).join('\n'),
    '</tool_result>'
  ].filter(Boolean).join('\n')).join('\n\n')
}

function normalizeToolHistory(messages: KiroHistoryMessage[], tools: KiroToolWrapper[]): KiroHistoryMessage[] {
  const toolNames = getToolNames(tools)
  const hasUnknownToolUse = messages.some(message => (
    message.assistantResponseMessage?.toolUses?.some(toolUse => !toolNames.has(toolUse.name)) ?? false
  ))
  if (!hasUnknownToolUse) return messages

  return messages.map(message => {
    if (message.assistantResponseMessage?.toolUses?.length) {
      return {
        assistantResponseMessage: {
          ...message.assistantResponseMessage,
          content: flattenContent(message.assistantResponseMessage.content, formatToolUses(message.assistantResponseMessage.toolUses)),
          toolUses: undefined
        }
      }
    }
    if (message.userInputMessage?.userInputMessageContext?.toolResults?.length) {
      return {
        userInputMessage: {
          ...message.userInputMessage,
          content: flattenContent(message.userInputMessage.content, formatToolResults(message.userInputMessage.userInputMessageContext.toolResults)),
          userInputMessageContext: {
            ...message.userInputMessage.userInputMessageContext,
            toolResults: undefined
          }
        }
      }
    }
    return message
  })
}

// 清理会话消息（参考 Kiro 官方实现）
function sanitizeConversation(messages: KiroHistoryMessage[]): KiroHistoryMessage[] {
  let sanitized = [...messages]
  sanitized = ensureStartsWithUserMessage(sanitized)
  sanitized = removeEmptyUserMessages(sanitized)
  sanitized = relocateToolResultMessages(sanitized)
  sanitized = removeInvalidToolResultMessages(sanitized)
  sanitized = ensureValidToolUsesAndResults(sanitized)
  sanitized = ensureAlternatingMessages(sanitized)
  sanitized = ensureEndsWithUserMessage(sanitized)
  const validationErrors = validateConversation(sanitized)
  if (validationErrors.length > 0) {
    throw new Error(`Invalid Kiro conversation after sanitization: ${validationErrors.join(', ')}`)
  }
  return sanitized
}

// 按 token 估算成对裁剪 history 最旧消息 (避免后端 CONTENT_LENGTH_EXCEEDS_THRESHOLD)
// 切点保证不破坏 toolUse↔toolResult 配对：assistant(toolUse) 必须连同后续 user(toolResult) 一起裁
// 裁剪后用 ensureStartsWithUserMessage 兜底重新规范化
function trimHistoryByTokens(payload: KiroPayload, maxTokens: number): { trimmed: number; finalTokens: number; iterations: number } {
  let history = payload.conversationState.history
  if (!history || history.length === 0) {
    return { trimmed: 0, finalTokens: estimatePayloadTokens(payload), iterations: 0 }
  }

  let totalTrimmed = 0
  let iterations = 0
  let currentTokens = estimatePayloadTokens(payload)
  const MAX_ITERATIONS = 100 // 防止极端情况死循环

  while (currentTokens > maxTokens && history.length >= 4 && iterations < MAX_ITERATIONS) {
    iterations++
    // 计算安全切点：从 index 0 开始至少裁掉 1 组 (user+assistant)，并连带 toolUse/toolResult 配对
    let cutAt = 0
    while (cutAt < history.length - 2) {
      const msg = history[cutAt]
      // assistant(toolUse) → 下一条 user(toolResult) 必须一起裁，避免配对断裂
      if (isAssistantResponseMessage(msg) && hasToolUses(msg)) {
        cutAt += 2
      } else {
        cutAt += 1
      }
      if (cutAt >= 2) break
    }

    if (cutAt === 0) break // 无法继续裁剪

    history = history.slice(cutAt)
    totalTrimmed += cutAt

    // 裁剪后 history 可能以 assistant 起头 → 补 HELLO 重新规范
    history = ensureStartsWithUserMessage(history)
    payload.conversationState.history = history
    currentTokens = estimatePayloadTokens(payload)
  }

  return { trimmed: totalTrimmed, finalTokens: currentTokens, iterations }
}

// ============= 构建 Kiro API 请求负载（参考 Kiro 官方实现）=============

export function buildKiroPayload(
  content: string,
  modelId: string,
  origin: string,
  history: KiroHistoryMessage[] = [],
  tools: KiroToolWrapper[] = [],
  toolResults: KiroToolResult[] = [],
  images: KiroImage[] = [],
  profileArn?: string,
  inferenceConfig?: { maxTokens?: number; temperature?: number; topP?: number },
  messageOptions?: { cachePoint?: KiroCachePoint | undefined; clientCacheConfig?: unknown; documents?: KiroDocument[]; conversationId?: string; context?: KiroRequestContext },
  additionalModelRequestFields?: Record<string, unknown>,
  // 客户端原始 model（含能力后缀如 "[1m]"）。仅用于按正确的 context window 算裁剪预算——
  // 后端 modelId 已被剥掉后缀，会让 1M 会话误用 200K 预算而过早裁剪 history。
  clientModelId?: string
): KiroPayload {
  // 裁剪预算按「客户端声明的 context window」算（含 [1m] 后缀），回退到后端 modelId
  const trimModelId = clientModelId || modelId
  // 构建当前消息
  const finalContent = content.trim() || (toolResults.length > 0 ? '' : 'Continue')
  
  const currentUserInputMessage: KiroUserInputMessage = {
    content: finalContent,
    modelId,
    origin
  }

  if (images.length > 0) {
    currentUserInputMessage.images = images
  }

  if (messageOptions?.documents?.length) {
    currentUserInputMessage.documents = messageOptions.documents
  }

  if (messageOptions?.cachePoint) {
    currentUserInputMessage.cachePoint = messageOptions.cachePoint
  }

  if (messageOptions?.clientCacheConfig !== undefined) {
    currentUserInputMessage.clientCacheConfig = messageOptions.clientCacheConfig
  }

  // 构建 userInputMessageContext（包含 tools 和 toolResults）
  // 注意：tools 只放在最后一条消息（currentMessage）的 userInputMessageContext 中
  if (tools.length > 0 || toolResults.length > 0) {
    currentUserInputMessage.userInputMessageContext = {}
    if (tools.length > 0) {
      currentUserInputMessage.userInputMessageContext.tools = tools
    }
    if (toolResults.length > 0) {
      currentUserInputMessage.userInputMessageContext.toolResults = toolResults
    }
  }

  if (messageOptions?.context) {
    currentUserInputMessage.userInputMessageContext = {
      ...currentUserInputMessage.userInputMessageContext,
      ...(messageOptions.context.editorState !== undefined ? { editorState: messageOptions.context.editorState } : {}),
      ...(messageOptions.context.shellState !== undefined ? { shellState: messageOptions.context.shellState } : {}),
      ...(messageOptions.context.gitState !== undefined ? { gitState: messageOptions.context.gitState } : {}),
      ...(messageOptions.context.envState !== undefined ? { envState: messageOptions.context.envState } : {}),
      ...(messageOptions.context.additionalContext !== undefined ? { additionalContext: messageOptions.context.additionalContext } : {})
    }
  }

  // 构建 currentMessage
  const currentMessage: KiroHistoryMessage = {
    userInputMessage: currentUserInputMessage
  }

  // 清理并准备所有消息（history + currentMessage）
  const allMessages = [...history, currentMessage]
  const sanitizedMessages = sanitizeConversation(normalizeToolHistory(allMessages, tools))
  
  // 分离 history 和 currentMessage
  // currentMessage 是最后一条消息，history 是其余的
  const sanitizedHistory = sanitizedMessages.slice(0, -1)
  let finalCurrentMessage = sanitizedMessages.at(-1)!

  // 确保 currentMessage 是 user 消息（sanitizeConversation 保证以 user 消息结束）
  // 并确保包含 tools
  if (!finalCurrentMessage.userInputMessage) {
    // 如果清理后最后一条不是 user 消息，创建一个新的
    finalCurrentMessage = {
      userInputMessage: {
        content: finalContent || 'Continue',
        modelId,
        origin
      }
    }
  }
  
  finalCurrentMessage.userInputMessage!.userInputMessageContext = {
    ...finalCurrentMessage.userInputMessage!.userInputMessageContext,
    ...(tools.length > 0 ? { tools } : {})
  }

  // conversationId 稳定化：同一会话的多轮请求复用同一个 conversationId
  // 优先级：客户端显式 conversation_id → sessionHint（header 提取）→ history fingerprint → 新 UUID
  const conversationId = resolveConversationId(history, messageOptions?.conversationId)
  const payload: KiroPayload = {
    conversationState: {
      agentContinuationId: uuidv4(),
      agentTaskType: 'vibe',
      chatTriggerType: 'MANUAL',
      conversationId,
      currentMessage: {
        userInputMessage: finalCurrentMessage.userInputMessage!
      },
      history: sanitizedHistory.length > 0 ? sanitizedHistory : undefined
    }
  }

  if (profileArn !== undefined) {
    payload.profileArn = profileArn
  }

  if (inferenceConfig && (inferenceConfig.maxTokens || inferenceConfig.temperature !== undefined || inferenceConfig.topP !== undefined)) {
    payload.inferenceConfig = {}
    if (inferenceConfig.maxTokens) {
      payload.inferenceConfig.maxTokens = inferenceConfig.maxTokens
    }
    if (inferenceConfig.temperature !== undefined) {
      payload.inferenceConfig.temperature = inferenceConfig.temperature
    }
    if (inferenceConfig.topP !== undefined) {
      payload.inferenceConfig.topP = inferenceConfig.topP
    }
  }

  // additionalModelRequestFields（thinking 等模型级参数）
  if (additionalModelRequestFields && Object.keys(additionalModelRequestFields).length > 0) {
    payload.additionalModelRequestFields = additionalModelRequestFields
  }

  // ====== 第一阶段：按 token 估算成对裁剪旧 history ======
  // 避免 Kiro 后端 CONTENT_LENGTH_EXCEEDS_THRESHOLD（token 维度的拒绝）
  // 注意：byte size 充足但 token 超限是常见情况（长对话+大量小消息）
  // effectiveLimit 按模型 context window 自动算：ctx - tokenBufferReserve（开关启用时，默认 20K）
  // 例：sonnet-4.5 (200K) → 180K, sonnet-4.5 with 1M beta → 980K
  // 开关关闭时完全跳过，超出 context window 由 Kiro 后端原样返回错误
  if (enableTokenBufferReserve) {
    const effectiveTokenLimit = getEffectiveTokenLimit(trimModelId)
    const tokenTrimResult = trimHistoryByTokens(payload, effectiveTokenLimit)
    if (tokenTrimResult.trimmed > 0) {
      const modelCtx = getModelContextLength(trimModelId)
      console.log(`[KiroPayload] Trimmed ${tokenTrimResult.trimmed} oldest history messages by token estimate (≈${tokenTrimResult.finalTokens.toLocaleString()} / ${effectiveTokenLimit.toLocaleString()} tokens [model ctx ${modelCtx.toLocaleString()} - buffer ${tokenBufferReserve.toLocaleString()}], ${tokenTrimResult.iterations} iter)`)
    }
  }

  // ====== 第二阶段：按 byte 截断 tool result 内容 ======
  // 避免 HTTP body 过大被 Kiro 网关拒绝
  // 用户可在高级设置中调整限制值（默认 1536KB = 1.5MB）
  const PAYLOAD_SIZE_LIMIT = (payloadSizeLimitKB || 1536) * 1024
  const TOOL_RESULT_TRUNCATE_LENGTH = 4000
  let initialPayloadSize = JSON.stringify(payload).length
  if (initialPayloadSize > PAYLOAD_SIZE_LIMIT && payload.conversationState.history) {
    const historyMessages = payload.conversationState.history
    let truncatedCount = 0
    for (const message of historyMessages) {
      if (initialPayloadSize <= PAYLOAD_SIZE_LIMIT) break
      const userToolResults = message.userInputMessage?.userInputMessageContext?.toolResults
      if (!userToolResults) continue
      for (const toolResult of userToolResults) {
        if (initialPayloadSize <= PAYLOAD_SIZE_LIMIT) break
        if (!toolResult.content) continue
        for (const contentItem of toolResult.content) {
          if (initialPayloadSize <= PAYLOAD_SIZE_LIMIT) break
          if (contentItem.text && contentItem.text.length > TOOL_RESULT_TRUNCATE_LENGTH) {
            const originalLen = contentItem.text.length
            contentItem.text = `${contentItem.text.slice(0, TOOL_RESULT_TRUNCATE_LENGTH)}\n\n[Truncated by proxy: original ${originalLen} chars]`
            truncatedCount++
            initialPayloadSize = JSON.stringify(payload).length
          }
        }
      }
    }
    if (truncatedCount > 0) {
      console.log(`[KiroPayload] Truncated ${truncatedCount} large tool results to fit payload size limit (final size: ${initialPayloadSize} bytes)`)
    }
  }

  // ====== 第三阶段：按 token 预算截断超大 tool_result（含「当前轮」）======
  // 关键修复（data-backed）：byte 阈值(1.5MB)远高于 Kiro 的 token 限制(~200k)，实测 payload
  // 最大仅 1.27MB 从不触发 byte 截断；且旧逻辑只截断 history、不截断「当前轮」的 tool_result。
  // playwright/MCP 等单个巨型工具结果会原样发出 → CONTENT_LENGTH_EXCEEDS_THRESHOLD(400)
  // → 流式无重试 → 客户端(Claude Code)中断。这里按 token 预算把最大的 tool_result 文本
  // 逐个截断（当前轮 + history 都纳入），直到估算 token 落入预算内。
  {
    const tokenBudget = getEffectiveTokenLimit(trimModelId) // 模型 ctx - buffer（ListAvailableModels 失败时回退 opus 200k → 180k）
    let estTokens = estimatePayloadTokens(payload)
    if (estTokens > tokenBudget) {
      const collectTexts = (msg?: { userInputMessage?: { userInputMessageContext?: { toolResults?: Array<{ content?: Array<{ text?: string }> }> } } }): Array<{ text?: string }> => {
        const out: Array<{ text?: string }> = []
        const trs = msg?.userInputMessage?.userInputMessageContext?.toolResults
        if (trs) for (const tr of trs) for (const c of (tr.content || [])) {
          if (c.text && c.text.length > TOOL_RESULT_TRUNCATE_LENGTH) out.push(c)
        }
        return out
      }
      const candidates: Array<{ text?: string }> = [
        ...collectTexts(payload.conversationState.currentMessage),
        ...(payload.conversationState.history || []).flatMap(m => collectTexts(m as { userInputMessage?: { userInputMessageContext?: { toolResults?: Array<{ content?: Array<{ text?: string }> }> } } }))
      ]
      // 大块优先截断
      candidates.sort((a, b) => (b.text?.length || 0) - (a.text?.length || 0))
      let truncated = 0
      for (const c of candidates) {
        if (estTokens <= tokenBudget) break
        const orig = c.text!.length
        c.text = `${c.text!.slice(0, TOOL_RESULT_TRUNCATE_LENGTH)}\n\n[Truncated by proxy: original ${orig} chars, trimmed to fit ~${tokenBudget} token limit]`
        truncated++
        estTokens = estimatePayloadTokens(payload)
      }
      if (truncated > 0) {
        console.log(`[KiroPayload] Token-budget truncated ${truncated} large tool results → est ${estTokens.toLocaleString()} / ${tokenBudget.toLocaleString()} tokens (prevents CONTENT_LENGTH 400)`)
      }
    }
  }

  // 调试日志
  console.log(`[KiroPayload] Built payload (native history mode):`, {
    contentLength: finalContent.length,
    originalHistoryLength: history.length,
    sanitizedHistoryLength: sanitizedHistory.length,
    toolsCount: tools.length,
    toolResultsCount: toolResults.length,
    hasProfileArn: payload.profileArn !== undefined,
    hasThinking: !!additionalModelRequestFields?.thinking,
    payloadSize: initialPayloadSize
  })

  return payload
}

// conversationId 稳定化：同一会话的多轮请求复用同一个 conversationId
// 策略：sessionHint（由 proxyServer 从 header/body 提取）→ 稳定映射到固定 conversationId
// 无 sessionHint 时用 history fingerprint 兜底
const conversationCache = new Map<string, { id: string; timestamp: number }>()
const CONVERSATION_CACHE_TTL = 2 * 60 * 60 * 1000 // 2 小时
const CONVERSATION_CACHE_MAX = 1000

function resolveConversationId(history: KiroHistoryMessage[], sessionHint?: string): string {
  // sessionHint 已包含 API Key hash 前缀（由 proxyServer 注入），天然隔离不同用户
  const key = sessionHint || fingerprintFromHistory(history)
  if (!key) return uuidv4()

  const now = Date.now()
  const cached = conversationCache.get(key)
  if (cached) {
    cached.timestamp = now
    return cached.id
  }

  // 清理过期缓存
  if (conversationCache.size > CONVERSATION_CACHE_MAX) {
    const cutoff = now - CONVERSATION_CACHE_TTL
    for (const [k, v] of conversationCache) {
      if (v.timestamp < cutoff) conversationCache.delete(k)
    }
  }

  const id = uuidv4()
  conversationCache.set(key, { id, timestamp: now })
  return id
}

function fingerprintFromHistory(history: KiroHistoryMessage[]): string | undefined {
  if (history.length === 0) return undefined
  const fp = history.slice(0, 2).map(msg =>
    `${msg.userInputMessage?.content || ''}|${msg.assistantResponseMessage?.content || ''}`
  ).join('::')
  const crypto = require('crypto')
  return crypto.createHash('sha256').update(fp).digest('hex').slice(0, 32)
}

// 清除所有内存缓存
export function clearAllCaches(): { conversation: number; model: number } {
  const conversationCount = conversationCache.size
  const modelCount = codeWhispererModelCache.size
  conversationCache.clear()
  codeWhispererModelCache.clear()
  return { conversation: conversationCount, model: modelCount }
}

// machineId 稳定生成缓存（用于无绑定 machineId 且 K-Proxy 不可用时的兆底）
const fallbackMachineIds = new Map<string, string>()

function generateStableMachineId(accountId: string): string {
  const cached = fallbackMachineIds.get(accountId)
  if (cached) return cached
  const crypto = require('crypto')
  const hash = crypto.createHash('sha256').update(`kiro-device-${accountId}`).digest('hex')
  fallbackMachineIds.set(accountId, hash)
  return hash
}

// 获取账号绑定的 Machine ID（保证永远不为空）
function getAccountMachineId(accountId: string, accountMachineId?: string): string {
  if (accountMachineId) return accountMachineId
  const kproxyService = getKProxyService()
  if (kproxyService) {
    const deviceId = kproxyService.getDeviceIdForAccount(accountId)
    if (deviceId) return deviceId
  }
  return generateStableMachineId(accountId)
}

// 获取认证方式对应的请求头
function getAuthHeaders(account: ProxyAccount, endpoint: KiroEndpoint): Record<string, string> {
  // 应用身份(application identity)由"端点"决定，而非账号的 authMethod。
  // 关键修复：旧代码对所有 IdC 账号强制使用 Amazon Q CLI 身份(vibe + aws-sdk-rust UA)，
  // 但很多 IdC/Enterprise 订阅只授权 Kiro IDE 应用，CLI 身份会被后端拒绝：
  //   403 "Your subscription does not support this application"
  // 已用真实 token 验证：IDE 身份(spec + KiroIDE UA) 返回 200，CLI 身份返回 403。
  // CLI 身份用于 AmazonQCLI 端点与新的 KiroRuntime(runtime.kiro.dev) 端点（identity:'cli'）。
  const useCliIdentity = endpoint.name === 'AmazonQCLI' || ('identity' in endpoint && endpoint.identity === 'cli')
  const machineId = getAccountMachineId(account.id, account.machineId)
  const agentMode = useCliIdentity ? AGENT_MODE_VIBE : AGENT_MODE_SPEC
  // RPC 端点（runtime.kiro.dev）：AWS JSON-1.0 协议——path '/' + x-amz-target 头 + 专用 content-type。
  // 抓包确认 runtime.kiro.dev 即用此风格，与旧 REST 端点（path 含动词）不同。
  const isRpc = 'rpc' in endpoint && endpoint.rpc === true

  const headers: Record<string, string> = {
    'content-type': isRpc ? 'application/x-amz-json-1.0' : 'application/json',
    'x-amzn-kiro-agent-mode': agentMode,
    'x-amz-user-agent': useCliIdentity ? KIRO_CLI_AMZ_USER_AGENT : getKiroAmzUserAgent(machineId),
    'user-agent': useCliIdentity ? KIRO_CLI_USER_AGENT : getKiroUserAgent(machineId),
    'amz-sdk-invocation-id': uuidv4(),
    'amz-sdk-request': 'attempt=1; max=3',
    'Authorization': `Bearer ${account.accessToken}`
  }
  if (isRpc) {
    // x-amz-target 把目标 operation 放进 header（RPC 风格的核心），抓包确认必带
    headers['x-amz-target'] = endpoint.amzTarget
    headers['x-amzn-codewhisperer-optout'] = 'false'
  }
  return headers
}

// 获取排序后的端点列表（根据首选端点配置）
// 默认 failover 链：KiroRuntime(runtime.kiro.dev) → CodeWhisperer → AmazonQ
// runtime.kiro.dev 为官方 kiro-cli 当前主端点，置于首位；若该账号订阅不支持 CLI 应用
// 返回 403，会自动回退到旧 codewhisperer/q 端点（IDE 身份）。
function getSortedEndpoints(preferredEndpoint?: 'codewhisperer' | 'amazonq' | 'amazonq-cli' | 'kiro-runtime'): KiroEndpoint[] {
  if (!preferredEndpoint) return KIRO_ENDPOINTS.filter(ep => ep.name !== 'AmazonQCLI')

  // AmazonQ CLI 模式：只用这一个端点，失败不回退
  if (preferredEndpoint === 'amazonq-cli') {
    return KIRO_ENDPOINTS.filter(ep => ep.name === 'AmazonQCLI')
  }

  const preferredName = preferredEndpoint === 'kiro-runtime' ? 'KiroRuntime'
    : preferredEndpoint === 'codewhisperer' ? 'CodeWhisperer'
    : 'AmazonQ'

  const sorted = KIRO_ENDPOINTS.filter(ep => ep.name !== 'AmazonQCLI')
  sorted.sort((a, b) => {
    if (a.name === preferredName) return -1
    if (b.name === preferredName) return 1
    return 0
  })

  return sorted
}

function getAbortError(signal?: AbortSignal): Error {
  if (signal?.reason instanceof Error) return signal.reason
  if (signal?.reason) return new Error(String(signal.reason))
  return new Error('Request aborted')
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw getAbortError(signal)
}

/**
 * 判断是否为「连接阶段」的瞬时网络错误（值得在同一 endpoint 重试一次，而非立刻 failover）。
 * 这些错误发生在 fetch 建连/发送阶段，此时 stream 尚未向客户端写出任何字节，重试是安全的
 * （parseEventStream 一旦开始就只经 onError 回调、绝不向上抛，故 callKiroApiStream 的 catch
 * 只会捕获 stream 开始前的错误）。涵盖 undici 的 socket 提前关闭 / EPIPE / 连接重置 / DNS 抖动。
 */
export function isTransientNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const codes = ['UND_ERR_SOCKET', 'ECONNRESET', 'EPIPE', 'ECONNREFUSED', 'ETIMEDOUT', 'EAI_AGAIN', 'ENOTFOUND', 'UND_ERR_CONNECT_TIMEOUT']
  const cause = (error as { cause?: { code?: string; message?: string } }).cause
  const code = cause?.code
  if (code && codes.includes(code)) return true
  const hay = `${error.message} ${cause?.message || ''}`
  return /other side closed|socket hang up|terminated|fetch failed|network|ECONNRESET|EPIPE|UND_ERR_SOCKET/i.test(hay)
}

// 调用 Kiro API（流式）
export async function callKiroApiStream(
  account: ProxyAccount,
  payload: KiroPayload,
  onChunk: (text: string, toolUse?: KiroToolUse, isThinking?: boolean, reasoningSignature?: string, redactedContent?: string) => void,
  onComplete: (usage: KiroUsage) => void,
  onError: (error: Error, partialUsage?: KiroUsage) => void,
  signal?: AbortSignal,
  preferredEndpoint?: 'codewhisperer' | 'amazonq' | 'amazonq-cli' | 'kiro-runtime',
  // 客户端原始 model（含能力后缀如 "[1m]"）。payload 里的 modelId 已被 mapModelId 剥掉后缀
  // 并映射成 Kiro 后端规范名，无法还原客户端声明的 1M 上下文，故单独透传给 contextUsageEvent
  // 反推使用——保证「分母」与客户端算 autocompact 阈值时一致（见 getModelContextLength）。
  clientModelId?: string
): Promise<void> {
  const endpoints = getSortedEndpoints(preferredEndpoint)
  let lastError: Error | null = null

  for (const endpoint of endpoints) {
    try {
      throwIfAborted(signal)
      const requestPayload = clonePayload(payload)
      // 流式端点对 BuilderId 占位符 ARN 返回 403，仅传真实 ARN 或 Social ARN
      const resolvedArn = resolveProfileArn(account)
      if (resolvedArn && !isPlaceholderProfileArn(resolvedArn)) {
        requestPayload.profileArn = resolvedArn
      } else {
        delete requestPayload.profileArn
      }
      const requestedModelId = getPayloadModelId(requestPayload)
      // CodeWhisperer 与 KiroRuntime 端点都需把别名（claude-sonnet-4.5 等）解析为后端真实 modelId。
      // 注：runtime.kiro.dev 经 ListAvailableModels 返回的就是 claude-sonnet-4.5 这类点号规范名，
      // 但 CodeWhisperer 旧端点要 CLAUDE_SONNET_4_..._V1_0 格式——保持各自既有解析逻辑。
      if (endpoint.name === 'CodeWhisperer') {
        applyPayloadModelId(requestPayload, await resolveCodeWhispererModelId(account, requestedModelId, signal))
      }

      applyPayloadOrigin(requestPayload, endpoint.origin)

      // AmazonQCLI 端点不支持 agentContinuationId/agentTaskType
      if (endpoint.name === 'AmazonQCLI') {
        delete (requestPayload.conversationState as unknown as Record<string, unknown>).agentContinuationId
        delete (requestPayload.conversationState as unknown as Record<string, unknown>).agentTaskType
      }

      const endpointUrl = resolveEndpointUrl(endpoint, account)
      const payloadStr = JSON.stringify(requestPayload)
      const headers = getAuthHeaders(account, endpoint)
      const currentUserInput = requestPayload.conversationState.currentMessage.userInputMessage
      const historyMessages = requestPayload.conversationState.history ?? []
      // 单条结构化 debug 日志取代原先 10 行 console.log。
      // 原实现每个请求都打印 10 行 raw console.log，而 console.log 已被 logger 拦截
      //（见 logger.ts）——每行都要 redact + buildEntry + 写 proxyLogStore，24/7 下是持续的同步开销。
      // 改为单条 debug：仅在 DEBUG 级别真正需要时记录，且重活（toolUse/toolResult 统计）只在记录时才算。
      proxyLogger.debug('KiroAPI', `Request to ${endpoint.name}`, {
        contentLength: currentUserInput?.content?.length || 0,
        toolsCount: currentUserInput?.userInputMessageContext?.tools?.length || 0,
        currentToolResults: currentUserInput?.userInputMessageContext?.toolResults?.length || 0,
        historyMessages: historyMessages.length,
        historyToolUses: historyMessages.reduce((c, m) => c + (m.assistantResponseMessage?.toolUses?.length ?? 0), 0),
        historyToolResults: historyMessages.reduce((c, m) => c + (m.userInputMessage?.userInputMessageContext?.toolResults?.length ?? 0), 0),
        modelId: currentUserInput?.modelId || 'default',
        hasProfileArn: requestPayload.profileArn !== undefined,
        agentMode: headers['x-amzn-kiro-agent-mode'],
        payloadSize: payloadStr.length
      })

      const agent = getNetworkAgent(account)
      if (agent) proxyLogger.debug('KiroAPI', `Stream request via proxy to ${endpoint.name}`)
      // 同 endpoint 瞬时网络错误重试：Kiro 偶尔在建连阶段提前关 socket（UND_ERR_SOCKET / EPIPE）。
      // 立刻 failover 到下一个 endpoint 往往切到 format 不同的 CodeWhisperer 或根本没有备用端点；
      // 对「连接阶段」的瞬时错误先就地重试 1 次（短 backoff）更稳。仅在 stream 尚未开始时安全（见上）。
      let response: Response | undefined
      const MAX_CONNECT_ATTEMPTS = 2
      for (let attempt = 1; attempt <= MAX_CONNECT_ATTEMPTS; attempt++) {
        try {
          response = agent
            ? await undiciFetch(endpointUrl, { method: 'POST', headers, body: payloadStr, signal, dispatcher: agent } as UndiciRequestInit) as unknown as Response
            : await undiciFetch(endpointUrl, { method: 'POST', headers, body: payloadStr, signal } as UndiciRequestInit) as unknown as Response
          break
        } catch (fetchErr) {
          if (signal?.aborted) throw fetchErr
          if (attempt < MAX_CONNECT_ATTEMPTS && isTransientNetworkError(fetchErr)) {
            proxyLogger.warn('KiroAPI', `Endpoint ${endpoint.name} transient connect error (attempt ${attempt}/${MAX_CONNECT_ATTEMPTS}), retrying same endpoint: ${formatError(fetchErr)}`)
            await new Promise(r => setTimeout(r, 300 * attempt))
            throwIfAborted(signal)
            continue
          }
          throw fetchErr
        }
      }
      if (!response) throw lastError || new Error(`Endpoint ${endpoint.name} produced no response`)

      if (response.status === 429) {
        console.log(`[KiroAPI] Endpoint ${endpoint.name} quota exhausted, trying next...`)
        lastError = new Error(`Quota exhausted on ${endpoint.name}`)
        continue
      }

      if (response.status === 401 || response.status === 403) {
        throwIfAborted(signal)
        const body = await response.text()
        throwIfAborted(signal)
        throw new Error(`Auth error ${response.status}: ${body}`)
      }

      if (!response.ok) {
        throwIfAborted(signal)
        const body = await response.text()
        throwIfAborted(signal)
        throw new Error(`API error ${response.status}: ${body}`)
      }

      // 解析 Event Stream
      // 关键：不再把 payloadStr 透传给 parseEventStream 做 tiktoken——payload 里可能含
      // 数 MB 的 base64 附件，对其同步 encode 会阻塞 event loop 让反代卡死。
      // 改为在此用 binary-aware 估算先算好 input token 兜底值（base64 按解码字节折算）。
      const bootstrapInputTokens = estimatePayloadTokens(requestPayload)
      // 反推 inputTokens 用的 model：优先用客户端原始 model（含 [1m] 等能力后缀），
      // 缺失时回退到 payload 里的后端 modelId。后缀决定 context 分母，必须与客户端一致。
      await parseEventStream(response.body!, onChunk, onComplete, onError, bootstrapInputTokens, signal, clientModelId || requestedModelId)
      return
    } catch (error) {
      if (signal?.aborted) {
        onError(getAbortError(signal))
        return
      }
      lastError = error as Error
      console.error(`[KiroAPI] Endpoint ${endpoint.name} failed: ${formatError(error)}`)
      
      // 如果是认证错误，不继续尝试其他端点
      if ((error as Error).message.includes('Auth error')) {
        onError(error as Error)
        return
      }
    }
  }

  if (lastError) {
    onError(lastError)
  }
}

// 从 headers 中提取 event type
function extractEventType(headers: Uint8Array): string {
  let offset = 0
  while (offset < headers.length) {
    if (offset >= headers.length) break
    const nameLen = headers[offset]
    offset++
    if (offset + nameLen > headers.length) break
    const name = new TextDecoder().decode(headers.slice(offset, offset + nameLen))
    offset += nameLen
    if (offset >= headers.length) break
    const valueType = headers[offset]
    offset++
    
    if (valueType === 7) { // String type
      if (offset + 2 > headers.length) break
      const valueLen = (headers[offset] << 8) | headers[offset + 1]
      offset += 2
      if (offset + valueLen > headers.length) break
      const value = new TextDecoder().decode(headers.slice(offset, offset + valueLen))
      offset += valueLen
      if (name === ':event-type') {
        return value
      }
      continue
    }
    
    // Skip other value types
    const skipSizes: Record<number, number> = { 0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16 }
    if (valueType === 6) {
      if (offset + 2 > headers.length) break
      const len = (headers[offset] << 8) | headers[offset + 1]
      offset += 2 + len
    } else if (skipSizes[valueType] !== undefined) {
      offset += skipSizes[valueType]
    } else {
      break
    }
  }
  return ''
}

// Tool Use 状态跟踪
interface ToolUseState {
  toolUseId: string
  name: string
  inputBuffer: string
}

// Token 估算（被 promptCacheTracker 等模块使用，用于 cache 块大小判定）
// 优先使用 tiktoken cl100k_base 精确计算（±5%），失败时自动降级到字符系数（±15%）
export function estimateTokens(text: string): number {
  return countTokens(text)
}

// 解析 AWS Event Stream 二进制格式
async function parseEventStream(
  body: ReadableStream<Uint8Array>,
  onChunk: (text: string, toolUse?: KiroToolUse, isThinking?: boolean, reasoningSignature?: string, redactedContent?: string) => void,
  onComplete: (usage: KiroUsage) => void,
  onError: (error: Error, partialUsage?: KiroUsage) => void,
  bootstrapInputTokens: number = 0,  // 调用方预先用 binary-aware 估算的 input token 兜底值
  signal?: AbortSignal,
  modelId?: string         // 模型 ID，用于 contextUsagePercentage 反推 inputTokens
): Promise<void> {
  const reader = body.getReader()
  const abort = () => {
    reader.cancel(getAbortError(signal)).catch(() => undefined)
  }
  let buffer = new Uint8Array(0)
  let bufStart = 0   // 读取游标：已消费数据的边界，避免每条消息 slice 整个 buffer（O(n²)→O(n)）
  let usage = { 
    inputTokens: 0, 
    outputTokens: 0, 
    credits: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    reasoningTokens: 0
  }
  
  // 累积输出文本长度，用于估算 tokens
  let totalOutputChars = 0
  // 累积输出文本内容，用于 tiktoken 精确计算 output tokens
  let collectedOutputText = ''
  // 是否已拿到 Kiro 真实 tokenUsage（最高优先级，锁定后不再被 contextUsage/tiktoken 覆盖）
  let hasRealTokenUsage = false
  
  // 流式事件聚合计数（logStreamEvents 开启时，结束后输出摘要而非逐条输出）
  const streamEventCounts: Record<string, number> = {}
  
  // 初始化 input tokens 估算（优先级链路：tokenUsage > contextUsage 反推 > 调用方 binary-aware 兜底）
  // 这里只是兜底初值，后续真实事件会覆盖。
  // 注意：绝不在此对 payload 跑 tiktoken——payload 可能含数 MB base64 附件，
  // 同步 encode 会阻塞事件循环导致反代卡死。调用方已用 estimatePayloadTokens 算好兜底值。
  if (bootstrapInputTokens > 0) {
    usage.inputTokens = bootstrapInputTokens
  }
  
  // Tool use 状态跟踪 - 用于累积输入片段
  let currentToolUse: ToolUseState | null = null
  const processedIds = new Set<string>()

  try {
    throwIfAborted(signal)
    signal?.addEventListener('abort', abort, { once: true })
    while (true) {
      throwIfAborted(signal)
      const { done, value } = await reader.read()
      throwIfAborted(signal)
      
      if (done) {
        break
      }

      // 合并缓冲区（O(n) 累积，避免 O(n²)）
      // bufStart 是已消费数据的读取游标：解析完一条消息只前移游标，不 slice 整个 buffer。
      // 收到新 chunk 时，先把「游标之后的未消费残片」搬到头部再追加新数据——
      // 这样每个字节最多被搬运一次，整段响应总成本是 O(n)，而非旧实现「每 chunk/每消息都
      // 重新分配并拷贝整个 buffer」的 O(n²)（长响应会明显拖慢主进程、卡住 UI）。
      const unconsumed = buffer.length - bufStart
      const newBuffer = new Uint8Array(unconsumed + value.length)
      if (unconsumed > 0) newBuffer.set(buffer.subarray(bufStart), 0)
      newBuffer.set(value, unconsumed)
      buffer = newBuffer
      bufStart = 0

      // 尝试解析消息（所有偏移均相对 bufStart，不再 slice 推进）
      while (buffer.length - bufStart >= 16) {
        // AWS Event Stream 格式：
        // - 4 bytes: total length
        // - 4 bytes: headers length
        // - 4 bytes: prelude CRC
        // - headers
        // - payload
        // - 4 bytes: message CRC

        const view = new DataView(buffer.buffer, buffer.byteOffset + bufStart)
        const totalLength = view.getUint32(0, false)

        if (buffer.length - bufStart < totalLength) {
          break // 等待更多数据
        }

        const headersLength = view.getUint32(4, false)

        // 从 headers 中提取 event type（偏移相对 bufStart）
        const headersStart = bufStart + 12
        const headersEnd = bufStart + 12 + headersLength
        const eventType = extractEventType(buffer.slice(headersStart, headersEnd))

        // 提取 payload（偏移相对 bufStart）
        const payloadStart = bufStart + 12 + headersLength
        const payloadEnd = bufStart + totalLength - 4 // 减去 message CRC

        if (payloadStart < payloadEnd) {
          const payloadBytes = buffer.slice(payloadStart, payloadEnd)
          
          try {
            const payloadText = new TextDecoder().decode(payloadBytes)
            const event = JSON.parse(payloadText)
            
            // 根据 event type 处理不同类型的事件
            if (eventType === 'assistantResponseEvent' || event.assistantResponseEvent) {
              const assistantResp = event.assistantResponseEvent || event
              const content = assistantResp.content
              if (content) {
                onChunk(content)
                // 累积输出字符长度（兜底估算用）
                totalOutputChars += content.length
                // 累积输出文本（tiktoken 精确计算用）
                collectedOutputText += content
              }
            }

            // AmazonQ CLI 协议特有：CodeEvent (代码片段流式输出)
            // 来自 amzn_qdeveloper_streaming_client 的 ChatResponseStream::CodeEvent { content: String }
            // CodeWhisperer/AmazonQ 端点用 AssistantResponseEvent 包代码，CLI 端点单独用 CodeEvent
            if (eventType === 'codeEvent' || event.codeEvent) {
              const codeResp = event.codeEvent || event
              const content = codeResp.content
              if (content) {
                onChunk(content)
                totalOutputChars += content.length
                collectedOutputText += content
              }
            }
            
            if (eventType === 'toolUseEvent' || event.toolUseEvent) {
              const toolUseData = event.toolUseEvent || event
              const toolUseId = toolUseData.toolUseId
              const toolName = toolUseData.name
              const isStop = toolUseData.stop === true
              
              // 获取输入 - 可能是字符串片段或完整对象
              let inputFragment = ''
              let inputObj: Record<string, unknown> | null = null
              if (typeof toolUseData.input === 'string') {
                inputFragment = toolUseData.input
              } else if (typeof toolUseData.input === 'object' && toolUseData.input !== null) {
                inputObj = toolUseData.input
              }
              
              // 新的 tool use 开始
              if (toolUseId && toolName) {
                if (currentToolUse && currentToolUse.toolUseId !== toolUseId) {
                  // 前一个 tool use 被中断，完成它
                  if (!processedIds.has(currentToolUse.toolUseId)) {
                    let finalInput: Record<string, unknown> = {}
                    try {
                      if (currentToolUse.inputBuffer) {
                        finalInput = JSON.parse(currentToolUse.inputBuffer)
                      }
                    } catch { /* 忽略解析错误 */ }
                    onChunk('', {
                      toolUseId: currentToolUse.toolUseId,
                      name: currentToolUse.name,
                      input: finalInput
                    })
                    totalOutputChars += currentToolUse.name.length + currentToolUse.inputBuffer.length
                    processedIds.add(currentToolUse.toolUseId)
                  }
                  currentToolUse = null
                }
                
                if (!currentToolUse) {
                  if (processedIds.has(toolUseId)) {
                    // 跳过重复的 tool use
                  } else {
                    currentToolUse = {
                      toolUseId,
                      name: toolName,
                      inputBuffer: ''
                    }
                  }
                }
              }
              
              // 累积输入片段
              if (currentToolUse && inputFragment) {
                currentToolUse.inputBuffer += inputFragment
              }
              
              // 如果直接提供了完整输入对象
              if (currentToolUse && inputObj) {
                currentToolUse.inputBuffer = JSON.stringify(inputObj)
              }
              
              // Tool use 完成
              if (isStop && currentToolUse) {
                let finalInput: Record<string, unknown> = {}
                let parseError = false
                try {
                  if (currentToolUse.inputBuffer) {
                    if (logStreamEvents) proxyLogger.debug('Kiro', 'Tool input buffer: ' + currentToolUse.inputBuffer.substring(0, 200))
                    finalInput = JSON.parse(currentToolUse.inputBuffer)
                    if (logStreamEvents) proxyLogger.debug('Kiro', 'Parsed tool input: ' + JSON.stringify(finalInput).substring(0, 200))
                  }
                } catch (e) {
                  parseError = true
                  console.error('[Kiro] Failed to parse tool input:', e, 'Buffer:', currentToolUse.inputBuffer?.substring(0, 100))
                  // 当 JSON 解析失败时，创建一个包含错误信息的 input
                  // 这样客户端可以看到工具调用失败的原因
                  finalInput = {
                    _error: 'Tool input truncated by Kiro API (output token limit exceeded)',
                    _partialInput: currentToolUse.inputBuffer?.substring(0, 500) || ''
                  }
                }
                
                // 只有在成功解析或有错误信息时才发送
                onChunk('', {
                  toolUseId: currentToolUse.toolUseId,
                  name: currentToolUse.name,
                  input: finalInput
                })
                totalOutputChars += currentToolUse.name.length + currentToolUse.inputBuffer.length
                
                // 如果解析失败，额外发送一条文本消息告知用户
                if (parseError) {
                  onChunk(`\n\n⚠️ Tool "${currentToolUse.name}" input was truncated by Kiro API. The output may be incomplete due to token limits.`)
                }
                
                processedIds.add(currentToolUse.toolUseId)
                currentToolUse = null
              }
            }
            
            // 处理 messageMetadataEvent - 包含 token 使用量
            if (eventType === 'messageMetadataEvent' || eventType === 'metadataEvent' || event.messageMetadataEvent || event.metadataEvent) {
              const metadata = event.messageMetadataEvent || event.metadataEvent || event
              proxyLogger.info('Kiro', 'messageMetadataEvent', metadata)
              
              // 检查 tokenUsage 对象
              if (metadata.tokenUsage) {
                const tokenUsage = metadata.tokenUsage
                proxyLogger.info('Kiro', 'tokenUsage', tokenUsage)
                // 计算 inputTokens = uncachedInputTokens + cacheReadInputTokens + cacheWriteInputTokens
                const uncached = tokenUsage.uncachedInputTokens || 0
                const cacheRead = tokenUsage.cacheReadInputTokens || 0
                const cacheWrite = tokenUsage.cacheWriteInputTokens || 0
                const calculatedInput = uncached + cacheRead + cacheWrite
                
                if (calculatedInput > 0) {
                  usage.inputTokens = calculatedInput
                  hasRealTokenUsage = true  // 真实值，锁定不再被 contextUsage/tiktoken 覆盖
                }
                if (tokenUsage.outputTokens) usage.outputTokens = tokenUsage.outputTokens
                if (tokenUsage.totalTokens) {
                  // 如果有 totalTokens，用它来推算
                  if (usage.inputTokens === 0 && usage.outputTokens > 0) {
                    usage.inputTokens = tokenUsage.totalTokens - usage.outputTokens
                    hasRealTokenUsage = true
                  }
                }
                
                // 保存 cache tokens
                usage.cacheReadTokens = cacheRead
                usage.cacheWriteTokens = cacheWrite
                
                // 记录上下文使用百分比
                if (tokenUsage.contextUsagePercentage !== undefined) {
                  proxyLogger.info('Kiro', 'Context usage: ' + tokenUsage.contextUsagePercentage.toFixed(2) + '%')
                }
                
                // 详细的 token 分解日志
                proxyLogger.info('Kiro', 'Token breakdown', {
                  uncached,
                  cacheRead,
                  cacheWrite,
                  inputTotal: calculatedInput,
                  output: tokenUsage.outputTokens || 0,
                  total: tokenUsage.totalTokens || 0,
                  contextUsage: tokenUsage.contextUsagePercentage ? `${tokenUsage.contextUsagePercentage.toFixed(2)}%` : 'N/A'
                })
              }
              
              // 直接在 metadata 中的 tokens
              if (metadata.inputTokens) {
                usage.inputTokens = metadata.inputTokens
                hasRealTokenUsage = true
              }
              if (metadata.outputTokens) usage.outputTokens = metadata.outputTokens
            }
            
            if (logStreamEvents) {
              // 聚合流式事件（不逐条输出，在 onComplete 时输出摘要）
              streamEventCounts[eventType || 'unknown'] = (streamEventCounts[eventType || 'unknown'] || 0) + 1
            }
            
            // 处理 usageEvent
            if (eventType === 'usageEvent' || eventType === 'usage' || event.usageEvent || event.usage) {
              const usageData = event.usageEvent || event.usage || event
              if (usageData.inputTokens) {
                usage.inputTokens = usageData.inputTokens
                hasRealTokenUsage = true
              }
              if (usageData.outputTokens) usage.outputTokens = usageData.outputTokens
            }
            
            // 处理 meteringEvent - Kiro API 返回 credit 使用量
            if (eventType === 'meteringEvent' || event.meteringEvent) {
              const metering = event.meteringEvent || event
              if (metering.usage && typeof metering.usage === 'number') {
                // 累加 credit 使用量
                usage.credits += metering.usage
                proxyLogger.info('Kiro', `meteringEvent - credit: ${metering.usage}, total: ${usage.credits}`)
              }
            }
            
            // 处理 supplementaryWebLinksEvent - 网页链接引用
            if (eventType === 'supplementaryWebLinksEvent' || event.supplementaryWebLinksEvent) {
              const webLinksEvent = event.supplementaryWebLinksEvent || event
              if (webLinksEvent.supplementaryWebLinks && Array.isArray(webLinksEvent.supplementaryWebLinks)) {
                // 格式化网页链接引用
                const links = webLinksEvent.supplementaryWebLinks
                  .filter((link: { url?: string; title?: string; snippet?: string }) => link.url)
                  .map((link: { url?: string; title?: string; snippet?: string }) => {
                    const title = link.title || link.url
                    return `- [${title}](${link.url})`
                  })
                if (links.length > 0) {
                  onChunk(`\n\n🔗 **Web References:**\n${links.join('\n')}`)
                }
              }
              proxyLogger.debug('Kiro', 'supplementaryWebLinksEvent', JSON.stringify(webLinksEvent).slice(0, 300))
            }
            
            // 处理 contextUsageEvent - 上下文使用百分比（反推真实 inputTokens）
            if (eventType === 'contextUsageEvent' || event.contextUsageEvent) {
              const contextEvent = event.contextUsageEvent || event
              if (contextEvent.contextUsagePercentage !== undefined) {
                const percentage = contextEvent.contextUsagePercentage
                // 若已拿到真实 tokenUsage，仅记录百分比，不覆盖 inputTokens
                if (hasRealTokenUsage) {
                  proxyLogger.info('Kiro', `contextUsageEvent - Context usage: ${percentage.toFixed(2)}% (real tokenUsage already received)`)
                } else {
                  // 反推真实 inputTokens：modelContext × percentage / 100
                  const contextLen = getModelContextLength(modelId)
                  const reverseInput = Math.round(contextLen * percentage / 100)
                  if (reverseInput > 0) {
                    usage.inputTokens = reverseInput
                    proxyLogger.info('Kiro', `contextUsageEvent ${percentage.toFixed(2)}% → inputTokens=${reverseInput} (modelContext=${contextLen}, model=${modelId || 'unknown'})`)
                  } else {
                    proxyLogger.info('Kiro', `contextUsageEvent - Context usage: ${percentage.toFixed(2)}%`)
                  }
                }
                // 如果上下文使用率超过 80%，发送警告
                if (percentage > 80) {
                  console.warn('[Kiro] Warning: Context usage is high:', percentage.toFixed(2) + '%')
                }
              }
            }
            
            // 处理 reasoningContentEvent - Thinking 模式的推理内容
            // Kiro ReasoningContentEvent 字段：[text, redactedContent, signature]
            if (eventType === 'reasoningContentEvent' || event.reasoningContentEvent) {
              const reasoning = event.reasoningContentEvent || event
              if (reasoning.text) {
                proxyLogger.info('Kiro', `Received reasoning content (isThinking=true): ${reasoning.text.slice(0, 50)}...`)
                onChunk(reasoning.text, undefined, true, reasoning.signature, undefined)
                totalOutputChars += reasoning.text.length
                usage.reasoningTokens += Math.max(1, Math.round(reasoning.text.length * 0.4))
              } else if (reasoning.signature && !reasoning.redactedContent) {
                onChunk('', undefined, true, reasoning.signature, undefined)
              }
              // 处理 redactedContent（重编辑的加密 thinking 内容）
              if (reasoning.redactedContent) {
                proxyLogger.info('Kiro', `Received redacted thinking content (len=${reasoning.redactedContent.length})`)
                onChunk('', undefined, true, undefined, reasoning.redactedContent)
              }
              proxyLogger.debug('Kiro', 'reasoningContentEvent', JSON.stringify(reasoning).slice(0, 200))
            }
            
            // 处理 codeReferenceEvent - 代码引用/许可证信息
            if (eventType === 'codeReferenceEvent' || event.codeReferenceEvent) {
              const codeRef = event.codeReferenceEvent || event
              if (codeRef.references && Array.isArray(codeRef.references)) {
                // 格式化代码引用信息
                const refTexts = codeRef.references
                  .filter((ref: { licenseName?: string; repository?: string; url?: string }) => ref.licenseName || ref.repository)
                  .map((ref: { licenseName?: string; repository?: string; url?: string }) => {
                    const parts: string[] = []
                    if (ref.licenseName) parts.push(`License: ${ref.licenseName}`)
                    if (ref.repository) parts.push(`Repo: ${ref.repository}`)
                    if (ref.url) parts.push(`URL: ${ref.url}`)
                    return parts.join(', ')
                  })
                if (refTexts.length > 0) {
                  onChunk(`\n\n📚 **Code References:**\n${refTexts.join('\n')}`)
                }
              }
              proxyLogger.debug('Kiro', 'codeReferenceEvent', JSON.stringify(codeRef).slice(0, 300))
            }
            
            // 处理 followupPromptEvent - 后续提示建议
            if (eventType === 'followupPromptEvent' || event.followupPromptEvent) {
              const followup = event.followupPromptEvent || event
              if (followup.followupPrompt) {
                const prompt = followup.followupPrompt
                if (prompt.content || prompt.userIntent) {
                  // 将后续提示作为建议输出
                  const suggestion = prompt.content || prompt.userIntent
                  onChunk(`\n\n💡 **Suggested follow-up:** ${suggestion}`)
                }
              }
              proxyLogger.debug('Kiro', 'followupPromptEvent', JSON.stringify(followup).slice(0, 200))
            }
            
            // 处理 intentsEvent - 意图事件（artifact、deeplinks 等）
            if (eventType === 'intentsEvent' || event.intentsEvent) {
              const intents = event.intentsEvent || event
              // 意图事件主要用于 UI 渲染，记录日志即可
              proxyLogger.debug('Kiro', 'intentsEvent', JSON.stringify(intents).slice(0, 300))
            }
            
            // 处理 interactionComponentsEvent - 交互组件事件
            if (eventType === 'interactionComponentsEvent' || event.interactionComponentsEvent) {
              const components = event.interactionComponentsEvent || event
              // 交互组件主要用于 UI 渲染，记录日志即可
              proxyLogger.debug('Kiro', 'interactionComponentsEvent', JSON.stringify(components).slice(0, 300))
            }
            
            // 处理 invalidStateEvent - 无效状态事件（错误处理）
            if (eventType === 'invalidStateEvent' || event.invalidStateEvent) {
              const invalid = event.invalidStateEvent || event
              const reason = invalid.reason || 'UNKNOWN'
              const message = invalid.message || 'Invalid state detected'
              console.error('[Kiro] invalidStateEvent:', reason, message)
              // 将无效状态作为错误消息输出
              onChunk(`\n\n⚠️ **Warning:** ${message} (reason: ${reason})`)
            }
            
            // 处理 citationEvent - 引用事件
            if (eventType === 'citationEvent' || event.citationEvent) {
              const citation = event.citationEvent || event
              if (citation.citations && Array.isArray(citation.citations)) {
                // 格式化引用信息
                const citationTexts = citation.citations
                  .filter((c: { title?: string; url?: string; content?: string }) => c.title || c.url)
                  .map((c: { title?: string; url?: string; content?: string }, i: number) => {
                    const parts = [`[${i + 1}]`]
                    if (c.title) parts.push(c.title)
                    if (c.url) parts.push(`(${c.url})`)
                    return parts.join(' ')
                  })
                if (citationTexts.length > 0) {
                  onChunk(`\n\n📖 **Citations:**\n${citationTexts.join('\n')}`)
                }
              }
              proxyLogger.debug('Kiro', 'citationEvent', JSON.stringify(citation).slice(0, 300))
            }
            
            // 检查错误
            if (event._type || event.error) {
              const errMsg = event.message || event.error?.message || 'Unknown stream error'
              throw new Error(errMsg)
            }
          } catch (parseError) {
            if (parseError instanceof SyntaxError) {
              // JSON 解析错误，忽略
              console.debug('[EventStream] JSON parse error:', parseError)
            } else {
              throw parseError
            }
          }
        }
        
        // 移动到下一条消息：只前移读取游标（O(1)），不再 slice 拷贝剩余 buffer
        bufStart += totalLength
      }
    }
    
    // 完成任何未完成的 tool use
    if (currentToolUse && !processedIds.has(currentToolUse.toolUseId)) {
      let finalInput: Record<string, unknown> = {}
      try {
        if (currentToolUse.inputBuffer) {
          finalInput = JSON.parse(currentToolUse.inputBuffer)
        }
      } catch { /* 忽略解析错误 */ }
      onChunk('', {
        toolUseId: currentToolUse.toolUseId,
        name: currentToolUse.name,
        input: finalInput
      })
      totalOutputChars += currentToolUse.name.length + currentToolUse.inputBuffer.length
    }
    
    // 如果 API 没有返回 token 信息，优先用 tiktoken 精确计算，兜底字符系数
    if (usage.outputTokens === 0 && totalOutputChars > 0) {
      if (collectedOutputText) {
        // tiktoken cl100k_base 精确计算（±5%）
        usage.outputTokens = Math.max(1, countTokens(collectedOutputText))
        proxyLogger.info('Kiro', `Estimated output tokens (tiktoken): ${totalOutputChars} chars -> ${usage.outputTokens} tokens`)
      } else {
        // 字符系数兜底（自然语言中英混合约 0.4 token/字符）
        usage.outputTokens = Math.max(1, Math.round(totalOutputChars * 0.4))
        proxyLogger.info('Kiro', `Estimated output tokens (fallback): ${totalOutputChars} chars -> ${usage.outputTokens} tokens`)
      }
    }
    
    // 流式事件聚合摘要
    if (logStreamEvents && Object.keys(streamEventCounts).length > 0) {
      const total = Object.values(streamEventCounts).reduce((a, b) => a + b, 0)
      proxyLogger.debug('Kiro', `Stream events summary (${total} total)`, streamEventCounts)
    }
    
    throwIfAborted(signal)
    proxyLogger.info('Kiro', 'Stream complete, final usage', usage)
    onComplete(usage)
  } catch (error) {
    // 把已累计的 usage 透传给 onError：流中途断开/报错时，Kiro 可能已发过 meteringEvent
    // （credits 已产生）。上层据此对已消耗用量结算计费，避免「中途断 = 白嫖」的漏计费。
    // 注意：onComplete 与此 catch 互斥（onComplete 成功则不进 catch），故不会与正常结算重复计费。
    onError(signal?.aborted ? getAbortError(signal) : error as Error, usage)
  } finally {
    signal?.removeEventListener('abort', abort)
    reader.releaseLock()
  }
}

// 非流式调用（等待完整响应）
export async function callKiroApi(
  account: ProxyAccount,
  payload: KiroPayload,
  signal?: AbortSignal
): Promise<{
  content: string
  toolUses: KiroToolUse[]
  usage: KiroUsage
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
}> {
  return new Promise((resolve, reject) => {
    let content = ''
    let reasoningText = ''
    let reasoningSignature: string | undefined
    let redactedContent = ''
    const toolUses: KiroToolUse[] = []
    let usage: KiroUsage = { inputTokens: 0, outputTokens: 0, credits: 0 }

    callKiroApiStream(
      account,
      payload,
      (text, toolUse, isThinking, signature, redacted) => {
        if (isThinking) {
          if (text) reasoningText += text
          if (signature) reasoningSignature = signature
          if (redacted) redactedContent += redacted
        } else {
          content += text
        }
        if (toolUse) {
          toolUses.push(toolUse)
        }
      },
      (u) => {
        usage = u
        if (reasoningText || redactedContent) {
          const rc: { text?: string; signature?: string; redactedContent?: string } = {}
          if (reasoningText) rc.text = reasoningText
          if (reasoningSignature) rc.signature = reasoningSignature
          if (redactedContent) rc.redactedContent = redactedContent
          resolve({ content, toolUses, usage, reasoningContent: rc })
          return
        }
        resolve({ content, toolUses, usage })
      },
      reject,
      signal
    ).catch(reject)
  })
}

// ============================================================================
// Web 工具 agentic 循环
//
// 当模型调用 web_search/web_fetch 时，由代理执行并把结果喂回 Kiro，循环到模型
// 不再调用 web 工具为止。客户端 custom 工具（非 web）原样返回给上层，由客户端执行。
//
// 返回最终聚合结果：content + 非 web 的 toolUses（需透传给客户端）+ 累计 usage。
// ============================================================================
const DEFAULT_MAX_WEB_TOOL_ROUNDS = 2

// 一次 web 工具调用的结构化记录，用于回传给客户端（server_tool_use + web_search_tool_result）
export interface WebToolSearchRecord {
  kind: 'web_search' | 'web_fetch'
  toolUseId: string
  input: Record<string, unknown>
  sources: Array<{ title: string; url: string; pageAge?: string }>
  isError: boolean
}

export interface WebToolLoopResult {
  content: string
  toolUses: KiroToolUse[]
  usage: KiroUsage
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
  webToolRounds: number
  // 本轮所有 web 工具调用的结构化记录（按执行顺序），handler 据此生成原生 content block
  searches: WebToolSearchRecord[]
}

// 把一次 Kiro 响应（assistant）+ 对应的 web 工具结果（user）追加进 history
function appendWebToolTurn(
  payload: KiroPayload,
  assistantContent: string,
  webToolUses: KiroToolUse[],
  toolResults: KiroToolResult[]
): void {
  payload.conversationState.history = payload.conversationState.history || []

  // 关键修复：先把"上一轮的 currentMessage"（用户原始问题，或上一轮的 web toolResults）
  // 落入 history，再追加本轮 assistant。否则 currentMessage 会被直接覆盖，导致：
  //   1) 用户原始问题 / 上一轮 toolResults 丢失；
  //   2) 上一轮 assistant 的 toolUse 失去配对的 toolResult → Bedrock 报
  //      TOOL_USE_RESULT_MISMATCH（"Expected toolResult blocks at messages.N.content"）。
  // 仅在第 2 轮（含）以上的 web 工具调用时触发，因此表现为"多次搜索才偶发 400"。
  const prevCurrent = payload.conversationState.currentMessage
  if (prevCurrent?.userInputMessage) {
    // tools 无需随 history 重复携带（新 currentMessage 会重新附带），去掉以免每轮膨胀
    const prevContext = prevCurrent.userInputMessage.userInputMessageContext
    let historyContext = prevContext
    if (prevContext && 'tools' in prevContext) {
      historyContext = { ...prevContext }
      delete (historyContext as Record<string, unknown>).tools
      if (Object.keys(historyContext).length === 0) historyContext = undefined
    }
    payload.conversationState.history.push({
      userInputMessage: {
        ...prevCurrent.userInputMessage,
        userInputMessageContext: historyContext
      }
    })
  }

  payload.conversationState.history.push({
    assistantResponseMessage: {
      content: assistantContent || '',
      toolUses: webToolUses
    }
  })
  // 把工具结果放入"当前消息"——下一轮请求就以它作为最新输入
  const modelId = getPayloadModelId(payload)
  // 关键：携带原始 tools 声明。Bedrock 要求只要消息含 toolUse/toolResult，
  // 请求就必须带 toolConfig（即 tools）；否则报 "toolConfig field must be defined"。
  const prevTools = prevCurrent?.userInputMessage?.userInputMessageContext?.tools
  payload.conversationState.currentMessage = {
    userInputMessage: {
      content: '',
      modelId,
      origin: 'AI_EDITOR',
      userInputMessageContext: {
        toolResults,
        ...(prevTools && prevTools.length > 0 ? { tools: prevTools } : {})
      }
    }
  }
}

// callKiroApi 的签名类型，供依赖注入（测试时可替换为 mock）
type KiroCaller = (account: ProxyAccount, payload: KiroPayload, signal?: AbortSignal) => Promise<{
  content: string
  toolUses: KiroToolUse[]
  usage: KiroUsage
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
}>

export async function runWebToolLoop(
  account: ProxyAccount,
  initialPayload: KiroPayload,
  webConfig: WebToolConfig,
  signal?: AbortSignal,
  kiroCaller: KiroCaller = callKiroApi
): Promise<WebToolLoopResult> {
  const maxRounds = webConfig.maxRounds ?? DEFAULT_MAX_WEB_TOOL_ROUNDS
  const payload = clonePayload(initialPayload)
  let aggUsage: KiroUsage = { inputTokens: 0, outputTokens: 0, credits: 0 }
  let webToolRounds = 0
  const searches: WebToolSearchRecord[] = []

  for (let round = 0; round <= maxRounds; round++) {
    if (signal?.aborted) throw new Error('aborted')
    const result = await kiroCaller(account, payload, signal)
    // 累计 usage（多轮 round-trip 都计入）
    aggUsage = {
      inputTokens: aggUsage.inputTokens + (result.usage.inputTokens || 0),
      outputTokens: aggUsage.outputTokens + (result.usage.outputTokens || 0),
      credits: (aggUsage.credits || 0) + (result.usage.credits || 0),
      cacheReadTokens: (aggUsage.cacheReadTokens || 0) + (result.usage.cacheReadTokens || 0),
      cacheWriteTokens: (aggUsage.cacheWriteTokens || 0) + (result.usage.cacheWriteTokens || 0),
      reasoningTokens: (aggUsage.reasoningTokens || 0) + (result.usage.reasoningTokens || 0)
    }

    // 分离 web 工具调用 与 需要透传客户端的工具调用
    const webCalls = result.toolUses.filter(tu => isServerWebTool(tu.name))
    const passthroughCalls = result.toolUses.filter(tu => !isServerWebTool(tu.name))

    // 没有 web 工具调用 → 循环结束，把结果（含透传 toolUses）交回上层
    if (webCalls.length === 0 || round === maxRounds) {
      if (webCalls.length > 0 && round === maxRounds) {
        proxyLogger.warn('WebToolLoop', `Hit max rounds (${maxRounds}); returning without executing remaining web tools`)
      }
      return { content: result.content, toolUses: passthroughCalls, usage: aggUsage, reasoningContent: result.reasoningContent, webToolRounds, searches }
    }

    // 执行所有 web 工具调用，收集 tool_result（同时记录结构化结果供回传客户端）
    webToolRounds++
    const toolResults: KiroToolResult[] = []
    for (const call of webCalls) {
      const kind = isServerWebTool(call.name)!
      proxyLogger.info('WebToolLoop', `Round ${round + 1}: executing ${kind} (${JSON.stringify(call.input).slice(0, 120)})`)
      const exec = await executeWebToolStructured(kind, call.input, webConfig, signal)
      toolResults.push({ content: [{ text: exec.text }], status: exec.isError ? 'error' : 'success', toolUseId: call.toolUseId })
      searches.push({ kind, toolUseId: call.toolUseId, input: call.input, sources: exec.sources, isError: exec.isError })
    }

    // 把本轮 assistant + 工具结果写入 history，准备下一轮
    appendWebToolTurn(payload, result.content, webCalls, toolResults)
  }

  // 理论不可达（循环内已 return）
  return { content: '', toolUses: [], usage: aggUsage, webToolRounds, searches }
}

// Kiro 官方模型信息
export interface KiroModel {
  modelId: string
  modelName: string
  description: string
  modelProvider?: string | null
  rateMultiplier?: number
  rateUnit?: string
  status?: string | null
  supportedInputTypes?: string[]
  tokenLimits?: {
    maxInputTokens?: number | null
    maxOutputTokens?: number | null
  }
  promptCaching?: {
    supportsPromptCaching: boolean
    maximumCacheCheckpointsPerRequest?: number | null
    minimumTokensPerCacheCheckpoint?: number | null
  } | null
  additionalModelRequestFieldsSchema?: Record<string, unknown> | null
  availableOrigins?: string[] | null
}

// 根据账号区域获取 Q Service 端点（官方插件使用 q.{region}.amazonaws.com）
function getQServiceEndpoint(region?: string): string {
  if (region?.startsWith('eu-')) return 'https://q.eu-central-1.amazonaws.com'
  return 'https://q.us-east-1.amazonaws.com'
}

// 获取 Kiro 官方模型列表（支持分页，与官方插件一致传递 profileArn）
export async function fetchKiroModels(account: ProxyAccount, signal?: AbortSignal): Promise<KiroModel[]> {
  const baseUrl = getQServiceEndpoint(account.region)
  const machineId = getAccountMachineId(account.id, account.machineId)
  
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${account.accessToken}`,
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': getKiroUserAgent(machineId),
    'x-amz-user-agent': getKiroAmzUserAgent(machineId),
    'x-amzn-codewhisperer-optout': 'true'
  }

  const allModels: KiroModel[] = []
  let nextToken: string | undefined

  try {
    do {
      const params = new URLSearchParams({ origin: 'AI_EDITOR', maxResults: '50' })
      const modelsProfileArn = resolveProfileArnForRead(account)
      params.set('profileArn', modelsProfileArn)
      if (nextToken) params.set('nextToken', nextToken)

      const url = `${baseUrl}/ListAvailableModels?${params.toString()}`
      throwIfAborted(signal)
      const response = await fetchWithProxy(url, { method: 'GET', headers, signal }, account)
      throwIfAborted(signal)
      
      if (!response.ok) {
        const body = await response.text().catch(() => '')
        console.error(`[KiroAPI] ListAvailableModels failed: ${response.status} ${body.slice(0, 300)}`)
        // 401/403：token 被撤销/轮换。抛出可识别的 Auth error，让上层 refresh + 重试一次
        // （否则静默 return [] 会让 ctx-window 缓存填不上 → opus 误判 200K，且 /v1/models 变空）。
        if (response.status === 401 || response.status === 403) {
          throw new Error(`Auth error ${response.status}: ${body.slice(0, 200)}`)
        }
        break
      }

      const data = await response.json()
      allModels.push(...(data.models || []))
      nextToken = data.nextToken
    } while (nextToken)

    return allModels
  } catch (error) {
    if (signal?.aborted) throw getAbortError(signal)
    // Auth error 需上抛给调用方做 refresh + 重试（见 ProxyServer.fetchKiroModelsWithRefresh）
    if (error instanceof Error && error.message.startsWith('Auth error')) throw error
    console.error('[KiroAPI] ListAvailableModels error:', error)
    return allModels.length > 0 ? allModels : []
  }
}

// 订阅计划信息
export interface SubscriptionPlan {
  name: string  // KIRO_FREE, KIRO_PRO, KIRO_PRO_PLUS, KIRO_POWER
  qSubscriptionType: string
  description: {
    title: string
    billingInterval: string
    featureHeader: string
    features: string[]
  }
  pricing: {
    amount: number
    currency: string
  }
}

// 订阅列表响应
export interface SubscriptionListResponse {
  disclaimer?: string[]
  subscriptionPlans?: SubscriptionPlan[]
}

// 订阅请求专用 User-Agent（匹配 Kiro IDE 实际报文格式）
const KIRO_SUBSCRIPTION_VERSION = '0.12.155'

function getSubscriptionUserAgent(machineId?: string): string {
  const suffix = machineId ? `KiroIDE-${KIRO_SUBSCRIPTION_VERSION}-${machineId}` : `KiroIDE-${KIRO_SUBSCRIPTION_VERSION}`
  return `aws-sdk-js/1.0.0 ua/2.1 os/win32#10.0.19043 lang/js md/nodejs#22.22.0 api/codewhispererruntime#1.0.0 m/N,E ${suffix}`
}

function getSubscriptionAmzUserAgent(machineId?: string): string {
  const suffix = machineId ? `KiroIDE-${KIRO_SUBSCRIPTION_VERSION}-${machineId}` : `KiroIDE-${KIRO_SUBSCRIPTION_VERSION}`
  return `aws-sdk-js/1.0.0 ${suffix}`
}

// 获取可用订阅列表
export async function fetchAvailableSubscriptions(account: ProxyAccount): Promise<SubscriptionListResponse> {
  const baseUrl = getQServiceEndpoint(account.region)
  const url = `${baseUrl}/listAvailableSubscriptions`
  const machineId = getAccountMachineId(account.id, account.machineId)
  
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${account.accessToken}`,
    'content-type': 'application/json',
    'user-agent': getSubscriptionUserAgent(machineId),
    'x-amz-user-agent': getSubscriptionAmzUserAgent(machineId),
    'amz-sdk-invocation-id': uuidv4(),
    'amz-sdk-request': 'attempt=1; max=1'
  }

  const profileArn = resolveProfileArnForRead(account)
  const body = JSON.stringify({ profileArn })

  console.log(`[KiroAPI] ListAvailableSubscriptions [${account.email || account.id.slice(0, 8)}]`, {
    url,
    hasProfileArn: profileArn !== undefined
  })

  try {
    const response = await fetchWithProxy(url, { method: 'POST', headers, body }, account)
    const responseText = await response.text()
    console.log(`[KiroAPI] ListAvailableSubscriptions → ${response.status}`, JSON.parse(responseText))
    
    if (!response.ok) {
      return {}
    }

    return JSON.parse(responseText)
  } catch (error) {
    console.error('[KiroAPI] ListAvailableSubscriptions error:', error)
    return {}
  }
}

// 订阅 Token 响应
export interface SubscriptionTokenResponse {
  encodedVerificationUrl?: string
  status?: string
  token?: string | null
  message?: string
}

// 获取订阅管理/支付链接
export async function fetchSubscriptionToken(
  account: ProxyAccount,
  subscriptionType?: string
): Promise<SubscriptionTokenResponse> {
  const baseUrl = getQServiceEndpoint(account.region)
  const url = `${baseUrl}/CreateSubscriptionToken`
  const machineId = getAccountMachineId(account.id, account.machineId)
  
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${account.accessToken}`,
    'content-type': 'application/json',
    'user-agent': getSubscriptionUserAgent(machineId),
    'x-amz-user-agent': getSubscriptionAmzUserAgent(machineId),
    'amz-sdk-invocation-id': uuidv4(),
    'amz-sdk-request': 'attempt=1; max=1'
  }

  const profileArn = resolveProfileArnForRead(account)

  // clientToken 是必需参数；profileArn 仅在解析出有效值时附带
  const payload: Record<string, string> = {
    clientToken: uuidv4(),
    provider: 'STRIPE'
  }
  if (profileArn) {
    payload.profileArn = profileArn
  }
  if (subscriptionType) {
    payload.subscriptionType = subscriptionType
  }

  try {
    const response = await fetchWithProxy(url, { method: 'POST', headers, body: JSON.stringify(payload) }, account)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error('[KiroAPI] CreateSubscriptionToken failed:', response.status, errorData)
      return { message: errorData.message || `Request failed with status ${response.status}` }
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('[KiroAPI] CreateSubscriptionToken error:', error)
    return { message: error instanceof Error ? error.message : 'Unknown error' }
  }
}

// 设置用户偏好（超额开启/关闭）
export async function setUserPreference(
  account: ProxyAccount,
  overageStatus: 'ENABLED' | 'DISABLED'
): Promise<{ success: boolean; error?: string }> {
  const baseUrl = getQServiceEndpoint(account.region)
  const url = `${baseUrl}/setUserPreference`
  const machineId = getAccountMachineId(account.id, account.machineId)

  const headers: Record<string, string> = {
    'Authorization': `Bearer ${account.accessToken}`,
    'content-type': 'application/json',
    'user-agent': getSubscriptionUserAgent(machineId),
    'x-amz-user-agent': getSubscriptionAmzUserAgent(machineId),
    'amz-sdk-invocation-id': uuidv4(),
    'amz-sdk-request': 'attempt=1; max=1'
  }

  const profileArn = resolveProfileArnForRead(account)
  const body = JSON.stringify({
    overageConfiguration: { overageStatus },
    profileArn
  })

  try {
    const response = await fetchWithProxy(url, { method: 'POST', headers, body }, account)
    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      return { success: false, error: `HTTP ${response.status}: ${errorText.substring(0, 200)}` }
    }
    return { success: true }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}
