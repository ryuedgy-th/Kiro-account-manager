// Kiro Proxy HTTP/HTTPS 服务器
import http from 'http'
import https from 'https'
import fs from 'fs'
import crypto from 'crypto'
import { v4 as uuidv4 } from 'uuid'
import type { Socket } from 'net'
import type {
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIContentPart,
  OpenAIResponsesRequest,
  ClaudeRequest,
  ClaudeContentBlock,
  ClaudeCacheControl,
  ClaudeStreamEvent,
  ProxyConfig,
  ProxyStats,
  ProxyAccount,
  TokenRefreshCallback
} from './types'
import { AccountPool, ErrorType, classifyError } from './accountPool'
import { callKiroApiStream, callKiroApi, runWebToolLoop, fetchKiroModels, setModelContextWindow, canonicalizeModelId, isTransientNetworkError, type KiroModel, type WebToolSearchRecord } from './kiroApi'
import { proxyLogger, formatError } from './logger'
import { getKProxyService, generateDeviceId } from '../kproxy'
import {
  openaiToKiro,
  claudeToKiro,
  kiroToOpenaiResponse,
  kiroToClaudeResponse,
  createOpenaiStreamChunk,
  createClaudeStreamEvent,
  responsesToOpenAIChat,
  openAIChatToResponsesResponse,
  setModelThinkingCapability,
  clearModelThinkingCapabilities,
  deriveClaudeEffort
} from './translator'
import { ToolNameRegistry } from './toolNameRegistry'
import { promptCacheTracker } from './promptCacheTracker'
import { estimateBase64DocumentTokens, IMAGE_TOKEN_ESTIMATE } from './tokenCounter'
import { isServerWebTool, type WebToolConfig } from './webTools'
import { perfDiag, PerfEvent, PerfPhase } from './perfDiag'
import * as portal from './portal'
import type { Customer, CustomerView, SlipTopupRecord } from './types'
import { fetch as undiciFetch, FormData as UndiciFormData, type Dispatcher } from 'undici'
import { getSystemProxy, safeCreateProxyAgent } from './systemProxy'
// Self-contained single-page templates (vanilla JS, no external deps), inlined at
// build time via Vite's `?raw`. Served verbatim by handlePortalPage/handleAdminPage.
//   portal.html — customer portal: login → balance/usage/rates → API-key mgmt (/portal/* JSON API)
//   admin.html  — operator dashboard behind Cloudflare Access; operator key → Bearer → /admin/*
import PORTAL_HTML from './templates/portal.html?raw'
import ADMIN_HTML from './templates/admin.html?raw'


// 把代理侧执行的 web 工具记录，转换为 Anthropic 原生 content block 序列：
// 每次调用产出一对 [server_tool_use, web_search_tool_result]，顺序与执行顺序一致。
// 客户端（Claude Code）据此统计搜索次数并渲染可点击来源。
function buildWebSearchContentBlocks(searches: WebToolSearchRecord[]): ClaudeContentBlock[] {
  const blocks: ClaudeContentBlock[] = []
  for (const s of searches) {
    // 1) server_tool_use：模型发起的查询调用（名义上仍叫 web_search，与客户端声明对齐）
    blocks.push({
      type: 'server_tool_use',
      id: s.toolUseId,
      name: s.kind, // 'web_search' | 'web_fetch'
      input: s.input
    })
    // 2) web_search_tool_result：搜索结果（或错误）
    if (s.isError) {
      blocks.push({
        type: 'web_search_tool_result',
        tool_use_id: s.toolUseId,
        content: { type: 'web_search_tool_result_error', error_code: 'unavailable' }
      })
    } else {
      blocks.push({
        type: 'web_search_tool_result',
        tool_use_id: s.toolUseId,
        content: s.sources.map(src => ({
          type: 'web_search_result' as const,
          url: src.url,
          title: src.title,
          ...(src.pageAge ? { page_age: src.pageAge } : {})
        }))
      })
    }
  }
  return blocks
}


export interface ProxyServerEvents {
  onRequest?: (info: { path: string; method: string; accountId?: string }) => void
  onResponse?: (info: { path: string; model?: string; status: number; tokens?: number; inputTokens?: number; outputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number; reasoningTokens?: number; credits?: number; responseTime?: number; error?: string; sessionId?: string }) => void
  onError?: (error: Error) => void
  onConfigChanged?: (config: ProxyConfig) => void  // API Key 用量更新时触发
  onStatusChange?: (running: boolean, port: number) => void
  onTokenRefresh?: TokenRefreshCallback
  onAccountUpdate?: (account: ProxyAccount) => void
  // 账号被 Kiro 后端长期封禁（如 TEMPORARILY_SUSPENDED / AccountSuspendedException）
  // 不同于临时 token 失效，需人工解封
  onAccountSuspended?: (info: { accountId: string; email?: string; reason: string; message: string }) => void
  onCreditsUpdate?: (totalCredits: number) => void
  onTokensUpdate?: (inputTokens: number, outputTokens: number) => void
  onRequestStatsUpdate?: (totalRequests: number, successRequests: number, failedRequests: number) => void
  onPoolEmpty?: () => Promise<void> // 账号池为空时触发（冷启动懒加载）
}

type ModelModality = 'text' | 'audio' | 'image' | 'video' | 'pdf'

type ClientModel = {
  id: string
  object: 'model'
  created: number
  owned_by: string
  name: string
  description: string
  model_name?: string
  family: string
  release_date: string
  attachment: boolean
  reasoning: boolean
  temperature: boolean
  tool_call: boolean
  interleaved: boolean | { field: 'reasoning_content' }
  cost: { input: number; output: number; cache_read: number; cache_write: number }
  limit: { context: number; input?: number; output: number }
  modalities: { input: ModelModality[]; output: ModelModality[] }
  capabilities: {
    temperature: boolean
    reasoning: boolean
    attachment: boolean
    toolcall: boolean
    input: Record<ModelModality, boolean>
    output: Record<ModelModality, boolean>
    interleaved: boolean | { field: 'reasoning_content' }
  }
  context_length: number
  max_tokens: number
  max_input_tokens?: number
  max_output_tokens: number
  inputTypes?: string[]
  rateMultiplier?: number
  rateUnit?: string
  supportsThinking?: boolean
  thinkingEfforts?: string[]
  supportsPromptCaching?: boolean
  modelProvider?: string
  permission: unknown[]
  root: string
  parent: null
}

// normalizeCreditsLimit 的哨兵返回值：表示客户传入的 creditsLimit 非法（非正数 / 非数字），调用方据此回 400。
const INVALID_LIMIT = Symbol('invalid-credits-limit')

// 从 modelId 解析 Claude 家族与版本号（仅匹配规范名 claude-{family}-{major}[.-{minor}]）。
// minor 限定 1~2 位且其后非数字，避免把日期快照（claude-sonnet-4-20250514）误读成 minor=20250514。
// 返回 versionLabel 用于显示（"4.8" / "4"），version 为数值用于比较取最新。
function parseClaudeFamilyVersion(id: string): { family: string; version: number; versionLabel: string } | null {
  const m = id.toLowerCase().match(/^claude-(opus|sonnet|haiku)-(\d+)(?:[.-](\d{1,2})(?=$|[^\d]))?/)
  if (!m) return null
  const major = m[2]
  const minor = m[3]
  const versionLabel = minor !== undefined ? `${major}.${minor}` : major
  return { family: m[1], version: parseFloat(versionLabel), versionLabel }
}

function modelDisplayName(id: string, modelName?: string): string {
  // Claude 规范模型：始终用 modelId 里的真实版本号生成显示名（"Opus 4.8"），
  // 不信任后端 modelName——它常把多个版本统一写成 "Opus 4"，导致下拉里无法区分版本。
  const cv = parseClaudeFamilyVersion(id)
  if (cv) {
    return `${cv.family.charAt(0).toUpperCase()}${cv.family.slice(1)} ${cv.versionLabel}`
  }
  if (modelName?.trim()) return modelName
  return id
    .split('-')
    .filter(Boolean)
    .map(part => part === 'gpt' ? 'GPT' : part === 'ai' ? 'AI' : part[0]?.toUpperCase() + part.slice(1))
    .join(' ')
}

// effort 变体后缀枚举（如 claude-opus-4.8-max 的 "max"）。取 Claude 模型 thinking effort 的并集
// （opus=[low,medium,high,xhigh,max]、sonnet=[low,medium,high,max]）。仅用于「解析客户端选中的变体 ID」；
// 生成 /v1/models 变体条目时按各模型真实枚举（见 expandEffortVariants），不依赖此常量。
const EFFORT_VARIANT_SUFFIXES = ['low', 'medium', 'high', 'xhigh', 'max']

// 从 model ID 拆出 effort 变体后缀，返回 { baseId, effort? }。
// 仅当 base 是规范 Claude 模型（parseClaudeFamilyVersion 命中）时才拆——避免把恰好以 -max 等结尾的
// 第三方模型名误拆。纯函数，不读 config：是否启用由调用方（resolveEffortVariant）判定。
// 注：suffix 必须是 effort 词；版本号短横形式（claude-opus-4-8 的 "8"）不是 effort 词 → 不会被误拆。
function splitEffortSuffix(modelId: string): { baseId: string; effort?: string } {
  const m = modelId.match(/^(.+)-(low|medium|high|xhigh|max)$/i)
  if (!m) return { baseId: modelId }
  const base = m[1]
  if (!parseClaudeFamilyVersion(base)) return { baseId: modelId }
  return { baseId: base, effort: m[2].toLowerCase() }
}


// 这些仍可经 /v1/messages 直接按 ID 调用，只是不进 picker / 费率表（避免列表冗杂、版本无法区分）。
function isInternalOrLegacyModelId(id: string): boolean {
  const lower = id.toLowerCase()
  return /^[A-Z0-9_]+$/.test(id)
    || lower === 'simple-task'
    || lower.startsWith('claude-3')
}

// 把完整模型列表精简为「面向客户端 picker / 费率表」的列表：
//   - 同一 Claude 家族（opus/sonnet/haiku）只保留版本号最高的一个；
//   - 剔除内部/历史 ID；
//   - 非 Claude（auto / gpt 别名 / deepseek 等）原样保留。
// 供 /v1/models（Cowork/Claude Code 下拉）与 /portal/rates（客户费率表）共用，保证两处口径一致。
function filterPickerModels<T extends { id: string }>(models: T[]): T[] {
  const latestByFamily = new Map<string, { version: number; id: string }>()
  for (const m of models) {
    const cv = parseClaudeFamilyVersion(m.id)
    if (!cv) continue
    const cur = latestByFamily.get(cv.family)
    if (!cur || cv.version > cur.version) latestByFamily.set(cv.family, { version: cv.version, id: m.id })
  }
  return models.filter(m => {
    if (isInternalOrLegacyModelId(m.id)) return false
    const cv = parseClaudeFamilyVersion(m.id)
    if (cv) return latestByFamily.get(cv.family)?.id === m.id
    return true
  }).filter((m, i, arr) => arr.findIndex(x => x.id === m.id) === i) // 去重（按 id 保留首个），供 /portal/rates 直接 map 时不出现重复行
}

function modelFamily(id: string): string {
  const lower = id.toLowerCase()
  if (lower.includes('opus')) return 'claude-opus'
  if (lower.includes('sonnet')) return 'claude-sonnet'
  if (lower.includes('haiku')) return 'claude-haiku'
  if (lower.includes('gpt-4o')) return 'gpt-4o'
  if (lower.includes('gpt-4')) return 'gpt-4'
  if (lower.includes('gpt-3.5')) return 'gpt-3.5'
  if (lower.includes('glm')) return 'glm'
  if (lower === 'auto') return 'auto'
  return lower.split(/[.-]/).slice(0, 2).join('-') || lower
}

function modelOutputLimit(id: string, output?: number | null): number {
  if (typeof output === 'number' && output > 0) return output
  const lower = id.toLowerCase()
  if (lower.includes('haiku') || lower.includes('gpt-3.5')) return 8192
  return 32000
}

function modelInputModalities(inputTypes?: string[]): ModelModality[] {
  const values = new Set<ModelModality>(['text'])
  for (const item of inputTypes ?? []) {
    const lower = item.toLowerCase()
    if (lower.includes('image')) values.add('image')
    if (lower.includes('pdf') || lower.includes('document') || lower.includes('file')) values.add('pdf')
    if (lower.includes('audio')) values.add('audio')
    if (lower.includes('video')) values.add('video')
  }
  return Array.from(values)
}

function modelCapabilityMap(modalities: ModelModality[]): Record<ModelModality, boolean> {
  return {
    text: modalities.includes('text'),
    audio: modalities.includes('audio'),
    image: modalities.includes('image'),
    video: modalities.includes('video'),
    pdf: modalities.includes('pdf')
  }
}

function extractThinkingEfforts(schema?: Record<string, unknown> | null): string[] | undefined {
  if (!schema) return undefined
  const props = schema.properties as Record<string, unknown> | undefined
  if (!props?.thinking) return undefined
  const thinking = props.thinking as Record<string, unknown>
  const thinkingProps = thinking.properties as Record<string, unknown> | undefined
  const typeField = thinkingProps?.type as Record<string, unknown> | undefined
  const enumValues = typeField?.enum as string[] | undefined
  if (enumValues?.includes('adaptive') || enumValues?.includes('disabled')) {
    const effortField = (props.output_config as Record<string, unknown> | undefined)?.properties as Record<string, unknown> | undefined
    const effortEnum = (effortField?.effort as Record<string, unknown> | undefined)?.enum as string[] | undefined
    return effortEnum || undefined
  }
  return undefined
}

// 从 Kiro 模型的 additionalModelRequestFieldsSchema 读取该模型是否支持 thinking。
function schemaSupportsThinking(schema?: Record<string, unknown> | null): boolean {
  return !!(schema?.properties as Record<string, unknown> | undefined)?.thinking
}

// 将后端返回的模型真实能力同步进 translator 的能力注册表，
// 供 buildAdditionalModelRequestFields 按真实 schema 决定是否下发 thinking / 哪些 effort 合法。
function syncModelThinkingCapabilities(models: KiroModel[]): void {
  clearModelThinkingCapabilities()
  for (const m of models) {
    if (!m.modelId) continue
    setModelThinkingCapability(m.modelId, {
      supportsThinking: schemaSupportsThinking(m.additionalModelRequestFieldsSchema),
      thinkingEfforts: extractThinkingEfforts(m.additionalModelRequestFieldsSchema) ?? []
    })
  }
}

function buildClientModel(input: {
  id: string
  created: number
  ownedBy: string
  description?: string
  modelName?: string
  supportedInputTypes?: string[]
  maxInputTokens?: number | null
  maxOutputTokens?: number | null
  rateMultiplier?: number
  rateUnit?: string
  promptCaching?: { supportsPromptCaching: boolean; maximumCacheCheckpointsPerRequest?: number | null; minimumTokensPerCacheCheckpoint?: number | null } | null
  additionalModelRequestFieldsSchema?: Record<string, unknown> | null
  modelProvider?: string | null
}): ClientModel {
  const name = modelDisplayName(input.id, input.modelName)
  const inputModalities = modelInputModalities(input.supportedInputTypes)
  const outputModalities: ModelModality[] = ['text']
  const output = modelOutputLimit(input.id, input.maxOutputTokens)
  const context = typeof input.maxInputTokens === 'number' && input.maxInputTokens > 0 ? input.maxInputTokens : 200000
  const reasoning = false
  const interleaved = false

  return {
    id: input.id,
    object: 'model',
    created: input.created,
    owned_by: input.ownedBy,
    name,
    description: input.description || name,
    model_name: input.modelName || name,
    family: modelFamily(input.id),
    release_date: '',
    attachment: inputModalities.some(item => item !== 'text'),
    reasoning,
    temperature: true,
    tool_call: true,
    interleaved,
    cost: { input: 0, output: 0, cache_read: 0, cache_write: 0 },
    limit: {
      context,
      ...(typeof input.maxInputTokens === 'number' && input.maxInputTokens > 0 ? { input: input.maxInputTokens } : {}),
      output
    },
    modalities: { input: inputModalities, output: outputModalities },
    capabilities: {
      temperature: true,
      reasoning,
      attachment: inputModalities.some(item => item !== 'text'),
      toolcall: true,
      input: modelCapabilityMap(inputModalities),
      output: modelCapabilityMap(outputModalities),
      interleaved
    },
    context_length: context,
    max_tokens: output,
    ...(typeof input.maxInputTokens === 'number' && input.maxInputTokens > 0 ? { max_input_tokens: input.maxInputTokens } : {}),
    max_output_tokens: output,
    inputTypes: input.supportedInputTypes,
    rateMultiplier: input.rateMultiplier,
    rateUnit: input.rateUnit,
    supportsThinking: !!(input.additionalModelRequestFieldsSchema?.properties as Record<string, unknown> | undefined)?.thinking,
    thinkingEfforts: extractThinkingEfforts(input.additionalModelRequestFieldsSchema),
    supportsPromptCaching: input.promptCaching?.supportsPromptCaching || false,
    modelProvider: input.modelProvider || undefined,
    permission: [],
    root: input.id,
    parent: null
  }
}

// 请求体超限错误（统一识别用，触发 413 响应）
class BodyTooLargeError extends Error {
  constructor(public readonly received: number, public readonly limit: number) {
    super(`Request body too large: ${received} bytes exceeds limit of ${limit} bytes`)
    this.name = 'BodyTooLargeError'
  }
}

export class ProxyServer {
  private server: http.Server | https.Server | null = null
  private fallbackServer: http.Server | null = null  // HTTPS 启用时同时监听 HTTP（可选）
  private accountPool: AccountPool
  private config: ProxyConfig
  private stats: ProxyStats
  private sessionStats: { totalRequests: number; successRequests: number; failedRequests: number; startTime: number }
  private events: ProxyServerEvents
  private refreshingTokens: Map<string, Promise<boolean>> = new Map() // 同账号并发刷新去重：共享同一个 in-flight promise
  private isHttps: boolean = false
  private isStopping: boolean = false
  private activeRequests: Set<AbortController> = new Set()
  private sockets: Set<Socket> = new Set()
  /** P1-7 按 API Key/IP 的滑动窗口限流（每分钟桶） */
  private rateLimitBuckets: Map<string, { count: number; windowStart: number }> = new Map()
  /** 门户登录按 IP 的失败限流桶（每分钟），独立于业务限流，防暴力破解 */
  private portalLoginBuckets: Map<string, { count: number; windowStart: number }> = new Map()
  /** 客户在途请求计数（信用预留），防并发请求穿透预付余额造成超额消费 */
  private customerInFlight: Map<string, number> = new Map()
  /**
   * slip2go 已入账的银行交易号集合（去重主键）。启动时从 slipTopupRecords(settled) 重建，
   * 入账时同步写入——保证同一笔真实转账只入账一次，且 restart 后仍生效。
   */
  private usedSlipTransRefs: Set<string> = new Set()
  /**
   * 正在验证中的slip指纹（sha256(图片字节)）。check→add 之间无 await，
   * 防同一slip并发提交触发重复 slip2go 调用与重复入账竞态。
   */
  private inFlightSlipKeys: Set<string> = new Set()
  /** slip 提交按客户的限流桶（每分钟 + 每日），独立于业务限流，防刷耗 slip2go 配额 */
  private slipSubmitBuckets: Map<string, { minuteCount: number; minuteStart: number; dayCount: number; dayStart: number }> = new Map()
  /** P1-8 会话粘性：session hint → accountId 的映射（10 分钟 TTL） */
  private sessionAffinity: Map<string, { accountId: string; lastAt: number }> = new Map()
  /** P2-17 审计日志（最近 200 条） */
  private auditLog: Array<{ ts: number; type: string; data: Record<string, unknown> }> = []
  /** Webhook 触发回调（由外部注入，避免 main → renderer 循环依赖） */
  private webhookTrigger?: (event: string, payload: Record<string, unknown>) => void
  /** 定期清理 timer */
  private cleanupTimer: NodeJS.Timeout | null = null
  /** 冷启动预热 ctx-window 缓存的重试 timer（见 warmModelContextCache） */
  private warmCacheTimer: NodeJS.Timeout | null = null

  /**
   * 从请求中提取 session hint，用于稳定 conversationId
   * 优先级 1：显式稳定 ID（header）
   * 优先级 2：请求体中的会话相关字段（body）
   * 优先级 3：返回 undefined（由 kiroApi 用 history fingerprint 兜底）
   */
  static extractSessionHint(req: http.IncomingMessage, body: unknown): string | undefined {
    const b = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>
    const h = req.headers
    // 优先级 1：显式稳定 header
    const headerHint =
      (h['x-claude-code-session-id'] as string) ||
      (h['x-opencode-session'] as string) ||
      (h['x-session-affinity'] as string) ||
      (h['x-conversation-id'] as string)
    if (headerHint) return headerHint

    // 优先级 2：body 中可靠的会话字段
    const bodyHint =
      (b.prompt_cache_key as string) ||
      (b.promptCacheKey as string) ||
      (b.conversation_id as string) ||
      (b.conversationId as string) ||
      (b.thread_id as string) ||
      (b.threadId as string) ||
      (b.session_id as string) ||
      (b.sessionId as string)
    if (bodyHint) return bodyHint

    // 优先级 2.5：metadata 中的 session/conversation
    const metadata = b.metadata as Record<string, unknown> | undefined
    if (metadata) {
      const metaHint =
        (metadata.session_id as string) ||
        (metadata.conversation_id as string)
      if (metaHint) return metaHint
    }

    // 优先级 3：无显式 ID，返回 undefined（kiroApi 用 history fingerprint 兜底）
    return undefined
  }

  /**
   * 把不同客户端的「推理强度」归一化成统一档位，供用量记录/dashboard 展示。
   * 目的：让 OpenAI（reasoning_effort: low/high）与 Claude Code（thinking.budget_tokens）
   * 在同一张表里口径一致（Maxplus 风格的 effort 列）。
   *
   * 取值优先级：
   *   1. 显式 effort 字符串（OpenAI reasoning_effort / Claude output_config.effort）→ 直接采用
   *   2. thinking.type==='disabled' 或缺省 → 'none'
   *   3. 仅有 thinking.budget_tokens → 按 token 预算映射到档位（阈值见下）
   *
   * 返回的档位是展示用文案，不回传给 Kiro 后端（后端枚举校验另由 translator 负责）。
   */
  static deriveEffortLevel(body: unknown): string {
    const b = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>

    // OpenAI / Responses 形态的显式 effort（Claude 形态由 deriveClaudeEffort 处理）
    const openaiEffort =
      (b.reasoning_effort as string | undefined) ||
      ((b.reasoning as Record<string, unknown> | undefined)?.effort as string | undefined)
    if (typeof openaiEffort === 'string' && openaiEffort.trim()) return openaiEffort.trim().toLowerCase()

    // Claude 形态：output_config.effort 显式值 + thinking.budget_tokens 折算
    // 复用 translator.deriveClaudeEffort，确保「下发给 Kiro 的 effort」与「dashboard 显示的档位」口径一致。
    const claudeEffort = deriveClaudeEffort(b as {
      output_config?: { effort?: string }
      thinking?: { type?: string; budget_tokens?: number }
    })
    // dashboard 用 'none' 表示「无推理强度」，translator 用 undefined 表示「不下发字段」——此处归一为 'none'
    return claudeEffort ?? 'none'
  }

  constructor(config: Partial<ProxyConfig> = {}, events: ProxyServerEvents = {}) {
    this.config = {
      enabled: false,
      port: 5580,
      host: '127.0.0.1',
      enableMultiAccount: true,
      selectedAccountIds: [],
      logRequests: true,
      maxConcurrent: 10,
      maxRetries: 3,
      retryDelayMs: 1000,
      tokenRefreshBeforeExpiry: 300, // 5分钟提前刷新
      autoStart: false, // 是否自动启动
      clientDrivenToolExecution: true,
      ...config
    }
    this.accountPool = new AccountPool()
    this.accountPool.setStrategy(this.config.accountSelectionStrategy || 'round-robin')
    this.stats = {
      totalRequests: 0,
      successRequests: 0,
      failedRequests: 0,
      totalTokens: 0,
      totalCredits: 0,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      reasoningTokens: 0,
      startTime: Date.now(),
      accountStats: new Map(),
      endpointStats: new Map(),
      modelStats: new Map(),
      recentRequests: []
    }
    this.sessionStats = {
      totalRequests: 0,
      successRequests: 0,
      failedRequests: 0,
      startTime: 0
    }
    this.events = events
    // 门户已启用但缺签名密钥（如从旧配置恢复）→ 立即生成，避免 /portal/login 503
    if (this.config.portalEnabled && !this.config.portalSessionSecret) {
      this.config.portalSessionSecret = crypto.randomBytes(32).toString('hex')
      this.events.onConfigChanged?.(this.config)
    }
  }

  /**
   * 检测当前绑定地址是否会暴露到本机以外
   * 0.0.0.0 / :: / 网卡地址 → true；127.0.0.1 / ::1 / localhost → false
   */
  private isBindingExternal(host?: string): boolean {
    if (!host) return false
    const h = host.toLowerCase().trim()
    return h === '0.0.0.0' || h === '::' || h === '*' || (
      h !== '127.0.0.1' && h !== '::1' && h !== 'localhost'
    )
  }

  // 启动服务器
  async start(): Promise<void> {
    if (this.server) {
      console.log('[ProxyServer] Server already running')
      return
    }

    // P0-2 安全护栏：外网绑定 + 无 API Key → 拒绝启动（用户可以显式 allowExternalWithoutApiKey 解除）
    if (this.isBindingExternal(this.config.host)) {
      const hasAnyKey = (this.config.apiKeys?.some(k => k.enabled && k.key) ?? false) || !!this.config.apiKey
      if (!hasAnyKey && !this.config.allowExternalWithoutApiKey) {
        const err = new Error(
          `[Security] Refused to start: host=${this.config.host} exposes to network but no API Key configured. ` +
          `Set at least one API Key, or change host to 127.0.0.1, or set allowExternalWithoutApiKey=true (NOT RECOMMENDED).`
        )
        console.error('[ProxyServer]', err.message)
        this.events.onError?.(err)
        throw err
      }
      if (!hasAnyKey) {
        console.warn(`[ProxyServer] [Security] WARNING: binding to ${this.config.host} without API Key (allowExternalWithoutApiKey=true). This exposes your accounts to the network!`)
      }
    }

    // 安全提示：门户启用但未启用 TLS。门户用 Bearer session token + 明文密码登录；若 origin 经
    // 非加密 hop 暴露（裸 HTTP / 某些 tunnel 不加密 edge→origin），token/密码会被中途截获。
    // 推荐：依赖 Cloudflare Tunnel（client→edge 强制 HTTPS）且不要再在 origin 前串接明文 HTTP 跳。
    if (this.config.portalEnabled && !this.config.tls?.enabled) {
      console.warn('[ProxyServer] [Security] Portal enabled over plain HTTP: session tokens & passwords are unencrypted on the wire. Rely on a TLS-terminating tunnel (e.g. Cloudflare) and avoid any extra cleartext HTTP hop.')
    }
    // 安全提示：/admin/* 对外暴露 + 经公网时，管理面（充值/删客户/改配置）仅靠 operator key 一道防线。
    if (this.config.adminApiExposed === true) {
      console.warn('[ProxyServer] [Security] adminApiExposed=true: /admin/* is reachable over HTTP. Ensure a strong operator key and restrict access at the tunnel/proxy layer.')
    }

    return new Promise((resolve, reject) => {
      this.isStopping = false
      const requestHandler = (req: http.IncomingMessage, res: http.ServerResponse) => 
        this.handleRequest(req, res)

      // 检查是否启用 TLS
      if (this.config.tls?.enabled) {
        try {
          const tlsOptions = this.getTlsOptions()
          this.server = https.createServer(tlsOptions, requestHandler)
          this.isHttps = true
        } catch (error) {
          reject(new Error(`TLS configuration error: ${(error as Error).message}`))
          return
        }
      } else {
        this.server = http.createServer(requestHandler)
        this.isHttps = false
      }

      this.server.on('error', (error: NodeJS.ErrnoException) => {
        if (error.code === 'EADDRINUSE') {
          console.error(`[ProxyServer] Port ${this.config.port} is already in use`)
          reject(new Error(`Port ${this.config.port} is already in use`))
        } else {
          console.error('[ProxyServer] Server error:', error)
          reject(error)
        }
        this.events.onError?.(error)
      })

      this.server.on('connection', (socket: Socket) => {
        // 关闭 Nagle 算法：SSE 是大量小 chunk 的流式输出，Nagle 会把小包攒一会儿再发，
        // 导致 token 一顿一顿地到（TTFT 变差、看起来卡）。关掉后每个 delta 立即发出，更顺滑。
        // 这是流式客户端的标准做法，不影响正确性，也不构成异常网络指纹。
        socket.setNoDelay(true)
        this.sockets.add(socket)
        socket.on('close', () => this.sockets.delete(socket))
        // P1-10 backpressure 监控：socket 写入缓冲区超过 1MB 时记录警告
        socket.on('drain', () => {
          if (socket.writableLength > 0) {
            proxyLogger.debug('ProxyServer', `Socket drain: bufferedLen=${socket.writableLength}`)
          }
        })
      })

      // 服务器关闭时尝试自动重启
      this.server.on('close', () => {
        if (!this.isStopping && this.config.autoStart && this.config.enabled) {
          console.log('[ProxyServer] Server closed unexpectedly, attempting restart in 3s...')
          setTimeout(() => {
            if (!this.isStopping && this.config.autoStart && !this.isRunning()) {
              console.log('[ProxyServer] Auto-restarting...')
              this.start().catch(err => {
                console.error('[ProxyServer] Auto-restart failed:', err)
              })
            }
          }, 3000)
        }
      })

      // P1-11 keep-alive / headers 空闲超时（避免长连接占用资源）
      const keepAliveMs = this.config.keepAliveTimeoutMs ?? 65_000
      const headersMs = this.config.headersTimeoutMs ?? 60_000
      this.server.keepAliveTimeout = keepAliveMs
      this.server.headersTimeout = Math.max(headersMs, keepAliveMs + 1000) // headers 必须 > keepAlive，否则 Node 会 warn
      this.server.requestTimeout = 0  // 流式响应可能很长，禁用 request 总超时

      // 启动定期清理（每 5 分钟）
      if (this.cleanupTimer) clearInterval(this.cleanupTimer)
      this.cleanupTimer = setInterval(() => this.cleanupExpiredCaches(), 5 * 60_000)
      // 让 timer 在 Node 退出时不阻塞
      this.cleanupTimer.unref?.()

      // 性能诊断（默认关闭）：开启时启动 event-loop 监视 + 周期汇总。零行为变更。
      if (this.config.perfDiagnostics) perfDiag.start()

      // 从持久化的slip充值流水重建 transRef 去重集合（restart 后仍防重复入账）
      this.rebuildSlipTransRefIndex()

      const protocol = this.isHttps ? 'https' : 'http'
      this.server.listen(this.config.port, this.config.host, () => {
        proxyLogger.info('ProxyServer', `Started on ${protocol}://${this.config.host}:${this.config.port} (keepAlive=${keepAliveMs}ms)`)
        this.stats.startTime = Date.now()
        // 重置会话统计
        this.sessionStats = {
          totalRequests: 0,
          successRequests: 0,
          failedRequests: 0,
          startTime: Date.now()
        }
        this.events.onStatusChange?.(true, this.config.port)
        resolve()
        // 预热 ctx-window 缓存（fire-and-forget，带退避重试）：让首个请求就能拿到正确的
        // context window（如 opus=1,000,000），避免冷窗口内反推 inputTokens 偏低 5 倍。
        this.warmModelContextCache()
      })

      // D4 启用 TLS 时同时监听 HTTP fallback 端口（如果配置了 fallbackPort）
      if (this.isHttps && this.config.fallbackPort && this.config.fallbackPort !== this.config.port) {
        const fallback = http.createServer(requestHandler)
        fallback.keepAliveTimeout = keepAliveMs
        fallback.headersTimeout = Math.max(headersMs, keepAliveMs + 1000)
        fallback.requestTimeout = 0
        fallback.on('connection', (socket) => {
          socket.setNoDelay(true)
          this.sockets.add(socket)
          socket.on('close', () => this.sockets.delete(socket))
        })
        fallback.on('error', (err) => proxyLogger.warn('ProxyServer', `Fallback HTTP error: ${err.message}`))
        fallback.listen(this.config.fallbackPort, this.config.host, () => {
          proxyLogger.info('ProxyServer', `Fallback HTTP listening on http://${this.config.host}:${this.config.fallbackPort}`)
        })
        this.fallbackServer = fallback
      }
    })
  }

  /**
   * 冷启动预热：在 server 开始监听后立即填充 modelContextWindowCache（modelId → maxInputTokens），
   * 供 contextUsageEvent 反推 inputTokens 与 token 裁剪预算使用。
   *
   * 背景（data-backed）：该缓存原本只在首次有人请求 /v1/models 时才由 getAvailableModels()
   * 懒填充。在那之前，opus 等模型走关键词兜底 → 200K，而 Kiro 后端实际对 opus 上报 1,000,000。
   * 于是 contextUsagePercentage 反推出的 inputTokens 偏低 5 倍，Claude Code 永远到不了 autocompact
   * 阈值（"不 compact 直到撑爆"）。预热把这个冷窗口收敛到启动瞬间。
   *
   * 冷启动账号可能尚未同步进 pool（index.ts 有最多 ~10s 的 setAccounts 重试），故这里带退避重试：
   * getAvailableModels() 拿不到账号会返回空且不缓存，重试直到缓存被真正填充或达到上限。
   * fire-and-forget，绝不阻塞 start()。
   */
  private warmModelContextCache(): void {
    if (this.warmCacheTimer) { clearTimeout(this.warmCacheTimer); this.warmCacheTimer = null }
    const maxAttempts = 6           // 1s,2s,4s,8s,16s,32s ≈ 覆盖 index.ts 的账号同步重试窗口
    const attempt = (n: number): void => {
      if (this.isStopping || !this.server) return
      this.getAvailableModels()
        .then(({ models, fromCache }) => {
          if (this.isStopping || !this.server) return
          // models 非空即说明 setModelContextWindow 已被填充（见 getAvailableModels）。
          if (models.length > 0) {
            proxyLogger.info('ProxyServer', `Warmed model context-window cache (${models.length} models${fromCache ? ', from cache' : ''})`)
            return
          }
          if (n < maxAttempts) {
            this.warmCacheTimer = setTimeout(() => attempt(n + 1), Math.min(32_000, 1000 * 2 ** (n - 1)))
            this.warmCacheTimer.unref?.()
          } else {
            proxyLogger.debug('ProxyServer', 'Warm cache: no models after retries; will lazy-fill on first /v1/models hit')
          }
        })
        .catch((err) => {
          if (this.isStopping || !this.server) return
          if (n < maxAttempts) {
            this.warmCacheTimer = setTimeout(() => attempt(n + 1), Math.min(32_000, 1000 * 2 ** (n - 1)))
            this.warmCacheTimer.unref?.()
          } else {
            proxyLogger.debug('ProxyServer', `Warm cache failed after retries: ${err instanceof Error ? err.message : String(err)}`)
          }
        })
    }
    attempt(1)
  }

  // 获取 TLS 配置选项
  // P1-13 当 tls.enabled 但未提供 cert/key 时，自动生成自签证书
  private getTlsOptions(): https.ServerOptions {
    const tls = this.config.tls!
    
    let cert: string
    let key: string

    // 优先使用直接提供的 PEM 内容
    if (tls.cert && tls.key) {
      cert = tls.cert
      key = tls.key
    } else if (tls.certPath && tls.keyPath) {
      // 从文件读取
      cert = fs.readFileSync(tls.certPath, 'utf8')
      key = fs.readFileSync(tls.keyPath, 'utf8')
    } else {
      // 自动生成自签证书（位于 dataDir/proxy-tls/）
      try {
        const { ensureProxySelfSignedCert } = require('./selfSignedCert')
        const hostnames = [this.config.host || '127.0.0.1']
        const result = ensureProxySelfSignedCert(this.getProxyDataDir(), hostnames)
        proxyLogger.info('ProxyServer', `Using self-signed TLS cert (SAN=${result.altNames.join(',')}, fingerprint=${result.fingerprint.slice(0, 19)}...)`)
        cert = result.cert
        key = result.key
      } catch (err) {
        throw new Error(`TLS enabled but no certificate/key provided and auto-generation failed: ${(err as Error).message}`)
      }
    }

    return { cert, key }
  }

  /**
   * 解析数据目录（cert/log 等）：headless 友好。
   * 优先 config.dataDir → 环境变量 KIRO_DATA_DIR → Electron userData（仅 Electron 内可用）→ tmpdir。
   */
  private getProxyDataDir(): string {
    if (this.config.dataDir) return this.config.dataDir
    if (process.env.KIRO_DATA_DIR) return process.env.KIRO_DATA_DIR
    try {
      const { app } = require('electron')
      if (app?.getPath) return app.getPath('userData')
    } catch { /* 非 Electron 环境，忽略 */ }
    return require('path').join(require('os').tmpdir(), 'kiro-proxy')
  }

  /**
   * 获取（或生成）反代自签证书信息（供 UI 显示/导出 PEM）
   */
  getSelfSignedCertInfo(): import('./selfSignedCert').ProxySelfSignedCert | null {
    try {
      const { ensureProxySelfSignedCert } = require('./selfSignedCert')
      return ensureProxySelfSignedCert(this.getProxyDataDir(), [this.config.host || '127.0.0.1'])
    } catch (err) {
      proxyLogger.warn('ProxyServer', `getSelfSignedCertInfo failed: ${(err as Error).message}`)
      return null
    }
  }

  /** 强制重新生成自签证书（用户在 UI 上点"重新生成"） */
  regenerateSelfSignedCert(): import('./selfSignedCert').ProxySelfSignedCert | null {
    try {
      const { ensureProxySelfSignedCert } = require('./selfSignedCert')
      this.appendAuditLog('regenerate_self_signed_cert', { host: this.config.host })
      return ensureProxySelfSignedCert(this.getProxyDataDir(), [this.config.host || '127.0.0.1'], true)
    } catch (err) {
      proxyLogger.warn('ProxyServer', `regenerateSelfSignedCert failed: ${(err as Error).message}`)
      return null
    }
  }

  /**
   * 优雅停止服务器
   * - 立刻拒绝新连接（server.close）
   * - 给正在进行中的请求 5 秒完成；超时后强制 destroy socket
   * - 同时停 fallback HTTP 服务器
   */
  async stop(gracefulMs: number = 5000): Promise<void> {
    if (!this.server) {
      return
    }

    this.isStopping = true

    const main = this.server
    const fallback = this.fallbackServer

    return new Promise((resolve) => {
      let done = false
      const finish = () => {
        if (done) return
        done = true
        proxyLogger.info('ProxyServer', 'Stopped')
        this.server = null
        this.fallbackServer = null
        this.isStopping = false
        this.activeRequests.clear()
        this.sockets.clear()
        if (this.cleanupTimer) { clearInterval(this.cleanupTimer); this.cleanupTimer = null }
        if (this.warmCacheTimer) { clearTimeout(this.warmCacheTimer); this.warmCacheTimer = null }
        perfDiag.stop()
        this.events.onStatusChange?.(false, this.config.port)
        resolve()
      }

      // 先停止接受新连接
      main.close(() => {
        fallback?.close(() => finish()) || finish()
      })
      fallback?.close()

      // P1-14 优雅停止：给正在进行中的请求时间完成，超时再强制
      this.activeRequests.forEach(controller => {
        // 给客户端一个明确的 stop 信号，但不立即中断已发送的响应流
        try { controller.abort(new Error('Proxy server stopped')) } catch { /* ignore */ }
      })

      // 超时强制 destroy
      setTimeout(() => {
        this.sockets.forEach(socket => { try { socket.destroy() } catch { /* ignore */ } })
        finish()
      }, Math.max(0, gracefulMs))
    })
  }

  // 更新配置
  // P2-18 检测到 port/host/tls 变更时，标记 needsRestart=true，UI 可读取并提示
  private _needsRestart = false
  updateConfig(config: Partial<ProxyConfig>): void {
    // 标记需要重启的字段
    const restartTriggerFields: Array<keyof ProxyConfig> = ['port', 'host', 'tls', 'fallbackPort']
    const willRestart = restartTriggerFields.some(k => k in config && JSON.stringify(this.config[k]) !== JSON.stringify(config[k]))
    if (willRestart && this.isRunning()) {
      this._needsRestart = true
      proxyLogger.warn('ProxyServer', `Config change requires restart: ${restartTriggerFields.filter(k => k in config).join(', ')}`)
    }
    this.appendAuditLog('config_changed', { fields: Object.keys(config), needsRestart: willRestart })
    // perfDiagnostics 可热切换（无需重启）：仅当本次确实改变了该值且服务在运行时，立刻 start/stop。
    // start()/stop() 自身幂等；此处比较 old→new 避免无谓地重置已累计的诊断数据。
    if ('perfDiagnostics' in config && this.isRunning()) {
      const wasOn = !!this.config.perfDiagnostics
      const willOn = !!config.perfDiagnostics
      if (willOn && !wasOn) perfDiag.start()
      else if (!willOn && wasOn) perfDiag.stop()
    }
    this.config = { ...this.config, ...config }
    // 门户启用时若未设签名密钥，自动生成一份（持久化靠 onConfigChanged）。
    // 缺少密钥会导致 /portal/login 返回 503，故在此兜底初始化。
    if (this.config.portalEnabled && !this.config.portalSessionSecret) {
      this.config.portalSessionSecret = crypto.randomBytes(32).toString('hex')
      this.events.onConfigChanged?.(this.config)
    }
    // 同步账号选择策略到 accountPool
    if (config.accountSelectionStrategy !== undefined) {
      this.accountPool.setStrategy(this.config.accountSelectionStrategy || 'round-robin')
    }
  }

  /** UI 可用此判断是否需提示用户重启 */
  needsRestart(): boolean {
    return this._needsRestart
  }

  /** 重启后调用清除 needsRestart 标记 */
  async restartServer(): Promise<void> {
    if (!this.isRunning()) {
      await this.start()
      this._needsRestart = false
      return
    }
    await this.stop()
    await this.start()
    this._needsRestart = false
  }

  // 获取配置
  getConfig(): ProxyConfig {
    return { ...this.config }
  }

  private validateCacheControl(cacheControl?: ClaudeCacheControl): void {
    if (!cacheControl) return
    if (cacheControl.type !== 'ephemeral') {
      throw new Error(`Unsupported cache_control type: ${cacheControl.type}`)
    }
  }


  private validateClaudeContentBlocks(blocks: ClaudeContentBlock[]): void {
    blocks.forEach(block => {
      this.validateCacheControl(block.cache_control)
      // 仅对嵌套的 Claude content block 递归校验；web_search_result / error 等 server-tool
      // 专有子结构没有 cache_control，跳过避免类型不匹配。
      if (Array.isArray(block.content) && block.type !== 'web_search_tool_result') {
        this.validateClaudeContentBlocks(block.content as ClaudeContentBlock[])
      }
    })
  }

  private validateOpenAICacheControls(request: OpenAIChatRequest): void {
    request.messages.forEach(message => {
      this.validateCacheControl(message.cache_control)
      if (Array.isArray(message.content)) {
        message.content.forEach(part => this.validateCacheControl(part.cache_control))
      }
    })
    request.tools?.forEach(tool => this.validateCacheControl(tool.cache_control))
  }

  private validateClaudeCacheControls(request: ClaudeRequest): void {
    if (Array.isArray(request.system)) {
      request.system.forEach(block => this.validateCacheControl(block.cache_control))
    }
    request.messages.forEach(message => {
      this.validateCacheControl(message.cache_control)
      if (Array.isArray(message.content)) {
        this.validateClaudeContentBlocks(message.content)
      }
    })
    request.tools?.forEach(tool => this.validateCacheControl(tool.cache_control))
  }

  // 通用 HTTP 附件下载（图片/文档共用）：带超时、代理、content-type 校验与大小上限。
  // 关键：必须有 maxBytes 上限——否则恶意/超大 URL 会把整文件读进内存（内存放大），
  // 且下载是请求处理的一部分，过大附件会拖慢/拖垮服务。
  private async downloadHttpAttachment(
    url: string,
    opts: { allowedTypes?: string[]; maxBytes: number; label: string },
    signal?: AbortSignal
  ): Promise<{ contentType: string; base64: string }> {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 15000)
    const abort = () => controller.abort(this.getAbortError(signal))
    try {
      if (signal?.aborted) throw this.getAbortError(signal)
      signal?.addEventListener('abort', abort, { once: true })
      const agent = (() => {
        const { getSystemProxy, safeCreateProxyAgent } = require('./systemProxy')
        const envProxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy
        const envAgent = safeCreateProxyAgent(envProxy)
        if (envAgent) return envAgent
        return safeCreateProxyAgent(getSystemProxy())
      })()
      const { fetch: undiciFetch } = require('undici')
      const response = agent
        ? await undiciFetch(url, { signal: controller.signal, dispatcher: agent }) as unknown as globalThis.Response
        : await undiciFetch(url, { signal: controller.signal }) as unknown as globalThis.Response
      if (!response.ok) {
        throw new Error(`Failed to download ${opts.label.toLowerCase()}: HTTP ${response.status}`)
      }
      const contentType = response.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase() || ''
      if (opts.allowedTypes && (!contentType || !opts.allowedTypes.includes(contentType))) {
        throw new Error(`Unsupported ${opts.label.toLowerCase()} content-type: ${contentType || 'unknown'}`)
      }
      // 提前用 Content-Length 拒绝超大文件（部分服务器会提供）
      const declared = parseInt(response.headers.get('content-length') || '0', 10)
      if (Number.isFinite(declared) && declared > opts.maxBytes) {
        throw new Error(`${opts.label} exceeds ${(opts.maxBytes / (1024 * 1024)).toFixed(0)}MB limit`)
      }
      const arrayBuffer = await response.arrayBuffer()
      if (arrayBuffer.byteLength > opts.maxBytes) {
        throw new Error(`${opts.label} exceeds ${(opts.maxBytes / (1024 * 1024)).toFixed(0)}MB limit`)
      }
      return { contentType: contentType || 'application/octet-stream', base64: Buffer.from(arrayBuffer).toString('base64') }
    } finally {
      clearTimeout(timeout)
      signal?.removeEventListener('abort', abort)
    }
  }

  private async downloadImageDataUrl(url: string, signal?: AbortSignal): Promise<string> {
    const { contentType, base64 } = await this.downloadHttpAttachment(url, {
      allowedTypes: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
      maxBytes: 10 * 1024 * 1024,
      label: 'Image'
    }, signal)
    return `data:${contentType};base64,${base64}`
  }

  private async resolveOpenAIHttpImages(request: OpenAIChatRequest, signal?: AbortSignal): Promise<OpenAIChatRequest> {
    await Promise.all(request.messages.map(async message => {
      if (!Array.isArray(message.content)) return
      await Promise.all(message.content.map(async part => {
        if (part.type !== 'image_url' || !part.image_url?.url.startsWith('http')) return
        part.image_url.url = await this.downloadImageDataUrl(part.image_url.url, signal)
      }))
    }))
    return request
  }

  private async resolveClaudeHttpImages(request: ClaudeRequest, signal?: AbortSignal): Promise<ClaudeRequest> {
    await Promise.all(request.messages.map(async message => {
      if (!Array.isArray(message.content)) return
      await Promise.all(message.content.map(async block => {
        if (block.source?.type !== 'url') return
        if (block.type === 'image') {
          const dataUrl = await this.downloadImageDataUrl(block.source.url, signal)
          const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/)
          if (!match) {
            throw new Error('Downloaded image produced invalid data URL')
          }
          block.source = { type: 'base64', media_type: match[1], data: match[2] }
        } else if (block.type === 'document') {
          // PDF/文档 URL：下载并转 base64（旧实现只处理图片，文档 URL 会被原样透传给 Kiro 而失败）
          const { contentType, base64 } = await this.downloadHttpAttachment(block.source.url, {
            maxBytes: 32 * 1024 * 1024,
            label: 'Document'
          }, signal)
          block.source = { type: 'base64', media_type: contentType, data: base64 }
        }
      }))
    }))
    return request
  }

  private prepareOpenAIRequest(request: OpenAIChatRequest): OpenAIChatRequest {
    this.validateOpenAICacheControls(request)

    if (this.config.disableTools || request.tool_choice === 'none') {
      return { ...request, tools: undefined, tool_choice: undefined }
    }

    if (request.tool_choice && typeof request.tool_choice === 'object' && request.tool_choice.type === 'function' && !request.tool_choice.function?.name) {
      throw new Error('tool_choice function requires a tool name')
    }

    if (request.tool_choice && typeof request.tool_choice === 'object' && request.tool_choice.function?.name) {
      const selectedToolName = request.tool_choice.function.name
      if (!request.tools?.some(tool => tool.function.name === selectedToolName)) {
        throw new Error(`tool_choice references unknown tool: ${selectedToolName}`)
      }
      return {
        ...request,
        tools: request.tools?.filter(tool => tool.function.name === selectedToolName)
      }
    }

    return request
  }

  private prepareClaudeRequest(request: ClaudeRequest): ClaudeRequest {
    this.validateClaudeCacheControls(request)

    if (this.config.disableTools || request.tool_choice?.type === 'none') {
      return { ...request, tools: undefined, tool_choice: undefined }
    }

    if (request.tool_choice?.type === 'tool' && !request.tool_choice.name) {
      throw new Error('tool_choice tool requires a tool name')
    }

    if (request.tool_choice?.name) {
      const selectedToolName = request.tool_choice.name
      if (!request.tools?.some(tool => tool.name === selectedToolName)) {
        throw new Error(`tool_choice references unknown tool: ${selectedToolName}`)
      }
      return {
        ...request,
        tools: request.tools?.filter(tool => tool.name === selectedToolName)
      }
    }

    return request
  }

  // 获取统计信息
  getStats(): ProxyStats {
    // 返回可序列化的统计信息（Map 对象在 IPC 中无法正确序列化）
    return {
      totalRequests: this.stats.totalRequests,
      successRequests: this.stats.successRequests,
      failedRequests: this.stats.failedRequests,
      totalTokens: this.stats.totalTokens,
      totalCredits: this.stats.totalCredits,
      inputTokens: this.stats.inputTokens,
      outputTokens: this.stats.outputTokens,
      cacheReadTokens: this.stats.cacheReadTokens,
      cacheWriteTokens: this.stats.cacheWriteTokens,
      reasoningTokens: this.stats.reasoningTokens,
      startTime: this.stats.startTime,
      accountStats: this.stats.accountStats,
      endpointStats: this.stats.endpointStats,
      modelStats: this.stats.modelStats,
      recentRequests: this.stats.recentRequests
    }
  }

  // 获取账号池
  getAccountPool(): AccountPool {
    return this.accountPool
  }

  // 设置初始累计 credits（用于从持久化存储恢复）
  setTotalCredits(credits: number): void {
    this.stats.totalCredits = credits
  }

  // 重置累计 credits
  resetTotalCredits(): void {
    this.stats.totalCredits = 0
    this.events.onCreditsUpdate?.(0)
  }

  // 设置初始累计 tokens（用于从持久化存储恢复）
  setTotalTokens(inputTokens: number, outputTokens: number): void {
    this.stats.inputTokens = inputTokens
    this.stats.outputTokens = outputTokens
    this.stats.totalTokens = inputTokens + outputTokens
  }

  // 重置累计 tokens
  resetTotalTokens(): void {
    this.stats.inputTokens = 0
    this.stats.outputTokens = 0
    this.stats.totalTokens = 0
  }

  // 设置请求统计（用于从持久化存储恢复）
  setRequestStats(totalRequests: number, successRequests: number, failedRequests: number): void {
    this.stats.totalRequests = totalRequests
    this.stats.successRequests = successRequests
    this.stats.failedRequests = failedRequests
  }

  // 重置请求统计
  resetRequestStats(): void {
    this.stats.totalRequests = 0
    this.stats.successRequests = 0
    this.stats.failedRequests = 0
    this.notifyRequestStatsUpdate()
  }

  // 通知请求统计更新
  private notifyRequestStatsUpdate(): void {
    this.events.onRequestStatsUpdate?.(
      this.stats.totalRequests,
      this.stats.successRequests,
      this.stats.failedRequests
    )
  }

  // 记录请求成功
  private recordRequestSuccess(): void {
    this.stats.successRequests++
    this.sessionStats.successRequests++
    this.notifyRequestStatsUpdate()
  }

  // 记录请求失败
  private recordRequestFailed(): void {
    this.stats.failedRequests++
    this.sessionStats.failedRequests++
    this.notifyRequestStatsUpdate()
  }

  // 记录新请求
  private recordNewRequest(): void {
    this.stats.totalRequests++
    this.sessionStats.totalRequests++
    this.notifyRequestStatsUpdate()
  }

  // 获取会话统计（当前服务运行期间的统计）
  getSessionStats(): { totalRequests: number; successRequests: number; failedRequests: number; startTime: number } {
    return { ...this.sessionStats }
  }

  // 是否运行中
  isRunning(): boolean {
    return this.server !== null
  }

  private getAbortError(signal?: AbortSignal): Error {
    if (signal?.reason instanceof Error) return signal.reason
    if (signal?.reason) return new Error(String(signal.reason))
    return new Error('Request aborted')
  }

  private isAbortError(error: unknown, signal?: AbortSignal): boolean {
    return signal?.aborted === true
      || (error instanceof Error && (error.message.includes('Client disconnected') || error.message.includes('Proxy server stopped')))
  }

  private throwIfAborted(signal?: AbortSignal): void {
    if (signal?.aborted) throw this.getAbortError(signal)
  }

  private throwIfResponseClosed(res: http.ServerResponse, signal?: AbortSignal): void {
    this.throwIfAborted(signal)
    if (res.writableEnded || res.destroyed) throw new Error('Client disconnected')
  }

  private isResponseClosed(res: http.ServerResponse): boolean {
    return res.writableEnded || res.destroyed
  }

  // 检测错误消息中是否包含账号被长期封禁的特征
  // 返回 { reason, message } 表示需要标记 suspended；返回 null 表示非封禁错误
  // 覆盖：
  //   - Kiro 后端 HTTP 403 + body: { reason: "TEMPORARILY_SUSPENDED", message: "..." }
  //   - CodeWhisperer AccountSuspendedException
  //   - 423 Locked
  private detectSuspendedError(errMsg: string): { reason: string; message: string } | null {
    if (!errMsg) return null

    // 1) 显式 reason: "TEMPORARILY_SUSPENDED" (Kiro 风控)
    const reasonMatch = errMsg.match(/"reason"\s*:\s*"(TEMPORARILY_SUSPENDED|ACCOUNT_SUSPENDED|PERMANENTLY_SUSPENDED)"/i)
    if (reasonMatch) {
      // 尝试提取 message 字段
      const msgMatch = errMsg.match(/"message"\s*:\s*"([^"]+)"/)
      return { reason: reasonMatch[1].toUpperCase(), message: msgMatch?.[1] || errMsg }
    }

    // 2) 文本特征 "temporarily suspended" / "user id is ... suspended"
    if (/User\s+ID\s+is\s+(temporarily\s+)?suspended/i.test(errMsg) || /temporarily\s+suspended/i.test(errMsg)) {
      const msgMatch = errMsg.match(/"message"\s*:\s*"([^"]+)"/)
      return { reason: 'TEMPORARILY_SUSPENDED', message: msgMatch?.[1] || errMsg }
    }

    // 3) AccountSuspendedException (CodeWhisperer)
    if (errMsg.includes('AccountSuspendedException') || errMsg.includes('Account suspended')) {
      const msgMatch = errMsg.match(/"message"\s*:\s*"([^"]+)"/)
      return { reason: 'AccountSuspendedException', message: msgMatch?.[1] || errMsg }
    }

    // 4) HTTP 423 Locked
    if (/\b423\b/.test(errMsg) && /locked|suspended/i.test(errMsg)) {
      return { reason: 'ACCOUNT_LOCKED', message: errMsg }
    }

    return null
  }

  private waitForRetry(ms: number, signal?: AbortSignal): Promise<void> {
    this.throwIfAborted(signal)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        signal?.removeEventListener('abort', abort)
        resolve()
      }, ms)
      const abort = () => {
        clearTimeout(timeout)
        reject(this.getAbortError(signal))
      }
      signal?.addEventListener('abort', abort, { once: true })
    })
  }

  private async abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
    this.throwIfAborted(signal)
    if (!signal) return promise
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        const abort = () => reject(this.getAbortError(signal))
        signal.addEventListener('abort', abort, { once: true })
        promise.then(
          () => signal.removeEventListener('abort', abort),
          () => signal.removeEventListener('abort', abort)
        )
      })
    ])
  }

  // 清除模型缓存，强制下次请求重新获取
  clearModelCache(): void {
    this.modelCache = null
    clearModelThinkingCapabilities()
    console.log('[ProxyServer] Model cache cleared')
  }

  // 获取可用模型列表
  private static mapKiroModelToApi(m: KiroModel) {
    return {
      id: m.modelId,
      name: m.modelName,
      description: m.description,
      inputTypes: m.supportedInputTypes,
      maxInputTokens: m.tokenLimits?.maxInputTokens,
      maxOutputTokens: m.tokenLimits?.maxOutputTokens,
      rateMultiplier: m.rateMultiplier,
      rateUnit: m.rateUnit,
      supportsThinking: !!(m.additionalModelRequestFieldsSchema?.properties as Record<string, unknown> | undefined)?.thinking,
      thinkingEfforts: extractThinkingEfforts(m.additionalModelRequestFieldsSchema),
      supportsPromptCaching: m.promptCaching?.supportsPromptCaching || false,
      modelProvider: m.modelProvider || undefined
    }
  }

  /**
   * fetchKiroModels + 401/403 自愈：token 被外部 Kiro IDE 轮换后，原 accessToken 会 403。
   * 静默失败会让 ctx-window 缓存填不上（opus 误判 200K）且 /v1/models 变空，故这里 refresh 一次再重试。
   * 非 auth 错误（网络/超时）原样上抛，由调用方既有 try/catch 处理。
   */
  private async fetchKiroModelsWithRefresh(account: ProxyAccount, signal?: AbortSignal): Promise<KiroModel[]> {
    try {
      return await fetchKiroModels(account, signal)
    } catch (error) {
      if (this.isAbortError(error, signal)) throw error
      const msg = error instanceof Error ? error.message : String(error)
      if (!msg.startsWith('Auth error')) throw error
      // 401/403 → refresh token 一次再重试
      const full = this.accountPool.getAccount(account.id) || account
      let refreshed = false
      try {
        refreshed = await this.refreshToken(full, signal)
      } catch (refreshErr) {
        console.error('[ProxyServer] Token refresh for model fetch failed:', formatError(refreshErr))
      }
      if (!refreshed) throw error
      const fresh = this.accountPool.getAccount(account.id) || full
      console.log('[ProxyServer] Retrying ListAvailableModels after token refresh')
      return await fetchKiroModels(fresh, signal)
    }
  }

  async getAvailableModels(signal?: AbortSignal): Promise<{ models: ReturnType<typeof ProxyServer.mapKiroModelToApi>[]; fromCache: boolean }> {
    const now = Date.now()
    
    let kiroModels: KiroModel[]
    let fromCache = false

    if (this.modelCache && (now - this.modelCache.timestamp) < this.MODEL_CACHE_TTL) {
      kiroModels = this.modelCache.models
      fromCache = true
    } else {
      this.throwIfAborted(signal)
      const account = await this.getAvailableAccount(signal)
      this.throwIfAborted(signal)
      if (!account) {
        return { models: [], fromCache: false }
      }

      try {
        kiroModels = await this.fetchKiroModelsWithRefresh(account, signal)
        if (kiroModels.length > 0) {
          this.modelCache = { models: kiroModels, timestamp: now }
          // 同步到 kiroApi 的 ctx cache, 供 token 裁剪逻辑使用
          for (const m of kiroModels) {
            if (m.tokenLimits?.maxInputTokens) {
              setModelContextWindow(m.modelId, m.tokenLimits.maxInputTokens)
            }
          }
          // 同步真实 thinking/effort 能力到 translator，供 additionalModelRequestFields 构建时校验
          syncModelThinkingCapabilities(kiroModels)
        }
      } catch (error) {
        if (this.isAbortError(error, signal)) throw error
        console.error('[ProxyServer] Failed to fetch models:', error)
        return { models: [], fromCache: false }
      }
    }

    // 合并隐藏模型（与 /v1/models 端点一致）
    const modelIds = new Set(kiroModels.map(m => m.modelId))
    const hiddenModels: KiroModel[] = [
      { modelId: 'claude-3.7-sonnet', modelName: 'Claude 3.7 Sonnet', description: 'Claude 3.7 Sonnet (hidden)', supportedInputTypes: ['TEXT', 'IMAGE'], tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 64000 } } as KiroModel,
      { modelId: 'simple-task', modelName: 'Simple Task', description: 'Kiro fast model (routes to Haiku)', supportedInputTypes: ['TEXT'], tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 4096 } } as KiroModel,
      { modelId: 'CLAUDE_SONNET_4_20250514_V1_0', modelName: 'Claude Sonnet 4 (CW)', description: 'CodeWhisperer internal ID', supportedInputTypes: ['TEXT', 'IMAGE'], tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 64000 } } as KiroModel,
      { modelId: 'CLAUDE_HAIKU_4_5_20251001_V1_0', modelName: 'Claude Haiku 4.5 (CW)', description: 'CodeWhisperer internal ID', supportedInputTypes: ['TEXT', 'IMAGE'], tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 64000 } } as KiroModel,
      { modelId: 'CLAUDE_3_7_SONNET_20250219_V1_0', modelName: 'Claude 3.7 Sonnet (CW)', description: 'CodeWhisperer internal ID', supportedInputTypes: ['TEXT', 'IMAGE'], tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 64000 } } as KiroModel
    ]
    const merged = [...kiroModels, ...hiddenModels.filter(m => !modelIds.has(m.modelId))]

    return { models: merged.map(ProxyServer.mapKiroModelToApi), fromCache }
  }

  // Token 刷新阈值分两档（stale-while-revalidate）：
  //   soft（isTokenExpiringSoon）：进入 tokenRefreshBeforeExpiry（默认 300s）窗口 → 「应该尽快刷」，
  //     但当前 token 仍可用，故先用旧 token 立即服务请求、把刷新作为后台 fire-and-forget（不阻塞 TTFT）。
  //   hard（isTokenExpired）：剩余 < HARD_REFRESH_BUFFER_MS（或已过期）→ 「这次请求前必须刷」，
  //     此时才在请求路径上 await 刷新。正常使用下 soft 后台刷新会赶在进入 hard 窗口前完成，
  //     hard 阻塞仅在「长时间闲置到 token 逼近过期」这类罕见场景触发。
  //
  // 反检测说明：后台刷新复用同一条 refreshToken() 路径（同 jitter、同并发去重、同 device-id 绑定），
  // 出网指纹与原阻塞式刷新完全一致；只是从「每个请求各等一次」变成「不阻塞用户」。不新增任何定时器，
  // 刷新仍由真实请求驱动 → refresh-to-use 比例 ≈ 1:1、节奏随真实流量，不产生可被识别的周期性/突发。
  private static readonly HARD_REFRESH_BUFFER_MS = 60_000

  // soft 窗口：进入 tokenRefreshBeforeExpiry 但尚未触及 hard 窗口 → 后台刷新即可
  private isTokenExpiringSoon(account: ProxyAccount): boolean {
    if (!account.expiresAt) return false
    const refreshBeforeMs = (this.config.tokenRefreshBeforeExpiry || 300) * 1000
    return Date.now() + refreshBeforeMs >= account.expiresAt
  }

  // hard 窗口：token 已过期或剩余不足 HARD_REFRESH_BUFFER_MS → 必须在请求前阻塞刷新
  private isTokenExpired(account: ProxyAccount): boolean {
    if (!account.expiresAt) return false
    return Date.now() + ProxyServer.HARD_REFRESH_BUFFER_MS >= account.expiresAt
  }

  // 后台刷新（fire-and-forget）：soft 窗口内触发，不阻塞当前请求。
  // - 复用 refreshToken() 的并发去重（refreshingTokens map）：若已有在途刷新（请求路径或上一次后台）
  //   则不会重复发起，天然防止「自造突发」。
  // - 绝不传入请求的 AbortSignal：否则当前请求结束/中断会把后台刷新一并 abort，留下半刷新状态。
  //   传 undefined 让刷新跑完其自然生命周期。
  // - 调用方必须已先执行 syncKProxyDeviceId(account)，使本次刷新出网带正确的 device-id（与阻塞式刷新一致）。
  private triggerBackgroundRefresh(account: ProxyAccount): void {
    // 已有在途刷新（含本函数上一次触发）→ refreshToken 内部会复用，这里无需重复进入
    if (this.refreshingTokens.has(account.id)) return
    // 不传 signal；catch 收敛所有异常，避免 fire-and-forget 产生 unhandledRejection
    void this.refreshToken(account, undefined).then(
      (ok) => { if (!ok) perfDiag.incr(PerfEvent.BackgroundRefreshFailure) },
      () => { perfDiag.incr(PerfEvent.BackgroundRefreshFailure) /* 后台刷新失败由请求路径的 hard 刷新兜底 */ }
    )
  }

  // 刷新 Token —— 同账号并发去重：所有等待方共享同一个 in-flight refresh promise，
  // 拿到真实结果（不再盲等 1s 后猜测，避免「猜成失败 → 误判 account 不可用 → 账号乱跳」）。
  private async refreshToken(account: ProxyAccount, signal?: AbortSignal): Promise<boolean> {
    this.throwIfAborted(signal)
    if (!this.events.onTokenRefresh) {
      console.warn('[ProxyServer] No token refresh callback configured')
      return false
    }

    // 已有同账号刷新在途 → 复用同一 promise（不自己再发起、也不盲等）
    const inFlight = this.refreshingTokens.get(account.id)
    if (inFlight) {
      console.log(`[ProxyServer] Awaiting in-flight token refresh for ${account.email || account.id}`)
      try {
        return await inFlight
      } catch {
        return false
      }
    }

    // 发起新的刷新，把 promise 存入 map 供其他 waiter 复用；finally 清理
    const p = this.doRefreshToken(account, signal)
    this.refreshingTokens.set(account.id, p)
    try {
      return await p
    } finally {
      this.refreshingTokens.delete(account.id)
    }
  }

  private async doRefreshToken(account: ProxyAccount, signal?: AbortSignal): Promise<boolean> {
    console.log(`[ProxyServer] Refreshing token for ${account.email || account.id}`)
    try {
      // 随机延迟 0-3 秒，避免多账号同时刷新被识别为批量操作。注意：因并发去重，同账号的
      // N 个等待方共享这一次刷新，只「集体」承担一次 jitter，而不是每个请求各等一次。
      const jitter = Math.floor(Math.random() * 3000)
      if (jitter > 0) await this.waitForRetry(jitter, signal)

      const result = await this.abortable(this.events.onTokenRefresh!(account), signal)
      if (result.success && result.accessToken) {
        // 更新账号池中的 Token
        this.accountPool.updateAccount(account.id, {
          accessToken: result.accessToken,
          refreshToken: result.refreshToken || account.refreshToken,
          expiresAt: result.expiresAt
        })
        // 通知外部更新
        this.events.onAccountUpdate?.({
          ...account,
          accessToken: result.accessToken,
          refreshToken: result.refreshToken || account.refreshToken,
          expiresAt: result.expiresAt
        })
        console.log(`[ProxyServer] Token refreshed for ${account.email || account.id}`)
        return true
      } else {
        console.error(`[ProxyServer] Token refresh failed for ${account.email || account.id}: ${result.error}`)
        this.accountPool.markNeedsRefresh(account.id)
        return false
      }
    } catch (error) {
      if (this.isAbortError(error, signal)) throw error
      console.error(`[ProxyServer] Token refresh error for ${account.email || account.id}:`, error)
      this.accountPool.markNeedsRefresh(account.id)
      return false
    }
  }

  /**
   * 计算 API Key 允许使用的账号 ID 集合（P2-21）
   * 返回 undefined = 不限制（允许所有账号）
   */
  private getAllowedAccountIds(apiKeyId?: string): Set<string> | undefined {
    if (!apiKeyId) return undefined
    const bindings = this.config.apiKeyAccountBindings?.[apiKeyId]
    if (!bindings || bindings.length === 0) return undefined
    return new Set(bindings)
  }

  /**
   * 把"允许账号白名单"转换成 getNextAccount(excludeIds) 需要的"排除集合"。
   * 排除 = 当前池中所有不在白名单内的账号。
   * 返回 undefined = 不限制（无绑定时）。
   */
  private excludeSetForAllowed(allowedIds?: Set<string>): Set<string> | undefined {
    if (!allowedIds) return undefined
    const exclude = new Set<string>()
    for (const acc of this.accountPool.getAllAccounts()) {
      if (!allowedIds.has(acc.id)) exclude.add(acc.id)
    }
    return exclude
  }

  /**
   * 合并"白名单外排除集合"与额外要排除的账号 ID（如当前已失败的账号）。
   * base 为 undefined（无绑定）时，仅排除 extra（若有），否则返回 undefined 表示不限制。
   */
  private excludeWith(base: Set<string> | undefined, ...extraIds: string[]): Set<string> | undefined {
    if (!base && extraIds.length === 0) return undefined
    const merged = new Set<string>(base ?? [])
    for (const id of extraIds) merged.add(id)
    return merged
  }

  // 获取可用账号（包含 Token 刷新检查）
  // P1-8 sessionHint：相同会话尽量复用同一账号（命中 prompt cache + 防风控）
  // P2-21 apiKeyId：用于过滤 API Key 允许使用的账号子集
  private async getAvailableAccount(signal?: AbortSignal, sessionHint?: string, apiKeyId?: string): Promise<ProxyAccount | null> {
    const allowedIds = this.getAllowedAccountIds(apiKeyId)
    const groupMode = this.config.multiAccountSelectionMode === 'groups'
    const allowedGroupIds = groupMode ? new Set(this.config.multiAccountGroupIds || []) : null
    const isAllowed = (acc: ProxyAccount | null): boolean => {
      if (!acc) return true
      // API Key 白名单（apiKeyAccountBindings）
      if (allowedIds && !allowedIds.has(acc.id)) return false
      // 分组过滤（双保险：即便前端忘了重新同步账号池，这里也能拦住非选中分组的账号）
      if (groupMode && allowedGroupIds) {
        const gid = acc.groupId || '__ungrouped__'
        if (!allowedGroupIds.has(gid)) return false
      }
      return true
    }
    this.throwIfAborted(signal)
    // 如果 pool 为空，触发懒加载回调尝试同步账号（冷启动场景）
    if (this.accountPool.size === 0 && this.events.onPoolEmpty) {
      console.log('[ProxyServer] Account pool empty, triggering lazy sync...')
      await this.abortable(this.events.onPoolEmpty(), signal)
    }
    this.throwIfAborted(signal)

    // P1-8 会话粘性：优先复用已绑定的账号（同时受 API Key 绑定过滤）
    if (this.config.sessionAffinityEnabled && sessionHint) {
      const sticky = this.pickAccountWithAffinity(sessionHint)
      if (sticky && isAllowed(sticky)) {
        proxyLogger.debug('ProxyServer', `Session affinity hit: ${sessionHint.slice(0, 16)} → ${sticky.email || sticky.id.slice(0, 8)}`)
        // 仍需检查 token 是否需要刷新（stale-while-revalidate）
        if (this.isTokenExpired(sticky)) {
          // hard 窗口：token 逼近过期/已过期 → 这次请求前必须阻塞刷新
          const refreshed = await this.refreshToken(sticky, signal)
          if (refreshed) {
            return this.accountPool.getAccount(sticky.id) || sticky
          }
          // 刷新失败 → 不在此 return，落到下方常规挑选逻辑（保持原「失败后另寻账号」语义）
        } else {
          // token 仍可用：先用旧 token 立即服务（不阻塞 TTFT）；
          // 若已进入 soft 窗口则顺带后台刷新，让下一个请求用上新 token。
          if (this.isTokenExpiringSoon(sticky)) this.triggerBackgroundRefresh(sticky)
          return sticky
        }
      }
    }

    let account: ProxyAccount | null

    if (this.config.enableMultiAccount) {
      account = this.accountPool.getNextAccount()
      if (account && !isAllowed(account)) {
        // 尝试找一个允许的账号（白名单 + 分组都已合并进 isAllowed）
        const allAccounts = this.accountPool.getAllAccounts()
        const exclude = new Set<string>()
        for (const a of allAccounts) {
          if (!isAllowed(a)) exclude.add(a.id)
        }
        account = this.accountPool.getNextAccount(exclude)
      }
      if (!account) {
        const status = this.accountPool.getQuotaStatus()
        if (status.exhausted > 0 && status.available === 0) {
          console.log(`[ProxyServer] All accounts quota exhausted (${status.exhausted}/${status.total}), no available accounts`)
        }
      }
    } else {
      // 禁用多账号轮询时，优先使用指定的账号
      // P2-21: 若该 API Key 绑定了账号白名单，则所有挑选/兜底都必须落在白名单内
      const pickFirstAllowed = (): ProxyAccount | null =>
        this.accountPool.getAllAccounts().find(isAllowed) || null

      if (this.config.selectedAccountIds && this.config.selectedAccountIds.length > 0) {
        // 使用指定的第一个账号
        account = this.accountPool.getAccount(this.config.selectedAccountIds[0])
        // P2-21: 指定账号不在 API Key 白名单内时，改用白名单内账号（绑定优先于全局 selectedAccountIds）
        if (account && !isAllowed(account)) {
          console.log(`[ProxyServer] Selected account ${account.email || account.id} not in API key whitelist, switching to an allowed account`)
          account = pickFirstAllowed()
        }
        // 检查指定账号是否配额耗尽，若是则尝试自动切换（切换目标仍须在白名单内）
        if (account && this.accountPool.isQuotaExhausted(account) && this.config.autoSwitchOnQuotaExhausted) {
          const exhaustedId = account.id
          let nextAccount = this.accountPool.getNextAvailableAccount(exhaustedId)
          if (nextAccount && !isAllowed(nextAccount)) {
            nextAccount = this.accountPool.getAllAccounts().find(a =>
              a.id !== exhaustedId && isAllowed(a) && !this.accountPool.isQuotaExhausted(a)
            ) || null
          }
          if (nextAccount) {
            console.log(`[ProxyServer] Selected account ${account.email || account.id} quota exhausted, auto-switching to ${nextAccount.email || nextAccount.id}`)
            // 绑定白名单是按 API Key 维度的，不应改写全局 selectedAccountIds；仅无白名单时持久化
            if (!allowedIds) {
              this.config.selectedAccountIds = [nextAccount.id]
              this.events.onAccountUpdate?.(nextAccount)
            }
            account = nextAccount
          }
        }
        if (!account) {
          console.log(`[ProxyServer] Selected account ${this.config.selectedAccountIds[0]} not usable, using first allowed/available`)
          account = allowedIds ? pickFirstAllowed() : (this.accountPool.getAllAccounts()[0] || null)
        }
      } else {
        // 没有指定账号，使用第一个可用账号（受白名单约束）
        account = allowedIds ? pickFirstAllowed() : (this.accountPool.getAllAccounts()[0] || null)
      }
    }
    
    if (!account) return null

    // 自动切换 K-Proxy 设备 ID（如果 K-Proxy 服务可用）
    // 注意：必须在任何刷新（阻塞式或后台）之前执行，使刷新出网带本账号正确的 device-id。
    this.syncKProxyDeviceId(account)

    // 检查是否需要刷新 Token（stale-while-revalidate）
    if (this.isTokenExpired(account)) {
      // hard 窗口：token 逼近过期/已过期 → 这次请求前必须阻塞刷新
      // 嫌疑计数：区分「已有在途刷新（含后台 SWR）时被迫等待」与「自己发起新刷新」——
      // 前者多说明后台软刷新没能在 hard 窗口前刷完（jitter+RTT 过长），是 TTFT 尾延迟来源。
      if (perfDiag.enabled) {
        perfDiag.incr(PerfEvent.HardWindowBlockingRefresh)
        if (this.refreshingTokens.has(account.id)) perfDiag.incr(PerfEvent.HardWindowAwaitInflight)
      }
      const refreshed = await this.refreshToken(account, signal)
      if (!refreshed) {
        // 刷新失败，如果启用多账号才尝试获取下一个账号（受 API Key 白名单约束）
        if (this.config.enableMultiAccount) {
          const exclude = this.excludeSetForAllowed(allowedIds)
          return this.accountPool.getNextAccount(exclude)
        }
        return null
      }
      // 返回更新后的账号
      const refreshedAccount = this.accountPool.getAccount(account.id)
      if (refreshedAccount && sessionHint) this.rememberAffinity(sessionHint, refreshedAccount.id)
      return refreshedAccount
    }

    // token 仍可用：用旧 token 立即服务（不阻塞 TTFT）。
    // 若已进入 soft 窗口，顺带后台刷新（device-id 已 sync），让后续请求用上新 token，
    // 正常使用下可在 token 进入 hard 窗口前完成，从而避免请求路径上的阻塞刷新。
    if (this.isTokenExpiringSoon(account)) this.triggerBackgroundRefresh(account)

    if (sessionHint) this.rememberAffinity(sessionHint, account.id)
    return account
  }

  // 同步 K-Proxy 设备 ID（根据账号自动切换）
  private syncKProxyDeviceId(account: ProxyAccount): void {
    const kproxyService = getKProxyService()
    if (!kproxyService || !kproxyService.isRunning()) {
      return // K-Proxy 未初始化或未运行
    }

    // 尝试切换到账号绑定的设备 ID
    const switched = kproxyService.switchToAccount(account.id)
    
    if (!switched) {
      // 账号没有绑定设备 ID，自动生成并绑定
      const newDeviceId = generateDeviceId()
      kproxyService.addDeviceIdMapping({
        accountId: account.id,
        deviceId: newDeviceId,
        description: account.email || `Account ${account.id.substring(0, 8)}`,
        createdAt: Date.now()
      })
      kproxyService.setDeviceId(newDeviceId)
      proxyLogger.info('ProxyServer', `Auto-generated device ID for account ${account.email || account.id.substring(0, 8)}`)
    } else {
      proxyLogger.debug('ProxyServer', `Switched to device ID for account ${account.email || account.id.substring(0, 8)}`)
    }
  }

  // 带重试的 API 调用
  // P2-21 allowedIds：API Key 绑定的账号白名单；切换账号时不得越界，undefined = 不限制
  private async callWithRetry<T>(
    account: ProxyAccount,
    apiCall: (acc: ProxyAccount, endpointIndex: number) => Promise<T>,
    _path: string,
    signal?: AbortSignal,
    allowedIds?: Set<string>
  ): Promise<{ result: T; account: ProxyAccount }> {
    const maxRetries = this.config.maxRetries || 3
    const retryDelay = this.config.retryDelayMs || 1000
    let lastError: Error | null = null
    let currentAccount = account
    let endpointIndex = 0
    // 绝对迭代上限：防御性兜底，避免任何意外导致死循环。
    // quota 分支每个账号最多消耗 2 次迭代（两个端点各试一次），因此按 池大小 × 2 计；
    // 再加同账号重试预算与余量，确保正常 failover 永远不会被本上限误截断。
    let iterations = 0
    const maxIterations = maxRetries + this.accountPool.getAllAccounts().length * 2 + 5
    // 切换账号时用于排除白名单外的账号
    const isAllowed = (acc: ProxyAccount | null): boolean => !acc || !allowedIds || allowedIds.has(acc.id)
    const excludeOutOfWhitelist = this.excludeSetForAllowed(allowedIds)
    // 本次请求累计已尝试的账号 ID，避免重试时循环命中已经失败过的账号
    const triedIds = new Set<string>([account.id])
    /**
     * 切到下一个可用账号：同时满足
     *  - API Key 白名单约束（excludeOutOfWhitelist）
     *  - 排除本请求已试过的账号（triedIds）
     * 多账号模式用 getNextAccount；单账号 + 自动切换用 getNextAvailableAccount；否则不切换。
     */
    const switchToNextAccount = (): ProxyAccount | null => {
      const exclude = this.excludeWith(excludeOutOfWhitelist, ...triedIds)
      let next: ProxyAccount | null = null
      if (this.config.enableMultiAccount) {
        next = this.accountPool.getNextAccount(exclude)
      } else if (this.config.autoSwitchOnQuotaExhausted) {
        next = this.accountPool.getNextAvailableAccount(exclude ?? new Set<string>())
      }
      return isAllowed(next) ? next : null
    }

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      // 切换账号不应消耗"同账号重试预算"(maxRetries)：否则账号多于 maxRetries 时，
      // 遇到连续 suspend/quota 会在试完整个池之前就耗尽预算，导致仍有可用账号却返回错误。
      // 账号切换由 triedIds（池大小）自然封顶；这里加绝对迭代上限做防御性兜底，杜绝死循环。
      if (++iterations > maxIterations) {
        console.warn(`[ProxyServer] callWithRetry hit absolute iteration cap (${maxIterations}), giving up`)
        break
      }
      this.throwIfAborted(signal)
      try {
        const result = await apiCall(currentAccount, endpointIndex)
        return { result, account: currentAccount }
      } catch (error) {
        if (this.isAbortError(error, signal)) throw error
        lastError = error as Error
        const errMsg = lastError.message || ''

        console.log(`[ProxyServer] API call failed (attempt ${attempt + 1}/${maxRetries}): ${errMsg}`)

        // 优先检测账号被长期封禁（不是 token 问题，刷新也没用）
        // 特征：HTTP 403 + reason: "TEMPORARILY_SUSPENDED" 或 AccountSuspendedException / 423
        const suspendInfo = this.detectSuspendedError(errMsg)
        if (suspendInfo) {
          const newlyMarked = this.accountPool.markSuspended(currentAccount.id, suspendInfo.reason, suspendInfo.message)
          if (newlyMarked) {
            this.events.onAccountSuspended?.({
              accountId: currentAccount.id,
              email: currentAccount.email,
              reason: suspendInfo.reason,
              message: suspendInfo.message
            })
            // P1-6 关键事件 → 触发 webhook
            this.appendAuditLog('account_suspended', {
              accountId: currentAccount.id,
              email: currentAccount.email,
              reason: suspendInfo.reason
            })
            this.triggerWebhook('proxy-account-suspended', {
              title: '反代账号被风控',
              message: `账号 ${currentAccount.email || currentAccount.id.slice(0, 8)} 被 Kiro 后端标记为 ${suspendInfo.reason}，需要人工解封`,
              level: 'error',
              fields: {
                邮箱: currentAccount.email || '-',
                账号ID: currentAccount.id.slice(0, 8),
                封禁原因: suspendInfo.reason,
                详情: this.sanitizeErrorMessage(suspendInfo.message || '').slice(0, 200)
              }
            })
          }
          console.warn(`[ProxyServer] Account ${currentAccount.email || currentAccount.id} suspended (${suspendInfo.reason}), switching to next available account`)
          // 账号被封 → 池容量下降，检查是否需要预警补号
          this.checkPoolLow()
          // 切到下个可用账号（跳过被 suspended 的 + 白名单外 + 本请求已试过的）
          const nextAccount = switchToNextAccount()
          if (nextAccount && !triedIds.has(nextAccount.id) && isAllowed(nextAccount)) {
            currentAccount = nextAccount
            triedIds.add(nextAccount.id)
            attempt--  // 切号不消耗同账号重试预算（由 triedIds/maxIterations 兜底）
            // 绑定白名单按 API Key 维度，不改写全局 selectedAccountIds
            if (!this.config.enableMultiAccount && !allowedIds) {
              this.config.selectedAccountIds = [nextAccount.id]
              this.events.onAccountUpdate?.(nextAccount)
            }
            continue
          }
          // 无可切换的账号 → 直接抛出错误给客户端
          break
        }

        // 401/403: 尝试刷新 Token
        if (errMsg.includes('401') || errMsg.includes('403') || errMsg.includes('Auth')) {
          console.log('[ProxyServer] Auth error, attempting token refresh')
          const refreshed = await this.refreshToken(currentAccount, signal)
          if (refreshed) {
            currentAccount = this.accountPool.getAccount(currentAccount.id) || currentAccount
            continue
          }
          // 刷新失败 → 切到没试过的下个账号（受 API Key 白名单约束）
          const nextAccount = switchToNextAccount()
          if (nextAccount && !triedIds.has(nextAccount.id) && isAllowed(nextAccount)) {
            currentAccount = nextAccount
            triedIds.add(nextAccount.id)
            attempt--  // 切号不消耗同账号重试预算
            continue
          }
        }

        // 402/429: 额度耗尽，切换端点或账号
        if (errMsg.includes('402') || errMsg.includes('429') || errMsg.includes('quota') || errMsg.includes('ThrottlingException') || errMsg.includes('reached the limit') || errMsg.includes('ServiceQuotaExceededException') || errMsg.includes('limit exceeded') || errMsg.includes('rate limit')) {
          console.log('[ProxyServer] Quota/throttle error, switching endpoint or account')
          this.accountPool.recordError(currentAccount.id, ErrorType.RECOVERABLE, 429)
          // 配额耗尽 → 可用账号可能下降，检查是否需要预警补号
          this.checkPoolLow()
          endpointIndex = (endpointIndex + 1) % 2 // 切换端点
          if (endpointIndex !== 0) {
            // 切到备用端点重试同一账号：端点数固定为 2，天然有界，
            // 不应消耗"跨账号遍历预算"，否则账号多于 maxRetries 时会在试完整个池前耗尽。
            attempt--
            continue
          }
          // endpointIndex 回到 0：两个端点都已试过 → 切换到没试过的下个账号
          const nextAccount = switchToNextAccount()
          if (nextAccount && !triedIds.has(nextAccount.id) && isAllowed(nextAccount)) {
            console.log(`[ProxyServer] Auto-switching to ${nextAccount.email || nextAccount.id.slice(0, 8)} due to quota exhausted`)
            currentAccount = nextAccount
            triedIds.add(nextAccount.id)
            attempt--  // 切号不消耗同账号重试预算（由 triedIds/maxIterations 兜底）
            // 绑定白名单按 API Key 维度，不改写全局 selectedAccountIds
            if (!this.config.enableMultiAccount && !allowedIds) {
              this.config.selectedAccountIds = [nextAccount.id]
              this.events.onAccountUpdate?.(nextAccount)
            }
            continue
          }
          // 两端点皆试过且无可切换账号 → 停止（与 suspend/auth 分支一致地 break，
          // 避免无脑 continue 把剩余 maxIterations 全部空耗）
          break
        }

        // 5xx: 同账号短退避重试一次；再次 5xx 直接 fallback 到没试过的账号（瞬时故障跨账号绕过）
        if (errMsg.includes('500') || errMsg.includes('502') || errMsg.includes('503') || errMsg.includes('504')) {
          console.log('[ProxyServer] Server error, retrying')
          // 第二次及以后的 5xx → 切换账号（旧逻辑会同账号撞死）
          if (attempt > 0) {
            const nextAccount = switchToNextAccount()
            if (nextAccount && !triedIds.has(nextAccount.id)) {
              console.log(`[ProxyServer] Persistent 5xx on ${currentAccount.email || currentAccount.id.slice(0, 8)}, switching account`)
              currentAccount = nextAccount
              triedIds.add(nextAccount.id)
              attempt--  // 切号不消耗同账号重试预算
              continue
            }
          }
          await this.waitForRetry(retryDelay * (attempt + 1), signal)
          continue
        }

        // 其他错误，不重试
        break
      }
    }

    throw lastError || new Error('Unknown error')
  }

  /**
   * 常数时间字符串比较（防时序攻击）
   * 长度不同时返回 false 但仍走一次 timingSafeEqual 防止旁路
   */
  private safeStringEq(a: string, b: string): boolean {
    // Buffer.from 处理 UTF-8 编码
    const ab = Buffer.from(a, 'utf8')
    const bb = Buffer.from(b, 'utf8')
    if (ab.length !== bb.length) {
      // 仍执行一次比较保证常数时间（用 a 自身比，结果不影响）
      try { crypto.timingSafeEqual(ab, ab) } catch { /* ignore */ }
      return false
    }
    try {
      return crypto.timingSafeEqual(ab, bb)
    } catch {
      return false
    }
  }

  // 验证 API Key 并返回匹配的 Key（用于统计）
  // P0-3 使用 timingSafeEqual 防止时序攻击逐字猜 Key
  private validateApiKey(req: http.IncomingMessage): { valid: boolean; apiKey?: import('./types').ApiKey; reason?: string } {
    // 如果没有配置任何 API Key，则跳过验证
    const hasApiKeys = this.config.apiKeys && this.config.apiKeys.length > 0
    const hasLegacyKey = !!this.config.apiKey
    if (!hasApiKeys && !hasLegacyKey) return { valid: true }

    // 从 Authorization 头或 X-Api-Key 头获取 API Key
    const authHeader = req.headers['authorization'] || ''
    const apiKeyHeader = (req.headers['x-api-key'] as string) || ''

    let providedKey = ''
    // Bearer token 格式
    if (authHeader.startsWith('Bearer ')) {
      providedKey = authHeader.slice(7)
    }
    // 直接 API Key 格式
    if (!providedKey && apiKeyHeader) {
      providedKey = apiKeyHeader
    }

    if (!providedKey) return { valid: false }

    // 检查多 API Key（常数时间比较）
    if (hasApiKeys) {
      let matched: import('./types').ApiKey | undefined
      for (const k of this.config.apiKeys!) {
        if (!k.enabled || !k.key) continue
        if (this.safeStringEq(k.key, providedKey)) {
          matched = k
          // 不 break：继续遍历保持时间一致（小数量数组 OK）
        }
      }
      if (matched) {
        if (matched.creditsLimit && matched.usage.totalCredits >= matched.creditsLimit) {
          return { valid: false, reason: 'Credits limit exceeded' }
        }
        // 客户门户预付余额校验：Key 归属某客户时，余额 <= 0 或客户被禁用则拒绝
        if (matched.customerId) {
          const customer = (this.config.customers || []).find(c => c.id === matched!.customerId)
          if (!customer || !customer.enabled) {
            return { valid: false, reason: 'Credits limit exceeded' }
          }
          if (customer.creditBalance <= 0) {
            return { valid: false, reason: 'Credits limit exceeded' }
          }
        }
        return { valid: true, apiKey: matched }
      }
    }

    // 兼容旧的单 API Key（常数时间比较）
    if (hasLegacyKey && this.safeStringEq(this.config.apiKey!, providedKey)) {
      return { valid: true }
    }

    return { valid: false }
  }

  /**
   * 管理员鉴权（独立于业务 validateApiKey）。
   * 放行条件（任一）：
   *   - 提供的 Key 等于 legacy config.apiKey（运营方主 Key）
   *   - 提供的 Key 命中 config.apiKeys 中某个【未绑定 customerId】的启用 Key
   * 明确拒绝：客户在门户自助创建的 Key（带 customerId）—— 防提权。
   * 若两种管理员凭证都没配置（hasApiKeys=false 且无 legacy），保持与原 validateApiKey 一致：
   *   无任何 Key 配置时视为开放（本地默认场景），交由上层 IP 白名单等控制。
   */
  private validateAdminApiKey(req: http.IncomingMessage): boolean {
    const hasApiKeys = !!(this.config.apiKeys && this.config.apiKeys.length > 0)
    const hasLegacyKey = !!this.config.apiKey
    // 安全：管理接口 fail-closed。未配置任何 Key 时拒绝 admin（旧行为是放行，会让暴露在
    // 公网 tunnel 上的 /admin/* 完全无防护）。普通 LLM 端点的「无 Key=开放」逻辑见 validateApiKey，
    // 此处仅约束管理面。
    if (!hasApiKeys && !hasLegacyKey) return false

    const authHeader = req.headers['authorization'] || ''
    const apiKeyHeader = (req.headers['x-api-key'] as string) || ''
    let providedKey = ''
    if (typeof authHeader === 'string' && authHeader.startsWith('Bearer ')) providedKey = authHeader.slice(7)
    if (!providedKey && apiKeyHeader) providedKey = apiKeyHeader
    if (!providedKey) return false

    // legacy 主 Key = 管理员
    if (hasLegacyKey && this.safeStringEq(this.config.apiKey!, providedKey)) return true

    // config.apiKeys 中【未绑定 customerId】的启用 Key = 运营方 Key = 管理员
    // 常数时间遍历，且必须 customerId 为空才放行
    if (hasApiKeys) {
      let isAdmin = false
      for (const k of this.config.apiKeys!) {
        if (!k.enabled || !k.key) continue
        if (this.safeStringEq(k.key, providedKey) && !k.customerId) {
          isAdmin = true
          // 不 break，保持时序一致
        }
      }
      return isAdmin
    }
    return false
  }

  /**
   * P0-4 IP 访问控制
   * - deniedIPs 优先：命中即拒绝
   * - allowedIPs 配置后：必须在列表内（白名单模式）
   * - 都未配置：允许
   * 支持单 IP 和 CIDR（IPv4 / IPv6 简化处理）
   */
  private isClientIPAllowed(clientIP: string): { allowed: boolean; reason?: string } {
    if (!clientIP) return { allowed: true }
    // 规范化（::ffff:1.2.3.4 → 1.2.3.4）
    const ip = clientIP.startsWith('::ffff:') ? clientIP.slice(7) : clientIP

    const matchEntry = (entry: string): boolean => {
      const e = entry.trim()
      if (!e) return false
      // CIDR
      if (e.includes('/')) {
        return this.ipInCidr(ip, e)
      }
      return e === ip
    }

    const denied = this.config.deniedIPs?.find(matchEntry)
    if (denied) return { allowed: false, reason: `IP ${ip} matches denied entry ${denied}` }

    const allowList = this.config.allowedIPs
    if (allowList && allowList.length > 0) {
      const allowed = allowList.some(matchEntry)
      if (!allowed) return { allowed: false, reason: `IP ${ip} not in allowed list` }
    }
    return { allowed: true }
  }

  /**
   * 简化 IPv4/IPv6 CIDR 匹配（不依赖外部库）
   * IPv4 CIDR：1.2.3.0/24；IPv6 CIDR：仅前缀逐 bit 比较
   */
  private ipInCidr(ip: string, cidr: string): boolean {
    const [range, bitsStr] = cidr.split('/')
    const bits = parseInt(bitsStr, 10)
    if (!Number.isFinite(bits)) return false

    const isV4 = ip.includes('.') && range.includes('.')
    if (isV4) {
      const ipNum = this.ipv4ToInt(ip)
      const rangeNum = this.ipv4ToInt(range)
      if (ipNum < 0 || rangeNum < 0) return false
      const mask = bits === 0 ? 0 : (~0 << (32 - bits)) >>> 0
      return (ipNum & mask) === (rangeNum & mask)
    }
    // IPv6 简化：转字节数组 + 前缀逐 bit 比较
    const ipBytes = this.ipv6ToBytes(ip)
    const rangeBytes = this.ipv6ToBytes(range)
    if (!ipBytes || !rangeBytes) return false
    let bitsLeft = bits
    for (let i = 0; i < 16 && bitsLeft > 0; i++) {
      if (bitsLeft >= 8) {
        if (ipBytes[i] !== rangeBytes[i]) return false
        bitsLeft -= 8
      } else {
        const mask = (0xff << (8 - bitsLeft)) & 0xff
        if ((ipBytes[i] & mask) !== (rangeBytes[i] & mask)) return false
        bitsLeft = 0
      }
    }
    return true
  }

  private ipv4ToInt(ip: string): number {
    const parts = ip.split('.').map(p => parseInt(p, 10))
    if (parts.length !== 4 || parts.some(p => !Number.isFinite(p) || p < 0 || p > 255)) return -1
    return ((parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]) >>> 0
  }

  private ipv6ToBytes(ip: string): Uint8Array | null {
    try {
      // 简化处理：支持 :: 缩写
      const parts = ip.split('::')
      let head: string[] = []
      let tail: string[] = []
      if (parts.length === 1) {
        head = parts[0].split(':')
      } else if (parts.length === 2) {
        head = parts[0] ? parts[0].split(':') : []
        tail = parts[1] ? parts[1].split(':') : []
      } else {
        return null
      }
      const missing = 8 - head.length - tail.length
      if (missing < 0) return null
      const segments = [...head, ...new Array(missing).fill('0'), ...tail]
      const bytes = new Uint8Array(16)
      for (let i = 0; i < 8; i++) {
        const v = parseInt(segments[i] || '0', 16)
        if (!Number.isFinite(v) || v < 0 || v > 0xffff) return null
        bytes[i * 2] = (v >> 8) & 0xff
        bytes[i * 2 + 1] = v & 0xff
      }
      return bytes
    } catch {
      return null
    }
  }

  /** 取客户端真实 IP（不信任 X-Forwarded-For，仅取 socket address） */
  private getClientIP(req: http.IncomingMessage): string {
    return req.socket.remoteAddress || ''
  }

  // 记录 API Key 用量
  /**
   * 计费层：返回某模型的加价倍率。计费未启用、未配置或非正数时一律返回 1.0，
   * 因此默认行为与旧逻辑完全一致（实扣 = 原始 credit × 1）。
   */
  private modelMarkupFor(model?: string): number {
    const pricing = this.config.pricing
    if (!pricing || pricing.enabled !== true || !model) return 1
    const m = pricing.modelMarkup?.[model]
    return typeof m === 'number' && m > 0 ? m : 1
  }

  /**
   * 门户对外暴露的定价视图：仅含面向客户的安全字段（售价 / 汇率 / 官方对比 / 加价倍率），
   * 绝不包含成本价 costPerCredit 或网关费等内部毛利信息。计费未启用时返回 { enabled:false }。
   */
  private publicPricing(): {
    enabled: boolean
    bahtPerCredit?: number
    usdToBaht?: number
    kiroRetailUsdPerCredit?: number
    modelMarkup?: Record<string, number>
  } {
    const p = this.config.pricing
    if (!p || p.enabled !== true) return { enabled: false }
    return {
      enabled: true,
      bahtPerCredit: typeof p.bahtPerCredit === 'number' && p.bahtPerCredit > 0 ? p.bahtPerCredit : 0.47,
      usdToBaht: typeof p.usdToBaht === 'number' && p.usdToBaht > 0 ? p.usdToBaht : 36,
      kiroRetailUsdPerCredit: typeof p.kiroRetailUsdPerCredit === 'number' && p.kiroRetailUsdPerCredit > 0 ? p.kiroRetailUsdPerCredit : 0.02,
      modelMarkup: p.modelMarkup || {}
    }
  }

  recordApiKeyUsage(apiKeyId: string, credits: number, inputTokens: number, outputTokens: number, model?: string, path?: string, effort?: string, cacheReadTokens?: number, cacheWriteTokens?: number, sessionId?: string): void {
    if (!this.config.apiKeys) return
    const apiKey = this.config.apiKeys.find(k => k.id === apiKeyId)
    if (!apiKey) return

    const today = new Date().toISOString().split('T')[0]
    const now = Date.now()
    
    // 更新总计
    apiKey.usage.totalRequests++
    apiKey.usage.totalCredits += credits
    apiKey.usage.totalInputTokens += inputTokens
    apiKey.usage.totalOutputTokens += outputTokens
    const cacheRead = cacheReadTokens || 0
    const cacheWrite = cacheWriteTokens || 0
    apiKey.usage.totalCacheReadTokens = (apiKey.usage.totalCacheReadTokens || 0) + cacheRead
    apiKey.usage.totalCacheWriteTokens = (apiKey.usage.totalCacheWriteTokens || 0) + cacheWrite
    apiKey.lastUsedAt = now

    // 更新日统计
    if (!apiKey.usage.daily[today]) {
      apiKey.usage.daily[today] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
    }
    apiKey.usage.daily[today].requests++
    apiKey.usage.daily[today].credits += credits
    apiKey.usage.daily[today].inputTokens += inputTokens
    apiKey.usage.daily[today].outputTokens += outputTokens

    // 更新模型统计
    if (model) {
      if (!apiKey.usage.byModel) {
        apiKey.usage.byModel = {}
      }
      if (!apiKey.usage.byModel[model]) {
        apiKey.usage.byModel[model] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
      }
      apiKey.usage.byModel[model].requests++
      apiKey.usage.byModel[model].credits += credits
      apiKey.usage.byModel[model].inputTokens += inputTokens
      apiKey.usage.byModel[model].outputTokens += outputTokens
    }

    // 更新推理强度档位统计（effort 缺省时归到 'none'，保证旧客户端也有一致口径）
    const effortKey = effort && effort.trim() ? effort.trim().toLowerCase() : 'none'
    if (!apiKey.usage.byEffort) {
      apiKey.usage.byEffort = {}
    }
    if (!apiKey.usage.byEffort[effortKey]) {
      apiKey.usage.byEffort[effortKey] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
    }
    apiKey.usage.byEffort[effortKey].requests++
    apiKey.usage.byEffort[effortKey].credits += credits
    apiKey.usage.byEffort[effortKey].inputTokens += inputTokens
    apiKey.usage.byEffort[effortKey].outputTokens += outputTokens

    // 添加用量历史记录（保留最近 100 条）
    if (!apiKey.usageHistory) {
      apiKey.usageHistory = []
    }
    // markup 在此算一次，record 与 balance 扣减共用同一值，保证对账一致
    const markup = this.modelMarkupFor(model)
    apiKey.usageHistory.unshift({
      timestamp: now,
      model: model || 'unknown',
      inputTokens,
      outputTokens,
      credits,
      path: path || 'unknown',
      effort: effortKey,
      cacheReadTokens: cacheRead,
      cacheWriteTokens: cacheWrite,
      effectiveCredits: credits * markup,
      markupAtTime: markup,
      sessionId: sessionId || undefined
    })
    if (apiKey.usageHistory.length > 100) {
      apiKey.usageHistory = apiKey.usageHistory.slice(0, 100)
    }

    // 客户门户预付扣费：Key 归属某客户时，从其 creditBalance 扣减本次消耗的 credit。
    // 允许扣成负数（最后一次请求可能略微超额），下次请求 validateApiKey 会因 <=0 拒绝。
    // v1.10：启用计费层后，按模型加价倍率扣减（实扣 = 原始 credit × markup）。
    if (apiKey.customerId && credits > 0 && this.config.customers) {
      const customer = this.config.customers.find(c => c.id === apiKey.customerId)
      if (customer) {
        customer.creditBalance -= credits * markup
      }
    }

    // 触发配置保存事件
    this.events.onConfigChanged?.(this.config)
  }

  /**
   * 客户端中途断开时的用量结算：AWS 已经实际消耗了 token/credit，
   * 即便我们不再向已关闭的连接写响应，也必须把这笔用量记到客户账上，
   * 否则客户可借"发起请求→立即断开"白嫖额度。仅在有 matchedApiKey 且有正向用量时计费。
   * 账号侧统计也补记 success（请求确实成功了，只是客户没收完）。
   */
  private settleAbortedUsage(
    matchedApiKey: import('./types').ApiKey | undefined,
    accountId: string,
    usage: { credits?: number; inputTokens: number; outputTokens: number; cacheReadTokens?: number; cacheWriteTokens?: number },
    model: string | undefined,
    path: string,
    effort?: string,
    sessionId?: string
  ): void {
    const credits = usage.credits || 0
    if (credits <= 0 && usage.inputTokens <= 0 && usage.outputTokens <= 0) return
    this.accountPool.recordSuccess(accountId, usage.inputTokens + usage.outputTokens)
    if (matchedApiKey) {
      this.recordApiKeyUsage(matchedApiKey.id, credits, usage.inputTokens, usage.outputTokens, model, path, effort, usage.cacheReadTokens, usage.cacheWriteTokens, sessionId)
    }
  }

  // 返回可用的 web 工具配置（启用且有 apiKey 时），否则 null
  private getWebToolConfig(): WebToolConfig | null {
    const ws = this.config.webSearch
    if (ws && ws.enabled && ws.apiKey?.trim()) {
      return { enabled: true, provider: ws.provider, apiKey: ws.apiKey, maxRounds: ws.maxRounds }
    }
    return null
  }

  // 判断 Claude 请求是否声明了需代理执行的 web 工具（web_search/web_fetch）
  private claudeRequestUsesWebTools(request: ClaudeRequest): boolean {
    const tools = request.tools as Array<{ name?: string; type?: string }> | undefined
    if (!tools?.length) return false
    return tools.some(t => isServerWebTool(t.name, t.type) !== null)
  }

  // 应用模型映射
  /**
   * 模型白名单判定：config.allowedModels 未设置或为空 = 放行全部（向后兼容）。
   * 设置后按「精确 model ID（大小写不敏感）」比对——避免 family 级放行误开放
   * 同家族的历史/内部版本（如只想开 claude-sonnet-4.6 却连带放行 claude-3.x）。
   */
  private isModelAllowed(modelId: string): boolean {
    const allow = this.config.allowedModels
    if (!allow || allow.length === 0) return true
    // 规范化两侧再比对：剥离客户端能力后缀（[1m]）+ 版本短横转点号 + 小写，
    // 避免 Claude Code 发来的 "claude-opus-4-8[1m]" 匹配不上白名单里的 "claude-opus-4.8" 而误拦 403。
    const target = canonicalizeModelId(modelId)
    if (!target) return false
    return allow.some(m => canonicalizeModelId(m) === target)
  }

  /**
   * 解析客户端选中的 effort 变体模型 ID。
   * 仅当 effortVariantsExposed=true 且 ID 形如「{Claude base}-{effort}」时拆分，否则原样返回。
   * 调用点：每个请求 path 在 applyModelMapping / isModelAllowed 之前调用，把 model 还原成 base，
   * 并把 effort 注入请求体（见各 handler）。这样 base 走原有白名单 / 映射逻辑，零额外改动。
   */
  private resolveEffortVariant(modelId: string): { baseId: string; effort?: string } {
    if (this.config.effortVariantsExposed !== true) return { baseId: modelId }
    return splitEffortSuffix(modelId)
  }

  /**
   * 为 /v1/models 列表追加 effort 变体条目。
   * 规则：对每个「规范 Claude 模型」且「后端返回了真实 effort 枚举（thinkingEfforts 非空）」的 base，
   * 按其枚举顺序追加 {base.id}-{effort} 条目（保留 base 本身）。其余模型（auto / 第三方 / 无 effort 枚举）
   * 原样保留、不加变体。必须在 filterPickerModels 之后调用——否则变体 ID 会被按家族收敛掉。
   */
  private expandEffortVariants(models: ClientModel[]): ClientModel[] {
    const EFFORT_ORDER = EFFORT_VARIANT_SUFFIXES
    const out: ClientModel[] = []
    for (const m of models) {
      out.push(m)
      if (!parseClaudeFamilyVersion(m.id)) continue
      const efforts = (m.thinkingEfforts || []).filter(e => EFFORT_ORDER.includes(e.toLowerCase()))
      if (efforts.length === 0) continue
      // 按 EFFORT_ORDER 稳定排序，避免后端枚举顺序抖动导致列表顺序变化
      const ordered = EFFORT_ORDER.filter(e => efforts.some(x => x.toLowerCase() === e))
      for (const eff of ordered) {
        const cap = eff.charAt(0).toUpperCase() + eff.slice(1)
        out.push({
          ...m,
          id: `${m.id}-${eff}`,
          name: `${m.name} (${cap})`,
          model_name: `${m.model_name || m.name} (${cap})`,
          description: `${m.description} — effort: ${eff}`,
          reasoning: true,
          capabilities: { ...m.capabilities, reasoning: true },
          root: `${m.id}-${eff}`
        })
      }
    }
    return out
  }

  private applyModelMapping(requestedModel: string, apiKeyId?: string): string {
    const mappings = this.config.modelMappings
    if (!mappings || mappings.length === 0) return requestedModel

    // 按优先级排序（数字越小优先级越高）
    const sortedMappings = [...mappings].sort((a, b) => a.priority - b.priority)

    for (const rule of sortedMappings) {
      // 检查规则是否启用
      if (!rule.enabled) continue

      // 检查是否适用于当前 API Key
      if (rule.apiKeyIds && rule.apiKeyIds.length > 0 && apiKeyId) {
        if (!rule.apiKeyIds.includes(apiKeyId)) continue
      }

      // 检查源模型是否匹配（支持通配符 *）
      const sourcePattern = rule.sourceModel.replace(/\*/g, '.*')
      const regex = new RegExp(`^${sourcePattern}$`, 'i')
      if (!regex.test(requestedModel)) continue

      // 匹配成功，根据类型选择目标模型
      const validTargets = rule.targetModels.filter(t => t.trim())
      if (validTargets.length === 0) continue

      let targetModel: string

      if (rule.type === 'loadbalance' && validTargets.length > 1) {
        // 负载均衡：根据权重随机选择
        const weights = rule.weights || validTargets.map(() => 1)
        const totalWeight = weights.reduce((a, b) => a + b, 0)
        let random = Math.random() * totalWeight
        let selectedIndex = 0
        for (let i = 0; i < weights.length; i++) {
          random -= weights[i]
          if (random <= 0) {
            selectedIndex = i
            break
          }
        }
        targetModel = validTargets[selectedIndex]
      } else {
        // replace 或 alias：直接使用第一个目标
        targetModel = validTargets[0]
      }

      proxyLogger.info('ProxyServer', `Model mapping applied: ${requestedModel} -> ${targetModel} (rule: ${rule.name}, type: ${rule.type})`)
      return targetModel
    }

    return requestedModel
  }

  // 处理请求
  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const path = req.url || '/'
    const method = req.method || 'GET'
    const clientIP = this.getClientIP(req)
    // 客户信用预留：在 finally 中释放（仅当成功 acquire 时）
    let reservedCustomerId: string | null = null
    const controller = new AbortController()
    const abortRequest = () => {
      if (!this.isStopping && res.writableEnded) return
      if (!controller.signal.aborted) {
        controller.abort(new Error(this.isStopping ? 'Proxy server stopped' : 'Client disconnected'))
      }
    }
    this.activeRequests.add(controller)
    req.on('aborted', abortRequest)
    res.on('close', abortRequest)

    // CORS 预检
    if (method === 'OPTIONS') {
      this.setCorsHeaders(res, path, req)
      res.writeHead(204)
      res.end()
      req.off('aborted', abortRequest)
      res.off('close', abortRequest)
      this.activeRequests.delete(controller)
      return
    }

    try {
      this.setCorsHeaders(res, path, req)

      // P0-4 IP 访问控制（健康检查也走，防止扫描器）
      const ipCheck = this.isClientIPAllowed(clientIP)
      if (!ipCheck.allowed) {
        proxyLogger.warn('ProxyServer', `Blocked request from ${clientIP}: ${ipCheck.reason}`)
        this.appendAuditLog('ip_blocked', { ip: clientIP, path, reason: ipCheck.reason })
        this.sendError(res, 403, 'Forbidden')
        return
      }

      // API Key 验证（健康检查端点除外）。
      // /portal（页面+API）走自身 session；/admin 与 /admin/ 是管理面【页面本身】，登录在客户端做
      // （粘贴 operator key → 调 /admin/* 时才带 Authorization），故页面 HTML 不经此 Key 门。
      // 注意：仅放行裸 /admin 页面；/admin/* 的 API 仍需经 validateApiKey + validateAdminApiKey 两道。
      // 用去 query 后的路径判断，与下方路由（pathWithoutQuery）口径一致，避免 /admin?x 走岔。
      const bareForAuth = path.split('?')[0]
      const isAdminPage = bareForAuth === '/admin' || bareForAuth === '/admin/'
      if (path !== '/health' && path !== '/' && !path.startsWith('/portal') && !isAdminPage) {
        const authResult = this.validateApiKey(req)
        if (!authResult.valid) {
          const errorMsg = authResult.reason || 'Invalid or missing API key'
          const statusCode = authResult.reason === 'Credits limit exceeded' ? 429 : 401
          // 401 不返回 reason 详情（防止指纹爬取）
          this.sendError(res, statusCode, statusCode === 401 ? 'Unauthorized' : errorMsg,
            this.isAnthropicPath(path) ? 'anthropic' : 'openai')
          return
        }
        // 将匹配的 API Key 存储到请求对象中，用于后续统计
        ;(req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey = authResult.apiKey

        // P1-7 按 API Key（或匿名时按 IP）请求限流
        const rateLimitId = authResult.apiKey?.id || `ip:${clientIP || 'unknown'}`
        const rl = this.checkRateLimit(rateLimitId)
        if (!rl.allowed) {
          res.setHeader('Retry-After', String(Math.ceil(rl.retryAfterMs / 1000)))
          res.setHeader('X-RateLimit-Limit', String(this.config.rateLimitPerKeyPerMinute || 0))
          res.setHeader('X-RateLimit-Remaining', '0')
          this.sendError(res, 429, 'Rate limit exceeded',
            this.isAnthropicPath(path) ? 'anthropic' : 'openai')
          return
        }

        // 客户信用预留：限制单客户并发在途请求，防止并发穿透接近耗尽的预付余额
        const ownerId = authResult.apiKey?.customerId
        if (ownerId) {
          if (!this.acquireCustomerSlot(ownerId)) {
            res.setHeader('Retry-After', '5')
            this.sendError(res, 429, 'Too many concurrent requests',
              this.isAnthropicPath(path) ? 'anthropic' : 'openai')
            return
          }
          reservedCustomerId = ownerId
        }
      }

      // 记录请求
      if (this.config.logRequests) {
        proxyLogger.info('ProxyServer', `${method} ${path}`)
      }

      // 路由（移除查询参数）
      const pathWithoutQuery = path.split('?')[0]
      
      if (pathWithoutQuery === '/v1/models' || pathWithoutQuery === '/models') {
        await this.handleModels(res, controller.signal)
      } else if (pathWithoutQuery === '/v1/chat/completions' || pathWithoutQuery === '/chat/completions') {
        await this.handleOpenAIChat(req, res, controller.signal)
      } else if (pathWithoutQuery === '/v1/responses' || pathWithoutQuery === '/responses') {
        await this.handleOpenAIResponses(req, res, controller.signal)
      } else if (pathWithoutQuery === '/v1/messages' || pathWithoutQuery === '/messages' || pathWithoutQuery === '/anthropic/v1/messages') {
        await this.handleClaudeMessages(req, res, controller.signal)
      } else if (pathWithoutQuery === '/v1/messages/count_tokens' || pathWithoutQuery === '/messages/count_tokens') {
        // Claude Code token 计数端点 - 返回模拟响应
        await this.handleCountTokens(req, res, controller.signal)
      } else if (pathWithoutQuery === '/api/event_logging/batch') {
        // Claude Code 遥测端点 - 直接返回 200 OK
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ status: 'ok' }))
      } else if (pathWithoutQuery.startsWith('/v1beta/models/')) {
        // Gemini v1beta 兼容路由
        await this.handleGeminiRequest(req, res, pathWithoutQuery, controller.signal)
      } else if (pathWithoutQuery === '/v1beta/models') {
        // Gemini 模型列表
        await this.handleGeminiModels(res, controller.signal)
      } else if (pathWithoutQuery === '/health' || pathWithoutQuery === '/') {
        this.handleHealth(res)
      } else if (pathWithoutQuery === '/metrics' && this.config.enableMetrics) {
        // P2-16 Prometheus metrics
        res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4; charset=utf-8' })
        res.end(this.renderPrometheusMetrics())
      } else if (pathWithoutQuery === '/admin' || pathWithoutQuery === '/admin/') {
        // 运营方管理面页面（静态 HTML）。与 /admin/* API 同一道 adminApiExposed 开关：
        // 关闭时一律 404（不透露存在）。生产建议在隧道层再套 Cloudflare Access。
        if (this.config.adminApiExposed !== true) {
          this.sendError(res, 404, `Not Found: ${pathWithoutQuery}`)
        } else {
          this.handleAdminPage(res)
        }
      } else if (pathWithoutQuery.startsWith('/admin/')) {
        // 管理 API 端点。默认对外关闭（adminApiExposed!==true → 404，与「无此路由」无差别，
        // 不向公网透露其存在）。应用自身管理界面走 Electron IPC，不依赖此 HTTP 接口，故关闭无副作用。
        if (this.config.adminApiExposed !== true) {
          this.sendError(res, 404, `Not Found: ${pathWithoutQuery}`)
        } else {
          await this.handleAdminApi(req, res, pathWithoutQuery, controller.signal)
        }
      } else if (pathWithoutQuery === '/portal' || pathWithoutQuery === '/portal/') {
        // 客户门户页面（静态 HTML）
        if (!this.config.portalEnabled) {
          this.sendError(res, 404, `Not Found: ${pathWithoutQuery}`)
        } else {
          this.handlePortalPage(res)
        }
      } else if (pathWithoutQuery.startsWith('/portal/')) {
        // 客户门户 API
        if (!this.config.portalEnabled) {
          this.sendError(res, 404, `Not Found: ${pathWithoutQuery}`)
        } else {
          await this.handlePortalApi(req, res, pathWithoutQuery, controller.signal)
        }
      } else {
        // 记录未知路径以便调试
        console.log(`[ProxyServer] Unknown path: ${path} (method: ${method})`)
        this.sendError(res, 404, `Not Found: ${pathWithoutQuery}`)
      }
    } catch (error) {
      if (this.isAbortError(error, controller.signal)) {
        proxyLogger.info('ProxyServer', `Request aborted: ${method} ${path}`)
        return
      }
      // P0-1 body 超限 → 413
      if (error instanceof BodyTooLargeError) {
        proxyLogger.warn('ProxyServer', `Body too large from ${clientIP}: ${error.received}/${error.limit} bytes (${path})`)
        const limitMB = (error.limit / (1024 * 1024)).toFixed(0)
        const gotMB = (error.received / (1024 * 1024)).toFixed(1)
        this.sendError(res, 413,
          `Request body too large: ${gotMB}MB exceeds the ${limitMB}MB limit. Attachments (PDF/images) are base64-encoded and ~33% larger than the original file. Reduce the file size or raise "maxRequestBodyBytes" in proxy settings.`,
          this.isAnthropicPath(path) ? 'anthropic' : 'openai')
        return
      }
      // P0-5 错误响应 sanitize：500 类不吐内部 message
      console.error('[ProxyServer] Request error:', error)
      this.sendError(res, 500, 'Internal server error', this.isAnthropicPath(path) ? 'anthropic' : 'openai')
      this.events.onError?.(error as Error)
    } finally {
      req.off('aborted', abortRequest)
      res.off('close', abortRequest)
      this.activeRequests.delete(controller)
      if (reservedCustomerId) this.releaseCustomerSlot(reservedCustomerId)
    }
  }

  // 管理 API 端点
  private async handleAdminApi(req: http.IncomingMessage, res: http.ServerResponse, path: string, signal?: AbortSignal): Promise<void> {
    const method = req.method || 'GET'

    // 管理 API 需要"管理员"凭证：不能用客户在门户里自助创建的 Key。
    // 否则付费客户可凭自己的 Key 调 /admin/customers/:id/credit 给自己充值、
    // 枚举所有客户、改配置 —— 提权漏洞。validateAdminApiKey 仅放行
    // legacy apiKey 或未绑定 customerId 的 Key（运营方自己的 Key）。
    if (!this.validateAdminApiKey(req)) {
      this.sendError(res, 401, 'Admin API requires authentication')
      return
    }

    if (path === '/admin/stats' && method === 'GET') {
      // 获取详细统计
      this.handleAdminStats(res)
    } else if (path === '/admin/accounts' && method === 'GET') {
      // 获取账号列表
      this.handleAdminAccounts(res)
    } else if (path === '/admin/config' && method === 'GET') {
      // 获取配置
      this.handleAdminConfig(res)
    } else if (path === '/admin/config' && method === 'POST') {
      // 更新配置（P1-9 schema 白名单校验，防止任意字段注入）
      const body = await this.readBody(req, signal)
      let parsed: Record<string, unknown>
      try { parsed = JSON.parse(body) } catch {
        this.sendError(res, 400, 'Invalid JSON body')
        return
      }
      const safeUpdate = this.filterAdminConfigUpdate(parsed)
      this.updateConfig(safeUpdate)
      this.appendAuditLog('config_updated', { fields: Object.keys(safeUpdate) })
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, applied: Object.keys(safeUpdate), config: this.handleAdminConfigPayload() }))
    } else if (path === '/admin/audit' && method === 'GET') {
      // P2-17 审计日志
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ entries: this.auditLog.slice(-100) }))
    } else if (path === '/admin/logs' && method === 'GET') {
      // 获取最近日志
      this.handleAdminLogs(res)
    } else if (path === '/admin/cache/clear' && method === 'POST') {
      // 清除内存缓存（conversationId 映射、模型缓存、prompt cache、tiktoken 记忆缓存）
      const { clearAllCaches } = require('./kiroApi')
      const { clearTokenMemo } = require('./tokenCounter')
      const cleared = clearAllCaches()
      const promptCacheCleared = promptCacheTracker.clear()
      clearTokenMemo()
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, cleared: { ...cleared, promptCache: promptCacheCleared } }))
    } else if (path === '/admin/customers' && method === 'GET') {
      // 客户列表（脱敏：不返回密码哈希）
      this.handleAdminListCustomers(res)
    } else if (path === '/admin/customers' && method === 'POST') {
      // 创建客户（email + password [+ 初始 credit]）
      await this.handleAdminCreateCustomer(req, res, signal)
    } else if (/^\/admin\/customers\/[^/]+\/credit$/.test(path) && method === 'POST') {
      // 人工充值/扣减 credit
      let customerId: string
      try { customerId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'Customer not found')
        return
      }
      await this.handleAdminTopupCustomer(req, res, customerId, signal)
    } else if (/^\/admin\/customers\/[^/]+\/(enable|disable)$/.test(path) && method === 'POST') {
      // 启用/停用客户
      let customerId: string
      try { customerId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'Customer not found')
        return
      }
      this.handleAdminSetCustomerEnabled(res, customerId, path.endsWith('/enable'))
    } else if (/^\/admin\/customers\/[^/]+\/password$/.test(path) && method === 'POST') {
      // 管理员重置客户密码
      let customerId: string
      try { customerId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'Customer not found')
        return
      }
      await this.handleAdminResetCustomerPassword(req, res, customerId, signal)
    } else if (/^\/admin\/customers\/[^/]+$/.test(path) && method === 'DELETE') {
      // 删除客户（同时吊销其名下 Key）
      let customerId: string
      try { customerId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'Customer not found')
        return
      }
      this.handleAdminDeleteCustomer(res, customerId)
    } else if (path === '/admin/api-keys' && method === 'GET') {
      // 运营方 Key 列表（脱敏：仅 customerId 为空的 operator Key）
      this.handleAdminListApiKeys(res)
    } else if (path === '/admin/api-keys' && method === 'POST') {
      // 创建运营方 Key（format sk/simple/token；返回一次完整 key）
      await this.handleAdminCreateApiKey(req, res, signal)
    } else if (/^\/admin\/api-keys\/[^/]+\/reset-usage$/.test(path) && method === 'POST') {
      // 重置某 Key 的用量统计
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'API key not found')
        return
      }
      this.handleAdminResetApiKeyUsage(res, keyId)
    } else if (/^\/admin\/api-keys\/[^/]+$/.test(path) && method === 'PUT') {
      // 更新运营方 Key（name/enabled/creditsLimit）
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'API key not found')
        return
      }
      await this.handleAdminUpdateApiKey(req, res, keyId, signal)
    } else if (/^\/admin\/api-keys\/[^/]+$/.test(path) && method === 'DELETE') {
      // 删除运营方 Key
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'API key not found')
        return
      }
      this.handleAdminDeleteApiKey(res, keyId)
    } else if (path === '/admin/invites' && method === 'GET') {
      // 邀请码列表
      this.sendJson(res, 200, { invites: this.listInvites() })
    } else if (path === '/admin/invites' && method === 'POST') {
      // 创建邀请码
      await this.handleAdminCreateInvite(req, res, signal)
    } else if (/^\/admin\/invites\/[^/]+$/.test(path) && method === 'DELETE') {
      // 撤销未使用的邀请码
      let code: string
      try { code = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendError(res, 404, 'Invite not found')
        return
      }
      this.handleAdminRevokeInvite(res, code)
    } else if (path === '/admin/slip-records' && method === 'GET') {
      // 收款单（slip）核销记录
      this.sendJson(res, 200, { records: this.listSlipTopupRecords(50) })
    } else {
      this.sendError(res, 404, 'Admin endpoint not found')
    }
  }

  // ============ 管理 API - 客户管理 ============
  //
  // 业务逻辑集中在「programmatic」方法（createCustomer / topupCustomer / ...），
  // 不耦合 http req/res：校验失败抛 Error，调用方（HTTP handler 或 主进程 IPC）
  // 自行转成各自的响应。这样 REST(/admin/*) 与 桌面端 IPC 共用同一份实现，
  // 不会出现「改一处漏一处」的重复逻辑。

  /** 单个客户脱敏视图（不含密码哈希/salt） */
  private toCustomerView(c: Customer): CustomerView {
    return {
      id: c.id,
      email: c.email,
      name: c.name,
      enabled: c.enabled,
      createdAt: c.createdAt,
      lastLoginAt: c.lastLoginAt,
      creditBalance: c.creditBalance,
      totalToppedUp: c.totalToppedUp || 0,
      keyCount: portal.customerKeys(this.config, c.id).length,
      maxKeys: portal.maxKeysFor(this.config, c),
      topupHistory: Array.isArray(c.topupHistory) ? c.topupHistory.slice(0, 100) : []
    }
  }

  /** 客户列表（脱敏）。供 HTTP 与 IPC 共用。 */
  listCustomers(): CustomerView[] {
    return (this.config.customers || []).map(c => this.toCustomerView(c))
  }

  /**
   * 创建客户。校验失败抛 Error（message 可直接展示）。成功返回脱敏视图。
   * 共用：HTTP POST /admin/customers 与 IPC proxy-create-customer。
   */
  async createCustomer(input: { email?: string; password?: string; name?: string; creditBalance?: number; maxKeys?: number }): Promise<CustomerView> {
    const email = portal.normalizeEmail(input.email || '')
    if (!portal.isValidEmail(email)) throw new Error('Invalid email')
    if (!portal.isStrongEnoughPassword(input.password || '')) throw new Error('Password too short (min 8 chars)')
    if (portal.findCustomerByEmail(this.config, email)) throw new Error('Email already exists')
    const now = Date.now()
    const customer = await portal.buildCustomer(email, input.password!, {
      name: input.name,
      creditBalance: typeof input.creditBalance === 'number' ? input.creditBalance : 0,
      maxKeys: input.maxKeys
    }, now)
    if (!this.config.customers) this.config.customers = []
    this.config.customers.push(customer)
    this.appendAuditLog('customer_created', { id: customer.id, email: customer.email })
    this.events.onConfigChanged?.(this.config)
    return this.toCustomerView(customer)
  }

  /**
   * 创建邀请码（invite-only 注册）。code 绑定 email，客户用 Google 登录时携带 code 完成首次注册。
   * 共用：IPC proxy-create-invite。返回完整 invite（含 code，供管理员复制发送）。
   */
  createInvite(input: { email?: string; name?: string; creditBalance?: number; maxKeys?: number; expiresInDays?: number }): import('./types').PortalInvite {
    const email = portal.normalizeEmail(input.email || '')
    if (!portal.isValidEmail(email)) throw new Error('Invalid email')
    // 已是客户的 email 不必再邀请
    if (portal.findCustomerByEmail(this.config, email)) throw new Error('Email already a customer')
    const now = Date.now()
    if (!this.config.portalInvites) this.config.portalInvites = []
    // 同 email 已有未使用邀请 → 撤销旧的，避免堆积
    this.config.portalInvites = this.config.portalInvites.filter(
      i => !(portal.normalizeEmail(i.email) === email && !i.usedAt)
    )
    const days = typeof input.expiresInDays === 'number' && input.expiresInDays > 0 ? input.expiresInDays : undefined
    const invite: import('./types').PortalInvite = {
      code: portal.generateInviteCode(),
      email,
      name: input.name,
      creditBalance: typeof input.creditBalance === 'number' && input.creditBalance > 0 ? input.creditBalance : 0,
      maxKeys: input.maxKeys,
      createdAt: now,
      expiresAt: days ? now + days * 24 * 3600 * 1000 : undefined
    }
    this.config.portalInvites.push(invite)
    this.appendAuditLog('invite_created', { email, code: invite.code.slice(0, 6) + '…' })
    this.events.onConfigChanged?.(this.config)
    return invite
  }

  /** 列出所有邀请（含 code，管理端用）。 */
  listInvites(): import('./types').PortalInvite[] {
    return (this.config.portalInvites || []).slice().sort((a, b) => b.createdAt - a.createdAt)
  }

  /** 撤销（删除）一个未使用的邀请。已使用的邀请保留作审计。 */
  revokeInvite(code: string): void {
    const invite = portal.findInviteByCode(this.config, code)
    if (!invite) throw new Error('Invite not found')
    if (invite.usedAt) throw new Error('Invite already used')
    this.config.portalInvites = (this.config.portalInvites || []).filter(i => i.code !== code)
    this.appendAuditLog('invite_revoked', { email: invite.email, code: code.slice(0, 6) + '…' })
    this.events.onConfigChanged?.(this.config)
  }

  /**
   * 人工充值/扣减 credit（amount 可为负）。返回最新余额。
   * 共用：HTTP POST /admin/customers/:id/credit 与 IPC proxy-topup-customer。
   */
  topupCustomer(customerId: string, amount: number, note?: string): { creditBalance: number } {
    const amt = Number(amount)
    if (!Number.isFinite(amt) || amt === 0) throw new Error('amount must be a non-zero number')
    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) throw new Error('Customer not found')
    customer.creditBalance += amt
    if (amt > 0) customer.totalToppedUp = (customer.totalToppedUp || 0) + amt
    if (!customer.topupHistory) customer.topupHistory = []
    customer.topupHistory.unshift({ timestamp: Date.now(), amount: amt, note, by: 'admin' })
    if (customer.topupHistory.length > 100) customer.topupHistory = customer.topupHistory.slice(0, 100)
    this.appendAuditLog('customer_topup', { id: customer.id, amount: amt })
    this.events.onConfigChanged?.(this.config)
    return { creditBalance: customer.creditBalance }
  }

  // ============ 转账slip自动充值（slip2go） ============

  /**
   * 设置slip自动充值配置（含 apiSecret）。仅本地 IPC 调用，不经 HTTP admin。
   * 部分更新：仅覆盖传入字段，未传字段保留原值（apiSecret 传空字符串视为「不修改」）。
   * 返回脱敏视图（不含 apiSecret 明文）。
   */
  setSlipTopupConfig(input: Partial<import('./types').SlipTopupConfig>): import('./types').SlipTopupConfig {
    const prev = this.config.slipTopup
    const next: import('./types').SlipTopupConfig = {
      enabled: typeof input.enabled === 'boolean' ? input.enabled : (prev?.enabled ?? false),
      // apiSecret 传空 = 保留旧值（避免前端回显空串误清空密钥）
      apiSecret: (typeof input.apiSecret === 'string' && input.apiSecret.length > 0) ? input.apiSecret : (prev?.apiSecret ?? ''),
      receiverAccounts: Array.isArray(input.receiverAccounts) ? input.receiverAccounts : (prev?.receiverAccounts ?? []),
      minAmountThb: typeof input.minAmountThb === 'number' ? input.minAmountThb : prev?.minAmountThb,
      maxAmountThb: typeof input.maxAmountThb === 'number' ? input.maxAmountThb : prev?.maxAmountThb,
      freshnessHours: typeof input.freshnessHours === 'number' ? input.freshnessHours : prev?.freshnessHours,
      dailyMaxSubmitsPerCustomer: typeof input.dailyMaxSubmitsPerCustomer === 'number' ? input.dailyMaxSubmitsPerCustomer : prev?.dailyMaxSubmitsPerCustomer,
      perMinuteMaxSubmitsPerCustomer: typeof input.perMinuteMaxSubmitsPerCustomer === 'number' ? input.perMinuteMaxSubmitsPerCustomer : prev?.perMinuteMaxSubmitsPerCustomer
    }
    this.config.slipTopup = next
    this.appendAuditLog('slip_topup_config_changed', { enabled: next.enabled, receiverCount: next.receiverAccounts.length, hasSecret: !!next.apiSecret })
    this.events.onConfigChanged?.(this.config)
    return this.slipTopupConfigView()
  }

  /** slip 配置脱敏视图（apiSecret 仅返回是否已设置，不返回明文）。供 IPC/UI 读取。 */
  slipTopupConfigView(): import('./types').SlipTopupConfig {
    const c = this.config.slipTopup
    return {
      enabled: c?.enabled ?? false,
      apiSecret: c?.apiSecret ? '***' : '',
      receiverAccounts: c?.receiverAccounts ?? [],
      minAmountThb: c?.minAmountThb,
      maxAmountThb: c?.maxAmountThb,
      freshnessHours: c?.freshnessHours,
      dailyMaxSubmitsPerCustomer: c?.dailyMaxSubmitsPerCustomer,
      perMinuteMaxSubmitsPerCustomer: c?.perMinuteMaxSubmitsPerCustomer
    }
  }

  /** 最近的slip充值流水（默认 50 条，已脱敏——apiSecret 从不在记录中）。供后台对账。 */
  listSlipTopupRecords(limit = 50): SlipTopupRecord[] {
    return (this.config.slipTopupRecords || []).slice(0, Math.max(1, Math.min(limit, 500)))
  }

  /**
   * 客户端可见的slip充值信息（绝不含 apiSecret）。enabled=false 时返回 { enabled:false }，
   * 前端据此隐藏「เติมเงิน」入口。含收款账号（供客户核对转账目标）与金额限制。
   */
  private publicSlipTopupInfo(): {
    enabled: boolean
    receiverAccounts?: Array<{ accountType?: string; accountNumber?: string; accountNameTH?: string; accountNameEN?: string }>
    minAmountThb?: number
    maxAmountThb?: number
  } {
    const c = this.config.slipTopup
    if (!c || !c.enabled || !c.apiSecret) return { enabled: false }
    return {
      enabled: true,
      receiverAccounts: c.receiverAccounts || [],
      minAmountThb: c.minAmountThb ?? 1,
      maxAmountThb: c.maxAmountThb && c.maxAmountThb > 0 ? c.maxAmountThb : undefined
    }
  }

  /** 启动时从持久化流水重建 transRef 去重集合（仅 settled）。幂等，可重复调用。 */
  private rebuildSlipTransRefIndex(): void {
    this.usedSlipTransRefs.clear()
    for (const r of this.config.slipTopupRecords || []) {
      if (r.status === 'settled' && r.transRef) this.usedSlipTransRefs.add(r.transRef)
    }
  }

  /**
   * THB → credit 换算（与门户计费口径一致）。委托 portal.bahtToCredits 做整数域计算，
   * 这里只负责取 rate（pricing 未启用时回退默认 0.47 ฿/credit）。
   */
  private bahtToCredits(bahtAmount: number): { credits: number; bahtPerCredit: number } {
    const rate = (this.config.pricing?.enabled ? this.config.pricing.bahtPerCredit : undefined) || 0.47
    return { credits: portal.bahtToCredits(bahtAmount, rate), bahtPerCredit: rate }
  }

  /**
   * slip 提交限流（每分钟 + 每日，按客户）。在调用 slip2go 之前检查，保护有限的验证配额。
   * 返回 allowed=false 时调用方回 429。limit=0 表示该维度不限制。
   */
  private checkSlipSubmitRate(customerId: string): { allowed: boolean; reason?: string } {
    const cfg = this.config.slipTopup
    const perMin = cfg?.perMinuteMaxSubmitsPerCustomer ?? 5
    const perDay = cfg?.dailyMaxSubmitsPerCustomer ?? 20
    const now = Date.now()
    let b = this.slipSubmitBuckets.get(customerId)
    if (!b) { b = { minuteCount: 0, minuteStart: now, dayCount: 0, dayStart: now }; this.slipSubmitBuckets.set(customerId, b) }
    if (now - b.minuteStart >= 60_000) { b.minuteCount = 0; b.minuteStart = now }
    if (now - b.dayStart >= 86_400_000) { b.dayCount = 0; b.dayStart = now }
    if (perMin > 0 && b.minuteCount >= perMin) return { allowed: false, reason: 'per_minute' }
    if (perDay > 0 && b.dayCount >= perDay) return { allowed: false, reason: 'per_day' }
    b.minuteCount++
    b.dayCount++
    return { allowed: true }
  }

  /**
   * 构造出网 dispatcher：优先环境代理（HTTPS_PROXY 等），回退系统代理；都没有则返回 undefined（直连）。
   * 与 downloadImageDataUrl 共用同一出网策略。
   */
  private createOutboundDispatcher(): Dispatcher | undefined {
    const envProxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy
    const envAgent = safeCreateProxyAgent(envProxy)
    if (envAgent) return envAgent
    return safeCreateProxyAgent(getSystemProxy()) || undefined
  }

  /**
   * 构造 slip2go 校验条件（checkDuplicate + checkReceiver=我方账号 + checkAmount(gte minAmount)）。
   * qr-code 模式包到 payload.checkCondition，qr-image 模式直接作为 payload JSON——两者条件口径一致。
   */
  private buildSlipCheckCondition(): Record<string, unknown> {
    const cfg = this.config.slipTopup
    const checkCondition: Record<string, unknown> = { checkDuplicate: true }
    if (cfg && Array.isArray(cfg.receiverAccounts) && cfg.receiverAccounts.length > 0) {
      checkCondition.checkReceiver = cfg.receiverAccounts
    }
    const minAmount = cfg?.minAmountThb ?? 1
    if (minAmount > 0) checkCondition.checkAmount = { type: 'gte', amount: minAmount }
    return checkCondition
  }

  /** 解析 slip2go 响应：业务结果（含拒绝码）走 HTTP 200，非 200 且无 code 视为系统/鉴权错误 → 抛出（调用方 fail closed）。 */
  private async parseSlipResponse(response: globalThis.Response): Promise<{ code: number; message?: string; data?: Record<string, unknown> }> {
    const text = await response.text()
    let parsed: { code?: string | number; message?: string; data?: Record<string, unknown> }
    try { parsed = JSON.parse(text) } catch { throw new Error(`slip2go bad response (HTTP ${response.status})`) }
    if (!response.ok && !parsed.code) throw new Error(`slip2go HTTP ${response.status}`)
    const code = Number(parsed.code)
    if (!Number.isFinite(code)) throw new Error('slip2go missing result code')
    return { code, message: parsed.message, data: parsed.data }
  }

  /**
   * 调用 slip2go 验证slip（QR-Image 图片模式）。仅服务端调用，apiSecret 不外泄。
   * 客户上传的slip图片由 slip2go 侧解码 QR，再按相同条件（buildSlipCheckCondition）校验。
   * 用 multipart/form-data：file=图片，payload=条件 JSON（图片模式条件不包 checkCondition 层）。
   * 返回 slip2go 的 { code, data }；网络/HTTP 错误抛出（调用方 fail closed，不入账）。
   */
  private async verifySlipByQrImage(image: Buffer, mimeType: string, signal?: AbortSignal): Promise<{ code: number; message?: string; data?: Record<string, unknown> }> {
    const cfg = this.config.slipTopup
    if (!cfg || !cfg.apiSecret) throw new Error('slip topup not configured')

    // file 字段名 + payload 条件 JSON（图片模式下条件直接是 payload，不再包一层 checkCondition）。
    // 用 undici 的 FormData/Blob（与 undiciFetch 同源），并以 Uint8Array 包裹避免 Buffer→BlobPart 类型不匹配。
    const ext = mimeType === 'image/png' ? 'png' : mimeType === 'image/webp' ? 'webp' : 'jpg'
    const form = new UndiciFormData()
    form.append('file', new Blob([new Uint8Array(image)], { type: mimeType }), `slip.${ext}`)
    form.append('payload', JSON.stringify(this.buildSlipCheckCondition()))

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 20000)
    const abort = (): void => controller.abort(this.getAbortError(signal))
    try {
      if (signal?.aborted) throw this.getAbortError(signal)
      signal?.addEventListener('abort', abort, { once: true })
      const agent = this.createOutboundDispatcher()
      const url = 'https://connect.slip2go.com/api/verify-slip/qr-image/info'
      // 不手动设 Content-Type：交给 FormData 生成带 boundary 的 multipart 头
      const opts = {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${cfg.apiSecret}` },
        body: form,
        signal: controller.signal,
        ...(agent ? { dispatcher: agent } : {})
      } as Parameters<typeof undiciFetch>[1]
      const response = await undiciFetch(url, opts)
      return await this.parseSlipResponse(response as unknown as globalThis.Response)
    } finally {
      clearTimeout(timeout)
      signal?.removeEventListener('abort', abort)
    }
  }

  /**
   * 入账一笔已验证的slip充值。**同步**完成 check→add→credit→record→persist，中途无 await，
   * 关闭并发 double-submit 竞态（事件循环只在 await 边界切换）。
   * 返回 alreadyCredited=true 表示该 transRef 已入过账（不重复加 credit）。
   */
  private creditFromSlip(
    customerId: string,
    bahtAmount: number,
    meta: { transRef: string; referenceId: string; code: number; receiverAccount?: string; senderName?: string; slipDateTime?: string }
  ): { ok: boolean; alreadyCredited?: boolean; creditsAdded?: number; creditBalance?: number; reason?: string } {
    const transRef = meta.transRef
    if (!transRef) return { ok: false, reason: 'missing_transRef' }
    if (this.usedSlipTransRefs.has(transRef)) return { ok: true, alreadyCredited: true }

    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) return { ok: false, reason: 'customer_not_found' }

    const { credits, bahtPerCredit } = this.bahtToCredits(bahtAmount)
    if (credits <= 0) return { ok: false, reason: 'amount_too_small' }

    // —— 关键临界区：先占用 transRef，再改余额，全程无 await ——
    this.usedSlipTransRefs.add(transRef)
    customer.creditBalance += credits
    customer.totalToppedUp = (customer.totalToppedUp || 0) + credits
    if (!customer.topupHistory) customer.topupHistory = []
    customer.topupHistory.unshift({ timestamp: Date.now(), amount: credits, note: 'slip2go', by: 'slip', transRef })
    if (customer.topupHistory.length > 100) customer.topupHistory = customer.topupHistory.slice(0, 100)

    this.recordSlipTopup({
      transRef, referenceId: meta.referenceId, customerId, bahtAmount,
      creditsAdded: credits, bahtPerCreditAtTime: bahtPerCredit, code: meta.code,
      status: 'settled', receiverAccount: meta.receiverAccount, senderName: meta.senderName,
      slipDateTime: meta.slipDateTime
    })
    this.appendAuditLog('customer_topup_slip', { id: customerId, baht: bahtAmount, credits, transRef })
    this.events.onConfigChanged?.(this.config)
    return { ok: true, creditsAdded: credits, creditBalance: customer.creditBalance }
  }

  /** 记录一条slip充值流水（capped 500），含拒绝记录便于排查与对账。不单独 persist（由调用方触发）。 */
  private recordSlipTopup(rec: Omit<SlipTopupRecord, 'id' | 'verifiedAt'>): void {
    if (!this.config.slipTopupRecords) this.config.slipTopupRecords = []
    this.config.slipTopupRecords.unshift({ ...rec, id: uuidv4(), verifiedAt: Date.now() })
    if (this.config.slipTopupRecords.length > 500) {
      this.config.slipTopupRecords = this.config.slipTopupRecords.slice(0, 500)
    }
  }

  /** 启用/停用客户（停用后该客户无法登录门户、名下 Key 全部被拒）。 */
  setCustomerEnabled(customerId: string, enabled: boolean): CustomerView {
    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) throw new Error('Customer not found')
    customer.enabled = enabled
    this.appendAuditLog('customer_enabled_changed', { id: customer.id, enabled })
    this.events.onConfigChanged?.(this.config)
    return this.toCustomerView(customer)
  }

  /** 管理员重置客户密码（旧密码立即失效）。 */
  async resetCustomerPassword(customerId: string, newPassword: string): Promise<void> {
    if (!portal.isStrongEnoughPassword(newPassword || '')) throw new Error('Password too short (min 8 chars)')
    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) throw new Error('Customer not found')
    const { salt, hash } = await portal.hashPassword(newPassword)
    customer.passwordSalt = salt
    customer.passwordHash = hash
    this.appendAuditLog('customer_password_reset', { id: customer.id })
    this.events.onConfigChanged?.(this.config)
  }

  /**
   * 删除客户，并吊销其名下全部 Key（避免遗留可用 Key 仍能计费扣到已删客户）。
   * 返回被吊销的 Key 数量。
   */
  deleteCustomer(customerId: string): { revokedKeys: number } {
    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) throw new Error('Customer not found')
    const before = (this.config.apiKeys || []).length
    this.config.apiKeys = (this.config.apiKeys || []).filter(k => k.customerId !== customerId)
    const revokedKeys = before - (this.config.apiKeys || []).length
    this.config.customers = (this.config.customers || []).filter(c => c.id !== customerId)
    this.appendAuditLog('customer_deleted', { id: customerId, revokedKeys })
    this.events.onConfigChanged?.(this.config)
    return { revokedKeys }
  }

  /** HTTP 适配：客户列表 */
  private handleAdminListCustomers(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ customers: this.listCustomers() }))
  }

  /** HTTP 适配：创建客户 */
  private async handleAdminCreateCustomer(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { email?: string; password?: string; name?: string; creditBalance?: number; maxKeys?: number }
    try { parsed = JSON.parse(body) } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    try {
      const customer = await this.createCustomer(parsed)
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, customer: { id: customer.id, email: customer.email, creditBalance: customer.creditBalance } }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to create customer'
      this.sendError(res, msg === 'Email already exists' ? 409 : 400, msg)
    }
  }

  /** HTTP 适配：充值/扣减 credit */
  private async handleAdminTopupCustomer(req: http.IncomingMessage, res: http.ServerResponse, customerId: string, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { amount?: number; note?: string }
    try { parsed = JSON.parse(body) } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    try {
      const { creditBalance } = this.topupCustomer(customerId, Number(parsed.amount), parsed.note)
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, creditBalance }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to top up'
      this.sendError(res, msg === 'Customer not found' ? 404 : 400, msg)
    }
  }

  /** HTTP 适配：启用/停用客户（path 末段 enable|disable） */
  private handleAdminSetCustomerEnabled(res: http.ServerResponse, customerId: string, enabled: boolean): void {
    try {
      const view = this.setCustomerEnabled(customerId, enabled)
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, customer: view }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed'
      this.sendError(res, msg === 'Customer not found' ? 404 : 400, msg)
    }
  }

  /** HTTP 适配：重置客户密码 */
  private async handleAdminResetCustomerPassword(req: http.IncomingMessage, res: http.ServerResponse, customerId: string, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { password?: string }
    try { parsed = JSON.parse(body) } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    try {
      await this.resetCustomerPassword(customerId, parsed.password || '')
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed'
      this.sendError(res, msg === 'Customer not found' ? 404 : 400, msg)
    }
  }

  /** HTTP 适配：删除客户 */
  private handleAdminDeleteCustomer(res: http.ServerResponse, customerId: string): void {
    try {
      const { revokedKeys } = this.deleteCustomer(customerId)
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ success: true, revokedKeys }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed'
      this.sendError(res, msg === 'Customer not found' ? 404 : 400, msg)
    }
  }

  // ============ 管理 API - 运营方 API Key 管理 ============
  //
  // 运营方 Key = config.apiKeys 中【未绑定 customerId】的 Key（与门户客户自助 Key 区分）。
  // 这套 HTTP 端点把桌面端 IPC（proxy-add/update/delete/reset-api-key）的能力搬到 /admin，
  // 共用同一份 config.apiKeys 存储；变更后统一走 onConfigChanged 持久化 + appendAuditLog 审计。

  /** 运营方 Key 列表（脱敏 key，仅返回 customerId 为空的 operator Key） */
  private handleAdminListApiKeys(res: http.ServerResponse): void {
    const keys = (this.config.apiKeys || [])
      .filter(k => !k.customerId)
      .map(k => ({
        id: k.id,
        name: k.name,
        keyMasked: portal.maskKey(k.key),
        format: k.format,
        enabled: k.enabled,
        createdAt: k.createdAt,
        lastUsedAt: k.lastUsedAt,
        creditsLimit: k.creditsLimit ?? null,
        usage: {
          totalRequests: k.usage?.totalRequests || 0,
          totalCredits: k.usage?.totalCredits || 0,
          totalInputTokens: k.usage?.totalInputTokens || 0,
          totalOutputTokens: k.usage?.totalOutputTokens || 0
        }
      }))
    this.sendJson(res, 200, { apiKeys: keys })
  }

  /** 创建运营方 Key。format 决定 key 字符串格式；返回一次完整 key（之后列表只给脱敏值）。 */
  private async handleAdminCreateApiKey(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { name?: string; format?: 'sk' | 'simple' | 'token'; creditsLimit?: unknown } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    const creditsLimit = this.normalizeCreditsLimit(parsed.creditsLimit)
    if (creditsLimit === INVALID_LIMIT) {
      this.sendError(res, 400, 'creditsLimit must be a positive number')
      return
    }
    const crypto = require('crypto') as typeof import('crypto')
    const format: 'sk' | 'simple' | 'token' = parsed.format === 'simple' || parsed.format === 'token' ? parsed.format : 'sk'
    const randomHex = crypto.randomBytes(24).toString('hex')
    let key: string
    switch (format) {
      case 'simple': key = `PROXY_KEY_${randomHex.toUpperCase().substring(0, 32)}`; break
      case 'token': key = `KEY:${randomHex.substring(0, 16)}:TOKEN:${randomHex.substring(16, 32)}`; break
      default: key = `sk-${randomHex}`
    }
    const existing = this.config.apiKeys || []
    const newKey: import('./types').ApiKey = {
      id: crypto.randomUUID(),
      name: (parsed.name || '').trim().slice(0, 64) || `operator-key-${existing.length + 1}`,
      key,
      format,
      enabled: true,
      createdAt: Date.now(),
      usage: { totalRequests: 0, totalCredits: 0, totalInputTokens: 0, totalOutputTokens: 0, daily: {} }
    }
    if (typeof creditsLimit === 'number') newKey.creditsLimit = creditsLimit
    if (!this.config.apiKeys) this.config.apiKeys = []
    this.config.apiKeys.push(newKey)
    this.appendAuditLog('admin_key_created', { keyId: newKey.id, name: newKey.name, format })
    this.events.onConfigChanged?.(this.config)
    // 仅此一次返回完整 key
    this.sendJson(res, 200, { id: newKey.id, name: newKey.name, key: newKey.key, format, creditsLimit: newKey.creditsLimit ?? null })
  }

  /** 更新运营方 Key：name / enabled / creditsLimit。拒绝改 customer Key（customerId 非空）。 */
  private async handleAdminUpdateApiKey(req: http.IncomingMessage, res: http.ServerResponse, keyId: string, signal?: AbortSignal): Promise<void> {
    const key = (this.config.apiKeys || []).find(k => k.id === keyId)
    // 仅允许操作 operator Key（customerId 为空）；客户 Key 不在此端点管理范围内
    if (!key || key.customerId) {
      this.sendError(res, 404, 'API key not found')
      return
    }
    const body = await this.readBody(req, signal)
    let parsed: { name?: unknown; enabled?: unknown; creditsLimit?: unknown } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    if (typeof parsed.name === 'string') key.name = parsed.name.trim().slice(0, 64) || key.name
    if (typeof parsed.enabled === 'boolean') key.enabled = parsed.enabled
    if (parsed.creditsLimit !== undefined) {
      const creditsLimit = this.normalizeCreditsLimit(parsed.creditsLimit)
      if (creditsLimit === INVALID_LIMIT) {
        this.sendError(res, 400, 'creditsLimit must be a positive number')
        return
      }
      if (typeof creditsLimit === 'number') key.creditsLimit = creditsLimit
      else delete key.creditsLimit
    }
    this.appendAuditLog('admin_key_updated', { keyId: key.id, name: key.name, enabled: key.enabled, creditsLimit: key.creditsLimit ?? null })
    this.events.onConfigChanged?.(this.config)
    this.sendJson(res, 200, { id: key.id, name: key.name, enabled: key.enabled, creditsLimit: key.creditsLimit ?? null })
  }

  /** 删除运营方 Key。拒绝删 customer Key（应走客户门户/删客户流程）。 */
  private handleAdminDeleteApiKey(res: http.ServerResponse, keyId: string): void {
    const key = (this.config.apiKeys || []).find(k => k.id === keyId)
    if (!key || key.customerId) {
      this.sendError(res, 404, 'API key not found')
      return
    }
    this.config.apiKeys = (this.config.apiKeys || []).filter(k => k.id !== keyId)
    this.appendAuditLog('admin_key_deleted', { keyId })
    this.events.onConfigChanged?.(this.config)
    this.sendJson(res, 200, { success: true })
  }

  /** 重置运营方 Key 的用量统计（不影响 key 本身）。 */
  private handleAdminResetApiKeyUsage(res: http.ServerResponse, keyId: string): void {
    const key = (this.config.apiKeys || []).find(k => k.id === keyId)
    if (!key || key.customerId) {
      this.sendError(res, 404, 'API key not found')
      return
    }
    key.usage = { totalRequests: 0, totalCredits: 0, totalInputTokens: 0, totalOutputTokens: 0, daily: {} }
    key.usageHistory = []
    this.appendAuditLog('admin_key_usage_reset', { keyId })
    this.events.onConfigChanged?.(this.config)
    this.sendJson(res, 200, { success: true })
  }

  // ============ 管理 API - 邀请码 ============

  /** 创建邀请码。校验失败抛 Error（message 可直接展示）。 */
  private async handleAdminCreateInvite(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { email?: string; name?: string; creditBalance?: number; maxKeys?: number; expiresInDays?: number } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    try {
      const invite = this.createInvite(parsed)
      this.sendJson(res, 200, { success: true, invite })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to create invite'
      this.sendError(res, 400, msg)
    }
  }

  /** 撤销未使用的邀请码。 */
  private handleAdminRevokeInvite(res: http.ServerResponse, code: string): void {
    try {
      this.revokeInvite(code)
      this.sendJson(res, 200, { success: true })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed'
      this.sendError(res, msg === 'Invite not found' ? 404 : 400, msg)
    }
  }

  // ============ 客户门户 API ============

  /** 小工具：发送 JSON 响应 */
  private sendJson(res: http.ServerResponse, status: number, payload: unknown): void {
    res.writeHead(status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(payload))
  }

  /** 取并校验门户会话，返回当前登录客户或 null */
  private getPortalCustomer(req: http.IncomingMessage): Customer | null {
    const secret = this.config.portalSessionSecret
    if (!secret) return null
    const auth = req.headers['authorization'] || ''
    let token = ''
    if (typeof auth === 'string' && auth.startsWith('Bearer ')) token = auth.slice(7)
    if (!token) return null
    const cid = portal.verifySession(secret, token, Date.now())
    if (!cid) return null
    const customer = portal.findCustomerById(this.config, cid)
    if (!customer || !customer.enabled) return null
    return customer
  }

  private async handlePortalApi(req: http.IncomingMessage, res: http.ServerResponse, path: string, signal?: AbortSignal): Promise<void> {
    const method = req.method || 'GET'

    // 门户前端配置（公开）：是否启用 Google 登录 + client id，供登录页渲染按钮
    if (path === '/portal/config' && method === 'GET') {
      this.sendJson(res, 200, {
        googleEnabled: !!this.config.portalGoogleEnabled && !!this.config.googleClientId,
        googleClientId: this.config.portalGoogleEnabled ? (this.config.googleClientId || '') : ''
      })
      return
    }

    // Google 登录（invite-only）：不需会话，按 IP 限流
    if (path === '/portal/google' && method === 'POST') {
      const grl = this.checkPortalLoginRate(this.getClientIP(req))
      if (!grl.allowed) {
        res.setHeader('Retry-After', String(Math.ceil(grl.retryAfterMs / 1000)))
        this.sendJson(res, 429, { error: 'Too many login attempts, try again later' })
        return
      }
      await this.handlePortalGoogleLogin(req, res, signal)
      return
    }

    // 登录：不需要会话，但按 IP 限流防暴力破解
    if (path === '/portal/login' && method === 'POST') {
      const loginRl = this.checkPortalLoginRate(this.getClientIP(req))
      if (!loginRl.allowed) {
        res.setHeader('Retry-After', String(Math.ceil(loginRl.retryAfterMs / 1000)))
        this.sendJson(res, 429, { error: 'Too many login attempts, try again later' })
        return
      }
      await this.handlePortalLogin(req, res, signal)
      return
    }

    // 其余端点需要有效会话
    const customer = this.getPortalCustomer(req)
    if (!customer) {
      this.sendJson(res, 401, { error: 'Unauthorized' })
      return
    }

    if (path === '/portal/me' && method === 'GET') {
      this.sendJson(res, 200, {
        id: customer.id,
        email: customer.email,
        name: customer.name,
        creditBalance: customer.creditBalance,
        totalToppedUp: customer.totalToppedUp || 0,
        maxKeys: portal.maxKeysFor(this.config, customer),
        keyCount: portal.customerKeys(this.config, customer.id).length,
        pricing: this.publicPricing(),
        slipTopup: this.publicSlipTopupInfo()
      })
    } else if (path === '/portal/keys' && method === 'GET') {
      this.handlePortalListKeys(res, customer)
    } else if (path === '/portal/keys' && method === 'POST') {
      await this.handlePortalCreateKey(req, res, customer, signal)
    } else if (/^\/portal\/keys\/[^/]+$/.test(path) && method === 'PUT') {
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendJson(res, 404, { error: 'Key not found' })
        return
      }
      const body = await this.readBody(req, signal)
      this.handlePortalUpdateKey(res, customer, keyId, body)
    } else if (/^\/portal\/keys\/[^/]+$/.test(path) && method === 'DELETE') {
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendJson(res, 404, { error: 'Key not found' })
        return
      }
      this.handlePortalDeleteKey(res, customer, keyId)
    } else if (path === '/portal/usage' && method === 'GET') {
      this.handlePortalUsage(res, customer)
    } else if (path === '/portal/rates' && method === 'GET') {
      await this.handlePortalRates(res, signal)
    } else if (path === '/portal/topup/slip' && method === 'POST') {
      await this.handlePortalSlipTopup(req, res, customer, signal)
    } else if (path === '/portal/topup/slip/history' && method === 'GET') {
      // 仅返回该客户自己的slip充值流水（严格按 customerId 过滤，防越权查看他人）
      const records = (this.config.slipTopupRecords || [])
        .filter(r => r.customerId === customer.id)
        .slice(0, 20)
        .map(r => ({
          transRef: r.transRef,
          bahtAmount: r.bahtAmount,
          creditsAdded: r.creditsAdded,
          status: r.status,
          rejectReason: r.rejectReason,
          slipDateTime: r.slipDateTime,
          verifiedAt: r.verifiedAt
        }))
      this.sendJson(res, 200, { records })
    } else {
      this.sendJson(res, 404, { error: 'Not found' })
    }
  }

  private async handlePortalLogin(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const secret = this.config.portalSessionSecret
    if (!secret) {
      this.sendJson(res, 503, { error: 'Portal not initialized' })
      return
    }
    const body = await this.readBody(req, signal)
    let parsed: { email?: string; password?: string }
    try { parsed = JSON.parse(body) } catch {
      this.sendJson(res, 400, { error: 'Invalid JSON body' })
      return
    }
    const customer = portal.findCustomerByEmail(this.config, parsed.email || '')
    // 不区分"用户不存在"与"密码错误"，统一返回 401，防止枚举邮箱。
    // passwordless 客户（仅 Google 登录）没有 salt/hash → verifyPassword 安全返回 false。
    const ok = !!customer && customer.enabled && !!customer.passwordSalt && !!customer.passwordHash &&
      await portal.verifyPassword(parsed.password || '', customer.passwordSalt, customer.passwordHash)
    if (!ok || !customer) {
      this.sendJson(res, 401, { error: 'Invalid email or password' })
      return
    }
    customer.lastLoginAt = Date.now()
    this.events.onConfigChanged?.(this.config)
    const token = portal.signSession(secret, customer.id, portal.sessionTtlHours(this.config), Date.now())
    this.sendJson(res, 200, {
      token,
      customer: { id: customer.id, email: customer.email, name: customer.name, creditBalance: customer.creditBalance }
    })
  }

  /**
   * Google 登录（invite-only）。流程：
   *   1. 校验 Google ID token（签名 + aud + iss + exp + email_verified）
   *   2. 已绑定 googleSub 的客户 → 直接登录
   *   3. email 命中现有启用客户但未绑定 → 首次绑定 googleSub 后登录
   *   4. 否则需 invite code（与同一 email 绑定）→ 创建 passwordless 客户后登录
   *   5. 都不满足 → 403（invite-only，不放行任意 Google 账号）
   */
  private async handlePortalGoogleLogin(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const secret = this.config.portalSessionSecret
    if (!secret) { this.sendJson(res, 503, { error: 'Portal not initialized' }); return }
    if (!this.config.portalGoogleEnabled || !this.config.googleClientId) {
      this.sendJson(res, 503, { error: 'Google login not enabled' }); return
    }
    const body = await this.readBody(req, signal)
    let parsed: { credential?: string; inviteCode?: string }
    try { parsed = JSON.parse(body) } catch { this.sendJson(res, 400, { error: 'Invalid JSON body' }); return }

    const identity = await portal.verifyGoogleIdToken(parsed.credential || '', this.config.googleClientId, Date.now())
    if (!identity) { this.sendJson(res, 401, { error: 'Invalid Google token' }); return }
    if (!identity.emailVerified) { this.sendJson(res, 403, { error: 'Google email not verified' }); return }

    const now = Date.now()
    let customer = portal.findCustomerByEmail(this.config, identity.email)

    if (customer) {
      // 邮箱命中现有客户：校验 googleSub 一致或首次绑定
      if (customer.googleSub && customer.googleSub !== identity.sub) {
        this.sendJson(res, 403, { error: 'Account mismatch' }); return
      }
      if (!customer.enabled) { this.sendJson(res, 403, { error: 'Account disabled' }); return }
      if (!customer.googleSub) {
        customer.googleSub = identity.sub
        this.appendAuditLog('customer_google_linked', { id: customer.id, email: customer.email })
      }
    } else {
      // 新邮箱：必须有与该 email 绑定的有效 invite（invite-only）
      const invite = portal.findInviteByCode(this.config, parsed.inviteCode || '')
      const v = portal.validateInvite(invite, identity.email, now)
      if (!v.ok || !invite) {
        // 统一 403，不泄露 invite 状态细节（reason 仅写审计日志）
        this.appendAuditLog('google_login_rejected', { email: identity.email, reason: v.reason || 'no_invite' })
        this.sendJson(res, 403, { error: 'Invitation required' }); return
      }
      customer = portal.buildGoogleCustomer(identity.email, identity.sub, {
        name: invite.name || identity.name,
        creditBalance: invite.creditBalance,
        maxKeys: invite.maxKeys
      }, now)
      if (!this.config.customers) this.config.customers = []
      this.config.customers.push(customer)
      invite.usedAt = now
      invite.usedByCustomerId = customer.id
      this.appendAuditLog('customer_created_via_invite', { id: customer.id, email: customer.email })
    }

    customer.lastLoginAt = now
    this.events.onConfigChanged?.(this.config)
    const token = portal.signSession(secret, customer.id, portal.sessionTtlHours(this.config), now)
    this.sendJson(res, 200, {
      token,
      customer: { id: customer.id, email: customer.email, name: customer.name, creditBalance: customer.creditBalance }
    })
  }

  private handlePortalListKeys(res: http.ServerResponse, customer: Customer): void {
    const keys = portal.customerKeys(this.config, customer.id).map(k => ({
      id: k.id,
      name: k.name,
      keyMasked: portal.maskKey(k.key),
      enabled: k.enabled,
      createdAt: k.createdAt,
      lastUsedAt: k.lastUsedAt,
      totalCredits: k.usage.totalCredits,
      totalRequests: k.usage.totalRequests,
      creditsLimit: k.creditsLimit ?? null
    }))
    this.sendJson(res, 200, { keys })
  }

  private async handlePortalCreateKey(req: http.IncomingMessage, res: http.ServerResponse, customer: Customer, signal?: AbortSignal): Promise<void> {
    const max = portal.maxKeysFor(this.config, customer)
    // 先粗筛（早拒，省去读 body）
    if (portal.customerKeys(this.config, customer.id).length >= max) {
      this.sendJson(res, 403, { error: `Key limit reached (max ${max})` })
      return
    }
    const body = await this.readBody(req, signal)
    let parsed: { name?: string; creditsLimit?: unknown } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendJson(res, 400, { error: 'Invalid JSON body' })
      return
    }
    // 客户自助设置的 credit 上限（按该 Key 累计扣费计）。仅接受正数；缺省/0/null = 无限制。
    const creditsLimit = this.normalizeCreditsLimit(parsed.creditsLimit)
    if (creditsLimit === INVALID_LIMIT) {
      this.sendJson(res, 400, { error: 'creditsLimit must be a positive number' })
      return
    }
    // 读 body 后、push 前再次校验（readBody 是 await，期间可能有并发创建）——闭合竞态
    const existing = portal.customerKeys(this.config, customer.id)
    if (existing.length >= max) {
      this.sendJson(res, 403, { error: `Key limit reached (max ${max})` })
      return
    }
    const name = (parsed.name || '').trim().slice(0, 64) || `key-${existing.length + 1}`
    const newKey = portal.buildApiKey(name, customer.id, Date.now())
    if (typeof creditsLimit === 'number') newKey.creditsLimit = creditsLimit
    if (!this.config.apiKeys) this.config.apiKeys = []
    this.config.apiKeys.push(newKey)
    this.appendAuditLog('portal_key_created', { customerId: customer.id, keyId: newKey.id })
    this.events.onConfigChanged?.(this.config)
    // 创建时返回一次完整 key（仅此一次），之后列表只给脱敏值
    this.sendJson(res, 200, { id: newKey.id, name: newKey.name, key: newKey.key, creditsLimit: newKey.creditsLimit ?? null })
  }

  private handlePortalUpdateKey(res: http.ServerResponse, customer: Customer, keyId: string, body: string): void {
    const key = (this.config.apiKeys || []).find(k => k.id === keyId)
    // 严格校验归属：只能改自己的 Key
    if (!key || key.customerId !== customer.id) {
      this.sendJson(res, 404, { error: 'Key not found' })
      return
    }
    let parsed: { creditsLimit?: unknown } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendJson(res, 400, { error: 'Invalid JSON body' })
      return
    }
    const creditsLimit = this.normalizeCreditsLimit(parsed.creditsLimit)
    if (creditsLimit === INVALID_LIMIT) {
      this.sendJson(res, 400, { error: 'creditsLimit must be a positive number' })
      return
    }
    // number → 设上限；undefined（缺省/null/0）→ 清除上限（无限制）
    if (typeof creditsLimit === 'number') key.creditsLimit = creditsLimit
    else delete key.creditsLimit
    this.appendAuditLog('portal_key_updated', { customerId: customer.id, keyId: key.id, creditsLimit: key.creditsLimit ?? null })
    this.events.onConfigChanged?.(this.config)
    this.sendJson(res, 200, { id: key.id, creditsLimit: key.creditsLimit ?? null })
  }

  // 解析客户传入的 creditsLimit：正有限数 → 该数值；缺省/null/<=0 → undefined（无限制）；
  // 其它（字符串/NaN/负数以外的非法值）→ INVALID_LIMIT 哨兵，调用方据此回 400。
  private normalizeCreditsLimit(raw: unknown): number | undefined | typeof INVALID_LIMIT {
    if (raw === undefined || raw === null) return undefined
    if (typeof raw !== 'number' || !Number.isFinite(raw)) return INVALID_LIMIT
    if (raw <= 0) return undefined
    return raw
  }

  private handlePortalDeleteKey(res: http.ServerResponse, customer: Customer, keyId: string): void {
    const key = (this.config.apiKeys || []).find(k => k.id === keyId)
    // 严格校验归属：只能删自己的 Key
    if (!key || key.customerId !== customer.id) {
      this.sendJson(res, 404, { error: 'Key not found' })
      return
    }
    this.config.apiKeys = (this.config.apiKeys || []).filter(k => k.id !== keyId)
    this.appendAuditLog('portal_key_deleted', { customerId: customer.id, keyId })
    this.events.onConfigChanged?.(this.config)
    this.sendJson(res, 200, { success: true })
  }

  private handlePortalUsage(res: http.ServerResponse, customer: Customer): void {
    const keys = portal.customerKeys(this.config, customer.id)
    // 汇总名下所有 Key 的按日用量
    const dailyAgg: Record<string, { requests: number; credits: number; inputTokens: number; outputTokens: number }> = {}
    // 按模型汇总（dashboard 展示每个模型用了多少 request / credit / token）
    const modelAgg: Record<string, { requests: number; credits: number; inputTokens: number; outputTokens: number }> = {}
    // 按推理强度档位汇总（Maxplus 风格：客户能看到每个 effort 用了多少）
    const effortAgg: Record<string, { requests: number; credits: number; inputTokens: number; outputTokens: number }> = {}
    // 合并所有 Key 的最近请求历史（含 effort），供"最近用量"明细表
    const historyAll: import('./types').ApiKeyUsageRecord[] = []
    let totalRequests = 0
    let totalInputTokens = 0
    let totalOutputTokens = 0
    let totalCacheReadTokens = 0
    let totalCacheWriteTokens = 0
    for (const k of keys) {
      totalRequests += k.usage.totalRequests
      // ?? 0 兜底：token 字段是后加的，旧持久化的 key 可能缺这两个字段，
      // 直接相加会得到 NaN 并污染整份用量响应。
      totalInputTokens += k.usage.totalInputTokens ?? 0
      totalOutputTokens += k.usage.totalOutputTokens ?? 0
      totalCacheReadTokens += k.usage.totalCacheReadTokens ?? 0
      totalCacheWriteTokens += k.usage.totalCacheWriteTokens ?? 0
      for (const [day, d] of Object.entries(k.usage.daily || {})) {
        if (!dailyAgg[day]) dailyAgg[day] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
        dailyAgg[day].requests += d.requests ?? 0
        dailyAgg[day].credits += d.credits ?? 0
        dailyAgg[day].inputTokens += d.inputTokens ?? 0
        dailyAgg[day].outputTokens += d.outputTokens ?? 0
      }
      for (const [model, m] of Object.entries(k.usage.byModel || {})) {
        if (!modelAgg[model]) modelAgg[model] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
        modelAgg[model].requests += m.requests ?? 0
        modelAgg[model].credits += m.credits ?? 0
        modelAgg[model].inputTokens += m.inputTokens ?? 0
        modelAgg[model].outputTokens += m.outputTokens ?? 0
      }
      for (const [eff, e] of Object.entries(k.usage.byEffort || {})) {
        if (!effortAgg[eff]) effortAgg[eff] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
        effortAgg[eff].requests += e.requests ?? 0
        effortAgg[eff].credits += e.credits ?? 0
        effortAgg[eff].inputTokens += e.inputTokens ?? 0
        effortAgg[eff].outputTokens += e.outputTokens ?? 0
      }
      for (const rec of k.usageHistory || []) historyAll.push(rec)
    }

    // ── 把所有对客户展示的 credit 统一换算成「实际扣费 credit」（含模型加价倍率），
    //    让 dashboard 的数字与余额扣减口径一致；不向客户暴露 markup 倍率本身。
    //    byModel：每条只对应一个模型 → 精确乘该模型倍率。
    //    daily / byEffort：跨模型混合 → 用 byModel 的「实扣/原始」混合比例缩放，保证总额一致。
    let rawModelSum = 0
    let effModelSum = 0
    const byModel: Record<string, { requests: number; credits: number; inputTokens: number; outputTokens: number }> = {}
    for (const [model, m] of Object.entries(modelAgg)) {
      const eff = m.credits * this.modelMarkupFor(model)
      rawModelSum += m.credits
      effModelSum += eff
      byModel[model] = { requests: m.requests, credits: eff, inputTokens: m.inputTokens, outputTokens: m.outputTokens }
    }
    const blendedRatio = rawModelSum > 0 ? effModelSum / rawModelSum : 1
    const daily: typeof dailyAgg = {}
    for (const [day, d] of Object.entries(dailyAgg)) {
      daily[day] = { requests: d.requests, credits: d.credits * blendedRatio, inputTokens: d.inputTokens, outputTokens: d.outputTokens }
    }
    const byEffort: typeof effortAgg = {}
    for (const [eff, e] of Object.entries(effortAgg)) {
      byEffort[eff] = { requests: e.requests, credits: e.credits * blendedRatio, inputTokens: e.inputTokens, outputTokens: e.outputTokens }
    }
    const totalCredits = effModelSum

    // 最近 50 条请求明细（按时间倒序）；credits 换算成实扣值（× 模型倍率）。
    // input 拆成 uncached / cacheRead / cacheWrite 三段。注意：MITM 实测确认 Kiro 不做 prompt cache
    // 也不返回 cache 字段，因此目前 cacheRead/cacheWrite 恒为 0、uncached == 全量 input。
    // 保留这三个字段是为了：若将来 Kiro 真的返回 cache 用量，前端无需改动即可如实展示。
    const recentHistory = historyAll
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 50)
      .map(r => {
        const cacheRead = r.cacheReadTokens ?? 0
        const cacheWrite = r.cacheWriteTokens ?? 0
        const uncached = Math.max(0, r.inputTokens - cacheRead - cacheWrite)
        return {
          timestamp: r.timestamp,
          model: r.model,
          effort: r.effort || 'none',
          inputTokens: r.inputTokens,
          uncachedInputTokens: uncached,
          cacheReadTokens: cacheRead,
          cacheWriteTokens: cacheWrite,
          outputTokens: r.outputTokens,
          credits: r.credits * this.modelMarkupFor(r.model),
          // 会话分组键（MaxPlus 风格 session 视图）。旧记录无此字段 → 前端按「单条 = 单 session」兜底。
          sessionId: r.sessionId
        }
      })

    this.sendJson(res, 200, {
      creditBalance: customer.creditBalance,
      totalCredits,
      totalRequests,
      totalInputTokens,
      totalOutputTokens,
      totalCacheReadTokens,
      totalCacheWriteTokens,
      daily,
      byModel,
      byEffort,
      recentHistory,
      pricing: this.publicPricing()
    })
  }

  /**
   * 门户费率表：返回当前 Kiro 各模型的计费倍率（rateMultiplier / rateUnit），
   * 供客户在 dashboard 查看"哪个模型每次请求大概扣多少 credit"。
   * 复用 getAvailableModels（带 5 分钟缓存）+ filterPickerModels，口径与 /v1/models 一致；
   * 列表无法获取时返回空数组（dashboard 端会显示占位提示），不报错阻塞页面。
   */
  private async handlePortalRates(res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    let rates: Array<{ id: string; name: string; rateMultiplier: number | null; rateUnit: string | null; maxInputTokens: number | null; maxOutputTokens: number | null }> = []
    try {
      const { models } = await this.getAvailableModels(signal)
      // 客户费率表只展示我们实际对外提供的 3 个 Claude 家族（Opus/Sonnet/Haiku）。
      // filterPickerModels 已把每个家族收敛到最新版本，这里再按 Claude 家族过滤掉
      // Auto / 第三方模型（Deepseek/MiniMax/GLM/Qwen 等）——它们不在售卖范围内。
      rates = filterPickerModels(models)
        .filter(m => parseClaudeFamilyVersion(m.id))
        .filter(m => this.isModelAllowed(m.id))
        .map(m => ({
        id: m.id,
        name: modelDisplayName(m.id, m.name),
        rateMultiplier: typeof m.rateMultiplier === 'number' ? m.rateMultiplier : null,
        rateUnit: m.rateUnit || null,
        maxInputTokens: m.maxInputTokens ?? null,
        maxOutputTokens: m.maxOutputTokens ?? null
      }))
    } catch (e) {
      if (this.isAbortError(e, signal)) throw e
      proxyLogger.warn('ProxyServer', `handlePortalRates failed: ${(e as Error).message}`)
    }
    this.sendJson(res, 200, { models: rates })
  }

  /**
   * POST /portal/topup/slip — 客户上传转账slip图片，经 slip2go（qr-image）验证后自动入账 credit。
   * 安全要点（详见各 gate 注释）：金额只取 slip2go 返回值；收款人二次核对；transRef 去重；
   * 并发 in-flight 锁；限流保护 slip2go 配额；任何 slip2go 异常 fail closed（不入账）。
   */
  private async handlePortalSlipTopup(req: http.IncomingMessage, res: http.ServerResponse, customer: Customer, signal?: AbortSignal): Promise<void> {
    const cfg = this.config.slipTopup
    // 1) 功能开关：未配置/未启用 → 404（对外表现为「该端点不存在」）
    if (!cfg || !cfg.enabled || !cfg.apiSecret) {
      this.sendJson(res, 404, { error: 'Not found' })
      return
    }

    // 2) 限流（调用 slip2go 之前，保护有限配额）
    const rl = this.checkSlipSubmitRate(customer.id)
    if (!rl.allowed) {
      this.sendJson(res, 429, { error: 'Too many slip submissions, try again later' })
      return
    }

    // 3) 读取并校验输入（本地，不耗配额）：JSON { imageBase64, mimeType }
    let body: string
    try { body = await this.readBody(req, signal) } catch (e) {
      if (this.isAbortError(e, signal)) return
      this.sendJson(res, 400, { error: 'Invalid request body' }); return
    }
    let parsed: { imageBase64?: string; mimeType?: string }
    try { parsed = JSON.parse(body) } catch { this.sendJson(res, 400, { error: 'Invalid JSON body' }); return }

    const ALLOWED_MIME = ['image/jpeg', 'image/png', 'image/webp']
    const mimeType = (parsed.mimeType || '').toLowerCase()
    if (!ALLOWED_MIME.includes(mimeType)) {
      this.sendJson(res, 400, { error: 'Unsupported image type (jpeg/png/webp only)' }); return
    }
    // 去掉可能的 data URL 前缀（data:image/...;base64,），仅保留 base64 主体
    const rawB64 = (parsed.imageBase64 || '').replace(/^data:[^,]*,/, '').trim()
    if (!rawB64) { this.sendJson(res, 400, { error: 'Missing image' }); return }
    // base64 体积上限 ~8MB（解码后约 6MB），足够手机截图；过大直接拒，保护内存与上游
    if (rawB64.length > 8 * 1024 * 1024) {
      this.sendJson(res, 413, { error: 'Image too large (max ~6MB)' }); return
    }
    let image: Buffer
    try {
      image = Buffer.from(rawB64, 'base64')
    } catch { this.sendJson(res, 400, { error: 'Invalid image data' }); return }
    if (image.length < 100) { this.sendJson(res, 400, { error: 'Invalid image data' }); return }

    // 4) in-flight 锁（同步 check→add，无 await，防同一slip并发触发重复验证/入账）
    const slipKey = crypto.createHash('sha256').update(image).digest('hex')
    if (this.inFlightSlipKeys.has(slipKey)) {
      this.sendJson(res, 409, { error: 'This slip is being processed' }); return
    }
    this.inFlightSlipKeys.add(slipKey)

    try {
      // 5) 调用 slip2go（fail closed：网络/HTTP/鉴权异常 → 502，不入账）
      let result: { code: number; message?: string; data?: Record<string, unknown> }
      try {
        result = await this.verifySlipByQrImage(image, mimeType, signal)
      } catch (e) {
        if (this.isAbortError(e, signal)) return
        proxyLogger.warn('ProxyServer', `slip2go verify failed: ${(e as Error).message}`)
        this.sendJson(res, 502, { error: 'Slip verification service unavailable' }); return
      }

      const code = result.code
      const data = (result.data || {}) as Record<string, unknown>

      // 6) 结果码 gate：仅 200200（Slip is Valid，含 checkReceiver/checkAmount/checkDuplicate 全通过）才入账
      if (!portal.isSlipCreditable(code)) {
        const reason = this.slipRejectReason(code)
        // 记录拒绝流水（便于客户查询与对账），但不入账、不持久化 transRef
        this.recordSlipRejection(customer.id, data, code, reason)
        this.events.onConfigChanged?.(this.config)
        this.sendJson(res, 200, { ok: false, code, reason })
        return
      }

      // 7) 金额 gate：只取 slip2go 的 data.amount（绝不信任客户端），并校验范围
      const amount = typeof data.amount === 'number' ? data.amount : Number(data.amount)
      if (!Number.isFinite(amount) || amount <= 0) {
        this.sendJson(res, 200, { ok: false, code, reason: 'invalid_amount' }); return
      }
      const minAmt = cfg.minAmountThb ?? 1
      const maxAmt = cfg.maxAmountThb ?? 0
      if (amount < minAmt) { this.sendJson(res, 200, { ok: false, code, reason: 'below_min' }); return }
      if (maxAmt > 0 && amount > maxAmt) {
        // 异常大额 → 不自动入账，转人工核对（fail closed）
        this.recordSlipRejection(customer.id, data, code, 'above_max')
        this.events.onConfigChanged?.(this.config)
        this.sendJson(res, 200, { ok: false, code, reason: 'above_max' }); return
      }

      // 8) 收款人二次核对（defense in depth，不只信 slip2go 的 200200）
      if (!this.slipReceiverMatchesOurs(data)) {
        this.recordSlipRejection(customer.id, data, code, 'receiver_mismatch')
        this.events.onConfigChanged?.(this.config)
        this.sendJson(res, 200, { ok: false, code, reason: 'receiver_mismatch' }); return
      }

      // 9) 新鲜度 gate：slip 日期需在 freshnessHours 内，且不可是未来（容忍少量时钟偏移）
      const freshnessHours = cfg.freshnessHours ?? 48
      const slipDateTime = typeof data.dateTime === 'string' ? data.dateTime : undefined
      const freshness = portal.slipFreshness(slipDateTime, freshnessHours, Date.now())
      if (freshness === 'too_old') { this.sendJson(res, 200, { ok: false, code, reason: 'slip_too_old' }); return }
      if (freshness === 'future') { this.sendJson(res, 200, { ok: false, code, reason: 'slip_future_date' }); return }

      // 10) transRef gate + 入账（creditFromSlip 内部为同步临界区，去重 + 加余额 + 流水 + persist）
      const transRef = typeof data.transRef === 'string' ? data.transRef : ''
      if (!transRef) { this.sendJson(res, 200, { ok: false, code, reason: 'missing_transRef' }); return }

      const credited = this.creditFromSlip(customer.id, amount, {
        transRef,
        referenceId: typeof data.referenceId === 'string' ? data.referenceId : '',
        code,
        receiverAccount: this.extractReceiverAccount(data),
        senderName: this.extractSenderName(data),
        slipDateTime
      })

      if (credited.alreadyCredited) {
        this.sendJson(res, 200, { ok: false, code, reason: 'already_credited', transRef }); return
      }
      if (!credited.ok) {
        this.sendJson(res, 200, { ok: false, code, reason: credited.reason || 'credit_failed' }); return
      }

      // 11) 成功响应（不回传 apiSecret 或任何内部字段）
      this.sendJson(res, 200, {
        ok: true,
        creditsAdded: credited.creditsAdded,
        bahtAmount: amount,
        newBalance: credited.creditBalance,
        transRef
      })
    } finally {
      this.inFlightSlipKeys.delete(slipKey)
    }
  }

  /** slip2go 结果码 → 客户可读的拒绝原因（委托 portal 纯函数）。 */
  private slipRejectReason(code: number): string {
    return portal.slipRejectReason(code)
  }

  /** 记录一条slip拒绝流水（不入账、不写 transRef 去重集合）。 */
  private recordSlipRejection(customerId: string, data: Record<string, unknown>, code: number, reason: string): void {
    this.recordSlipTopup({
      transRef: typeof data.transRef === 'string' ? data.transRef : '',
      referenceId: typeof data.referenceId === 'string' ? data.referenceId : '',
      customerId,
      bahtAmount: typeof data.amount === 'number' ? data.amount : Number(data.amount) || 0,
      creditsAdded: 0,
      bahtPerCreditAtTime: (this.config.pricing?.enabled ? this.config.pricing.bahtPerCredit : undefined) || 0.47,
      code,
      status: 'rejected',
      rejectReason: reason,
      receiverAccount: this.extractReceiverAccount(data),
      senderName: this.extractSenderName(data),
      slipDateTime: typeof data.dateTime === 'string' ? data.dateTime : undefined
    })
  }

  /** 从 slip2go data 提取收款账号（bank.account 优先，回退 proxy.account），用于流水与核对。 */
  private extractReceiverAccount(data: Record<string, unknown>): string | undefined {
    const receiver = data.receiver as Record<string, unknown> | undefined
    const account = receiver?.account as Record<string, unknown> | undefined
    const bank = account?.bank as Record<string, unknown> | undefined
    const proxy = account?.proxy as Record<string, unknown> | undefined
    const bankAcc = typeof bank?.account === 'string' ? bank.account : undefined
    const proxyAcc = typeof proxy?.account === 'string' ? proxy.account : undefined
    return bankAcc || proxyAcc
  }

  /** 从 slip2go data 提取付款人姓名（部分脱敏），用于流水显示。 */
  private extractSenderName(data: Record<string, unknown>): string | undefined {
    const sender = data.sender as Record<string, unknown> | undefined
    const account = sender?.account as Record<string, unknown> | undefined
    return typeof account?.name === 'string' ? account.name : undefined
  }

  /**
   * 服务端二次核对收款人是否为我方账号（委托 portal.slipReceiverMatches 纯函数）。
   * 这里只负责从 slip2go data 抽出收款账号/姓名。
   */
  private slipReceiverMatchesOurs(data: Record<string, unknown>): boolean {
    const receiver = data.receiver as Record<string, unknown> | undefined
    const account = receiver?.account as Record<string, unknown> | undefined
    const recvName = typeof account?.name === 'string' ? account.name : undefined
    return portal.slipReceiverMatches(
      this.config.slipTopup?.receiverAccounts || [],
      this.extractReceiverAccount(data),
      recvName
    )
  }

  /** 客户门户静态页面（自包含 HTML，无外部依赖） */
  private handlePortalPage(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(PORTAL_HTML)
  }

  /** 运营方管理面静态页面（自包含 HTML）。仅在 adminApiExposed=true 时由路由放行；建议外层套 Cloudflare Access。 */
  private handleAdminPage(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(ADMIN_HTML)
  }

  // 管理 API - 详细统计
  private handleAdminStats(res: http.ServerResponse): void {
    const stats = this.getStats()
    const accountStats: Record<string, unknown> = {}
    stats.accountStats.forEach((v, k) => { accountStats[k] = v })

    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      totalRequests: stats.totalRequests,
      successRequests: stats.successRequests,
      failedRequests: stats.failedRequests,
      totalTokens: stats.totalTokens,
      inputTokens: stats.inputTokens,
      outputTokens: stats.outputTokens,
      uptime: Date.now() - stats.startTime,
      startTime: stats.startTime,
      accountStats,
      recentRequests: stats.recentRequests.slice(-50)
    }))
  }

  // 管理 API - 账号列表
  private handleAdminAccounts(res: http.ServerResponse): void {
    const accounts = this.accountPool.getAllAccounts().map(acc => ({
      id: acc.id,
      email: acc.email,
      isAvailable: acc.isAvailable !== false,
      lastUsed: acc.lastUsed,
      requestCount: acc.requestCount || 0,
      errorCount: acc.errorCount || 0,
      expiresAt: acc.expiresAt,
      authMethod: acc.authMethod
    }))

    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      total: accounts.length,
      available: accounts.filter(a => a.isAvailable).length,
      accounts
    }))
  }

  /**
   * P1-12 构造脱敏后的配置（apiKeys[].key 全部脱敏，tls 私钥不返回）
   * 暴露给 /admin/config GET
   */
  private handleAdminConfigPayload(): Record<string, unknown> {
    const config = this.getConfig()
    const maskKey = (k: string | undefined): string | undefined => {
      if (!k) return undefined
      if (k.length <= 8) return '***'
      return `${k.slice(0, 4)}***${k.slice(-4)}`
    }
    return {
      ...config,
      apiKey: maskKey(config.apiKey),
      apiKeys: config.apiKeys?.map(k => ({ ...k, key: maskKey(k.key) || '***' })),
      tls: config.tls ? { enabled: config.tls.enabled, hasCert: !!(config.tls.cert || config.tls.certPath), hasKey: !!(config.tls.key || config.tls.keyPath) } : undefined,
      // 绝不外泄门户签名密钥（泄露即可伪造任意客户会话）
      portalSessionSecret: config.portalSessionSecret ? '***' : undefined,
      // 客户列表脱敏：移除密码 salt/hash，仅保留运营可见的非敏感字段
      customers: config.customers?.map(c => ({
        id: c.id,
        email: c.email,
        name: c.name,
        enabled: c.enabled,
        createdAt: c.createdAt,
        lastLoginAt: c.lastLoginAt,
        creditBalance: c.creditBalance,
        totalToppedUp: c.totalToppedUp || 0
      }))
    }
  }

  // 管理 API - 配置
  private handleAdminConfig(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(this.handleAdminConfigPayload()))
  }

  /**
   * P1-9 admin/config POST 字段白名单过滤
   * 仅允许"可远程改"的字段；apiKeys/apiKey 等敏感字段必须通过本地 IPC 改
   */
  private filterAdminConfigUpdate(input: Record<string, unknown>): Partial<ProxyConfig> {
    const allowed: Array<keyof ProxyConfig> = [
      'enabled', 'enableMultiAccount', 'logRequests', 'logStreamEvents',
      'maxConcurrent', 'maxRetries', 'retryDelayMs', 'preferredEndpoint',
      'tokenRefreshBeforeExpiry', 'autoStart', 'clientDrivenToolExecution',
      'disableTools', 'payloadSizeLimitKB', 'enableTokenBufferReserve',
      'tokenBufferReserve', 'autoSwitchOnQuotaExhausted', 'accountSelectionStrategy',
      'multiAccountSelectionMode', 'multiAccountGroupIds', 'modelMappings', 'allowedModels',
      'effortVariantsExposed',
      'maxRequestBodyBytes', 'allowedIPs', 'deniedIPs',
      'rateLimitPerKeyPerMinute', 'sessionAffinityEnabled',
      'keepAliveTimeoutMs', 'headersTimeoutMs', 'recentRequestsLimit',
      'enableMetrics', 'apiKeyGroupBindings', 'enableAuditLog', 'poolLowThreshold',
      'portalEnabled', 'portalSessionTtlHours', 'portalDefaultMaxKeys',
      'portalGoogleEnabled', 'googleClientId', 'adminApiExposed', 'portalAllowedOrigins',
      'portalMaxConcurrentPerCustomer', 'perfDiagnostics'
      // 故意排除：port / host / apiKey / apiKeys / tls / fallbackPort / allowExternalWithoutApiKey
      // 也排除：customers / portalSessionSecret —— 含密码哈希与签名密钥，仅本地 IPC / 专用 admin 端点改
    ]
    const out: Partial<ProxyConfig> = {}
    for (const key of allowed) {
      if (key in input) {
        (out as Record<string, unknown>)[key] = input[key as string]
      }
    }
    return out
  }

  // 管理 API - 日志
  private handleAdminLogs(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      recentRequests: this.stats.recentRequests.slice(-100)
    }))
  }

  // 设置 CORS 头
  private setCorsHeaders(res: http.ServerResponse, path?: string, req?: http.IncomingMessage): void {
    const p = (path || '').split('?')[0]
    const isSensitive = p.startsWith('/admin/') || p === '/admin' || p.startsWith('/portal')
    if (isSensitive) {
      // 管理面 / 门户：不发通配 origin。仅当请求 Origin 命中白名单才回显该 origin（带 Vary: Origin）。
      // 防止任意网站用浏览器脚本读取 portal/admin 的响应（凭证/客户数据）。
      const origin = (req?.headers['origin'] as string) || ''
      const allow = this.config.portalAllowedOrigins || []
      if (origin && allow.includes(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin)
        res.setHeader('Vary', 'Origin')
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Api-Key')
      }
      // 未命中白名单：不设任何 CORS 头（同源仍可用；跨源浏览器读不到）
      return
    }
    // LLM 代理路径（/v1/* 等）：保留通配，兼容各类客户端 SDK（无 cookie，凭证走 header）
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Api-Key, anthropic-version, anthropic-beta, x-api-key, x-stainless-os, x-stainless-lang, x-stainless-package-version, x-stainless-runtime, x-stainless-runtime-version, x-stainless-arch')
    res.setHeader('Access-Control-Expose-Headers', 'x-request-id, x-ratelimit-limit-requests, x-ratelimit-limit-tokens, x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens, x-ratelimit-reset-requests, x-ratelimit-reset-tokens')
  }

  private isAnthropicPath(path: string): boolean {
    const pathWithoutQuery = path.split('?')[0]
    return pathWithoutQuery === '/v1/messages'
      || pathWithoutQuery === '/messages'
      || pathWithoutQuery === '/anthropic/v1/messages'
      || pathWithoutQuery === '/v1/messages/count_tokens'
      || pathWithoutQuery === '/messages/count_tokens'
  }

  private getAnthropicErrorType(status: number): string {
    if (status === 400) return 'invalid_request_error'
    if (status === 401) return 'authentication_error'
    if (status === 403) return 'permission_error'
    if (status === 404) return 'not_found_error'
    if (status === 429) return 'rate_limit_error'
    return 'api_error'
  }

  private buildClaudeUsage(
    usage: { inputTokens: number; outputTokens: number; cacheWriteTokens?: number; cacheReadTokens?: number }
  ): { input_tokens?: number; output_tokens: number; cache_creation_input_tokens?: number; cache_read_input_tokens?: number } {
    // 仅使用 Kiro 后端返回的真实 cache tokens。
    // 实测（MITM 抓取真实 kiro-cli，2026-06）：runtime.kiro.dev 不返回任何 cache 字段，
    // 也不读取请求里的 cachePoint —— Kiro 没有 Anthropic 风格的 prompt cache。
    // 因此这里不再用本地 promptCacheTracker 的模拟值兜底，避免向客户/客户端报告并不存在的缓存命中。
    // 若将来 Kiro 真的返回 cacheReadTokens/cacheWriteTokens，这里会如实透传。
    const cacheWrite = usage.cacheWriteTokens || 0
    const cacheRead = usage.cacheReadTokens || 0
    // Kiro 的 inputTokens 是全量（含缓存），Anthropic API 规范中 input_tokens 不含缓存部分
    // 需要扣除 cache tokens 避免客户端双重计费
    const adjustedInput = Math.max(0, usage.inputTokens - cacheWrite - cacheRead)
    return {
      input_tokens: adjustedInput,
      output_tokens: usage.outputTokens,
      ...(cacheWrite ? { cache_creation_input_tokens: cacheWrite } : {}),
      ...(cacheRead ? { cache_read_input_tokens: cacheRead } : {})
    }
  }

  private estimateTokenCount(value: unknown): number {
    if (value === null || value === undefined) return 0
    if (typeof value === 'string') return Math.ceil(value.length / 4)
    if (typeof value === 'number' || typeof value === 'boolean') return 1
    if (Array.isArray(value)) {
      return value.reduce<number>((total, item) => total + this.estimateTokenCount(item), 0)
    }
    if (typeof value !== 'object') return 0
    const record = value as Record<string, unknown>
    if (record.type === 'text' || record.type === 'input_text' || record.type === 'output_text') return this.estimateTokenCount(record.text) + 4
    if (record.type === 'thinking') return this.estimateTokenCount(record.thinking) + this.estimateTokenCount(record.signature) + 4
    if (record.type === 'redacted_thinking') return 8
    if (record.type === 'image' || record.type === 'input_image') return IMAGE_TOKEN_ESTIMATE
    if (record.type === 'document' || record.type === 'input_file') return this.estimateDocumentTokens(record)
    if (record.type === 'tool_use') return this.estimateTokenCount(record.name) + this.estimateTokenCount(record.input) + 12
    if (record.type === 'tool_result') return this.estimateTokenCount(record.content) + 8
    if (typeof record.role === 'string' && 'content' in record) return this.estimateTokenCount(record.content) + 4
    if (typeof record.name === 'string' && 'input_schema' in record) return this.estimateTokenCount(record.name) + this.estimateTokenCount(record.description) + this.estimateTokenCount(record.input_schema) + 32
    return Object.entries(record).reduce<number>((total, [key, item]) => key === 'cache_control' ? total : total + this.estimateTokenCount(item), 0)
  }

  /**
   * 估算 document / input_file 块的 token 数（按 base64 解码后的字节数）。
   *
   * 关键：绝不对 base64 字符串本身按 length/4 计 token——一个 3MB 的 PDF 的 base64
   * 约 400 万字符，会被算成 ~100 万 token，导致 count_tokens / message_start 报出虚高的
   * 上下文用量，进而打乱 Claude Code 的 autocompact 判断（看起来“超过 1M”）。
   * 这里把附件按解码字节折算，文件名/标题等元数据再单独计入。
   */
  private estimateDocumentTokens(record: Record<string, unknown>): number {
    // 元数据（标题/文件名）正常按文本估算
    let meta = this.estimateTokenCount(record.title) + this.estimateTokenCount(record.name) + this.estimateTokenCount(record.filename) + 8

    // 提取 base64 与格式提示，支持 Claude（source.data/media_type）与 OpenAI（file_data）两种结构
    let b64 = ''
    let format: string | undefined
    const source = record.source as Record<string, unknown> | undefined
    if (source && typeof source === 'object') {
      if (typeof source.data === 'string') b64 = source.data
      else if (typeof source.bytes === 'string') b64 = source.bytes
      if (typeof source.media_type === 'string') format = source.media_type
      // source.type === 'text'（纯文本附件）：data 不是 base64，按文本估算即可
      if (source.type === 'text' && typeof source.data === 'string') {
        return meta + this.estimateTokenCount(source.data)
      }
    }
    if (!b64 && typeof record.file_data === 'string') {
      // OpenAI: 可能是 data URL（data:application/pdf;base64,xxx）
      const m = record.file_data.match(/^data:([^;]+);base64,(.+)$/)
      if (m) { format = format || m[1]; b64 = m[2] } else { b64 = record.file_data }
    }
    if (typeof record.filename === 'string' && !format) format = record.filename

    return meta + estimateBase64DocumentTokens(b64, format)
  }

  /**
   * 估算 Kiro payload 的 token 数（binary-aware 兜底用）。
   * 把 images/documents 的 base64 单独按解码字节折算，其余文本走 ~3.5 字符/token，
   * 避免 JSON.stringify(payload) 把数 MB 的 base64 计成上百万 token。
   */
  private estimateKiroPayloadTokens(payload: ReturnType<typeof claudeToKiro>): number {
    type KiroMsg = ReturnType<typeof claudeToKiro>['conversationState']['currentMessage']
    type KiroInput = NonNullable<KiroMsg['userInputMessage']>
    let tokens = 0
    const accountFor = (input?: KiroInput): void => {
      if (!input) return
      const { images, documents, ...rest } = input
      tokens += Math.ceil(Buffer.byteLength(JSON.stringify(rest), 'utf-8') / 3.5)
      tokens += (images?.length ?? 0) * IMAGE_TOKEN_ESTIMATE
      for (const doc of documents ?? []) tokens += estimateBase64DocumentTokens(doc.source?.bytes || '', doc.format)
    }
    const cs = payload.conversationState
    accountFor(cs.currentMessage?.userInputMessage)
    for (const msg of cs.history ?? []) {
      if (msg.userInputMessage) accountFor(msg.userInputMessage)
      else if (msg.assistantResponseMessage) tokens += Math.ceil(Buffer.byteLength(JSON.stringify(msg.assistantResponseMessage), 'utf-8') / 3.5)
    }
    return Math.max(1, tokens)
  }

  // 健康检查
  private handleHealth(res: http.ServerResponse): void {
    const stats = this.getStats()
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      status: 'ok',
      version: '1.0.0',
      accounts: this.accountPool.size,
      availableAccounts: this.accountPool.availableCount,
      stats: {
        totalRequests: stats.totalRequests,
        successRequests: stats.successRequests,
        failedRequests: stats.failedRequests,
        totalTokens: stats.totalTokens,
        uptime: Date.now() - stats.startTime
      }
    }))
  }

  // Claude Code token 计数（模拟响应）
  private async handleCountTokens(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    try {
      this.throwIfAborted(signal)
      const body = await this.readBody(req, signal)
      this.throwIfAborted(signal)
      const request = JSON.parse(body) as Partial<ClaudeRequest>
      if (!Array.isArray(request.messages)) {
        throw new Error('count_tokens requires messages')
      }
      const estimatedTokens = Math.max(1, this.estimateTokenCount(request.system) + this.estimateTokenCount(request.messages) + this.estimateTokenCount(request.tools))
      
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ input_tokens: estimatedTokens }))
    } catch (error) {
      if (this.isAbortError(error, signal)) return
      this.sendError(res, 400, error instanceof Error ? error.message : 'Invalid request body', 'anthropic')
    }
  }

  // Gemini v1beta 模型列表
  private async handleGeminiModels(res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const result = await this.getAvailableModels(signal)
    const geminiModels = result.models.filter(m => this.isModelAllowed(m.id)).map(m => ({
      name: `models/${m.id}`,
      version: '001',
      displayName: m.name || m.id,
      description: m.description || '',
      inputTokenLimit: m.maxInputTokens || 200000,
      outputTokenLimit: m.maxOutputTokens || 64000,
      supportedGenerationMethods: ['generateContent', 'streamGenerateContent']
    }))
    this.throwIfResponseClosed(res, signal)
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ models: geminiModels }))
  }

  // Gemini v1beta generateContent / streamGenerateContent
  private async handleGeminiRequest(req: http.IncomingMessage, res: http.ServerResponse, path: string, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const geminiReq = JSON.parse(body)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey

    // 解析路径: /v1beta/models/{model}:{method}
    const match = path.match(/\/v1beta\/models\/([^:]+):(\w+)/)
    if (!match) {
      this.sendError(res, 400, 'Invalid Gemini endpoint path')
      return
    }
    const [, rawModelId, method] = match
    const isStream = method === 'streamGenerateContent'

    // effort 变体：从 Gemini model 段还原 base + 取出 effort，注入下方 openaiRequest。
    const evGem = this.resolveEffortVariant(rawModelId)
    const modelId = evGem.baseId

    // 将 Gemini 请求转为 OpenAI 格式
    const messages: OpenAIMessage[] = []
    if (geminiReq.systemInstruction?.parts) {
      const sysText = geminiReq.systemInstruction.parts.map((p: { text?: string }) => p.text || '').join('\n')
      if (sysText) messages.push({ role: 'system', content: sysText })
    }
    for (const content of geminiReq.contents || []) {
      const role = content.role === 'model' ? 'assistant' : 'user'
      const msg = ProxyServer.geminiPartsToOpenAIMessage(role, content.parts || [])
      if (msg) messages.push(msg)
    }
    if (messages.length === 0) {
      messages.push({ role: 'user', content: 'Hello' })
    }

    const openaiRequest: OpenAIChatRequest = {
      model: this.applyModelMapping(modelId, matchedApiKey?.id),
      messages,
      stream: isStream,
      temperature: geminiReq.generationConfig?.temperature,
      top_p: geminiReq.generationConfig?.topP,
      max_tokens: geminiReq.generationConfig?.maxOutputTokens,
      ...(evGem.effort ? { reasoning_effort: evGem.effort } : {})
    }

    // 模型白名单拦截
    if (!this.isModelAllowed(openaiRequest.model)) {
      this.sendError(res, 403, `Model not available: ${openaiRequest.model}`)
      return
    }

    // 复用 OpenAI 流程
    const startTime = Date.now()
    this.recordNewRequest()
    this.throwIfAborted(signal)
    const account = await this.getAvailableAccount(signal)
    this.throwIfAborted(signal)
    if (!account) {
      this.sendError(res, 503, 'No available accounts')
      return
    }

    try {
      const toolNameRegistry = new ToolNameRegistry()
      const kiroPayload = openaiToKiro(openaiRequest, account.profileArn, toolNameRegistry)

      if (isStream) {
        // SSE 流式
        this.writeSseHead(res)
        return new Promise<void>((resolve) => {
          callKiroApiStream(
            account as ProxyAccount,
            kiroPayload,
            (text) => {
              if (signal?.aborted || this.isResponseClosed(res)) return
              if (text) {
                const chunk = { candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: null }] }
                res.write(`data: ${JSON.stringify(chunk)}\n\n`)
              }
            },
            (usage) => {
              if (signal?.aborted || this.isResponseClosed(res)) {
                // 客户端已断开：仍按已消耗用量给客户计费，防白嫖
                this.settleAbortedUsage(matchedApiKey, account.id, usage, openaiRequest.model, '/v1beta/models', ProxyServer.deriveEffortLevel(geminiReq), kiroPayload.conversationState.conversationId)
                resolve()
                return
              }
              const finalChunk = { candidates: [{ content: { parts: [{ text: '' }], role: 'model' }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: usage.inputTokens, candidatesTokenCount: usage.outputTokens, totalTokenCount: usage.inputTokens + usage.outputTokens } }
              res.write(`data: ${JSON.stringify(finalChunk)}\n\n`)
              res.end()
              this.recordRequestSuccess()
              this.stats.totalTokens += usage.inputTokens + usage.outputTokens
              this.stats.inputTokens += usage.inputTokens
              this.stats.outputTokens += usage.outputTokens
              this.stats.totalCredits += usage.credits || 0
              this.accountPool.recordSuccess(account.id, usage.inputTokens + usage.outputTokens)
              const respTime = Date.now() - startTime
              this.events.onResponse?.({ path: '/v1beta/models', model: openaiRequest.model, status: 200, tokens: usage.inputTokens + usage.outputTokens, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, cacheReadTokens: usage.cacheReadTokens, reasoningTokens: usage.reasoningTokens, credits: usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
              this.recordRequest({ path: '/v1beta/models', model: openaiRequest.model, accountId: account.id, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, credits: usage.credits, responseTime: respTime, success: true })
              // 记录 API Key 用量（缺失会导致 Gemini 流式正常结束时不计费，而中断却计费——激励反向）
              if (matchedApiKey) {
                this.recordApiKeyUsage(matchedApiKey.id, usage.credits || 0, usage.inputTokens, usage.outputTokens, openaiRequest.model, '/v1beta/models', ProxyServer.deriveEffortLevel(geminiReq), usage.cacheReadTokens, usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
              }
              resolve()
            },
            (error, partialUsage) => {
              if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
                if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, openaiRequest.model, '/v1beta/models', ProxyServer.deriveEffortLevel(geminiReq), kiroPayload.conversationState.conversationId)
                resolve()
                return
              }
              res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
              res.end()
              this.recordRequestFailed()
              // mid-stream 报错但 Kiro 已计 credit：结算已消耗用量，防漏计费
              if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, openaiRequest.model, '/v1beta/models', ProxyServer.deriveEffortLevel(geminiReq), kiroPayload.conversationState.conversationId)
              resolve()
            },
            signal,
            this.config.preferredEndpoint
          ).catch(error => {
            if (!this.isAbortError(error, signal) && !this.isResponseClosed(res)) {
              res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
              res.end()
              this.recordRequestFailed()
            }
            resolve()
          })
        })
      } else {
        // 非流式
        const result = await callKiroApi(account as ProxyAccount, kiroPayload, signal)
        this.throwIfResponseClosed(res, signal)
        this.recordRequestSuccess()
        this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
        this.stats.inputTokens += result.usage.inputTokens
        this.stats.outputTokens += result.usage.outputTokens
        this.stats.totalCredits += result.usage.credits || 0
        this.accountPool.recordSuccess(account.id, result.usage.inputTokens + result.usage.outputTokens)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({
          candidates: [{ content: { parts: [{ text: result.content }], role: 'model' }, finishReason: 'STOP' }],
          usageMetadata: { promptTokenCount: result.usage.inputTokens, candidatesTokenCount: result.usage.outputTokens, totalTokenCount: result.usage.inputTokens + result.usage.outputTokens }
        }))
        const respTime = Date.now() - startTime
        this.events.onResponse?.({ path: '/v1beta/models', model: openaiRequest.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
        this.recordRequest({ path: '/v1beta/models', model: openaiRequest.model, accountId: account.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        // 记录 API Key 用量（旧实现只更新 totalTokens，导致 Gemini 非流式既不计费也无日志）
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, openaiRequest.model, '/v1beta/models', ProxyServer.deriveEffortLevel(geminiReq), result.usage.cacheReadTokens, result.usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
        }
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1beta', modelId, startTime, signal)
    }
  }

  // 把 Gemini parts 转成 OpenAI message，保留 inlineData（图片/PDF 等 base64 附件）。
  // 旧实现只取 part.text，会把 inlineData 整段丢弃 —— 用户通过 Gemini 接口传 PDF/图片时
  // 附件被静默吞掉，模型看不到文件却照常作答（结果是“看起来回了但没读文件”）。
  private static geminiPartsToOpenAIMessage(
    role: 'user' | 'assistant',
    parts: Array<{ text?: string; inlineData?: { mimeType?: string; data?: string }; inline_data?: { mime_type?: string; data?: string } }>
  ): OpenAIMessage | null {
    const contentParts: OpenAIContentPart[] = []
    let textOnly = ''
    let fileIndex = 0
    for (const p of parts) {
      if (p.text) {
        contentParts.push({ type: 'text', text: p.text })
        textOnly += p.text
        continue
      }
      // 兼容驼峰 inlineData（REST JSON）与下划线 inline_data（部分 SDK）
      const inline = p.inlineData || p.inline_data
      const mime = (p.inlineData?.mimeType || p.inline_data?.mime_type || '').toLowerCase()
      const data = inline?.data
      if (!data) continue
      if (mime.startsWith('image/')) {
        contentParts.push({ type: 'image_url', image_url: { url: `data:${mime};base64,${data}` } })
      } else {
        // 文档（PDF 等）：转成 OpenAI file part，由 extractOpenAIContent → KiroDocument 处理
        const ext = mime === 'application/pdf' ? 'pdf' : (mime.split('/')[1] || 'bin')
        contentParts.push({ type: 'file', file: { filename: `attachment-${++fileIndex}.${ext}`, file_data: `data:${mime || 'application/octet-stream'};base64,${data}` } })
      }
    }
    if (contentParts.length === 0) return null
    // 纯文本时退回 string content，保持与旧行为一致（避免下游对单元素数组的差异）
    if (contentParts.every(cp => cp.type === 'text')) {
      return { role, content: textOnly }
    }
    return { role, content: contentParts }
  }


  private modelCache: { models: KiroModel[]; timestamp: number } | null = null
  private readonly MODEL_CACHE_TTL = 5 * 60 * 1000 // 5 分钟缓存

  // 模型列表
  private async handleModels(res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const now = Date.now()
    
    // Kiro 官方模型（与 UI 保持一致）
    const kiroOfficialModels = [
      buildClientModel({ id: 'auto', created: now, ownedBy: 'kiro-api', description: 'Auto select best model' }),
      buildClientModel({ id: 'claude-sonnet-4.5', created: now, ownedBy: 'kiro-api', description: 'The latest Claude Sonnet model' }),
      buildClientModel({ id: 'claude-sonnet-4', created: now, ownedBy: 'kiro-api', description: 'Hybrid reasoning and coding' }),
      buildClientModel({ id: 'claude-haiku-4.5', created: now, ownedBy: 'kiro-api', description: 'The latest Claude Haiku model' }),
      buildClientModel({ id: 'claude-opus-4.5', created: now, ownedBy: 'kiro-api', description: 'The most powerful model' })
    ]

    // 隐藏模型（未在官方 ListAvailableModels 中返回，但后端可能支持）
    const hiddenModels = [
      buildClientModel({ id: 'claude-3.7-sonnet', created: now, ownedBy: 'kiro-api', description: 'Claude 3.7 Sonnet (hidden)', modelName: 'Claude 3.7 Sonnet', supportedInputTypes: ['TEXT', 'IMAGE'], maxInputTokens: 200000, maxOutputTokens: 64000 }),
      buildClientModel({ id: 'simple-task', created: now, ownedBy: 'kiro-api', description: 'Kiro fast model for intent classification and lightweight tasks (routes to Haiku)', modelName: 'Simple Task', supportedInputTypes: ['TEXT'], maxInputTokens: 200000, maxOutputTokens: 4096 }),
      buildClientModel({ id: 'CLAUDE_SONNET_4_20250514_V1_0', created: now, ownedBy: 'kiro-api', description: 'Claude Sonnet 4 (CodeWhisperer internal ID)', modelName: 'Claude Sonnet 4 (CW)', supportedInputTypes: ['TEXT', 'IMAGE'], maxInputTokens: 200000, maxOutputTokens: 64000 }),
      buildClientModel({ id: 'CLAUDE_HAIKU_4_5_20251001_V1_0', created: now, ownedBy: 'kiro-api', description: 'Claude Haiku 4.5 (CodeWhisperer internal ID)', modelName: 'Claude Haiku 4.5 (CW)', supportedInputTypes: ['TEXT', 'IMAGE'], maxInputTokens: 200000, maxOutputTokens: 64000 }),
      buildClientModel({ id: 'CLAUDE_3_7_SONNET_20250219_V1_0', created: now, ownedBy: 'kiro-api', description: 'Claude 3.7 Sonnet (CodeWhisperer internal ID)', modelName: 'Claude 3.7 Sonnet (CW)', supportedInputTypes: ['TEXT', 'IMAGE'], maxInputTokens: 200000, maxOutputTokens: 64000 })
    ]

    // 预设模型（GPT 兼容别名）
    const presetModels = [
      buildClientModel({ id: 'gpt-4o', created: now, ownedBy: 'kiro-proxy', description: 'GPT-compatible alias for Kiro' }),
      buildClientModel({ id: 'gpt-4', created: now, ownedBy: 'kiro-proxy', description: 'GPT-compatible alias for Kiro' }),
      buildClientModel({ id: 'gpt-4-turbo', created: now, ownedBy: 'kiro-proxy', description: 'GPT-compatible alias for Kiro' }),
      buildClientModel({ id: 'gpt-3.5-turbo', created: now, ownedBy: 'kiro-proxy', description: 'GPT-compatible alias for Kiro' })
    ]

    // 尝试从 Kiro API 获取动态模型
    let kiroModels: KiroModel[] = []
    
    // 检查缓存
    if (this.modelCache && (now - this.modelCache.timestamp) < this.MODEL_CACHE_TTL) {
      kiroModels = this.modelCache.models
    } else {
      // 获取一个可用账号来请求模型列表
      const account = this.accountPool.getNextAccount()
      if (account) {
        try {
          kiroModels = await this.fetchKiroModelsWithRefresh(account, signal)
          if (kiroModels.length > 0) {
            this.modelCache = { models: kiroModels, timestamp: now }
            // 同步到 kiroApi 的 ctx cache, 供 token 裁剪逻辑使用
            for (const m of kiroModels) {
              if (m.tokenLimits?.maxInputTokens) {
                setModelContextWindow(m.modelId, m.tokenLimits.maxInputTokens)
              }
            }
            // 同步真实 thinking/effort 能力到 translator，供 additionalModelRequestFields 构建时校验
            syncModelThinkingCapabilities(kiroModels)
            proxyLogger.info('ProxyServer', `Fetched ${kiroModels.length} models from Kiro API`)
          }
        } catch (error) {
          if (this.isAbortError(error, signal)) throw error
          console.error('[ProxyServer] Failed to fetch Kiro models:', error)
        }
      }
    }

    // 转换 Kiro 模型为 OpenAI 格式（保持原始 modelId）
    const dynamicModels = kiroModels.map(m => buildClientModel({
      id: m.modelId,
      created: now,
      ownedBy: 'kiro-api',
      description: m.description,
      modelName: m.modelName,
      supportedInputTypes: m.supportedInputTypes,
      maxInputTokens: m.tokenLimits?.maxInputTokens,
      maxOutputTokens: m.tokenLimits?.maxOutputTokens,
      rateMultiplier: m.rateMultiplier,
      rateUnit: m.rateUnit,
      promptCaching: m.promptCaching,
      additionalModelRequestFieldsSchema: m.additionalModelRequestFieldsSchema,
      modelProvider: m.modelProvider
    }))

    // 合并模型列表，去重
    const modelIds = new Set<string>()
    const allModels: ClientModel[] = []
    
    // 1. 优先添加动态模型（从 API 获取的，包含真实 token limit / input types）
    for (const m of dynamicModels) {
      if (!modelIds.has(m.id)) {
        modelIds.add(m.id)
        allModels.push(m)
      }
    }
    
    // 2. 添加隐藏模型（未在官方 ListAvailableModels 中返回，但后端可能支持）
    for (const m of hiddenModels) {
      if (!modelIds.has(m.id)) {
        modelIds.add(m.id)
        allModels.push(m)
      }
    }
    
    // 3. 动态模型缺失时才添加静态兜底
    if (dynamicModels.length === 0) {
      for (const m of [...kiroOfficialModels, ...presetModels]) {
        if (!modelIds.has(m.id)) {
          modelIds.add(m.id)
          allModels.push(m)
        }
      }
    }

    // 4. 精简下拉列表（与 /portal/rates 共用 filterPickerModels，保证口径一致）：
    //    同一 Claude 家族只留最新版本 + 剔除内部/历史 ID，避免 Cowork/Claude Code
    //    下拉里出现一堆无法区分的 "Opus 4"。被剔除的模型仍可经 /v1/messages 按 ID 直调。
    const pickerModels = filterPickerModels(allModels).filter(m => this.isModelAllowed(m.id))
    // effort 变体（如 claude-opus-4.8-max）：仅在开关开启时追加，按各模型真实 effort 枚举生成。
    // 必须在 filterPickerModels 之后——否则变体 ID 会被按 Claude 家族收敛掉。
    const listModels = this.config.effortVariantsExposed === true
      ? this.expandEffortVariants(pickerModels)
      : pickerModels

    this.throwIfResponseClosed(res, signal)
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ object: 'list', data: listModels }))
  }

  // 处理 OpenAI Chat Completions 请求
  private async handleOpenAIChat(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const request: OpenAIChatRequest = JSON.parse(body)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey

    // effort 变体：把 claude-opus-4.8-max 还原成 base + 强制注入 effort（覆盖客户端自带值）。
    // 必须在 applyModelMapping / isModelAllowed 之前——让 base 走原有逻辑。
    const evChat = this.resolveEffortVariant(request.model)
    request.model = evChat.baseId
    if (evChat.effort) request.reasoning_effort = evChat.effort

    // 提取 session hint（用于稳定 conversationId），拼入 API Key hash 隔离不同用户
    const rawHintChat = ProxyServer.extractSessionHint(req, request)
    if (!request.conversation_id && rawHintChat) {
      const keyPrefix = matchedApiKey?.id?.slice(0, 8) || 'default'
      request.conversation_id = `${keyPrefix}:${rawHintChat}`
    }
    const affinityHintChat = request.conversation_id

    // 应用模型映射
    request.model = this.applyModelMapping(request.model, matchedApiKey?.id)

    // 模型白名单拦截
    if (!this.isModelAllowed(request.model)) {
      this.recordRequestFailed()
      const msg = `Model not available: ${request.model}`
      this.sendError(res, 403, msg)
      this.events.onResponse?.({ path: '/v1/chat/completions', model: request.model, status: 403, error: msg })
      this.recordRequest({ path: '/v1/chat/completions', model: request.model, responseTime: 0, success: false, error: msg })
      return
    }

    const startTime = Date.now()

    this.recordNewRequest()
    this.events.onRequest?.({ path: '/v1/chat/completions', method: 'POST' })

    let processedRequest: OpenAIChatRequest
    try {
      processedRequest = await this.resolveOpenAIHttpImages(this.prepareOpenAIRequest(request), signal)
    } catch (error) {
      if (this.isAbortError(error, signal)) return
      this.recordRequestFailed()
      const message = error instanceof Error ? error.message : 'Invalid request'
      this.sendError(res, 400, message)
      this.events.onResponse?.({ path: '/v1/chat/completions', model: request.model, status: 400, error: message })
      this.recordRequest({ path: '/v1/chat/completions', model: request.model, responseTime: Date.now() - startTime, success: false, error: message })
      return
    }

    // 获取账号（包含 Token 刷新检查 + 会话粘性 + API Key 账号白名单）
    this.throwIfAborted(signal)
    const account = await this.getAvailableAccount(signal, affinityHintChat, matchedApiKey?.id)
    this.throwIfAborted(signal)
    if (!account) {
      this.recordRequestFailed()
      const quotaStatus = this.accountPool.getQuotaStatus()
      const errorMsg = quotaStatus.exhausted > 0 && quotaStatus.available === 0
        ? `All accounts quota exhausted (${quotaStatus.exhausted}/${quotaStatus.total} exhausted, ${quotaStatus.cooldown} in cooldown)`
        : 'No available accounts'
      this.sendError(res, 503, errorMsg)
      this.events.onResponse?.({ path: '/v1/chat/completions', model: request.model, status: 503, error: errorMsg })
      this.recordRequest({ path: '/v1/chat/completions', model: request.model, success: false, error: errorMsg })
      return
    }

    this.events.onRequest?.({ path: '/v1/chat/completions', method: 'POST', accountId: account.id })

    try {
      const toolNameRegistry = new ToolNameRegistry()

      // 转换为 Kiro 格式
      const kiroPayload = openaiToKiro(processedRequest, account.profileArn, toolNameRegistry)

      // 记录请求详情到日志
      if (this.config.logRequests) {
        const userInput = kiroPayload.conversationState.currentMessage?.userInputMessage
        const contentLength = typeof userInput?.content === 'string' ? userInput.content.length : 0
        const toolsCount = userInput?.userInputMessageContext?.tools?.length || 0
        const historyLength = kiroPayload.conversationState.history?.length || 0
        const hasImages = (userInput?.images?.length || 0) > 0
        
        proxyLogger.info('ProxyServer', `OpenAI API: ${request.model}`, {
          model: request.model,
          stream: request.stream,
          contentLength,
          toolsCount,
          historyLength,
          hasImages,
          accountId: account.id
        })
      }

      if (request.stream) {
        // 流式响应（流式不使用重试机制，错误由流处理）
        await this.handleOpenAIStream(res, account, kiroPayload, request.model, startTime, 0, undefined, false, matchedApiKey, toolNameRegistry, signal, ProxyServer.deriveEffortLevel(request))
      } else {
        // 非流式响应（带重试机制）
        const { result, account: usedAccount } = await this.callWithRetry(
          account,
          async (acc) => {
            const retryPayload = openaiToKiro(processedRequest, acc.profileArn, toolNameRegistry)
            return callKiroApi(acc, retryPayload, signal)
          },
          '/v1/chat/completions',
          signal,
          this.getAllowedAccountIds(matchedApiKey?.id)
        )
        const response = kiroToOpenaiResponse(result.content, result.toolUses, result.usage, request.model, toolNameRegistry, result.reasoningContent)

        this.throwIfResponseClosed(res, signal)
        this.recordRequestSuccess()
        this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
        this.stats.inputTokens += result.usage.inputTokens
        this.stats.outputTokens += result.usage.outputTokens
        this.accountPool.recordSuccess(usedAccount.id, result.usage.inputTokens + result.usage.outputTokens)

        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(response))
        const respTime = Date.now() - startTime
        this.events.onResponse?.({ path: '/v1/chat/completions', model: request.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
        this.recordRequest({ path: '/v1/chat/completions', model: request.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        // 记录 API Key 用量
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, request.model, '/v1/chat/completions', ProxyServer.deriveEffortLevel(request), result.usage.cacheReadTokens, result.usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
        }
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1/chat/completions', request.model, startTime, signal)
    }
  }

  private async handleOpenAIResponses(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey
    const startTime = Date.now()

    this.recordNewRequest()
    this.events.onRequest?.({ path: '/v1/responses', method: 'POST' })

    let responseRequest: OpenAIResponsesRequest
    let chatRequest: OpenAIChatRequest
    let processedRequest: OpenAIChatRequest
    let affinityHintResp: string | undefined
    try {
      responseRequest = JSON.parse(body)
      chatRequest = responsesToOpenAIChat(responseRequest)
      // effort 变体：还原 base + 注入 reasoning_effort（与 chat path 同口径）。
      const evResp = this.resolveEffortVariant(chatRequest.model)
      chatRequest.model = evResp.baseId
      if (evResp.effort) chatRequest.reasoning_effort = evResp.effort
      // session hint：用于会话粘性
      const rawHintResp = ProxyServer.extractSessionHint(req, responseRequest)
      if (rawHintResp) {
        const keyPrefix = matchedApiKey?.id?.slice(0, 8) || 'default'
        affinityHintResp = `${keyPrefix}:${rawHintResp}`
      }
      chatRequest.model = this.applyModelMapping(chatRequest.model, matchedApiKey?.id)
      processedRequest = await this.resolveOpenAIHttpImages(this.prepareOpenAIRequest(chatRequest), signal)
    } catch (error) {
      if (this.isAbortError(error, signal)) return
      this.recordRequestFailed()
      const message = error instanceof Error ? error.message : 'Invalid request'
      this.sendError(res, 400, message)
      this.events.onResponse?.({ path: '/v1/responses', status: 400, error: message })
      this.recordRequest({ path: '/v1/responses', responseTime: Date.now() - startTime, success: false, error: message })
      return
    }

    // 模型白名单拦截
    if (!this.isModelAllowed(chatRequest.model)) {
      this.recordRequestFailed()
      const msg = `Model not available: ${chatRequest.model}`
      this.sendError(res, 403, msg)
      this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 403, error: msg })
      this.recordRequest({ path: '/v1/responses', model: chatRequest.model, responseTime: Date.now() - startTime, success: false, error: msg })
      return
    }

    this.throwIfAborted(signal)
    const account = await this.getAvailableAccount(signal, affinityHintResp, matchedApiKey?.id)
    this.throwIfAborted(signal)
    if (!account) {
      this.recordRequestFailed()
      const quotaStatus = this.accountPool.getQuotaStatus()
      const errorMsg = quotaStatus.exhausted > 0 && quotaStatus.available === 0
        ? `All accounts quota exhausted (${quotaStatus.exhausted}/${quotaStatus.total} exhausted, ${quotaStatus.cooldown} in cooldown)`
        : 'No available accounts'
      this.sendError(res, 503, errorMsg)
      this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 503, error: errorMsg })
      this.recordRequest({ path: '/v1/responses', model: chatRequest.model, success: false, error: 'No available accounts' })
      return
    }

    this.events.onRequest?.({ path: '/v1/responses', method: 'POST', accountId: account.id })

    try {
      const toolNameRegistry = new ToolNameRegistry()
      if (processedRequest.stream) {
        this.writeSseHead(res)
        const responseId = `resp_${uuidv4()}`
        res.write(`event: response.created\ndata: ${JSON.stringify({ type: 'response.created', response: { id: responseId, object: 'response', created_at: Math.floor(Date.now() / 1000), model: chatRequest.model, output: [] } })}\n\n`)
        const { result, account: usedAccount } = await this.callWithRetry(
          account,
          async (acc) => {
            const retryPayload = openaiToKiro(processedRequest, acc.profileArn, toolNameRegistry)
            return callKiroApi(acc, retryPayload, signal)
          },
          '/v1/responses',
          signal,
          this.getAllowedAccountIds(matchedApiKey?.id)
        )
        const chatResponse = kiroToOpenaiResponse(result.content, result.toolUses, result.usage, chatRequest.model, toolNameRegistry, result.reasoningContent)
        this.throwIfResponseClosed(res, signal)
        const response = openAIChatToResponsesResponse(chatResponse, responseRequest.previous_response_id)
        const streamedResponse = { ...response, id: responseId }
        streamedResponse.output.forEach((item, outputIndex) => {
          this.throwIfResponseClosed(res, signal)
          res.write(`event: response.output_item.added\ndata: ${JSON.stringify({ type: 'response.output_item.added', output_index: outputIndex, item })}\n\n`)
          if (item.type === 'message') {
            item.content.forEach((part, contentIndex) => {
              this.throwIfResponseClosed(res, signal)
              res.write(`event: response.content_part.added\ndata: ${JSON.stringify({ type: 'response.content_part.added', item_id: item.id, output_index: outputIndex, content_index: contentIndex, part: { type: part.type, text: '' } })}\n\n`)
              if (part.text) {
                res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({ type: 'response.output_text.delta', item_id: item.id, output_index: outputIndex, content_index: contentIndex, delta: part.text })}\n\n`)
              }
              res.write(`event: response.output_text.done\ndata: ${JSON.stringify({ type: 'response.output_text.done', item_id: item.id, output_index: outputIndex, content_index: contentIndex, text: part.text })}\n\n`)
              res.write(`event: response.content_part.done\ndata: ${JSON.stringify({ type: 'response.content_part.done', item_id: item.id, output_index: outputIndex, content_index: contentIndex, part })}\n\n`)
            })
          } else {
            if (item.arguments) {
              res.write(`event: response.function_call_arguments.delta\ndata: ${JSON.stringify({ type: 'response.function_call_arguments.delta', item_id: item.id, output_index: outputIndex, delta: item.arguments })}\n\n`)
            }
            res.write(`event: response.function_call_arguments.done\ndata: ${JSON.stringify({ type: 'response.function_call_arguments.done', item_id: item.id, output_index: outputIndex, arguments: item.arguments })}\n\n`)
          }
          this.throwIfResponseClosed(res, signal)
          res.write(`event: response.output_item.done\ndata: ${JSON.stringify({ type: 'response.output_item.done', output_index: outputIndex, item })}\n\n`)
        })
        this.throwIfResponseClosed(res, signal)
        res.write(`event: response.completed\ndata: ${JSON.stringify({ type: 'response.completed', response: streamedResponse })}\n\n`)
        res.end()
        this.recordRequestSuccess()
        this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
        this.stats.inputTokens += result.usage.inputTokens
        this.stats.outputTokens += result.usage.outputTokens
        this.accountPool.recordSuccess(usedAccount.id, result.usage.inputTokens + result.usage.outputTokens)
        const respTime = Date.now() - startTime
        this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime, sessionId: affinityHintResp })
        this.recordRequest({ path: '/v1/responses', model: chatRequest.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, chatRequest.model, '/v1/responses', ProxyServer.deriveEffortLevel(chatRequest), result.usage.cacheReadTokens, result.usage.cacheWriteTokens, affinityHintResp)
        }
        return
      }

      const { result, account: usedAccount } = await this.callWithRetry(
        account,
        async (acc) => {
          const retryPayload = openaiToKiro(processedRequest, acc.profileArn, toolNameRegistry)
          return callKiroApi(acc, retryPayload, signal)
        },
        '/v1/responses',
        signal,
        this.getAllowedAccountIds(matchedApiKey?.id)
      )
      const chatResponse = kiroToOpenaiResponse(result.content, result.toolUses, result.usage, chatRequest.model, toolNameRegistry, result.reasoningContent)
      this.throwIfResponseClosed(res, signal)
      const response = openAIChatToResponsesResponse(chatResponse, responseRequest.previous_response_id)

      this.recordRequestSuccess()
      this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
      this.stats.inputTokens += result.usage.inputTokens
      this.stats.outputTokens += result.usage.outputTokens
      this.accountPool.recordSuccess(usedAccount.id, result.usage.inputTokens + result.usage.outputTokens)

      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(response))
      const respTime = Date.now() - startTime
      this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime, sessionId: affinityHintResp })
      this.recordRequest({ path: '/v1/responses', model: chatRequest.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
      if (matchedApiKey) {
        this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, chatRequest.model, '/v1/responses', ProxyServer.deriveEffortLevel(chatRequest), result.usage.cacheReadTokens, result.usage.cacheWriteTokens, affinityHintResp)
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1/responses', chatRequest.model, startTime, signal)
    }
  }

  // 处理 OpenAI 流式响应
  private async handleOpenAIStream(
    res: http.ServerResponse,
    account: { id: string; accessToken: string; profileArn?: string },
    kiroPayload: ReturnType<typeof openaiToKiro>,
    model: string,
    startTime: number,
    currentRound: number = 0,
    streamId?: string,
    headersSent: boolean = false,
    matchedApiKey?: import('./types').ApiKey,
    toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
    signal?: AbortSignal,
    effort?: string,
    authRetried: boolean = false,
    // 瞬时网络错误已重试标记：防止 transient-close 重试无限放大（仅重试一次）。
    transientRetried: boolean = false
  ): Promise<void> {
    if (!headersSent) {
      this.writeSseHead(res)
    }

    const id = streamId || `chatcmpl-${uuidv4()}`
    let toolCallIndex = 0
    const pendingToolCalls: Map<string, { index: number; name: string; arguments: string }> = new Map()
    let collectedContent = ''
    // 是否已向客户端发出真实内容（文本/思考/工具）——auth 重试只在「尚未发出真实内容」时安全
    let sentRealContent = false
    // 发送初始 chunk（仅首轮；auth 重试时不重发，避免重复 role chunk）
    // 发送初始 chunk（仅首轮、仅首次进入；任何重试继续（headersSent=true）都不重发，避免重复 role chunk）
    if (currentRound === 0 && !authRetried && !headersSent) {
      const initialChunk = createOpenaiStreamChunk(id, model, { role: 'assistant' })
      res.write(`data: ${JSON.stringify(initialChunk)}\n\n`)
    }

    return new Promise((resolve) => {
      callKiroApiStream(
        account as any,
        kiroPayload,
        (text, toolUse, isThinking) => {
          if (signal?.aborted || this.isResponseClosed(res)) return
          if (text && text.trim()) {
            sentRealContent = true
            if (isThinking) {
              // 原生 thinking 内容 → 输出为 reasoning_content
              const chunk = createOpenaiStreamChunk(id, model, { reasoning_content: text })
              res.write(`data: ${JSON.stringify(chunk)}\n\n`)
            } else {
              // 普通文本内容
              collectedContent += text
              const chunk = createOpenaiStreamChunk(id, model, { content: text })
              res.write(`data: ${JSON.stringify(chunk)}\n\n`)
            }
          }
          if (toolUse) {
            sentRealContent = true
            const idx = toolCallIndex++
            const restoredToolUse = toolNameRegistry.restoreToolUse(toolUse)
            pendingToolCalls.set(toolUse.toolUseId, {
              index: idx,
              name: toolUse.name,
              arguments: JSON.stringify(toolUse.input)
            })
            const toolChunk = createOpenaiStreamChunk(id, model, {
              tool_calls: [{
                index: idx,
                id: toolUse.toolUseId,
                type: 'function',
                function: {
                  name: restoredToolUse.name,
                  arguments: JSON.stringify(toolUse.input)
                }
              }]
            })
            res.write(`data: ${JSON.stringify(toolChunk)}\n\n`)
          }
        },
        async (usage) => {
          if (signal?.aborted || this.isResponseClosed(res)) {
            // 客户端已断开：仍按已消耗用量给客户计费，防白嫖
            this.settleAbortedUsage(matchedApiKey, account.id, usage, model, '/v1/chat/completions', effort, kiroPayload.conversationState.conversationId)
            resolve()
            return
          }

          this.recordRequestSuccess()
          this.stats.totalTokens += usage.inputTokens + usage.outputTokens
          this.stats.inputTokens += usage.inputTokens
          this.stats.outputTokens += usage.outputTokens
          this.stats.cacheReadTokens += usage.cacheReadTokens || 0
          this.stats.cacheWriteTokens += usage.cacheWriteTokens || 0
          this.stats.reasoningTokens += usage.reasoningTokens || 0
          this.stats.totalCredits += usage.credits || 0
          this.events.onCreditsUpdate?.(this.stats.totalCredits)
          this.events.onTokensUpdate?.(this.stats.inputTokens, this.stats.outputTokens)
          this.accountPool.recordSuccess(account.id, usage.inputTokens + usage.outputTokens)
          const oaiRespTime = Date.now() - startTime
          this.events.onResponse?.({ path: '/v1/chat/completions', model, status: 200, tokens: usage.inputTokens + usage.outputTokens, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, cacheReadTokens: usage.cacheReadTokens, reasoningTokens: usage.reasoningTokens, credits: usage.credits, responseTime: oaiRespTime, sessionId: kiroPayload.conversationState.conversationId })
          this.recordRequest({ path: '/v1/chat/completions', model, accountId: account.id, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, credits: usage.credits, responseTime: oaiRespTime, success: true })
          // 记录 API Key 用量
          if (matchedApiKey) {
            this.recordApiKeyUsage(matchedApiKey.id, usage.credits || 0, usage.inputTokens, usage.outputTokens, model, '/v1/chat/completions', effort, usage.cacheReadTokens, usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
          }

          // 发送结束 chunk（包含完整 usage 信息）
          const hasToolCalls = pendingToolCalls.size > 0
          const finishReason = hasToolCalls ? 'tool_calls' : 'stop'
          const usageInfo: {
            prompt_tokens: number
            completion_tokens: number
            total_tokens: number
            prompt_tokens_details?: { cached_tokens?: number }
            completion_tokens_details?: { reasoning_tokens?: number }
          } = {
            prompt_tokens: usage.inputTokens,
            completion_tokens: usage.outputTokens,
            total_tokens: usage.inputTokens + usage.outputTokens
          }
          // 添加 cache tokens 详情
          if (usage.cacheReadTokens && usage.cacheReadTokens > 0) {
            usageInfo.prompt_tokens_details = { cached_tokens: usage.cacheReadTokens }
          }
          // 添加 reasoning tokens 详情
          if (usage.reasoningTokens && usage.reasoningTokens > 0) {
            usageInfo.completion_tokens_details = { reasoning_tokens: usage.reasoningTokens }
          }
          const finalChunk = createOpenaiStreamChunk(id, model, {}, finishReason, usageInfo)
          res.write(`data: ${JSON.stringify(finalChunk)}\n\n`)
          res.write('data: [DONE]\n\n')
          res.end()
          resolve()
        },
        async (error, partialUsage) => {
          if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
            // 客户端断开但 Kiro 可能已计 credit：结算已消耗用量，防漏计费
            if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, model, '/v1/chat/completions', effort, kiroPayload.conversationState.conversationId)
            resolve()
            return
          }

          const errStatusCode = error.message.match(/(\d{3})/)?.[1]
          const statusNum = errStatusCode ? parseInt(errStatusCode) : 0

          // 401/403 自愈：token 被服务端撤销/轮换（常见于同账号被外部 Kiro IDE 轮换 refreshToken）。
          // 仅在「尚未向客户端发出真实内容」且「本请求未重试过」时安全——此时刷新 token 后可
          // 透明地重跑整个 stream，客户端无感知。已发出内容则无法回退，只能照常报错。
          if (
            (statusNum === 401 || statusNum === 403) &&
            !authRetried &&
            !sentRealContent &&
            !signal?.aborted &&
            !this.isResponseClosed(res)
          ) {
            console.log(`[ProxyServer] Stream auth error ${statusNum} before content; refreshing token and retrying once`)
            let refreshed = false
            const fullAccount = this.accountPool.getAccount(account.id)
            try {
              if (fullAccount) refreshed = await this.refreshToken(fullAccount, signal)
            } catch (refreshErr) {
              console.error('[ProxyServer] Token refresh during stream failed:', formatError(refreshErr))
            }
            if (signal?.aborted || this.isResponseClosed(res)) {
              resolve()
              return
            }
            if (refreshed) {
              const refreshedAccount = this.accountPool.getAccount(account.id) || account
              // 重跑整个 stream：headersSent=true 不重写响应头，authRetried=true 不重发 role chunk
              this.handleOpenAIStream(
                res, refreshedAccount as typeof account, kiroPayload, model, startTime,
                currentRound, id, true, matchedApiKey, toolNameRegistry, signal, effort, true
              ).then(resolve).catch(() => resolve())
              return
            }
            // 刷新失败 → 落到下方常规错误处理，记账并切断该账号
          }

          // 瞬时上游断连自愈：读流途中上游/中间层 idle 切断（UND_ERR_SOCKET / "other side closed"）。
          // 仅在「尚未发出真实内容」且「未因瞬时错误重试过」时透明重跑整个 stream（与 401/403 自愈同构）。
          // 不在重试前 settle partialUsage，避免与重试成功后的正式计费重复。已发出内容则照常报错。
          if (
            statusNum === 0 &&
            !transientRetried &&
            !sentRealContent &&
            !signal?.aborted &&
            !this.isResponseClosed(res) &&
            isTransientNetworkError(error)
          ) {
            console.log(`[ProxyServer] Stream transient close before content; retrying once: ${formatError(error)}`)
            this.handleOpenAIStream(
              res, account, kiroPayload, model, startTime,
              currentRound, id, true, matchedApiKey, toolNameRegistry, signal, effort, authRetried, true
            ).then(resolve).catch(() => resolve())
            return
          }

          console.error('[ProxyServer] Stream error:', formatError(error))
          res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
          res.end()

          this.recordRequestFailed()
          this.accountPool.recordError(account.id, errStatusCode ? classifyError(statusNum) : ErrorType.RECOVERABLE, errStatusCode ? statusNum : undefined)
          this.events.onResponse?.({ path: '/v1/chat/completions', model, status: 500, error: error.message })
          this.recordRequest({ path: '/v1/chat/completions', model, accountId: account.id, responseTime: Date.now() - startTime, success: false, error: error.message })
          // mid-stream 报错但 Kiro 已计 credit（meteringEvent 先于断开）：结算已消耗用量，防漏计费
          if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, model, '/v1/chat/completions', effort, kiroPayload.conversationState.conversationId)
          resolve()
        },
        signal,
        this.config.preferredEndpoint,
        model // clientModelId：保留 [1m] 后缀，让 contextUsageEvent 反推用正确的 context 分母
      ).catch(error => {
        if (!this.isAbortError(error, signal) && !this.isResponseClosed(res)) {
          res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
          res.end()
          this.recordRequestFailed()
        }
        resolve()
      })
    })
  }

  // 处理 Claude Messages 请求
  private async handleClaudeMessages(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const request: ClaudeRequest = JSON.parse(body)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey

    // effort 变体：还原 base + 注入 effort。Claude path 经 output_config.effort 下发
    // （deriveClaudeEffort 优先读 output_config.effort），覆盖客户端自带值。
    const evClaude = this.resolveEffortVariant(request.model)
    request.model = evClaude.baseId
    if (evClaude.effort) request.output_config = { ...request.output_config, effort: evClaude.effort }

    // 提取 session hint（用于稳定 conversationId），拼入 API Key hash 隔离不同用户
    const rawHint = ProxyServer.extractSessionHint(req, request)
    if (!request.conversation_id && rawHint) {
      const keyPrefix = matchedApiKey?.id?.slice(0, 8) || 'default'
      request.conversation_id = `${keyPrefix}:${rawHint}`
    }
    // P1-8 会话粘性使用 conversation_id 作为粘性 key（已包含 API Key 前缀）
    const affinityHint = request.conversation_id

    // 应用模型映射
    request.model = this.applyModelMapping(request.model, matchedApiKey?.id)

    // 模型白名单拦截：映射后的模型不在白名单 → 403（既隐藏于 /v1/models，也拦截直调）
    if (!this.isModelAllowed(request.model)) {
      this.recordRequestFailed()
      const msg = `Model not available: ${request.model}`
      this.sendError(res, 403, msg, 'anthropic')
      this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 403, error: msg })
      this.recordRequest({ path: '/v1/messages', model: request.model, responseTime: 0, success: false, error: msg })
      return
    }

    const startTime = Date.now()

    this.recordNewRequest()
    this.events.onRequest?.({ path: '/v1/messages', method: 'POST' })

    let processedRequest: ClaudeRequest
    try {
      processedRequest = await this.resolveClaudeHttpImages(this.prepareClaudeRequest(request), signal)
    } catch (error) {
      if (this.isAbortError(error, signal)) return
      this.recordRequestFailed()
      const message = error instanceof Error ? error.message : 'Invalid request'
      this.sendError(res, 400, message, 'anthropic')
      this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 400, error: message })
      this.recordRequest({ path: '/v1/messages', model: request.model, responseTime: Date.now() - startTime, success: false, error: message })
      return
    }

    // 获取账号（包含 Token 刷新检查 + 会话粘性 + API Key 账号白名单）
    this.throwIfAborted(signal)
    const account = await this.getAvailableAccount(signal, affinityHint, matchedApiKey?.id)
    this.throwIfAborted(signal)
    if (!account) {
      this.recordRequestFailed()
      const quotaStatus = this.accountPool.getQuotaStatus()
      const errorMsg = quotaStatus.exhausted > 0 && quotaStatus.available === 0
        ? `All accounts quota exhausted (${quotaStatus.exhausted}/${quotaStatus.total} exhausted, ${quotaStatus.cooldown} in cooldown)`
        : 'No available accounts'
      this.sendError(res, 503, errorMsg, 'anthropic')
      this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 503, error: errorMsg })
      this.recordRequest({ path: '/v1/messages', model: request.model, success: false, error: errorMsg })
      return
    }

    this.events.onRequest?.({ path: '/v1/messages', method: 'POST', accountId: account.id })

    try {
      const toolNameRegistry = new ToolNameRegistry()
      const webToolConfig = this.getWebToolConfig()
      const useWebTools = !!webToolConfig && this.claudeRequestUsesWebTools(request)

      const _tTranslate = perfDiag.enabled ? perfDiag.now() : 0
      const kiroPayload = claudeToKiro(processedRequest, account.profileArn, toolNameRegistry, useWebTools)
      if (perfDiag.enabled) perfDiag.recordTiming(PerfPhase.Translate, perfDiag.now() - _tTranslate)

      // 估算本轮请求的 input token 数（仅用于 message_start 的 usage 展示）。
      // 用 binary-aware 的 estimateTokenCount（与 count_tokens / message_start 同源），
      // 不能用 JSON.stringify(kiroPayload).length——payload 里的 base64 附件会让估算虚高数十倍。
      // 包成 thunk（只算一次）：stream 路径把这段同步开销推迟到「请求已发出后」再算，
      // 与上游网络 RTT 重叠，降低 TTFT 并减少对并发请求的事件循环阻塞；非流式即时调用。
      // 注：原先这里还会跑 promptCacheTracker 模拟 prompt cache usage，但 MITM 实测确认
      // Kiro 没有 prompt cache、也不返回 cache 字段，模拟值只会误导客户，故已移除。
      let _estimatedInputTokens: number | undefined
      const prepareInputEstimate = (): number => {
        if (_estimatedInputTokens !== undefined) return _estimatedInputTokens
        _estimatedInputTokens = Math.max(1,
          this.estimateTokenCount(processedRequest.system) +
          this.estimateTokenCount(processedRequest.messages) +
          this.estimateTokenCount(processedRequest.tools))
        return _estimatedInputTokens
      }

      // 记录请求详情到日志
      if (this.config.logRequests) {
        const userInput = kiroPayload.conversationState.currentMessage?.userInputMessage
        const contentLength = typeof userInput?.content === 'string' ? userInput.content.length : 0
        const toolsCount = userInput?.userInputMessageContext?.tools?.length || 0
        const historyLength = kiroPayload.conversationState.history?.length || 0
        const hasImages = (userInput?.images?.length || 0) > 0
        
        proxyLogger.info('ProxyServer', `Claude API: ${request.model}`, {
          model: request.model,
          stream: request.stream,
          contentLength,
          toolsCount,
          historyLength,
          hasImages,
          accountId: account.id.substring(0, 8) + '...'
        })
      }

      if (useWebTools && webToolConfig) {
        // Web 工具路径：代理侧执行 web_search/web_fetch 循环，得到最终回答后再返回客户端。
        // stream 与 non-stream 共用同一循环，只是输出格式不同（SSE replay vs 单个 JSON）。
        await this.handleClaudeWebToolRequest(
          res, account, kiroPayload, processedRequest, webToolConfig, toolNameRegistry,
          startTime, matchedApiKey, signal
        )
      } else if (request.stream) {
        // 流式响应（流式不使用重试机制，错误由流处理）
        // 把 input token 估算推迟给 handleClaudeStream：先发上游请求，
        // 再在网络 RTT 期间计算并发出 message_start（见 ensureStarted / setImmediate）。
        await this.handleClaudeStream(res, account, kiroPayload, request.model, startTime, 0, undefined, false, 0, matchedApiKey, toolNameRegistry, signal,
          undefined, ProxyServer.deriveEffortLevel(request), prepareInputEstimate, request.max_tokens)
      } else {
        // 非流式响应（带重试机制）
        const { result, account: usedAccount } = await this.callWithRetry(
          account,
          async (acc) => {
            const retryPayload = claudeToKiro(processedRequest, acc.profileArn, toolNameRegistry)
            return callKiroApi(acc, retryPayload, signal)
          },
          '/v1/messages',
          signal,
          this.getAllowedAccountIds(matchedApiKey?.id)
        )
        const response = kiroToClaudeResponse(result.content, result.toolUses, result.usage, request.model, toolNameRegistry, result.reasoningContent)

        // 不再注入本地模拟的 cache usage：Kiro 不返回 cache 字段、也不做 prompt cache（MITM 实测确认）。
        // response.usage 直接反映 Kiro 真实用量；若将来 Kiro 返回 cacheReadTokens，kiroToClaudeResponse 会透传。

        this.throwIfResponseClosed(res, signal)
        this.recordRequestSuccess()
        this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
        this.stats.inputTokens += result.usage.inputTokens
        this.stats.outputTokens += result.usage.outputTokens
        this.accountPool.recordSuccess(usedAccount.id, result.usage.inputTokens + result.usage.outputTokens)

        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(response))
        const respTime = Date.now() - startTime
        this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
        this.recordRequest({ path: '/v1/messages', model: request.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        // 记录 API Key 用量（与 /v1/chat/completions 非流式一致；缺失会导致非流式 Anthropic 客户端不计费）
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, request.model, '/v1/messages', ProxyServer.deriveEffortLevel(request), result.usage.cacheReadTokens, result.usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
        }
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1/messages', request.model, startTime, signal)
    } finally {
      if (perfDiag.enabled) perfDiag.recordTiming(PerfPhase.Total, Date.now() - startTime)
    }
  }

  // 处理带 web 工具的 Claude 请求：代理侧执行 web_search/web_fetch 循环，再把最终回答返回客户端。
  // stream=false → 单个 JSON；stream=true → 把最终结果 replay 成 Anthropic SSE 事件。
  private async handleClaudeWebToolRequest(
    res: http.ServerResponse,
    account: { id: string; accessToken: string; profileArn?: string },
    kiroPayload: ReturnType<typeof claudeToKiro>,
    request: ClaudeRequest,
    webToolConfig: WebToolConfig,
    toolNameRegistry: ToolNameRegistry,
    startTime: number,
    matchedApiKey: import('./types').ApiKey | undefined,
    signal: AbortSignal | undefined
  ): Promise<void> {
    const loop = await runWebToolLoop(account as Parameters<typeof runWebToolLoop>[0], kiroPayload, webToolConfig, signal)
    if (loop.webToolRounds > 0) {
      proxyLogger.info('ProxyServer', `Web tool loop completed: ${loop.webToolRounds} round(s), ${loop.searches.length} search(es), model=${request.model}`)
    }

    const response = kiroToClaudeResponse(loop.content, loop.toolUses, loop.usage, request.model, toolNameRegistry, loop.reasoningContent)

    // 把代理侧执行的 web 工具调用，还原成 Anthropic 原生 server_tool_use + web_search_tool_result
    // content block，并加上 usage.server_tool_use.web_search_requests。
    // 这样 Claude Code 等客户端才能正确显示 "Did N searches" 与可点击来源（否则显示 0 次）。
    if (loop.searches.length > 0) {
      const searchBlocks = buildWebSearchContentBlocks(loop.searches)
      // server_tool_use / web_search_tool_result 放在最终回答文本之前
      response.content = [...searchBlocks, ...response.content]
      const searchCount = loop.searches.filter(s => s.kind === 'web_search').length
      if (searchCount > 0) {
        response.usage.server_tool_use = { web_search_requests: searchCount }
      }
    }

    // 不再注入本地模拟的 cache usage（Kiro 无 prompt cache，MITM 实测确认）。

    this.throwIfResponseClosed(res, signal)
    this.recordRequestSuccess()
    this.stats.totalTokens += loop.usage.inputTokens + loop.usage.outputTokens
    this.stats.inputTokens += loop.usage.inputTokens
    this.stats.outputTokens += loop.usage.outputTokens
    this.stats.totalCredits += loop.usage.credits || 0
    this.events.onCreditsUpdate?.(this.stats.totalCredits)
    this.accountPool.recordSuccess(account.id, loop.usage.inputTokens + loop.usage.outputTokens)
    const respTime = Date.now() - startTime
    this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 200, tokens: loop.usage.inputTokens + loop.usage.outputTokens, inputTokens: loop.usage.inputTokens, outputTokens: loop.usage.outputTokens, credits: loop.usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
    this.recordRequest({ path: '/v1/messages', model: request.model, accountId: account.id, inputTokens: loop.usage.inputTokens, outputTokens: loop.usage.outputTokens, credits: loop.usage.credits, responseTime: respTime, success: true })
    if (matchedApiKey) {
      this.recordApiKeyUsage(matchedApiKey.id, loop.usage.credits || 0, loop.usage.inputTokens, loop.usage.outputTokens, request.model, '/v1/messages', ProxyServer.deriveEffortLevel(request), loop.usage.cacheReadTokens, loop.usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
    }

    if (!request.stream) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(response))
      return
    }

    // stream=true：把最终 response replay 成 Anthropic SSE
    // 这是 web-tool 循环已 buffer 完结果后的「一次性同步 replay」，没有 idle 间隙，故无需心跳（heartbeat:false）；
    // 但仍需 X-Accel-Buffering:no，避免 Cloudflare/nginx 把整段 SSE 缓冲后才下发。
    this.writeSseHead(res, { heartbeat: false })
    // 整段最终结果是同步一次性 replay：cork 期间合并成尽量少的 TCP 段，uncork（在 res.end 前）一次性冲刷，
    // 减少系统调用/小包数量。字节输出完全不变；非实时增量，故不影响流式顺滑度。
    res.cork()
    const id = response.id || `msg_${uuidv4()}`
    const write = (event: string, data: unknown): void => { res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`) }
    // message_start.usage 与主流式路径口径一致：Kiro 无 prompt cache，input_tokens 即全量本轮 input。
    const replayStartUsage: { input_tokens: number; output_tokens: number } = {
      input_tokens: Math.max(1, loop.usage.inputTokens),
      output_tokens: 0
    }
    write('message_start', createClaudeStreamEvent('message_start', {
      message: { id, type: 'message', role: 'assistant', content: [], model: request.model, stop_reason: null, stop_sequence: null, usage: replayStartUsage }
    }))
    response.content.forEach((block, index) => {
      if (block.type === 'text') {
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'text', text: '' } }))
        write('content_block_delta', createClaudeStreamEvent('content_block_delta', { index, delta: { type: 'text_delta', text: block.text } }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      } else if (block.type === 'thinking') {
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'thinking', thinking: '' } }))
        write('content_block_delta', createClaudeStreamEvent('content_block_delta', { index, delta: { type: 'thinking_delta', thinking: block.thinking } }))
        if (block.signature) write('content_block_delta', createClaudeStreamEvent('content_block_delta', { index, delta: { type: 'signature_delta', signature: block.signature } }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      } else if (block.type === 'redacted_thinking') {
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'redacted_thinking', data: block.data } }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      } else if (block.type === 'tool_use') {
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} } }))
        write('content_block_delta', createClaudeStreamEvent('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) } as unknown as ClaudeStreamEvent['delta'] }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      } else if (block.type === 'server_tool_use') {
        // web_search 的查询调用：start 带 id/name，input 通过 input_json_delta 流式补齐
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'server_tool_use', id: block.id, name: block.name, input: {} } as unknown as ClaudeContentBlock }))
        write('content_block_delta', createClaudeStreamEvent('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input || {}) } as unknown as ClaudeStreamEvent['delta'] }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      } else if (block.type === 'web_search_tool_result') {
        // 搜索结果整体作为单个 content block 直接下发（含 content 数组）
        write('content_block_start', createClaudeStreamEvent('content_block_start', { index, content_block: { type: 'web_search_tool_result', tool_use_id: block.tool_use_id, content: block.content } as unknown as ClaudeContentBlock }))
        write('content_block_stop', createClaudeStreamEvent('content_block_stop', { index }))
      }
    })
    write('message_delta', createClaudeStreamEvent('message_delta', {
      delta: { stop_reason: response.stop_reason || 'end_turn', stop_sequence: null } as unknown as ClaudeStreamEvent['delta'],
      usage: response.usage.server_tool_use
        ? { output_tokens: loop.usage.outputTokens, server_tool_use: response.usage.server_tool_use }
        : { output_tokens: loop.usage.outputTokens }
    }))
    write('message_stop', createClaudeStreamEvent('message_stop', {}))
    res.uncork()
    res.end()
  }

  /**
   * 统一写出 SSE 响应头 + （可选）启动心跳。集中到一处，避免 5 个流式分支各自 copy-paste
   * 这组头——任何新增的 SSE 端点只要调用本方法，就不会漏掉 X-Accel-Buffering / 心跳，
   * 从而不会再退回「Cloudflare Tunnel 缓冲 / idle 切断流式连接」的老问题。
   *
   * X-Accel-Buffering:no —— 禁止 Cloudflare Tunnel / nginx 等中间层缓冲 SSE（否则 token 不实时、表现为「卡成一坨」）。
   * heartbeat —— 真正逐 token 流式的分支传 true（默认）；已 buffer 完一次性 replay 的分支传 false（无 idle 间隙）。
   * 幂等性由调用点保证（如 handleClaudeStream 仅在 !headersSent 时调用），本方法不重复防护。
   */
  private writeSseHead(res: http.ServerResponse, opts?: { heartbeat?: boolean }): void {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no'
    })
    if (opts?.heartbeat !== false) this.startSseHeartbeat(res)
  }

  /**
   * SSE 心跳：流式响应在「上游长时间无输出」（如模型 thinking、工具循环间隙）时，链路上没有字节流动，
   * Cloudflare Tunnel / undici 等中间层会按 idle 超时（cloudflared 默认连接、undici bodyTimeout=300s）
   * 主动关闭这条连接 → 客户端（Claude Code）看到 "socket connection closed unexpectedly"。
   * 这里每 intervalMs 写一个 SSE 注释行（": hb\n\n"，SSE 规范允许、所有客户端忽略，不进入消息流），
   * 保持链路有字节流动，避免被 idle 切断。与正常 content 写入互不干扰（注释行被客户端丢弃）。
   * 自动停止：res 触发 finish/close、或已 writableEnded/destroyed 时清理 timer，绝不泄漏。
   * 返回一个手动 stop 函数（一般无需调用，依赖 finish/close 即可）。
   */
  private startSseHeartbeat(res: http.ServerResponse, intervalMs = 15_000): () => void {
    let timer: ReturnType<typeof setInterval> | null = null
    const stop = (): void => {
      if (timer) { clearInterval(timer); timer = null }
    }
    timer = setInterval(() => {
      if (res.writableEnded || res.destroyed) { stop(); return }
      try { res.write(': hb\n\n') } catch { stop() }
    }, intervalMs)
    // 心跳不应阻止进程退出
    timer.unref?.()
    res.once('close', stop)
    res.once('finish', stop)
    return stop
  }

  // 处理 Claude 流式响应
  private async handleClaudeStream(
    res: http.ServerResponse,
    account: { id: string; accessToken: string; profileArn?: string },
    kiroPayload: ReturnType<typeof claudeToKiro>,
    model: string,
    startTime: number,
    currentRound: number = 0,
    msgId?: string,
    headersSent: boolean = false,
    contentBlockIndex: number = 0,
    matchedApiKey?: import('./types').ApiKey,
    toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
    signal?: AbortSignal,
    promptInputTokens?: number,
    effort?: string,
    // 推迟计算 input token 估算的 thunk（stream 路径专用）。给定时，先发上游请求，
    // 再在网络 RTT 期间调用它计算并发出 message_start——同步开销与网络往返重叠。
    prepareInputEstimate?: () => number,
    // client ที่ขอมา (จาก request.max_tokens) — ใช้ตรวจ truncation เพื่อรายงาน stop_reason='max_tokens'
    maxTokens?: number,
    authRetried: boolean = false,
    // 瞬时网络错误已重试标记：防止 transient-close 重试无限放大（仅重试一次）。
    transientRetried: boolean = false
  ): Promise<void> {
    if (!headersSent) {
      // 统一 SSE 头 + 心跳（auth 重试 headersSent=true 时不重复启动）
      this.writeSseHead(res)
    }

    const id = msgId || `msg_${uuidv4()}`
    let currentBlockIndex = contentBlockIndex
    let hasStartedTextBlock = false
    let hasStartedThinkingBlock = false
    let pendingThinkingSignature: string | undefined
    let collectedContent = ''
    const pendingToolCalls: Map<string, { name: string; input: Record<string, unknown> }> = new Map()
    // 是否已向客户端发出真实内容（文本/思考/工具/redacted）——auth 重试只在「尚未发出真实内容」时安全。
    // 注意：message_start（ensureStarted）不算真实内容，重试时无需回退它。
    let sentRealContent = false

    const flushThinkingSignature = () => {
      if (!pendingThinkingSignature) return
      const signatureDelta = createClaudeStreamEvent('content_block_delta', {
        index: currentBlockIndex,
        delta: { type: 'signature_delta', signature: pendingThinkingSignature }
      })
      res.write(`event: content_block_delta\ndata: ${JSON.stringify(signatureDelta)}\n\n`)
      pendingThinkingSignature = undefined
    }

    // message_start 的 usage 必须与 message_delta 的口径一致，否则客户端（如 Claude Code）
    // 统计上下文用量时会偏高。Anthropic 规范里 message_start.usage 就带有完整的
    // input_tokens + cache_read/creation 拆分，且 input_tokens 不含缓存部分。
    // 这里优先用调用方按 Claude 原始请求结构算出的 token 数（与 /v1/messages/count_tokens 同源），
    // 仅在缺失时回退到基于 payload 体积的粗略估算（≈3.5 字符/token，贴近 cl100k_base）。
    //
    // ensureStarted：幂等地解析 input token 估算（若给了 prepareInputEstimate 则此刻才计算——与上游 RTT 重叠）
    // 并发出 message_start（仅首轮、仅一次）。在「主动 setImmediate」与「首个 onChunk/onComplete」
    // 两处调用，谁先到谁触发，保证 message_start 必定先于任何 content delta。
    let messageStarted = false
    const ensureStarted = (): void => {
      if (messageStarted) return
      messageStarted = true
      let grossInputTokens = promptInputTokens
      if (prepareInputEstimate) {
        grossInputTokens = prepareInputEstimate()
      }
      if (currentRound !== 0) return
      const gross = (typeof grossInputTokens === 'number' && grossInputTokens > 0)
        ? grossInputTokens
        : this.estimateKiroPayloadTokens(kiroPayload)
      // Kiro 无 prompt cache，input_tokens 即全量本轮 input（不扣 cache）。
      const startUsage: { input_tokens: number; output_tokens: number } = {
        input_tokens: Math.max(1, gross),
        output_tokens: 0
      }
      const messageStart = createClaudeStreamEvent('message_start', {
        message: {
          id,
          type: 'message',
          role: 'assistant',
          content: [],
          model,
          stop_reason: null,
          stop_sequence: null,
          usage: startUsage
        }
      })
      res.write(`event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`)
    }

    return new Promise((resolve) => {
      callKiroApiStream(
        account as any,
        kiroPayload,
        (text, toolUse, isThinking, reasoningSignature, redactedContent) => {
          if (signal?.aborted || this.isResponseClosed(res)) return
          // 标记：一旦有真实内容（文本/思考/工具/redacted）即不可再透明重试
          if (redactedContent || (text && text.trim()) || toolUse) sentRealContent = true
          ensureStarted() // message_start 必须先于任何 content delta（幂等）
          // 优先处理 redacted_thinking（加密的 thinking 块，需单独 content_block）
          if (redactedContent) {
            if (hasStartedTextBlock) {
              const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
              res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
              currentBlockIndex++
              hasStartedTextBlock = false
            }
            if (hasStartedThinkingBlock) {
              flushThinkingSignature()
              const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
              res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
              currentBlockIndex++
              hasStartedThinkingBlock = false
            }
            const blockStart = createClaudeStreamEvent('content_block_start', {
              index: currentBlockIndex,
              content_block: { type: 'redacted_thinking', data: redactedContent }
            })
            res.write(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`)
            const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
            res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
            currentBlockIndex++
            return
          }
          if (text && text.trim()) {
            if (isThinking) {
              // 原生 thinking 内容 → 输出为 Anthropic thinking block
              if (hasStartedTextBlock) {
                const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
                res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
                currentBlockIndex++
                hasStartedTextBlock = false
              }
              if (!hasStartedThinkingBlock) {
                const blockStart = createClaudeStreamEvent('content_block_start', {
                  index: currentBlockIndex,
                  content_block: { type: 'thinking', thinking: '' }
                })
                res.write(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`)
                hasStartedThinkingBlock = true
              }
              const delta = createClaudeStreamEvent('content_block_delta', {
                index: currentBlockIndex,
                delta: { type: 'thinking_delta', thinking: text }
              })
              res.write(`event: content_block_delta\ndata: ${JSON.stringify(delta)}\n\n`)
              if (reasoningSignature) {
                pendingThinkingSignature = reasoningSignature
              }
            } else {
              // 普通文本内容
              if (hasStartedThinkingBlock) {
                flushThinkingSignature()
                const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
                res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
                currentBlockIndex++
                hasStartedThinkingBlock = false
              }
              collectedContent += text
              if (!hasStartedTextBlock) {
                const blockStart = createClaudeStreamEvent('content_block_start', {
                  index: currentBlockIndex,
                  content_block: { type: 'text', text: '' }
                })
                res.write(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`)
                hasStartedTextBlock = true
              }
              const delta = createClaudeStreamEvent('content_block_delta', {
                index: currentBlockIndex,
                delta: { type: 'text_delta', text }
              })
              res.write(`event: content_block_delta\ndata: ${JSON.stringify(delta)}\n\n`)
            }
          } else if (isThinking && reasoningSignature) {
            if (!hasStartedThinkingBlock) {
              const blockStart = createClaudeStreamEvent('content_block_start', {
                index: currentBlockIndex,
                content_block: { type: 'thinking', thinking: '' }
              })
              res.write(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`)
              hasStartedThinkingBlock = true
            }
            pendingThinkingSignature = reasoningSignature
          }
          if (toolUse) {
            const restoredToolUse = toolNameRegistry.restoreToolUse(toolUse)
            if (hasStartedThinkingBlock) {
              flushThinkingSignature()
              const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
              res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
              currentBlockIndex++
              hasStartedThinkingBlock = false
            }
            // 结束之前的文本块
            if (hasStartedTextBlock) {
              const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
              res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
              currentBlockIndex++
              hasStartedTextBlock = false
            }
            // 记录工具调用
            pendingToolCalls.set(toolUse.toolUseId, { name: toolUse.name, input: toolUse.input })
            // 开始工具块
            const toolBlockStart = createClaudeStreamEvent('content_block_start', {
              index: currentBlockIndex,
              content_block: { type: 'tool_use', id: toolUse.toolUseId, name: restoredToolUse.name, input: {} }
            })
            res.write(`event: content_block_start\ndata: ${JSON.stringify(toolBlockStart)}\n\n`)
            // 发送工具输入
            const toolDelta = createClaudeStreamEvent('content_block_delta', {
              index: currentBlockIndex,
              delta: { type: 'input_json_delta', partial_json: JSON.stringify(toolUse.input) } as any
            })
            res.write(`event: content_block_delta\ndata: ${JSON.stringify(toolDelta)}\n\n`)
            // 结束工具块
            const toolBlockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
            res.write(`event: content_block_stop\ndata: ${JSON.stringify(toolBlockStop)}\n\n`)
            currentBlockIndex++
          }
        },
        async (usage) => {
          if (signal?.aborted || this.isResponseClosed(res)) {
            // 客户端已断开：仍按已消耗用量给客户计费，防白嫖
            this.settleAbortedUsage(matchedApiKey, account.id, usage, model, '/v1/messages', effort, kiroPayload.conversationState.conversationId)
            resolve()
            return
          }
          ensureStarted() // 保证 message_start 已发出（空响应时也需先于 message_delta）
          if (hasStartedThinkingBlock) {
            flushThinkingSignature()
            const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
            res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
            currentBlockIndex++
            hasStartedThinkingBlock = false
          }

          // 结束最后的文本块
          if (hasStartedTextBlock) {
            const blockStop = createClaudeStreamEvent('content_block_stop', { index: currentBlockIndex })
            res.write(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`)
            currentBlockIndex++
          }

          this.recordRequestSuccess()
          this.stats.totalTokens += usage.inputTokens + usage.outputTokens
          this.stats.inputTokens += usage.inputTokens
          this.stats.outputTokens += usage.outputTokens
          this.stats.totalCredits += usage.credits || 0
          this.events.onCreditsUpdate?.(this.stats.totalCredits)
          this.events.onTokensUpdate?.(this.stats.inputTokens, this.stats.outputTokens)
          this.accountPool.recordSuccess(account.id, usage.inputTokens + usage.outputTokens)
          this.stats.cacheReadTokens += usage.cacheReadTokens || 0
          this.stats.cacheWriteTokens += usage.cacheWriteTokens || 0
          this.stats.reasoningTokens += usage.reasoningTokens || 0
          const respTime = Date.now() - startTime
          this.events.onResponse?.({ path: '/v1/messages', model, status: 200, tokens: usage.inputTokens + usage.outputTokens, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, cacheReadTokens: usage.cacheReadTokens, reasoningTokens: usage.reasoningTokens, credits: usage.credits, responseTime: respTime, sessionId: kiroPayload.conversationState.conversationId })
          this.recordRequest({ path: '/v1/messages', model, accountId: account.id, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, credits: usage.credits, responseTime: respTime, success: true })
          // 记录 API Key 用量
          if (matchedApiKey) {
            this.recordApiKeyUsage(matchedApiKey.id, usage.credits || 0, usage.inputTokens, usage.outputTokens, model, '/v1/messages', effort, usage.cacheReadTokens, usage.cacheWriteTokens, kiroPayload.conversationState.conversationId)
          }

          // 发送 message_delta（包含完整 usage 信息）
          // stop_reason 优先级（保守策略，不影响已工作的工具循环）：
          //   1) 有工具调用 → 'tool_use'（客户端继续执行工具）
          //   2) 否则若输出 token 达到/超过 client 请求的 max_tokens → 'max_tokens'（响应被截断）
          //      —— 关键修复：旧逻辑此时误报 'end_turn'，导致 Claude Code 以为正常结束而提前停工
          //   3) 否则 → 'end_turn'
          const hasToolCalls = pendingToolCalls.size > 0
          const truncatedByMaxTokens = !hasToolCalls && !!maxTokens && maxTokens > 0 && usage.outputTokens >= maxTokens
          const stopReason: 'tool_use' | 'max_tokens' | 'end_turn' =
            hasToolCalls ? 'tool_use' : (truncatedByMaxTokens ? 'max_tokens' : 'end_turn')
          if (truncatedByMaxTokens) {
            proxyLogger.info('ProxyServer', `stop_reason=max_tokens (output ${usage.outputTokens} >= max_tokens ${maxTokens}) — response truncated`)
          }
          const messageDelta = createClaudeStreamEvent('message_delta', {
            delta: { stop_reason: stopReason, stop_sequence: null } as any,
            usage: this.buildClaudeUsage(usage)
          })
          res.write(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`)
          // 发送 message_stop
          const messageStop = createClaudeStreamEvent('message_stop')
          res.write(`event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`)
          res.end()
          resolve()
        },
        async (error, partialUsage) => {
          if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
            // 客户端断开但 Kiro 可能已计 credit：结算已消耗用量，防漏计费
            if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, model, '/v1/messages', effort, kiroPayload.conversationState.conversationId)
            resolve()
            return
          }

          const errStatusCode2 = error.message.match(/(\d{3})/)?.[1]
          const statusNum2 = errStatusCode2 ? parseInt(errStatusCode2) : 0

          // 401/403 自愈：token 被服务端撤销/轮换。仅在「未发出真实内容」且「未重试过」时透明重跑整个 stream。
          // message_start 已发出无妨——Anthropic 客户端允许 message_start 后跟随完整内容；重试用 headersSent=true
          // 不重写响应头，authRetried=true 透传给重试分支（重试自身不再 retry）。
          if (
            (statusNum2 === 401 || statusNum2 === 403) &&
            !authRetried &&
            !sentRealContent &&
            !signal?.aborted &&
            !this.isResponseClosed(res)
          ) {
            console.log(`[ProxyServer] Claude stream auth error ${statusNum2} before content; refreshing token and retrying once`)
            let refreshed = false
            const fullAccount = this.accountPool.getAccount(account.id)
            try {
              if (fullAccount) refreshed = await this.refreshToken(fullAccount, signal)
            } catch (refreshErr) {
              console.error('[ProxyServer] Token refresh during stream failed:', formatError(refreshErr))
            }
            if (signal?.aborted || this.isResponseClosed(res)) {
              resolve()
              return
            }
            if (refreshed) {
              const refreshedAccount = this.accountPool.getAccount(account.id) || account
              this.handleClaudeStream(
                res, refreshedAccount as typeof account, kiroPayload, model, startTime,
                currentRound, id, true, currentBlockIndex, matchedApiKey, toolNameRegistry, signal,
                promptInputTokens, effort, undefined, maxTokens, true
              ).then(resolve).catch(() => resolve())
              return
            }
          }

          // 瞬时上游断连自愈：parseEventStream 读流途中上游/中间层 idle 切断
          // （UND_ERR_SOCKET / "other side closed" / terminated）。仅在「尚未向客户端发出真实内容」
          // 且「本请求未因瞬时错误重试过」时安全——此时透明重跑整个 stream，客户端无感知。
          // 与 401/403 自愈同构：不在重试前 settle partialUsage，避免与重试成功后的正式计费重复。
          // 已发出真实内容（sentRealContent）则无法回退，照常报错（重试会重复 content）。
          if (
            statusNum2 === 0 &&
            !transientRetried &&
            !sentRealContent &&
            !signal?.aborted &&
            !this.isResponseClosed(res) &&
            isTransientNetworkError(error)
          ) {
            console.log(`[ProxyServer] Claude stream transient close before content; retrying once: ${formatError(error)}`)
            this.handleClaudeStream(
              res, account, kiroPayload, model, startTime,
              currentRound, id, true, currentBlockIndex, matchedApiKey, toolNameRegistry, signal,
              promptInputTokens, effort, undefined, maxTokens, authRetried, true
            ).then(resolve).catch(() => resolve())
            return
          }

          console.error('[ProxyServer] Stream error:', formatError(error))
          this.recordRequest({ path: '/v1/messages', model, accountId: account.id, responseTime: Date.now() - startTime, success: false, error: error.message })
          // mid-stream 报错但 Kiro 已计 credit（meteringEvent 先于断开）：结算已消耗用量，防漏计费
          if (partialUsage) this.settleAbortedUsage(matchedApiKey, account.id, partialUsage, model, '/v1/messages', effort, kiroPayload.conversationState.conversationId)
          resolve()
        },
        signal,
        this.config.preferredEndpoint,
        model // clientModelId：保留 [1m] 后缀，让 contextUsageEvent 反推用正确的 context 分母
      ).catch(error => {
        if (!this.isAbortError(error, signal) && !this.isResponseClosed(res)) {
          const errorEvent = createClaudeStreamEvent('error', {
            error: { type: 'api_error', message: error.message }
          })
          res.write(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`)
          res.end()
          this.recordRequestFailed()
        }
        resolve()
      })
      // 上游请求已发出（callKiroApiStream 内部已 await fetch 并让出事件循环）。
      // 在下一个 macrotask 主动计算缓存 usage 并发出 message_start——此时请求正在网络上往返，
      // 这段同步计算与 RTT 重叠，缩短 TTFT；onChunk/onComplete 里的 ensureStarted 仅作兜底（幂等）。
      setImmediate(() => { if (!signal?.aborted && !this.isResponseClosed(res)) ensureStarted() })
    })
  }

  // 处理 API 错误
  private handleApiError(res: http.ServerResponse, account: { id: string }, error: Error, path: string, model?: string, startTime?: number, signal?: AbortSignal): void {
    if (this.isAbortError(error, signal) || this.isResponseClosed(res)) return
    this.recordRequestFailed()
    const errCode = error.message.match(/(\d{3})/)?.[1]
    const parsedCode = errCode ? parseInt(errCode) : 500
    const errorType = classifyError(parsedCode)
    const isAuthError = error.message.includes('401') || error.message.includes('403') || error.message.includes('Auth')

    this.accountPool.recordError(account.id, errorType, parsedCode)

    let statusCode = parsedCode
    if (isAuthError) statusCode = 401

    if (res.headersSent) {
      if (!this.isResponseClosed(res)) {
        if (path === '/v1/responses' || path === '/responses') {
          res.write(`event: response.failed\ndata: ${JSON.stringify({ type: 'response.failed', error: { type: 'api_error', message: error.message } })}\n\n`)
        }
        res.end()
      }
      this.events.onResponse?.({ path, status: statusCode, error: error.message })
      this.recordRequest({ path, model, accountId: account.id, responseTime: startTime ? Date.now() - startTime : 0, success: false, error: error.message })
      return
    }

    this.sendError(res, statusCode, error.message, this.isAnthropicPath(path) ? 'anthropic' : 'openai')
    this.events.onResponse?.({ path, status: statusCode, error: error.message })
    this.recordRequest({ path, model, accountId: account.id, responseTime: startTime ? Date.now() - startTime : 0, success: false, error: error.message })
  }

  // 读取请求体
  /**
   * 读取请求体，限制最大字节数以防 DoS
   * - Content-Length 头超限：立即 reject
   * - 流式累加超限：销毁连接并 reject
   * 触发 BodyTooLarge 错误时上层会发 413 Payload Too Large
   */
  private readBody(req: http.IncomingMessage, signal?: AbortSignal): Promise<string> {
    // 默认 50MB：base64 会让二进制膨胀 ~33%，10MB 旧默认会让 ~7.5MB 以上的 PDF/图片
    // 在 readBody 阶段就被 413 拒掉（用户侧表现为"附件请求直接失败"）。50MB 可容纳
    // 单个大 PDF 或多张图片；仍由 Content-Length 提前拒绝超大体，避免内存放大攻击。
    const maxBytes = Math.max(1024, this.config.maxRequestBodyBytes ?? 50 * 1024 * 1024)

    // 优先用 Content-Length 提前拒绝（避免分配缓冲）
    const declaredLen = parseInt(req.headers['content-length'] || '0', 10)
    if (Number.isFinite(declaredLen) && declaredLen > maxBytes) {
      return Promise.reject(new BodyTooLargeError(declaredLen, maxBytes))
    }

    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = []
      let total = 0
      const cleanup = () => {
        req.off('data', onData)
        req.off('end', onEnd)
        req.off('error', onError)
        req.off('aborted', onAborted)
        signal?.removeEventListener('abort', onAbort)
      }
      const onData = (chunk: Buffer) => {
        total += chunk.length
        if (total > maxBytes) {
          cleanup()
          try { req.destroy() } catch { /* ignore */ }
          reject(new BodyTooLargeError(total, maxBytes))
          return
        }
        chunks.push(chunk)
      }
      const onEnd = () => {
        cleanup()
        resolve(Buffer.concat(chunks, total).toString('utf8'))
      }
      const onError = (error: Error) => {
        cleanup()
        reject(error)
      }
      const onAborted = () => {
        cleanup()
        reject(new Error('Client disconnected'))
      }
      const onAbort = () => {
        cleanup()
        reject(this.getAbortError(signal))
      }
      if (signal?.aborted) {
        reject(this.getAbortError(signal))
        return
      }
      req.on('data', onData)
      req.on('end', onEnd)
      req.on('error', onError)
      req.on('aborted', onAborted)
      signal?.addEventListener('abort', onAbort, { once: true })
    })
  }

  // 发送错误响应
  // P0-5 自动 sanitize：500 类不吐 message 详情；4xx 客户端错误正常返回
  private sendError(res: http.ServerResponse, status: number, message: string, format: 'openai' | 'anthropic' = 'openai'): void {
    if (res.writableEnded || res.destroyed) return
    // 500-599 强制使用通用消息（防止泄露内部信息）
    const safeMessage = status >= 500 && status < 600
      ? this.sanitizeErrorMessage(message) || 'Internal server error'
      : message
    // P1-6 503 → 触发 webhook（已有 5 分钟去重）
    if (status === 503) {
      this.notifyAllAccountsExhausted('unknown')
    }
    res.writeHead(status, { 'Content-Type': 'application/json' })
    if (format === 'anthropic') {
      res.end(JSON.stringify({
        type: 'error',
        error: {
          type: this.getAnthropicErrorType(status),
          message: safeMessage
        }
      }))
      return
    }
    res.end(JSON.stringify({ error: { message: safeMessage, type: 'error', code: status } }))
  }

  /**
   * P0-5 / P2-19 错误消息脱敏（移除可能含的 Bearer/Token/路径等敏感信息）
   * 用于错误响应和日志输出
   */
  private sanitizeErrorMessage(msg: string): string {
    if (!msg) return ''
    return msg
      // Bearer xxxx → Bearer ***
      .replace(/Bearer\s+[A-Za-z0-9\-_.~+/]+=*/gi, 'Bearer ***')
      // access_token / refresh_token / api_key / x-api-key 字段值
      .replace(/(access[_-]?token|refresh[_-]?token|api[_-]?key|x-api-key)["'\s:=]+[^"',\s}]+/gi, '$1=***')
      // 长 base64/JWT（>= 40 chars）替换为占位
      .replace(/eyJ[A-Za-z0-9\-_]{20,}/g, 'eyJ***')
      // Windows 用户路径
      .replace(/C:\\Users\\[^\\/\s]+/gi, 'C:\\Users\\***')
      // Linux/Mac home 路径
      .replace(/\/home\/[^\s/]+/g, '/home/***')
      .replace(/\/Users\/[^\s/]+/g, '/Users/***')
  }

  /**
   * P1-7 滑动窗口限流：每分钟 N 次（按 API Key id 或 IP）
   * 0 = 不限制
   */
  private checkRateLimit(id: string): { allowed: boolean; retryAfterMs: number } {
    const limit = this.config.rateLimitPerKeyPerMinute || 0
    if (limit <= 0) return { allowed: true, retryAfterMs: 0 }

    const now = Date.now()
    const bucket = this.rateLimitBuckets.get(id)
    if (!bucket || now - bucket.windowStart >= 60_000) {
      this.rateLimitBuckets.set(id, { count: 1, windowStart: now })
      return { allowed: true, retryAfterMs: 0 }
    }
    if (bucket.count >= limit) {
      return { allowed: false, retryAfterMs: 60_000 - (now - bucket.windowStart) }
    }
    bucket.count++
    return { allowed: true, retryAfterMs: 0 }
  }

  /**
   * 门户登录限流（按 IP，固定每分钟 10 次尝试）。
   * 与业务 rateLimitPerKeyPerMinute 独立，且无论是否配置都生效，
   * 因为 /portal/* 走的是会话鉴权、绕过了 validateApiKey 那条限流路径。
   */
  private checkPortalLoginRate(clientIP: string): { allowed: boolean; retryAfterMs: number } {
    const LIMIT = 10
    const id = `login:${clientIP || 'unknown'}`
    const now = Date.now()
    const bucket = this.portalLoginBuckets.get(id)
    if (!bucket || now - bucket.windowStart >= 60_000) {
      this.portalLoginBuckets.set(id, { count: 1, windowStart: now })
      return { allowed: true, retryAfterMs: 0 }
    }
    if (bucket.count >= LIMIT) {
      return { allowed: false, retryAfterMs: 60_000 - (now - bucket.windowStart) }
    }
    bucket.count++
    return { allowed: true, retryAfterMs: 0 }
  }

  /**
   * 尝试为某客户占用一个在途请求槽位（信用预留）。
   * 返回 false 表示已达并发上限，调用方应拒绝（429），不要忘记仅在返回 true 时 release。
   * cap=0 表示不限制（始终返回 true 且不计数，release 也安全）。
   */
  private acquireCustomerSlot(customerId: string): boolean {
    // 默认 32：Claude Code 单会话常并发开多个 stream（子代理/并行工具调用），cap=6 会误触发 429。
    // 余额防护的真正防线在 validateApiKey 的 credit 检查；此 cap 仅防极端 runaway，故放宽默认值。
    // cap<=0 表示完全不限制。
    const cap = this.config.portalMaxConcurrentPerCustomer ?? 32
    if (cap <= 0) return true
    const cur = this.customerInFlight.get(customerId) || 0
    if (cur >= cap) return false
    this.customerInFlight.set(customerId, cur + 1)
    return true
  }

  /** 释放某客户的在途请求槽位（与 acquireCustomerSlot 成对，放在 finally）。 */
  private releaseCustomerSlot(customerId: string): void {
    const cur = this.customerInFlight.get(customerId) || 0
    if (cur <= 1) this.customerInFlight.delete(customerId)
    else this.customerInFlight.set(customerId, cur - 1)
  }

  /** 定期清理过期的限流桶 / 会话粘性条目（避免内存泄漏） */
  private cleanupExpiredCaches(): void {
    const now = Date.now()
    // 限流桶过期 2 分钟
    for (const [key, bucket] of this.rateLimitBuckets) {
      if (now - bucket.windowStart > 120_000) this.rateLimitBuckets.delete(key)
    }
    // 粘性会话过期 10 分钟
    for (const [key, entry] of this.sessionAffinity) {
      if (now - entry.lastAt > 600_000) this.sessionAffinity.delete(key)
    }
    // 审计日志最多 200 条
    if (this.auditLog.length > 200) {
      this.auditLog = this.auditLog.slice(-200)
    }
  }

  /**
   * P1-8 会话粘性账号选择：相同 session hint 优先复用同一账号
   * 实现方式：用 sessionHint hash 索引到固定账号；账号失效时自动失效粘性
   */
  private pickAccountWithAffinity(sessionHint: string | undefined): ProxyAccount | null {
    if (!this.config.sessionAffinityEnabled || !sessionHint) return null
    const entry = this.sessionAffinity.get(sessionHint)
    if (entry) {
      const account = this.accountPool.getAccount(entry.accountId)
      // 校验账号仍可用且未被封禁
      if (account && !this.accountPool.isSuspended(account) && account.isAvailable !== false) {
        entry.lastAt = Date.now()
        return account
      }
      // 已失效 → 清掉粘性
      this.sessionAffinity.delete(sessionHint)
    }
    return null
  }

  /** 记录粘性映射 */
  private rememberAffinity(sessionHint: string | undefined, accountId: string): void {
    if (!this.config.sessionAffinityEnabled || !sessionHint) return
    this.sessionAffinity.set(sessionHint, { accountId, lastAt: Date.now() })
  }

  /** P2-17 审计日志 */
  private appendAuditLog(type: string, data: Record<string, unknown>): void {
    if (!this.config.enableAuditLog) return
    this.auditLog.push({ ts: Date.now(), type, data })
    if (this.auditLog.length > 200) this.auditLog.shift()
  }

  /** 获取审计日志（供管理 API） */
  getAuditLog(): ReadonlyArray<{ ts: number; type: string; data: Record<string, unknown> }> {
    return this.auditLog
  }

  /** 注入 webhook 触发器（由 main/index.ts 注入，调用 renderer 的 webhook store） */
  setWebhookTrigger(fn: (event: string, payload: Record<string, unknown>) => void): void {
    this.webhookTrigger = fn
  }

  /** 关键事件去重时间戳（5 分钟内同事件不重复推） */
  private lastWebhookByEvent: Map<string, number> = new Map()

  /** P1-6 触发 webhook（封装错误处理 + 5 分钟去重） */
  private triggerWebhook(event: string, payload: Record<string, unknown>): void {
    const now = Date.now()
    const last = this.lastWebhookByEvent.get(event) || 0
    if (now - last < 5 * 60_000) return  // 同事件 5 分钟内不重复推
    this.lastWebhookByEvent.set(event, now)
    try { this.webhookTrigger?.(event, payload) } catch (err) {
      proxyLogger.warn('ProxyServer', `Webhook trigger failed: ${(err as Error).message}`)
    }
  }

  /** 全员配额耗尽 webhook（503 时调用） */
  private notifyAllAccountsExhausted(path: string, model?: string): void {
    const quota = this.accountPool.getQuotaStatus()
    this.appendAuditLog('all_accounts_exhausted', { path, model, ...quota })
    this.triggerWebhook('proxy-all-exhausted', {
      title: '反代账号全部不可用',
      message: `所有账号配额耗尽或冷却中（exhausted=${quota.exhausted}/${quota.total}，cooldown=${quota.cooldown}）`,
      level: 'error',
      fields: { 端点: path, 模型: model || '-', 总账号: quota.total, 配额耗尽: quota.exhausted, 冷却中: quota.cooldown, 可用: quota.available }
    })
  }

  /**
   * 池容量预警：可用账号数跌破 poolLowThreshold 时推送 webhook，提醒及早补充账号。
   * 在"全员耗尽（503）"之前就告警，给补号留时间。triggerWebhook 自带 5 分钟去重，
   * 不会刷屏。available 回升到阈值以上后，下次跌破会重新告警（中间的去重窗口过期即可）。
   */
  private checkPoolLow(): void {
    const threshold = this.config.poolLowThreshold || 0
    if (threshold <= 0) return
    const quota = this.accountPool.getQuotaStatus()
    // total=0（还没加账号）不算预警场景；available 为 0 已由 all-exhausted 覆盖，这里仍推以防漏报
    if (quota.total === 0) return
    if (quota.available <= threshold) {
      this.triggerWebhook('proxy-pool-low', {
        title: '反代可用账号偏低',
        message: `可用账号仅剩 ${quota.available} 个（阈值 ${threshold}，总 ${quota.total}），建议尽快补充账号`,
        level: 'warn',
        fields: { 可用: quota.available, 阈值: threshold, 总账号: quota.total, 配额耗尽: quota.exhausted, 冷却中: quota.cooldown }
      })
    }
  }

  /** P2-16 Prometheus metrics 文本 */
  private renderPrometheusMetrics(): string {
    const s = this.stats
    const ap = this.accountPool
    const lines: string[] = []
    lines.push('# HELP kiro_proxy_requests_total Total requests handled')
    lines.push('# TYPE kiro_proxy_requests_total counter')
    lines.push(`kiro_proxy_requests_total ${s.totalRequests}`)
    lines.push('# HELP kiro_proxy_requests_success_total Total successful requests')
    lines.push('# TYPE kiro_proxy_requests_success_total counter')
    lines.push(`kiro_proxy_requests_success_total ${s.successRequests}`)
    lines.push('# HELP kiro_proxy_requests_failed_total Total failed requests')
    lines.push('# TYPE kiro_proxy_requests_failed_total counter')
    lines.push(`kiro_proxy_requests_failed_total ${s.failedRequests}`)
    lines.push('# HELP kiro_proxy_tokens_total Total tokens consumed')
    lines.push('# TYPE kiro_proxy_tokens_total counter')
    lines.push(`kiro_proxy_tokens_total{type="input"} ${s.inputTokens}`)
    lines.push(`kiro_proxy_tokens_total{type="output"} ${s.outputTokens}`)
    lines.push(`kiro_proxy_tokens_total{type="cache_read"} ${s.cacheReadTokens}`)
    lines.push(`kiro_proxy_tokens_total{type="cache_write"} ${s.cacheWriteTokens}`)
    lines.push('# HELP kiro_proxy_credits_total Total credits consumed')
    lines.push('# TYPE kiro_proxy_credits_total counter')
    lines.push(`kiro_proxy_credits_total ${s.totalCredits}`)
    lines.push('# HELP kiro_proxy_accounts Accounts by status')
    lines.push('# TYPE kiro_proxy_accounts gauge')
    const quota = ap.getQuotaStatus()
    lines.push(`kiro_proxy_accounts{status="total"} ${quota.total}`)
    lines.push(`kiro_proxy_accounts{status="available"} ${quota.available}`)
    lines.push(`kiro_proxy_accounts{status="exhausted"} ${quota.exhausted}`)
    lines.push(`kiro_proxy_accounts{status="cooldown"} ${quota.cooldown}`)
    lines.push('# HELP kiro_proxy_uptime_seconds Server uptime in seconds')
    lines.push('# TYPE kiro_proxy_uptime_seconds gauge')
    lines.push(`kiro_proxy_uptime_seconds ${Math.floor((Date.now() - s.startTime) / 1000)}`)
    // 性能诊断指标（仅在 perfDiagnostics 开启时输出；关闭时返回空数组）
    for (const line of perfDiag.renderMetrics()) lines.push(line)
    return lines.join('\n') + '\n'
  }

  // 记录请求到 recentRequests
  private recordRequest(log: {
    path: string
    model?: string
    accountId?: string
    inputTokens?: number
    outputTokens?: number
    credits?: number
    responseTime?: number
    success: boolean
    error?: string
  }): void {
    this.stats.recentRequests.push({
      timestamp: Date.now(),
      path: log.path,
      model: log.model || 'unknown',
      accountId: log.accountId || 'unknown',
      inputTokens: log.inputTokens || 0,
      outputTokens: log.outputTokens || 0,
      credits: log.credits,
      responseTime: log.responseTime || 0,
      success: log.success,
      // P2-19 错误消息脱敏
      error: log.error ? this.sanitizeErrorMessage(log.error).slice(0, 500) : undefined
    })
    // P2-15 可配置上限（默认 100，最多 10000）
    const limit = Math.min(10000, Math.max(20, this.config.recentRequestsLimit || 100))
    if (this.stats.recentRequests.length > limit) {
      this.stats.recentRequests = this.stats.recentRequests.slice(-limit)
    }
  }
}


