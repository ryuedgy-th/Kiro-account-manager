// Kiro Proxy 类型定义

// ============ OpenAI 兼容格式 ============
export interface OpenAIChatRequest {
  model: string
  messages: OpenAIMessage[]
  temperature?: number
  top_p?: number
  max_tokens?: number
  stream?: boolean
  tools?: OpenAITool[]
  tool_choice?: string | { type: string; function: { name: string } }
  response_format?: { type: string; json_schema?: unknown }
  conversation_id?: string
  metadata?: Record<string, unknown>
  kiro_context?: KiroRequestContext
  reasoning_effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | string
  thinking?: { type: 'enabled'; budget_tokens?: number } | { type: 'adaptive' } | { type: 'disabled' }
}

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | OpenAIContentPart[]
  reasoning_content?: string
  name?: string
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
  cache_control?: ClaudeCacheControl
}

export interface OpenAIContentPart {
  type: 'text' | 'image_url' | 'file' | 'document'
  text?: string
  image_url?: { url: string; detail?: string }
  file?: { filename?: string; file_data?: string }
  source?: ClaudeDocumentSource
  name?: string
  cache_control?: ClaudeCacheControl
}

export interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: unknown
  }
  cache_control?: ClaudeCacheControl
}

export interface OpenAIToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

export interface OpenAIChatResponse {
  id: string
  object: 'chat.completion'
  created: number
  model: string
  choices: OpenAIChoice[]
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens?: number
    }
    completion_tokens_details?: {
      reasoning_tokens?: number
    }
  }
}

export interface OpenAIChoice {
  index: number
  message: {
    role: 'assistant'
    content: string | null
    reasoning_content?: string
    tool_calls?: OpenAIToolCall[]
  }
  finish_reason: 'stop' | 'length' | 'tool_calls' | null
}

export interface OpenAIStreamChunk {
  id: string
  object: 'chat.completion.chunk'
  created: number
  model: string
  choices: {
    index: number
    delta: {
      role?: 'assistant'
      content?: string
      reasoning_content?: string
      tool_calls?: Partial<OpenAIToolCall>[]
    }
    finish_reason: 'stop' | 'length' | 'tool_calls' | null
  }[]
}

export interface OpenAIResponsesRequest {
  model: string
  input: string | OpenAIResponseInputItem[]
  instructions?: string
  temperature?: number
  top_p?: number
  max_output_tokens?: number
  stream?: boolean
  tools?: OpenAITool[]
  tool_choice?: string | { type: string; name?: string; function?: { name: string } }
  previous_response_id?: string
  reasoning?: { effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | string } | null
  metadata?: Record<string, unknown>
  kiro_context?: KiroRequestContext
}

export interface OpenAIResponseInputItem {
  type?: 'message' | 'function_call' | 'function_call_output'
  role?: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OpenAIResponseContentPart[]
  call_id?: string
  name?: string
  arguments?: string
  output?: string
}

export interface OpenAIResponseContentPart {
  type: 'input_text' | 'output_text' | 'input_image' | 'input_file'
  text?: string
  image_url?: string
  file_data?: string
  filename?: string
}

export interface OpenAIResponsesResponse {
  id: string
  object: 'response'
  created_at: number
  model: string
  output: OpenAIResponseOutputItem[]
  previous_response_id?: string
  usage: {
    input_tokens: number
    output_tokens: number
    total_tokens: number
    input_tokens_details?: { cached_tokens?: number }
    output_tokens_details?: { reasoning_tokens?: number }
  }
}

export type OpenAIResponseOutputItem =
  | { type: 'message'; id: string; role: 'assistant'; content: { type: 'output_text'; text: string }[] }
  | { type: 'function_call'; id: string; call_id: string; name: string; arguments: string }

// ============ Claude 兼容格式 ============
export interface ClaudeRequest {
  model: string
  messages: ClaudeMessage[]
  max_tokens: number
  temperature?: number
  top_p?: number
  stream?: boolean
  system?: string | ClaudeSystemBlock[]
  tools?: ClaudeTool[]
  tool_choice?: { type: string; name?: string }
  thinking?: { type: 'enabled'; budget_tokens: number } | { type: 'adaptive'; display?: string } | { type: 'disabled' }
  conversation_id?: string
  metadata?: Record<string, unknown>
  kiro_context?: KiroRequestContext
  anthropic_beta?: string[]
  output_config?: { effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | string; task_budget?: { type: 'tokens'; total: number; remaining?: number } }
  context_management?: { type?: string; [key: string]: unknown }
}

export interface ClaudeMessage {
  role: 'user' | 'assistant'
  content: string | ClaudeContentBlock[]
  cache_control?: ClaudeCacheControl
}

export interface ClaudeSystemBlock {
  type: 'text'
  text: string
  cache_control?: ClaudeCacheControl
}

export interface ClaudeContentBlock {
  type: 'text' | 'image' | 'document' | 'tool_use' | 'tool_result' | 'thinking' | 'redacted_thinking' | 'server_tool_use' | 'web_search_tool_result'
  text?: string
  thinking?: string
  signature?: string
  data?: string
  source?: { type: 'base64'; media_type: string; data: string } | { type: 'url'; url: string } | ClaudeDocumentSource
  id?: string
  name?: string
  input?: unknown
  tool_use_id?: string
  content?: string | ClaudeContentBlock[] | ClaudeWebSearchResultBlock[] | ClaudeWebSearchToolResultError
  cache_control?: ClaudeCacheControl
  // server_tool_use / web_search 引用（Anthropic 原生 web_search 回包字段）
  citations?: ClaudeCitation[]
}

// web_search_tool_result.content 中的单条搜索结果
export interface ClaudeWebSearchResultBlock {
  type: 'web_search_result'
  url: string
  title: string
  encrypted_content?: string
  page_age?: string
}

// web_search_tool_result 出错时的 content 形态
export interface ClaudeWebSearchToolResultError {
  type: 'web_search_tool_result_error'
  error_code: string
}

// text block 上的 web_search 引用
export interface ClaudeCitation {
  type: 'web_search_result_location'
  url: string
  title: string
  encrypted_index?: string
  cited_text?: string
}

export type ClaudeDocumentSource =
  | { type: 'base64'; media_type: string; data: string }
  | { type: 'text'; media_type?: string; data: string }

export interface ClaudeTool {
  name: string
  description: string
  input_schema: unknown
  cache_control?: ClaudeCacheControl
}

export interface ClaudeCacheControl {
  type: string
}

export interface ClaudeResponse {
  id: string
  type: 'message'
  role: 'assistant'
  content: ClaudeContentBlock[]
  model: string
  stop_reason: 'end_turn' | 'max_tokens' | 'tool_use' | null
  stop_sequence: string | null
  usage: {
    input_tokens: number
    output_tokens: number
    cache_creation_input_tokens?: number
    cache_read_input_tokens?: number
    // Anthropic 原生 web_search 计数：客户端据此显示 "Did N searches"
    server_tool_use?: {
      web_search_requests: number
    }
  }
}

export interface ClaudeStreamEvent {
  type: 'message_start' | 'content_block_start' | 'content_block_delta' | 'content_block_stop' | 'message_delta' | 'message_stop' | 'ping' | 'error'
  message?: Partial<ClaudeResponse>
  index?: number
  content_block?: ClaudeContentBlock
  delta?: { type: string; text?: string; thinking?: string; signature?: string; data?: string; reasoning_content?: string; stop_reason?: string; stop_sequence?: string }
  usage?: { input_tokens?: number; output_tokens: number; cache_creation_input_tokens?: number; cache_read_input_tokens?: number; server_tool_use?: { web_search_requests: number } }
  error?: { type: string; message: string }
}

// ============ Kiro API 格式 ============
export interface KiroPayload {
  conversationState: KiroConversationState
  profileArn?: string
  inferenceConfig?: KiroInferenceConfig
  additionalModelRequestFields?: Record<string, unknown>
}

export interface KiroConversationState {
  agentContinuationId?: string
  agentTaskType?: string
  chatTriggerType: 'MANUAL'
  conversationId: string
  currentMessage: KiroCurrentMessage
  history?: KiroHistoryMessage[]
}

export interface KiroCurrentMessage {
  userInputMessage: KiroUserInputMessage
}

export interface KiroUserInputMessage {
  content: string
  modelId?: string  // 可选，占位消息不需要
  origin: string
  images?: KiroImage[]
  documents?: KiroDocument[]
  cachePoint?: KiroCachePoint
  clientCacheConfig?: unknown
  userInputMessageContext?: KiroUserInputMessageContext
}

export interface KiroImage {
  format: string
  source: { bytes: string }
}

export interface KiroDocument {
  format: string
  name: string
  source: { bytes: string }
}

export interface KiroUserInputMessageContext {
  toolResults?: KiroToolResult[]
  tools?: KiroToolWrapper[]
  editorState?: unknown
  shellState?: unknown
  gitState?: unknown
  envState?: unknown
  additionalContext?: unknown
}

export interface KiroToolResult {
  content: { text: string }[]
  status: 'success' | 'error'
  toolUseId: string
}

export type KiroToolWrapper = {
  toolSpecification: {
    name: string
    description: string
    inputSchema: { json: unknown }
  }
} | {
  cachePoint: KiroCachePoint
}

export interface KiroHistoryMessage {
  userInputMessage?: KiroUserInputMessage
  assistantResponseMessage?: KiroAssistantResponseMessage
}

export interface KiroAssistantResponseMessage {
  content: string
  cachePoint?: KiroCachePoint
  reasoningContent?: KiroReasoningContent
  toolUses?: KiroToolUse[]
}

export interface KiroToolUse {
  toolUseId: string
  name: string
  input: Record<string, unknown>
}

export interface KiroInferenceConfig {
  maxTokens?: number
  temperature?: number
  topP?: number
}

export interface KiroCachePoint {
  type: 'default'
}

export interface KiroReasoningContent {
  reasoningText?: {
    text: string
    signature?: string
  }
  redactedContent?: string
}

export interface KiroRequestContext {
  editorState?: unknown
  shellState?: unknown
  gitState?: unknown
  envState?: unknown
  additionalContext?: unknown
}

export interface KiroUsage {
  inputTokens: number
  outputTokens: number
  credits: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  reasoningTokens?: number
}

// ============ 账号和代理配置 ============
export interface ProxyAccount {
  id: string
  email?: string
  accessToken: string
  refreshToken?: string
  clientId?: string
  clientSecret?: string
  region?: string
  authMethod?: 'social' | 'idc' | 'IdC' | 'external_idp'
  provider?: string
  profileArn?: string
  expiresAt?: number
  machineId?: string  // 账户绑定的设备 ID（64位十六进制）
  /** 账号绑定的出口代理 URL（http/https）；为空则使用全局代理逻辑 */
  proxyUrl?: string
  /** 账号所属分组 ID；与 multiAccountSelectionMode='groups' + multiAccountGroupIds 配合做轮询分组过滤 */
  groupId?: string
  // 运行时状态
  lastUsed?: number
  requestCount?: number
  errorCount?: number
  isAvailable?: boolean
  cooldownUntil?: number
  // 配额追踪
  quotaUsed?: number
  quotaLimit?: number
  quotaExhaustedAt?: number // 配额耗尽时间戳
  quotaResetAt?: number // 下次配额重置时间
  // 长期封禁追踪（区分于临时 errorCount 冷却）
  // Kiro 后端 TEMPORARILY_SUSPENDED / AccountSuspendedException 等风控触发时设置
  // 需要联系 AWS Support 人工解封，账号池会持续跳过直到 clearSuspended
  suspendedAt?: number       // 封禁时间戳
  suspendReason?: string     // 封禁原因 (如 'TEMPORARILY_SUSPENDED')
  suspendMessage?: string    // 封禁完整错误消息 (含联系链接)
}

// API Key 格式类型
export type ApiKeyFormat = 'sk' | 'simple' | 'token'

// API Key 用量记录
export interface ApiKeyUsageRecord {
  timestamp: number
  model: string
  inputTokens: number
  outputTokens: number
  credits: number
  path: string
  /**
   * 归一化后的推理强度档位（none/low/medium/high/xhigh/max）。
   * 来源可能是 OpenAI reasoning_effort、Claude output_config.effort，
   * 或由 thinking.budget_tokens 推导（见 ProxyServer.deriveEffortLevel）。
   * 旧持久化记录无此字段 → 视为 undefined（前端显示 '–'）。
   */
  effort?: string
  /**
   * input 明细：inputTokens 是三者之和（uncached + cacheRead + cacheWrite）。
   * 拆出来便于客户理解「为何 input 很大但扣费很少」——cacheRead 计费远低于普通 input。
   * 旧记录无此字段 → 前端按 uncached = inputTokens 兜底。
   */
  cacheReadTokens?: number
  cacheWriteTokens?: number
  /**
   * 实际向客户扣减的 credit（= credits × 当时的 modelMarkup）。
   * credits 字段存的是 Kiro 原始计量值；当 markup≠1 时两者不同。
   * 存这个是为了让 session/历史成本视图与真实 creditBalance 扣减对账一致，
   * 且事后调整 modelMarkup 不会改写历史显示成本。旧记录无此字段 → 前端回退到 credits。
   */
  effectiveCredits?: number
  /** 记录该 request 时使用的 modelMarkup 倍率（用于审计 / 重算）。旧记录无 → 视为 1。 */
  markupAtTime?: number
  /**
   * 会话分组键（= kiroPayload.conversationState.conversationId，已含 API Key hash 前缀隔离）。
   * 同一会话的多轮 request 共享同一值，用于把 usageHistory 分组成 session 视图（MaxPlus 风格）。
   * 旧记录无此字段 → 前端按「单条 = 单 session」或按时间分桶兜底。
   */
  sessionId?: string
}

// API Key 类型
export interface ApiKey {
  id: string
  name: string
  key: string
  format: ApiKeyFormat  // 密钥格式
  enabled: boolean
  createdAt: number
  lastUsedAt?: number
  /**
   * 所属客户 ID（客户门户场景）。
   * 设置后该 Key 的扣费会从对应 Customer.creditBalance 预付余额中扣减，
   * 余额 <= 0 时拒绝请求。未设置 = 旧式独立 Key，仅受 creditsLimit 约束。
   */
  customerId?: string
  // 额度限制
  creditsLimit?: number  // Credits 上限（undefined 表示无限制）
  // 用量统计
  usage: {
    totalRequests: number
    totalCredits: number
    totalInputTokens: number
    totalOutputTokens: number
    // input 明细累计（totalInputTokens 含这两部分）；旧 key 缺省 → 前端按 0 兜底
    totalCacheReadTokens?: number
    totalCacheWriteTokens?: number
    // 按日期统计（YYYY-MM-DD -> usage）
    daily: Record<string, {
      requests: number
      credits: number
      inputTokens: number
      outputTokens: number
    }>
    // 按模型统计
    byModel?: Record<string, {
      requests: number
      credits: number
      inputTokens: number
      outputTokens: number
    }>
    // 按推理强度档位统计（none/low/medium/high/xhigh/max）
    byEffort?: Record<string, {
      requests: number
      credits: number
      inputTokens: number
      outputTokens: number
    }>
  }
  // 用量历史记录（最近 100 条）
  usageHistory?: ApiKeyUsageRecord[]
}

// 客户（门户登录用户）。每个客户可自助创建多个 API Key，
// 共用一份预付 credit 余额（creditBalance），按 Kiro 实际计费的 credit 扣减。
export interface Customer {
  id: string
  email: string             // 登录账号（唯一，比较时小写）
  name?: string             // 显示名
  // 密码使用 scrypt 派生，存 salt + hash（均为 hex），不存明文。
  // Google 登录的客户可能没有密码（passwordless）→ 两字段为 optional。
  passwordSalt?: string
  passwordHash?: string
  /** 绑定的 Google 账号 sub（首次 Google 登录时写入）。设置后该客户可用 Google 登录。 */
  googleSub?: string
  enabled: boolean
  createdAt: number
  lastLoginAt?: number
  /** 预付 credit 余额；每次请求按 Kiro 实际计费扣减，<= 0 时该客户所有 Key 被拒 */
  creditBalance: number
  /** 累计已充值 credit（仅统计用，便于对账） */
  totalToppedUp?: number
  /** 客户自助创建 Key 的数量上限（undefined = 用全局默认） */
  maxKeys?: number
  /**
   * 充值流水（人工充值/扣减记录），便于对账。
   * by: 'admin' = 后台人工；'slip' = 客户上传转账slip经 slip2go 验证后自动入账。
   * transRef: 仅 slip 来源有值——银行端唯一交易号，用于对账与防重复入账。
   */
  topupHistory?: Array<{ timestamp: number; amount: number; note?: string; by?: 'admin' | 'slip'; transRef?: string }>
}

/**
 * 转账slip自动充值记录（slip2go 验证）。
 * 每条对应一次「客户提交slip → 验证 → （成功入账 / 拒绝）」，便于客户查询与后台对账。
 * 持久化在 ProxyConfig.slipTopupRecords（capped）；服务端据 status==='settled' 的 transRef 重建去重集合。
 */
export interface SlipTopupRecord {
  id: string                 // uuid
  transRef: string           // 银行端唯一交易号——去重主键（同一笔真实转账只入账一次）
  referenceId: string        // slip2go 每次验证的 UUID（可用于 GET 复查，不作去重键）
  customerId: string
  bahtAmount: number         // slip2go 返回的转账金额（THB），唯一可信来源
  creditsAdded: number       // 实际入账 credit（settled 时 > 0；拒绝时 0）
  bahtPerCreditAtTime: number // 入账时锁定的换算汇率，便于审计
  code: number               // slip2go 结果码（200200 / 200401 / 200501 ...）
  status: 'settled' | 'rejected'
  rejectReason?: string      // status==='rejected' 时的原因（不回传 apiSecret/内部细节）
  receiverAccount?: string   // slip 上的收款账号（部分脱敏），核对用
  senderName?: string        // 付款人姓名（部分脱敏），核对用
  slipDateTime?: string      // slip2go data.dateTime（ISO/GMT）
  verifiedAt: number         // 本地处理时间戳
}

/**
 * 转账slip自动充值配置。仅本地 IPC 可写（含 apiSecret），
 * 绝不进 filterAdminConfigUpdate 白名单，也不可经 HTTP /portal/* 读出。
 */
export interface SlipTopupConfig {
  /** 总开关。false（默认）= 不暴露 /portal/topup/slip 端点。 */
  enabled: boolean
  /** slip2go API Secret（Bearer）。仅主进程持有，不序列化到任何 HTTP 响应/HTML/日志。 */
  apiSecret: string
  /**
   * 我方收款账号白名单——传给 slip2go checkReceiver，且服务端二次核对 data.receiver。
   * 任一条匹配即视为收款人正确（slip2go 语义：matched only 1 condition = valid）。
   */
  receiverAccounts: Array<{
    accountType?: string      // slip2go account type code，如 "01004"=กสิกร、"02001"=PromptPay 手机号
    accountNumber?: string    // 账号/手机号/citizenID（部分匹配；本地核对存后缀）
    accountNameTH?: string
    accountNameEN?: string
  }>
  /** 单笔最低入账金额（THB）；低于则拒绝（避免微额slip耗 slip2go 配额）。默认 1。 */
  minAmountThb?: number
  /** 单笔最高入账金额（THB）；高于则拒绝（防异常大额误入账，转人工）。0/未设 = 不限制。 */
  maxAmountThb?: number
  /** slip 有效期（小时）：data.dateTime 超过此时长则拒绝，防囤旧slip。默认 48。 */
  freshnessHours?: number
  /** 单客户每日最多提交次数（防刷耗配额）。默认 20，0 = 不限制。 */
  dailyMaxSubmitsPerCustomer?: number
  /** 单客户每分钟最多提交次数。默认 5，0 = 不限制。 */
  perMinuteMaxSubmitsPerCustomer?: number
}

/**
 * 门户邀请（invite-only 注册）。管理员生成 code 发给客户，
 * 客户用 Google 登录时携带 code 完成首次注册。
 * 安全：code 绑定 email —— 必须与 Google 账号的 email 一致才放行。
 */
export interface PortalInvite {
  code: string              // 随机不可猜测的邀请码
  email: string             // 绑定的客户 email（小写，必须与 Google 账号一致）
  name?: string             // 预设显示名
  creditBalance: number     // 注册后初始 credit 余额
  maxKeys?: number          // 该客户 Key 上限（undefined = 用全局默认）
  createdAt: number
  expiresAt?: number        // 过期时间戳（undefined = 不过期）
  usedAt?: number           // 已使用时间戳（undefined = 未使用）
  usedByCustomerId?: string // 注册后生成的 customer id
}

/**
 * 客户脱敏视图：用于管理端（HTTP /admin/customers 与 IPC）返回给前端，
 * 绝不包含 passwordSalt / passwordHash。附带名下 Key 数量与 maxKeys 上限。
 */
export interface CustomerView {
  id: string
  email: string
  name?: string
  enabled: boolean
  createdAt: number
  lastLoginAt?: number
  creditBalance: number
  totalToppedUp: number
  keyCount: number
  maxKeys: number
}

// 模型映射规则
export interface ModelMappingRule {
  id: string
  name: string  // 规则名称
  enabled: boolean
  // 映射类型：replace(替换), alias(别名), loadbalance(负载均衡)
  type: 'replace' | 'alias' | 'loadbalance'
  // 源模型（用户请求的模型名，支持通配符 *）
  sourceModel: string
  // 目标模型列表（负载均衡时随机选择）
  targetModels: string[]
  // 负载均衡权重（可选，默认平均）
  weights?: number[]
  // 优先级（数字越小优先级越高）
  priority: number
  // 适用的 API Key ID 列表（空表示全局）
  apiKeyIds?: string[]
}

export interface ProxyConfig {
  enabled: boolean
  port: number
  host: string
  /** 数据目录（cert/log 等）。headless 模式下显式指定；Electron 下留空走 userData。 */
  dataDir?: string
  apiKey?: string  // 保留兼容性
  apiKeys?: ApiKey[]  // 多 API Key 支持
  enableMultiAccount: boolean
  selectedAccountIds: string[]
  logRequests: boolean
  logStreamEvents?: boolean
  maxConcurrent: number
  // 重试配置
  maxRetries?: number
  retryDelayMs?: number
  // 首选端点配置
  // kiro-runtime: runtime.kiro.dev（官方 kiro-cli 当前主端点）；其余为旧 amazonaws.com 端点
  preferredEndpoint?: 'codewhisperer' | 'amazonq' | 'amazonq-cli' | 'kiro-runtime'
  // Token 刷新提前量（秒）
  tokenRefreshBeforeExpiry?: number
  // TLS/HTTPS 配置
  tls?: TlsConfig
  // 自动启动
  autoStart?: boolean
  clientDrivenToolExecution?: boolean
  // 禁用工具调用（移除 tools 参数）
  disableTools?: boolean
  // Payload 大小限制（KB），超过时截断工具结果（byte 维度）
  payloadSizeLimitKB?: number
  // Token buffer reserve 开关（默认 false = 完全跳过 trimHistoryByTokens）
  // 关闭时后端不再裁剪任何旧消息，超出 context window 由 Kiro 后端原样返回错误
  enableTokenBufferReserve?: boolean
  // Token buffer reserve（仅在 enableTokenBufferReserve=true 时生效）
  // effective limit = model.maxInputTokens - buffer
  // 默认 20K：覆盖 system + tools + current message + output + 估算偏差
  tokenBufferReserve?: number
  // 单账号模式下额度耗尽自动切换到下一个账号
  autoSwitchOnQuotaExhausted?: boolean
  // 多账号选择策略 (仅 enableMultiAccount=true 时生效)
  // - round-robin: 每次请求成功后切到下一个账号 (默认, 负载均衡)
  // - sticky: 一个账号成功就粘住, 直到失败才切换 (保留 prompt cache, 牺牲均衡)
  accountSelectionStrategy?: 'round-robin' | 'sticky'
  // 多账号轮询范围 (仅 enableMultiAccount=true 时生效)
  // - 'all': 使用所有 active 账号（默认）
  // - 'groups': 仅使用 multiAccountGroupIds 选中分组的账号；可包含特殊值 '__ungrouped__' 表示未分组账号
  multiAccountSelectionMode?: 'all' | 'groups'
  multiAccountGroupIds?: string[]
  // 模型映射规则
  modelMappings?: ModelMappingRule[]
  /**
   * 对外开放的模型 ID 白名单（精确匹配 model ID，大小写不敏感）。
   * 未设置或空数组 = 开放全部模型（向后兼容，零行为变更）。
   * 设置后：/v1/models 与门户费率表只列出白名单内的模型；
   * /v1/messages、/v1/chat/completions、/v1/responses、Gemini 等请求若指定
   * 白名单外的模型，直接返回 403 permission_error（既隐藏也拦截）。
   * 注意：比对的是「经 modelMappings 映射后」的最终模型 ID。
   */
  allowedModels?: string[]

  // 服务端 web 工具（web_search / web_fetch）配置
  // 启用后，代理会拦截模型发起的 web_search/web_fetch 调用并在代理侧执行（经第三方搜索 API），
  // 把结果喂回对话。未配置 apiKey 或 enabled=false 时，web 工具会被丢弃（避免 Bedrock 400）。
  webSearch?: {
    enabled: boolean
    provider: 'tavily'
    apiKey: string
    /** web 工具循环的最大轮数，防止无限循环（默认 5） */
    maxRounds?: number
  }

  // ============ 安全 / 限流 / 可观测（v1.8 新增） ============
  /** 入站请求体最大字节数（默认 10MB）。超过返回 413 */
  maxRequestBodyBytes?: number
  /** 允许访问的客户端 IP 列表（CIDR 或单 IP）；空数组或未设 = 不限制 */
  allowedIPs?: string[]
  /** 拒绝访问的客户端 IP 列表（CIDR 或单 IP）；优先级高于 allowedIPs */
  deniedIPs?: string[]
  /** 当绑定 host 是 0.0.0.0/外网接口时，是否允许无 API Key 启动（默认 false 拒绝） */
  allowExternalWithoutApiKey?: boolean
  /**
   * 是否对外暴露 /admin/* HTTP 管理接口（默认 false = 关闭，返回 404）。
   * 应用自身的管理界面走 Electron IPC，不依赖此 HTTP 接口；当反代经公共 tunnel 暴露时，
   * /admin/* 默认开放会让运营方管理面（充值/删客户/改配置）暴露在公网，仅靠 operator key 一道防线。
   * 仅在确有外部脚本/工具需要远程管理时显式开启，并务必配合强 operator key + tunnel access policy。
   */
  adminApiExposed?: boolean
  /**
   * /portal/* 与 /admin/* 的 CORS 允许来源白名单（精确 origin，如 "https://kiro.example.com"）。
   * 留空 = 不对这两条路径发送 Access-Control-Allow-Origin（浏览器跨站脚本无法读取响应）。
   * 注意：LLM 代理路径（/v1/*）不受此限制，仍发 "*" 以兼容各类客户端 SDK。
   */
  portalAllowedOrigins?: string[]
  /** 按 API Key（或匿名时按 IP）的请求频率限制：每分钟最大请求数。0=不限制 */
  rateLimitPerKeyPerMinute?: number
  /** 客户端会话粘性：true 时同一 session hint 总路由到同一账号子集 */
  sessionAffinityEnabled?: boolean
  /** keep-alive 连接空闲超时（毫秒），默认 65s */
  keepAliveTimeoutMs?: number
  /** request headers 接收超时（毫秒），默认 60s */
  headersTimeoutMs?: number
  /** recentRequests 保留条数（默认 100，最多 10000） */
  recentRequestsLimit?: number
  /** 是否暴露 /metrics（Prometheus 文本格式） */
  enableMetrics?: boolean
  /**
   * P2-21 API Key 与账号的精细绑定：apiKey id → 允许使用的账号 ID 数组（白名单）
   * 未配置或空数组 = 该 API Key 可使用所有账号；
   * 兼容旧名 apiKeyGroupBindings（按 group 绑定，需配合 group 同步）
   */
  apiKeyAccountBindings?: Record<string, string[]>
  /** @deprecated 改用 apiKeyAccountBindings；保留以兼容老配置 */
  apiKeyGroupBindings?: Record<string, string[]>
  /** HTTP + HTTPS 双端口：启用 TLS 时，仍同时监听 HTTP 端口在 fallbackPort */
  fallbackPort?: number
  /** 启用审计日志（管理 API 操作、config 变更） */
  enableAuditLog?: boolean
  /**
   * 可用账号数低于此阈值时触发 proxy-pool-low webhook 预警（提醒补充账号）。
   * 0 或未设 = 关闭预警。用于"账号会被风控、需持续补充"的运营场景：
   * 在池耗尽（503）之前就提前告警，给补号留出时间。
   */
  poolLowThreshold?: number

  // ============ 客户门户（v1.9 新增） ============
  /** 客户列表（门户登录账号 + 预付 credit 余额） */
  customers?: Customer[]
  /** 启用 /portal/* 客户门户端点（登录、自助管理 Key、查看用量/余额） */
  portalEnabled?: boolean
  /**
   * 门户会话签名密钥（HMAC）。首次启用门户时由主进程自动生成并持久化；
   * 轮换此值会使所有已签发的登录会话立即失效。
   */
  portalSessionSecret?: string
  /** 门户会话有效期（小时），默认 24 */
  portalSessionTtlHours?: number
  /** 客户默认可创建的 API Key 数量上限（默认 5），可被 Customer.maxKeys 覆盖 */
  portalDefaultMaxKeys?: number
  /**
   * 单客户并发在途请求上限（默认 6，0 = 不限制）。
   * 预付 credit 在请求结束后才扣减，并发请求会同时通过"余额>0"校验；
   * 此上限把"接近耗尽时的超额消费"约束在 N 个请求以内，而不会无上限透支。
   */
  portalMaxConcurrentPerCustomer?: number

  /** 门户邀请列表（invite-only 注册用） */
  portalInvites?: PortalInvite[]
  /** 启用 Google 登录（"Sign in with Google" 按钮）；需同时设置 googleClientId */
  portalGoogleEnabled?: boolean
  /** Google OAuth Web Client ID（用于前端 Google Identity 按钮 + 后端校验 ID token 的 aud） */
  googleClientId?: string

  // ============ 计费定价（v1.10 新增，转售加价层） ============
  /**
   * 转售定价配置。creditBalance 仍以 credit 为单位（与 Kiro 计费口径一致），
   * 此处叠加"加价/换算层"：把 Kiro 实扣的 credit 换算成对客户的售价（泰铢），
   * 并支持按模型设定加价倍率。enabled !== true 时一切按原样运行（零行为变更）。
   */
  pricing?: PricingConfig

  // ============ 转账slip自动充值（slip2go） ============
  /**
   * slip2go 验证配置（含 apiSecret）。仅本地 IPC 可写，不进 admin config 白名单。
   * 未设或 enabled !== true 时 /portal/topup/slip 返回 404（功能不存在）。
   */
  slipTopup?: SlipTopupConfig
  /** slip 自动充值流水（capped，最近 N 条）。服务端据此重建 transRef 去重集合。 */
  slipTopupRecords?: SlipTopupRecord[]
}

export interface PricingConfig {
  /** 计费层总开关。未开启（默认）= 完全沿用旧逻辑：扣减原始 credit、门户按 credit 显示。 */
  enabled?: boolean
  /** 对客户的售价：每 credit 多少泰铢（用于充值换算与门户金额显示）。默认 0.47。 */
  bahtPerCredit?: number
  /** 你的成本：每 credit 多少泰铢（仅用于后台毛利显示，不影响扣费）。默认 0.11（1100฿/10000 credits, Kiro Power）。 */
  costPerCredit?: number
  /** 美元兑泰铢汇率（用于把 Kiro 官方美元价换算成泰铢做对比）。默认 36。 */
  usdToBaht?: number
  /** 支付网关手续费百分比（用于后台净利预估，不影响扣费）。默认 0。 */
  gatewayFeePct?: number
  /** Kiro 官方零售单价（美元/credit），用于门户"比官方省 X%"对比。默认 0.02（$200/10000）。 */
  kiroRetailUsdPerCredit?: number
  /**
   * 按模型加价倍率：实扣 credit = Kiro 原始 credit × 此倍率（缺省/未配置 = 1.0）。
   * 仅在 enabled === true 时生效；倍率 <= 0 视为 1.0。
   */
  modelMarkup?: Record<string, number>
}

export interface TlsConfig {
  enabled: boolean
  certPath?: string // 证书文件路径
  keyPath?: string // 私钥文件路径
  // 或直接提供 PEM 内容
  cert?: string
  key?: string
}

// Token 刷新回调类型
export type TokenRefreshCallback = (account: ProxyAccount) => Promise<{
  success: boolean
  accessToken?: string
  refreshToken?: string
  expiresAt?: number
  error?: string
}>

export interface ProxyStats {
  totalRequests: number
  successRequests: number
  failedRequests: number
  totalTokens: number
  totalCredits: number // 累计总 credits（所有请求）
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  reasoningTokens: number
  startTime: number
  accountStats: Map<string, AccountStats>
  // 按端点统计
  endpointStats: Map<string, EndpointStats>
  // 按模型统计
  modelStats: Map<string, ModelStats>
  // 最近请求日志
  recentRequests: RequestLog[]
}

export interface AccountStats {
  requests: number
  tokens: number
  inputTokens: number
  outputTokens: number
  errors: number
  lastUsed: number
  avgResponseTime: number
  totalResponseTime: number
}

export interface EndpointStats {
  name: string
  requests: number
  successes: number
  failures: number
  quotaErrors: number
}

export interface ModelStats {
  model: string
  requests: number
  tokens: number
}

export interface RequestLog {
  timestamp: number
  path: string
  model: string
  accountId: string
  inputTokens: number
  outputTokens: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  reasoningTokens?: number
  credits?: number // Kiro API 返回的 credit 使用量
  responseTime: number
  success: boolean
  error?: string
}

// ============ Event Stream 解析 ============
export interface KiroEventStreamMessage {
  type: string
  payload: unknown
}

export interface KiroAssistantResponseEvent {
  content?: string
  toolUse?: KiroToolUse
}

export interface KiroUsageEvent {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
}
