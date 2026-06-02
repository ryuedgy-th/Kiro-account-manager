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
import { callKiroApiStream, callKiroApi, runWebToolLoop, fetchKiroModels, setModelContextWindow, type KiroModel, type WebToolSearchRecord } from './kiroApi'
import { proxyLogger } from './logger'
import { getKProxyService, generateDeviceId } from '../kproxy'
import {
  openaiToKiro,
  claudeToKiro,
  kiroToOpenaiResponse,
  kiroToClaudeResponse,
  createOpenaiStreamChunk,
  createClaudeStreamEvent,
  responsesToOpenAIChat,
  openAIChatToResponsesResponse
} from './translator'
import { ToolNameRegistry } from './toolNameRegistry'
import { promptCacheTracker } from './promptCacheTracker'
import { isServerWebTool, type WebToolConfig } from './webTools'
import * as portal from './portal'
import type { Customer } from './types'


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
  onResponse?: (info: { path: string; model?: string; status: number; tokens?: number; inputTokens?: number; outputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number; reasoningTokens?: number; credits?: number; responseTime?: number; error?: string }) => void
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

function modelDisplayName(id: string, modelName?: string): string {
  if (modelName?.trim()) return modelName
  return id
    .split('-')
    .filter(Boolean)
    .map(part => part === 'gpt' ? 'GPT' : part === 'ai' ? 'AI' : part[0]?.toUpperCase() + part.slice(1))
    .join(' ')
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
  private refreshingTokens: Set<string> = new Set() // 防止并发刷新
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
  /** P1-8 会话粘性：session hint → accountId 的映射（10 分钟 TTL） */
  private sessionAffinity: Map<string, { accountId: string; lastAt: number }> = new Map()
  /** P2-17 审计日志（最近 200 条） */
  private auditLog: Array<{ ts: number; type: string; data: Record<string, unknown> }> = []
  /** Webhook 触发回调（由外部注入，避免 main → renderer 循环依赖） */
  private webhookTrigger?: (event: string, payload: Record<string, unknown>) => void
  /** 定期清理 timer */
  private cleanupTimer: NodeJS.Timeout | null = null

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
      })

      // D4 启用 TLS 时同时监听 HTTP fallback 端口（如果配置了 fallbackPort）
      if (this.isHttps && this.config.fallbackPort && this.config.fallbackPort !== this.config.port) {
        const fallback = http.createServer(requestHandler)
        fallback.keepAliveTimeout = keepAliveMs
        fallback.headersTimeout = Math.max(headersMs, keepAliveMs + 1000)
        fallback.requestTimeout = 0
        fallback.on('connection', (socket) => {
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
      // 自动生成自签证书（位于 userData/proxy-tls/）
      try {
        const { app } = require('electron')
        const { ensureProxySelfSignedCert } = require('./selfSignedCert')
        const hostnames = [this.config.host || '127.0.0.1']
        const result = ensureProxySelfSignedCert(app.getPath('userData'), hostnames)
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
   * 获取（或生成）反代自签证书信息（供 UI 显示/导出 PEM）
   */
  getSelfSignedCertInfo(): import('./selfSignedCert').ProxySelfSignedCert | null {
    try {
      const { app } = require('electron')
      const { ensureProxySelfSignedCert } = require('./selfSignedCert')
      return ensureProxySelfSignedCert(app.getPath('userData'), [this.config.host || '127.0.0.1'])
    } catch (err) {
      proxyLogger.warn('ProxyServer', `getSelfSignedCertInfo failed: ${(err as Error).message}`)
      return null
    }
  }

  /** 强制重新生成自签证书（用户在 UI 上点"重新生成"） */
  regenerateSelfSignedCert(): import('./selfSignedCert').ProxySelfSignedCert | null {
    try {
      const { app } = require('electron')
      const { ensureProxySelfSignedCert } = require('./selfSignedCert')
      this.appendAuditLog('regenerate_self_signed_cert', { host: this.config.host })
      return ensureProxySelfSignedCert(app.getPath('userData'), [this.config.host || '127.0.0.1'], true)
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

  private async downloadImageDataUrl(url: string, signal?: AbortSignal): Promise<string> {
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
        throw new Error(`Failed to download image: HTTP ${response.status}`)
      }
      const contentType = response.headers.get('content-type')?.split(';')[0]?.toLowerCase()
      if (!contentType || !['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(contentType)) {
        throw new Error(`Unsupported image content-type: ${contentType || 'unknown'}`)
      }
      const arrayBuffer = await response.arrayBuffer()
      if (arrayBuffer.byteLength > 10 * 1024 * 1024) {
        throw new Error('Image exceeds 10MB limit')
      }
      return `data:${contentType};base64,${Buffer.from(arrayBuffer).toString('base64')}`
    } finally {
      clearTimeout(timeout)
      signal?.removeEventListener('abort', abort)
    }
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
        if (block.type !== 'image' || block.source?.type !== 'url') return
        const dataUrl = await this.downloadImageDataUrl(block.source.url, signal)
        const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/)
        if (!match) {
          throw new Error('Downloaded image produced invalid data URL')
        }
        block.source = { type: 'base64', media_type: match[1], data: match[2] }
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
        kiroModels = await fetchKiroModels(account, signal)
        if (kiroModels.length > 0) {
          this.modelCache = { models: kiroModels, timestamp: now }
          // 同步到 kiroApi 的 ctx cache, 供 token 裁剪逻辑使用
          for (const m of kiroModels) {
            if (m.tokenLimits?.maxInputTokens) {
              setModelContextWindow(m.modelId, m.tokenLimits.maxInputTokens)
            }
          }
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

  // 检查 Token 是否需要刷新
  private isTokenExpiringSoon(account: ProxyAccount): boolean {
    if (!account.expiresAt) return false
    const refreshBeforeMs = (this.config.tokenRefreshBeforeExpiry || 300) * 1000
    return Date.now() + refreshBeforeMs >= account.expiresAt
  }

  // 刷新 Token
  private async refreshToken(account: ProxyAccount, signal?: AbortSignal): Promise<boolean> {
    this.throwIfAborted(signal)
    if (!this.events.onTokenRefresh) {
      console.warn('[ProxyServer] No token refresh callback configured')
      return false
    }

    // 防止并发刷新
    if (this.refreshingTokens.has(account.id)) {
      console.log(`[ProxyServer] Token refresh already in progress for ${account.email || account.id}`)
      // 等待刷新完成
      await this.waitForRetry(1000, signal)
      return !this.isTokenExpiringSoon(this.accountPool.getAccount(account.id) || account)
    }

    this.refreshingTokens.add(account.id)
    console.log(`[ProxyServer] Refreshing token for ${account.email || account.id}`)

    try {
      // 随机延迟 0-3 秒，避免多账号同时刷新被识别为批量操作
      const jitter = Math.floor(Math.random() * 3000)
      if (jitter > 0) await this.waitForRetry(jitter, signal)
      
      const result = await this.abortable(this.events.onTokenRefresh(account), signal)
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
    } finally {
      this.refreshingTokens.delete(account.id)
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
        // 仍需检查 token 是否需要刷新
        if (this.isTokenExpiringSoon(sticky)) {
          const refreshed = await this.refreshToken(sticky, signal)
          if (refreshed) {
            return this.accountPool.getAccount(sticky.id) || sticky
          }
        } else {
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
    this.syncKProxyDeviceId(account)

    // 检查是否需要刷新 Token
    if (this.isTokenExpiringSoon(account)) {
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
    if (!hasApiKeys && !hasLegacyKey) return true // 未配置任何 Key：沿用旧行为（开放）

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
  recordApiKeyUsage(apiKeyId: string, credits: number, inputTokens: number, outputTokens: number, model?: string, path?: string): void {
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

    // 添加用量历史记录（保留最近 100 条）
    if (!apiKey.usageHistory) {
      apiKey.usageHistory = []
    }
    apiKey.usageHistory.unshift({
      timestamp: now,
      model: model || 'unknown',
      inputTokens,
      outputTokens,
      credits,
      path: path || 'unknown'
    })
    if (apiKey.usageHistory.length > 100) {
      apiKey.usageHistory = apiKey.usageHistory.slice(0, 100)
    }

    // 客户门户预付扣费：Key 归属某客户时，从其 creditBalance 扣减本次消耗的 credit。
    // 允许扣成负数（最后一次请求可能略微超额），下次请求 validateApiKey 会因 <=0 拒绝。
    if (apiKey.customerId && credits > 0 && this.config.customers) {
      const customer = this.config.customers.find(c => c.id === apiKey.customerId)
      if (customer) {
        customer.creditBalance -= credits
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
    usage: { credits?: number; inputTokens: number; outputTokens: number },
    model: string | undefined,
    path: string
  ): void {
    const credits = usage.credits || 0
    if (credits <= 0 && usage.inputTokens <= 0 && usage.outputTokens <= 0) return
    this.accountPool.recordSuccess(accountId, usage.inputTokens + usage.outputTokens)
    if (matchedApiKey) {
      this.recordApiKeyUsage(matchedApiKey.id, credits, usage.inputTokens, usage.outputTokens, model, path)
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
      this.setCorsHeaders(res)
      res.writeHead(204)
      res.end()
      req.off('aborted', abortRequest)
      res.off('close', abortRequest)
      this.activeRequests.delete(controller)
      return
    }

    try {
      this.setCorsHeaders(res)

      // P0-4 IP 访问控制（健康检查也走，防止扫描器）
      const ipCheck = this.isClientIPAllowed(clientIP)
      if (!ipCheck.allowed) {
        proxyLogger.warn('ProxyServer', `Blocked request from ${clientIP}: ${ipCheck.reason}`)
        this.appendAuditLog('ip_blocked', { ip: clientIP, path, reason: ipCheck.reason })
        this.sendError(res, 403, 'Forbidden')
        return
      }

      // API Key 验证（健康检查端点除外）
      if (path !== '/health' && path !== '/' && !path.startsWith('/portal')) {
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
      } else if (pathWithoutQuery.startsWith('/admin/')) {
        // 管理 API 端点
        await this.handleAdminApi(req, res, pathWithoutQuery, controller.signal)
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
        this.sendError(res, 413, `Request body too large (max ${error.limit} bytes)`,
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
      // 清除内存缓存（conversationId 映射、模型缓存、prompt cache）
      const { clearAllCaches } = require('./kiroApi')
      const cleared = clearAllCaches()
      const promptCacheCleared = promptCacheTracker.clear()
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
    } else {
      this.sendError(res, 404, 'Admin endpoint not found')
    }
  }

  // ============ 管理 API - 客户管理 ============

  /** 客户列表脱敏视图（不含密码哈希/salt，附带名下 Key 数量） */
  private handleAdminListCustomers(res: http.ServerResponse): void {
    const customers = (this.config.customers || []).map(c => ({
      id: c.id,
      email: c.email,
      name: c.name,
      enabled: c.enabled,
      createdAt: c.createdAt,
      lastLoginAt: c.lastLoginAt,
      creditBalance: c.creditBalance,
      totalToppedUp: c.totalToppedUp || 0,
      keyCount: portal.customerKeys(this.config, c.id).length,
      maxKeys: portal.maxKeysFor(this.config, c)
    }))
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ customers }))
  }

  private async handleAdminCreateCustomer(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { email?: string; password?: string; name?: string; creditBalance?: number; maxKeys?: number }
    try { parsed = JSON.parse(body) } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    const email = portal.normalizeEmail(parsed.email || '')
    if (!portal.isValidEmail(email)) {
      this.sendError(res, 400, 'Invalid email')
      return
    }
    if (!portal.isStrongEnoughPassword(parsed.password || '')) {
      this.sendError(res, 400, 'Password too short (min 8 chars)')
      return
    }
    if (portal.findCustomerByEmail(this.config, email)) {
      this.sendError(res, 409, 'Email already exists')
      return
    }
    const now = Date.now()
    const customer = await portal.buildCustomer(email, parsed.password!, {
      name: parsed.name,
      creditBalance: typeof parsed.creditBalance === 'number' ? parsed.creditBalance : 0,
      maxKeys: parsed.maxKeys
    }, now)
    if (!this.config.customers) this.config.customers = []
    this.config.customers.push(customer)
    this.appendAuditLog('customer_created', { id: customer.id, email: customer.email })
    this.events.onConfigChanged?.(this.config)
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ success: true, customer: { id: customer.id, email: customer.email, creditBalance: customer.creditBalance } }))
  }

  private async handleAdminTopupCustomer(req: http.IncomingMessage, res: http.ServerResponse, customerId: string, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    let parsed: { amount?: number; note?: string }
    try { parsed = JSON.parse(body) } catch {
      this.sendError(res, 400, 'Invalid JSON body')
      return
    }
    const amount = Number(parsed.amount)
    if (!Number.isFinite(amount) || amount === 0) {
      this.sendError(res, 400, 'amount must be a non-zero number')
      return
    }
    const customer = portal.findCustomerById(this.config, customerId)
    if (!customer) {
      this.sendError(res, 404, 'Customer not found')
      return
    }
    customer.creditBalance += amount
    if (amount > 0) customer.totalToppedUp = (customer.totalToppedUp || 0) + amount
    if (!customer.topupHistory) customer.topupHistory = []
    customer.topupHistory.unshift({ timestamp: Date.now(), amount, note: parsed.note, by: 'admin' })
    if (customer.topupHistory.length > 100) customer.topupHistory = customer.topupHistory.slice(0, 100)
    this.appendAuditLog('customer_topup', { id: customer.id, amount })
    this.events.onConfigChanged?.(this.config)
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ success: true, creditBalance: customer.creditBalance }))
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
        keyCount: portal.customerKeys(this.config, customer.id).length
      })
    } else if (path === '/portal/keys' && method === 'GET') {
      this.handlePortalListKeys(res, customer)
    } else if (path === '/portal/keys' && method === 'POST') {
      await this.handlePortalCreateKey(req, res, customer, signal)
    } else if (/^\/portal\/keys\/[^/]+$/.test(path) && method === 'DELETE') {
      let keyId: string
      try { keyId = decodeURIComponent(path.split('/')[3]) } catch {
        this.sendJson(res, 404, { error: 'Key not found' })
        return
      }
      this.handlePortalDeleteKey(res, customer, keyId)
    } else if (path === '/portal/usage' && method === 'GET') {
      this.handlePortalUsage(res, customer)
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
    // 不区分"用户不存在"与"密码错误"，统一返回 401，防止枚举邮箱
    const ok = !!customer && customer.enabled &&
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

  private handlePortalListKeys(res: http.ServerResponse, customer: Customer): void {
    const keys = portal.customerKeys(this.config, customer.id).map(k => ({
      id: k.id,
      name: k.name,
      keyMasked: portal.maskKey(k.key),
      enabled: k.enabled,
      createdAt: k.createdAt,
      lastUsedAt: k.lastUsedAt,
      totalCredits: k.usage.totalCredits,
      totalRequests: k.usage.totalRequests
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
    let parsed: { name?: string } = {}
    try { parsed = body ? JSON.parse(body) : {} } catch {
      this.sendJson(res, 400, { error: 'Invalid JSON body' })
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
    if (!this.config.apiKeys) this.config.apiKeys = []
    this.config.apiKeys.push(newKey)
    this.appendAuditLog('portal_key_created', { customerId: customer.id, keyId: newKey.id })
    this.events.onConfigChanged?.(this.config)
    // 创建时返回一次完整 key（仅此一次），之后列表只给脱敏值
    this.sendJson(res, 200, { id: newKey.id, name: newKey.name, key: newKey.key })
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
    let totalCredits = 0
    let totalRequests = 0
    for (const k of keys) {
      totalCredits += k.usage.totalCredits
      totalRequests += k.usage.totalRequests
      for (const [day, d] of Object.entries(k.usage.daily || {})) {
        if (!dailyAgg[day]) dailyAgg[day] = { requests: 0, credits: 0, inputTokens: 0, outputTokens: 0 }
        dailyAgg[day].requests += d.requests
        dailyAgg[day].credits += d.credits
        dailyAgg[day].inputTokens += d.inputTokens
        dailyAgg[day].outputTokens += d.outputTokens
      }
    }
    this.sendJson(res, 200, {
      creditBalance: customer.creditBalance,
      totalCredits,
      totalRequests,
      daily: dailyAgg
    })
  }

  /** 客户门户静态页面（自包含 HTML，无外部依赖） */
  private handlePortalPage(res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(PORTAL_HTML)
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
      'multiAccountSelectionMode', 'multiAccountGroupIds', 'modelMappings',
      'maxRequestBodyBytes', 'allowedIPs', 'deniedIPs',
      'rateLimitPerKeyPerMinute', 'sessionAffinityEnabled',
      'keepAliveTimeoutMs', 'headersTimeoutMs', 'recentRequestsLimit',
      'enableMetrics', 'apiKeyGroupBindings', 'enableAuditLog', 'poolLowThreshold',
      'portalEnabled', 'portalSessionTtlHours', 'portalDefaultMaxKeys'
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
  private setCorsHeaders(res: http.ServerResponse): void {
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
    usage: { inputTokens: number; outputTokens: number; cacheWriteTokens?: number; cacheReadTokens?: number },
    simulatedCache?: { cacheCreationInputTokens: number; cacheReadInputTokens: number }
  ): { input_tokens?: number; output_tokens: number; cache_creation_input_tokens?: number; cache_read_input_tokens?: number } {
    // 优先使用 Kiro 后端返回的真实 cache tokens，否则用模拟器的值
    const cacheWrite = usage.cacheWriteTokens || simulatedCache?.cacheCreationInputTokens || 0
    const cacheRead = usage.cacheReadTokens || simulatedCache?.cacheReadInputTokens || 0
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
    if (record.type === 'image' || record.type === 'input_image') return 170
    if (record.type === 'document' || record.type === 'input_file') return this.estimateTokenCount(record.title) + this.estimateTokenCount(record.name) + this.estimateTokenCount(record.filename) + this.estimateTokenCount(record.source) + this.estimateTokenCount(record.file_data) + 120
    if (record.type === 'tool_use') return this.estimateTokenCount(record.name) + this.estimateTokenCount(record.input) + 12
    if (record.type === 'tool_result') return this.estimateTokenCount(record.content) + 8
    if (typeof record.role === 'string' && 'content' in record) return this.estimateTokenCount(record.content) + 4
    if (typeof record.name === 'string' && 'input_schema' in record) return this.estimateTokenCount(record.name) + this.estimateTokenCount(record.description) + this.estimateTokenCount(record.input_schema) + 32
    return Object.entries(record).reduce<number>((total, [key, item]) => key === 'cache_control' ? total : total + this.estimateTokenCount(item), 0)
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
    const geminiModels = result.models.map(m => ({
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
    const [, modelId, method] = match
    const isStream = method === 'streamGenerateContent'

    // 将 Gemini 请求转为 OpenAI 格式
    const messages: OpenAIMessage[] = []
    if (geminiReq.systemInstruction?.parts) {
      const sysText = geminiReq.systemInstruction.parts.map((p: { text?: string }) => p.text || '').join('\n')
      if (sysText) messages.push({ role: 'system', content: sysText })
    }
    for (const content of geminiReq.contents || []) {
      const role = content.role === 'model' ? 'assistant' : 'user'
      const text = (content.parts || []).map((p: { text?: string }) => p.text || '').join('')
      if (text) messages.push({ role: role as 'user' | 'assistant', content: text })
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
      max_tokens: geminiReq.generationConfig?.maxOutputTokens
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
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' })
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
                this.settleAbortedUsage(matchedApiKey, account.id, usage, modelId, '/v1beta/models')
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
              resolve()
            },
            (error) => {
              if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
                resolve()
                return
              }
              res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
              res.end()
              this.recordRequestFailed()
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
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({
          candidates: [{ content: { parts: [{ text: result.content }], role: 'model' }, finishReason: 'STOP' }],
          usageMetadata: { promptTokenCount: result.usage.inputTokens, candidatesTokenCount: result.usage.outputTokens, totalTokenCount: result.usage.inputTokens + result.usage.outputTokens }
        }))
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1beta', modelId, startTime, signal)
    }
  }

  // 模型列表缓存
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
          kiroModels = await fetchKiroModels(account, signal)
          if (kiroModels.length > 0) {
            this.modelCache = { models: kiroModels, timestamp: now }
            // 同步到 kiroApi 的 ctx cache, 供 token 裁剪逻辑使用
            for (const m of kiroModels) {
              if (m.tokenLimits?.maxInputTokens) {
                setModelContextWindow(m.modelId, m.tokenLimits.maxInputTokens)
              }
            }
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

    this.throwIfResponseClosed(res, signal)
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ object: 'list', data: allModels }))
  }

  // 处理 OpenAI Chat Completions 请求
  private async handleOpenAIChat(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const request: OpenAIChatRequest = JSON.parse(body)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey

    // 提取 session hint（用于稳定 conversationId），拼入 API Key hash 隔离不同用户
    const rawHintChat = ProxyServer.extractSessionHint(req, request)
    if (!request.conversation_id && rawHintChat) {
      const keyPrefix = matchedApiKey?.id?.slice(0, 8) || 'default'
      request.conversation_id = `${keyPrefix}:${rawHintChat}`
    }
    const affinityHintChat = request.conversation_id

    // 应用模型映射
    request.model = this.applyModelMapping(request.model, matchedApiKey?.id)

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
        await this.handleOpenAIStream(res, account, kiroPayload, request.model, startTime, 0, undefined, false, matchedApiKey, toolNameRegistry, signal)
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
        this.events.onResponse?.({ path: '/v1/chat/completions', model: request.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime })
        this.recordRequest({ path: '/v1/chat/completions', model: request.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        // 记录 API Key 用量
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, request.model, '/v1/chat/completions')
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
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        })
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
        this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime })
        this.recordRequest({ path: '/v1/responses', model: chatRequest.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
        if (matchedApiKey) {
          this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, chatRequest.model, '/v1/responses')
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
      this.events.onResponse?.({ path: '/v1/responses', model: chatRequest.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime })
      this.recordRequest({ path: '/v1/responses', model: chatRequest.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
      if (matchedApiKey) {
        this.recordApiKeyUsage(matchedApiKey.id, result.usage.credits || 0, result.usage.inputTokens, result.usage.outputTokens, chatRequest.model, '/v1/responses')
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
    signal?: AbortSignal
  ): Promise<void> {
    if (!headersSent) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      })
    }

    const id = streamId || `chatcmpl-${uuidv4()}`
    let toolCallIndex = 0
    const pendingToolCalls: Map<string, { index: number; name: string; arguments: string }> = new Map()
    let collectedContent = ''
    // 发送初始 chunk（仅首轮）
    if (currentRound === 0) {
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
            this.settleAbortedUsage(matchedApiKey, account.id, usage, model, '/v1/chat/completions')
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
          this.events.onResponse?.({ path: '/v1/chat/completions', model, status: 200, tokens: usage.inputTokens + usage.outputTokens, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, cacheReadTokens: usage.cacheReadTokens, reasoningTokens: usage.reasoningTokens, credits: usage.credits, responseTime: oaiRespTime })
          this.recordRequest({ path: '/v1/chat/completions', model, accountId: account.id, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, credits: usage.credits, responseTime: oaiRespTime, success: true })
          // 记录 API Key 用量
          if (matchedApiKey) {
            this.recordApiKeyUsage(matchedApiKey.id, usage.credits || 0, usage.inputTokens, usage.outputTokens, model, '/v1/chat/completions')
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
        (error) => {
          if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
            resolve()
            return
          }
          console.error('[ProxyServer] Stream error:', error)
          res.write(`data: ${JSON.stringify({ error: { message: error.message } })}\n\n`)
          res.end()

          this.recordRequestFailed()
          const errStatusCode = error.message.match(/(\d{3})/)?.[1]
          this.accountPool.recordError(account.id, errStatusCode ? classifyError(parseInt(errStatusCode)) : ErrorType.RECOVERABLE, errStatusCode ? parseInt(errStatusCode) : undefined)
          this.events.onResponse?.({ path: '/v1/chat/completions', model, status: 500, error: error.message })
          this.recordRequest({ path: '/v1/chat/completions', model, accountId: account.id, responseTime: Date.now() - startTime, success: false, error: error.message })
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
  }

  // 处理 Claude Messages 请求
  private async handleClaudeMessages(req: http.IncomingMessage, res: http.ServerResponse, signal?: AbortSignal): Promise<void> {
    const body = await this.readBody(req, signal)
    this.throwIfAborted(signal)
    const request: ClaudeRequest = JSON.parse(body)
    const matchedApiKey = (req as unknown as { matchedApiKey?: import('./types').ApiKey }).matchedApiKey

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

      const kiroPayload = claudeToKiro(processedRequest, account.profileArn, toolNameRegistry, useWebTools)

      // 构建 prompt cache profile（用于模拟缓存 usage）
      const estimatedInputTokens = Math.max(1, Math.round(JSON.stringify(kiroPayload).length * 0.3))
      const cacheProfile = promptCacheTracker.buildClaudeProfile(
        processedRequest.system,
        processedRequest.messages,
        processedRequest.tools,
        estimatedInputTokens,
        processedRequest.model
      )
      const cacheUsage = promptCacheTracker.compute(account.id, cacheProfile)

      if (cacheProfile) {
        proxyLogger.info('ProxyServer', `Prompt cache: ${cacheProfile.breakpoints.length} breakpoints, creation=${cacheUsage.cacheCreationInputTokens}, read=${cacheUsage.cacheReadInputTokens}`)
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
          startTime, matchedApiKey, signal,
          cacheProfile && cacheUsage ? { cacheUsage, cacheProfile, accountId: account.id } : undefined
        )
      } else if (request.stream) {
        // 流式响应（流式不使用重试机制，错误由流处理）
        await this.handleClaudeStream(res, account, kiroPayload, request.model, startTime, 0, undefined, false, 0, matchedApiKey, toolNameRegistry, signal,
          cacheProfile ? { ...cacheUsage, cacheProfile, accountId: account.id } : undefined)
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

        // 用缓存模拟的 usage 覆盖（如果有 cache profile）
        if (cacheProfile && cacheUsage) {
          if (cacheUsage.cacheCreationInputTokens > 0) response.usage.cache_creation_input_tokens = cacheUsage.cacheCreationInputTokens
          if (cacheUsage.cacheReadInputTokens > 0) response.usage.cache_read_input_tokens = cacheUsage.cacheReadInputTokens
          promptCacheTracker.update(usedAccount.id, cacheProfile)
        }

        this.throwIfResponseClosed(res, signal)
        this.recordRequestSuccess()
        this.stats.totalTokens += result.usage.inputTokens + result.usage.outputTokens
        this.stats.inputTokens += result.usage.inputTokens
        this.stats.outputTokens += result.usage.outputTokens
        this.accountPool.recordSuccess(usedAccount.id, result.usage.inputTokens + result.usage.outputTokens)

        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(response))
        const respTime = Date.now() - startTime
        this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 200, tokens: result.usage.inputTokens + result.usage.outputTokens, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, cacheReadTokens: result.usage.cacheReadTokens, reasoningTokens: result.usage.reasoningTokens, credits: result.usage.credits, responseTime: respTime })
        this.recordRequest({ path: '/v1/messages', model: request.model, accountId: usedAccount.id, inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, credits: result.usage.credits, responseTime: respTime, success: true })
      }
    } catch (error) {
      this.handleApiError(res, account, error as Error, '/v1/messages', request.model, startTime, signal)
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
    signal: AbortSignal | undefined,
    cache?: { cacheUsage: { cacheCreationInputTokens: number; cacheReadInputTokens: number }; cacheProfile: unknown; accountId: string }
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

    if (cache?.cacheUsage) {
      if (cache.cacheUsage.cacheCreationInputTokens > 0) response.usage.cache_creation_input_tokens = cache.cacheUsage.cacheCreationInputTokens
      if (cache.cacheUsage.cacheReadInputTokens > 0) response.usage.cache_read_input_tokens = cache.cacheUsage.cacheReadInputTokens
      promptCacheTracker.update(cache.accountId, cache.cacheProfile as Parameters<typeof promptCacheTracker.update>[1])
    }

    this.throwIfResponseClosed(res, signal)
    this.recordRequestSuccess()
    this.stats.totalTokens += loop.usage.inputTokens + loop.usage.outputTokens
    this.stats.inputTokens += loop.usage.inputTokens
    this.stats.outputTokens += loop.usage.outputTokens
    this.stats.totalCredits += loop.usage.credits || 0
    this.events.onCreditsUpdate?.(this.stats.totalCredits)
    this.accountPool.recordSuccess(account.id, loop.usage.inputTokens + loop.usage.outputTokens)
    const respTime = Date.now() - startTime
    this.events.onResponse?.({ path: '/v1/messages', model: request.model, status: 200, tokens: loop.usage.inputTokens + loop.usage.outputTokens, inputTokens: loop.usage.inputTokens, outputTokens: loop.usage.outputTokens, credits: loop.usage.credits, responseTime: respTime })
    this.recordRequest({ path: '/v1/messages', model: request.model, accountId: account.id, inputTokens: loop.usage.inputTokens, outputTokens: loop.usage.outputTokens, credits: loop.usage.credits, responseTime: respTime, success: true })
    if (matchedApiKey) {
      this.recordApiKeyUsage(matchedApiKey.id, loop.usage.credits || 0, loop.usage.inputTokens, loop.usage.outputTokens, request.model, '/v1/messages')
    }

    if (!request.stream) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(response))
      return
    }

    // stream=true：把最终 response replay 成 Anthropic SSE
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' })
    const id = response.id || `msg_${uuidv4()}`
    const write = (event: string, data: unknown): void => { res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`) }
    write('message_start', createClaudeStreamEvent('message_start', {
      message: { id, type: 'message', role: 'assistant', content: [], model: request.model, stop_reason: null, stop_sequence: null, usage: { input_tokens: loop.usage.inputTokens, output_tokens: 0 } }
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
    res.end()
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
    simulatedCacheUsage?: { cacheCreationInputTokens: number; cacheReadInputTokens: number; cacheProfile?: unknown; accountId?: string }
  ): Promise<void> {
    if (!headersSent) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      })
    }

    const id = msgId || `msg_${uuidv4()}`
    let currentBlockIndex = contentBlockIndex
    let hasStartedTextBlock = false
    let hasStartedThinkingBlock = false
    let pendingThinkingSignature: string | undefined
    let collectedContent = ''
    const pendingToolCalls: Map<string, { name: string; input: Record<string, unknown> }> = new Map()

    const flushThinkingSignature = () => {
      if (!pendingThinkingSignature) return
      const signatureDelta = createClaudeStreamEvent('content_block_delta', {
        index: currentBlockIndex,
        delta: { type: 'signature_delta', signature: pendingThinkingSignature }
      })
      res.write(`event: content_block_delta\ndata: ${JSON.stringify(signatureDelta)}\n\n`)
      pendingThinkingSignature = undefined
    }

    // 估算输入 tokens（基于 payload 大小）
    const estimatedInputTokens = Math.max(1, Math.round(JSON.stringify(kiroPayload).length / 3))
    
    // 发送 message_start（仅首轮）
    if (currentRound === 0) {
      const messageStart = createClaudeStreamEvent('message_start', {
        message: {
          id,
          type: 'message',
          role: 'assistant',
          content: [],
          model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: estimatedInputTokens, output_tokens: 0 }
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
            this.settleAbortedUsage(matchedApiKey, account.id, usage, model, '/v1/messages')
            resolve()
            return
          }
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
          this.stats.cacheReadTokens += usage.cacheReadTokens || simulatedCacheUsage?.cacheReadInputTokens || 0
          this.stats.cacheWriteTokens += usage.cacheWriteTokens || simulatedCacheUsage?.cacheCreationInputTokens || 0
          this.stats.reasoningTokens += usage.reasoningTokens || 0
          const respTime = Date.now() - startTime
          this.events.onResponse?.({ path: '/v1/messages', model, status: 200, tokens: usage.inputTokens + usage.outputTokens, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, cacheReadTokens: usage.cacheReadTokens || simulatedCacheUsage?.cacheReadInputTokens, reasoningTokens: usage.reasoningTokens, credits: usage.credits, responseTime: respTime })
          this.recordRequest({ path: '/v1/messages', model, accountId: account.id, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, credits: usage.credits, responseTime: respTime, success: true })
          // 记录 API Key 用量
          if (matchedApiKey) {
            this.recordApiKeyUsage(matchedApiKey.id, usage.credits || 0, usage.inputTokens, usage.outputTokens, model, '/v1/messages')
          }

          // 成功后更新 prompt cache tracker
          if (simulatedCacheUsage?.cacheProfile && simulatedCacheUsage?.accountId) {
            promptCacheTracker.update(simulatedCacheUsage.accountId, simulatedCacheUsage.cacheProfile as any)
          }
          // 发送 message_delta（包含完整 usage 信息）
          const hasToolCalls = pendingToolCalls.size > 0
          const stopReason = hasToolCalls ? 'tool_use' : 'end_turn'
          const messageDelta = createClaudeStreamEvent('message_delta', {
            delta: { stop_reason: stopReason, stop_sequence: null } as any,
            usage: this.buildClaudeUsage(usage, simulatedCacheUsage)
          })
          res.write(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`)
          // 发送 message_stop
          const messageStop = createClaudeStreamEvent('message_stop')
          res.write(`event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`)
          res.end()
          resolve()
        },
        (error) => {
          if (this.isAbortError(error, signal) || this.isResponseClosed(res)) {
            resolve()
            return
          }
          console.error('[ProxyServer] Stream error:', error)
          const errorEvent = createClaudeStreamEvent('error', {
            error: { type: 'api_error', message: error.message }
          })
          res.write(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`)
          res.end()

          this.recordRequestFailed()
          const errStatusCode2 = error.message.match(/(\d{3})/)?.[1]
          this.accountPool.recordError(account.id, errStatusCode2 ? classifyError(parseInt(errStatusCode2)) : ErrorType.RECOVERABLE, errStatusCode2 ? parseInt(errStatusCode2) : undefined)
          this.events.onResponse?.({ path: '/v1/messages', model, status: 500, error: error.message })
          this.recordRequest({ path: '/v1/messages', model, accountId: account.id, responseTime: Date.now() - startTime, success: false, error: error.message })
          resolve()
        },
        signal,
        this.config.preferredEndpoint
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
    const maxBytes = Math.max(1024, this.config.maxRequestBodyBytes ?? 10 * 1024 * 1024)

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
    const cap = this.config.portalMaxConcurrentPerCustomer ?? 6
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

// ============ 客户门户静态页面 ============
// 自包含单页（vanilla JS，无外部依赖）：登录 → 查看余额 → 管理 API Key → 查看用量。
// 通过 fetch 调用 /portal/* JSON API；会话 token 存 localStorage。
const PORTAL_HTML = `<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Portal</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --txt:#e2e8f0; --muted:#94a3b8; --accent:#3b82f6; --danger:#ef4444; --ok:#22c55e; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--txt); }
  .wrap { max-width:760px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:20px; margin:0 0 16px; }
  h2 { font-size:15px; margin:24px 0 10px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; margin-bottom:16px; }
  label { display:block; font-size:13px; color:var(--muted); margin:10px 0 4px; }
  input { width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:8px; background:#0b1220; color:var(--txt); font-size:14px; }
  button { padding:9px 16px; border:none; border-radius:8px; background:var(--accent); color:#fff; font-size:14px; cursor:pointer; }
  button:hover { opacity:.9; }
  button.secondary { background:#475569; }
  button.danger { background:var(--danger); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .balance { font-size:32px; font-weight:700; }
  .balance.low { color:var(--danger); }
  .muted { color:var(--muted); font-size:13px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:500; }
  code { background:#0b1220; padding:2px 6px; border-radius:5px; font-size:12px; word-break:break-all; }
  .hide { display:none; }
  .err { color:var(--danger); font-size:13px; margin-top:8px; min-height:18px; }
  .keybox { background:#0b1220; border:1px solid var(--ok); border-radius:8px; padding:12px; margin-top:10px; }
  .topbar { display:flex; justify-content:space-between; align-items:center; }
</style>
</head>
<body>
<div class="wrap">
  <!-- Login view -->
  <div id="loginView">
    <h1>เข้าสู่ระบบ</h1>
    <div class="card">
      <label>อีเมล</label>
      <input id="email" type="email" autocomplete="username" placeholder="you@example.com">
      <label>รหัสผ่าน</label>
      <input id="password" type="password" autocomplete="current-password" placeholder="••••••••">
      <div style="margin-top:14px"><button id="loginBtn">เข้าสู่ระบบ</button></div>
      <div class="err" id="loginErr"></div>
    </div>
  </div>

  <!-- Dashboard view -->
  <div id="dashView" class="hide">
    <div class="topbar">
      <h1>แดชบอร์ด</h1>
      <button class="secondary" id="logoutBtn">ออกจากระบบ</button>
    </div>
    <div class="muted" id="whoami"></div>

    <h2>เครดิตคงเหลือ</h2>
    <div class="card">
      <div class="balance" id="balance">–</div>
      <div class="muted" id="balanceNote">เติมเครดิตติดต่อแอดมิน</div>
    </div>

    <h2>API Keys</h2>
    <div class="card">
      <div class="row">
        <input id="keyName" placeholder="ชื่อ key (เช่น my-app)" style="flex:1; min-width:160px">
        <button id="createKeyBtn">สร้าง Key</button>
      </div>
      <div class="err" id="keyErr"></div>
      <div id="newKeyBox"></div>
      <table style="margin-top:12px">
        <thead><tr><th>ชื่อ</th><th>Key</th><th>credits</th><th></th></tr></thead>
        <tbody id="keyRows"></tbody>
      </table>
    </div>

    <h2>การใช้งาน</h2>
    <div class="card">
      <div class="muted">รวมทั้งหมด: <span id="usageTotal">–</span></div>
      <table style="margin-top:10px">
        <thead><tr><th>วันที่</th><th>requests</th><th>credits</th></tr></thead>
        <tbody id="usageRows"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
(function(){
  var TOKEN_KEY = 'portal_token';
  var token = localStorage.getItem(TOKEN_KEY) || '';
  var $ = function(id){ return document.getElementById(id); };

  function api(path, opts){
    opts = opts || {};
    var headers = { 'Content-Type':'application/json' };
    if (token) headers['Authorization'] = 'Bearer ' + token;
    return fetch(path, { method: opts.method || 'GET', headers: headers, body: opts.body ? JSON.stringify(opts.body) : undefined })
      .then(function(r){
        return r.json().catch(function(){ return {}; }).then(function(j){
          return { ok: r.ok, status: r.status, data: j };
        });
      });
  }

  function show(view){
    $('loginView').classList.toggle('hide', view !== 'login');
    $('dashView').classList.toggle('hide', view !== 'dash');
  }

  function doLogin(){
    $('loginErr').textContent = '';
    var email = $('email').value.trim();
    var password = $('password').value;
    if (!email || !password){ $('loginErr').textContent = 'กรอกอีเมลและรหัสผ่าน'; return; }
    $('loginBtn').disabled = true;
    api('/portal/login', { method:'POST', body:{ email:email, password:password } }).then(function(r){
      $('loginBtn').disabled = false;
      if (!r.ok){ $('loginErr').textContent = (r.data && r.data.error) || 'เข้าสู่ระบบไม่สำเร็จ'; return; }
      token = r.data.token;
      localStorage.setItem(TOKEN_KEY, token);
      loadDash();
    });
  }

  function logout(){
    token = '';
    localStorage.removeItem(TOKEN_KEY);
    show('login');
  }

  function fmt(n){ return (Math.round((n||0)*1000)/1000).toLocaleString(); }

  function loadDash(){
    api('/portal/me').then(function(r){
      if (!r.ok){ logout(); return; }
      show('dash');
      var c = r.data;
      $('whoami').textContent = c.email + (c.name ? ' ('+c.name+')' : '');
      var bal = $('balance');
      bal.textContent = fmt(c.creditBalance) + ' credits';
      bal.classList.toggle('low', c.creditBalance <= 0);
      $('balanceNote').textContent = c.creditBalance <= 0
        ? 'เครดิตหมด — ติดต่อแอดมินเพื่อเติม'
        : 'เติมเครดิตติดต่อแอดมิน';
      loadKeys();
      loadUsage();
    });
  }

  function loadKeys(){
    api('/portal/keys').then(function(r){
      if (!r.ok) return;
      var rows = $('keyRows'); rows.innerHTML = '';
      (r.data.keys || []).forEach(function(k){
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>'+esc(k.name)+'</td><td><code>'+esc(k.keyMasked)+'</code></td><td>'+fmt(k.totalCredits)+'</td>'
          + '<td><button class="danger" data-id="'+esc(k.id)+'">ลบ</button></td>';
        rows.appendChild(tr);
      });
      Array.prototype.forEach.call(rows.querySelectorAll('button[data-id]'), function(btn){
        btn.addEventListener('click', function(){ delKey(btn.getAttribute('data-id')); });
      });
    });
  }

  function createKey(){
    $('keyErr').textContent = '';
    var name = $('keyName').value.trim();
    $('createKeyBtn').disabled = true;
    api('/portal/keys', { method:'POST', body:{ name:name } }).then(function(r){
      $('createKeyBtn').disabled = false;
      if (!r.ok){ $('keyErr').textContent = (r.data && r.data.error) || 'สร้างไม่สำเร็จ'; return; }
      $('keyName').value = '';
      $('newKeyBox').innerHTML = '<div class="keybox">คัดลอก key นี้เก็บไว้ (แสดงครั้งเดียว):<br><code>'+esc(r.data.key)+'</code></div>';
      loadKeys();
    });
  }

  function delKey(id){
    if (!confirm('ลบ key นี้?')) return;
    api('/portal/keys/' + encodeURIComponent(id), { method:'DELETE' }).then(function(r){
      if (r.ok){ loadKeys(); }
    });
  }

  function loadUsage(){
    api('/portal/usage').then(function(r){
      if (!r.ok) return;
      $('usageTotal').textContent = fmt(r.data.totalCredits) + ' credits / ' + (r.data.totalRequests||0) + ' requests';
      var rows = $('usageRows'); rows.innerHTML = '';
      var days = Object.keys(r.data.daily || {}).sort().reverse().slice(0, 14);
      days.forEach(function(day){
        var d = r.data.daily[day];
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>'+esc(day)+'</td><td>'+d.requests+'</td><td>'+fmt(d.credits)+'</td>';
        rows.appendChild(tr);
      });
    });
  }

  function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }

  $('loginBtn').addEventListener('click', doLogin);
  $('password').addEventListener('keydown', function(e){ if (e.key==='Enter') doLogin(); });
  $('logoutBtn').addEventListener('click', logout);
  $('createKeyBtn').addEventListener('click', createKey);

  if (token) loadDash(); else show('login');
})();
</script>
</body>
</html>`

