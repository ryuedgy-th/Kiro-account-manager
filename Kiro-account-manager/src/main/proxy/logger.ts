// 代理服务器日志模块
import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'
import { redactString, redactValue } from '../utils/redact'

// headless 友好：不在模块顶层 import electron。
// Electron 主进程通过 ProxyLogStore.initialize(userDataPath) 自动注入；
// headless service 可用 config.logDir 显式指定，或设环境变量 KIRO_DATA_DIR。
let injectedDataDir: string | undefined
export function setProxyLoggerDataDir(dir: string): void {
  if (dir) injectedDataDir = dir
}

export interface LogEntry {
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  category: string
  message: string
  data?: unknown
}

export interface LoggerConfig {
  enabled: boolean
  logDir?: string
  maxFileSize?: number // 最大文件大小 (bytes)
  maxFiles?: number // 最大文件数量
  logToConsole?: boolean
  logDebug?: boolean // 是否处理 DEBUG 级日志（默认 false：热路径上的 debug 直接丢弃，省去 redact+存储开销）
}

const DEFAULT_CONFIG: LoggerConfig = {
  enabled: false,
  maxFileSize: 10 * 1024 * 1024, // 10MB
  maxFiles: 5,
  logToConsole: true,
  logDebug: false
}

class ProxyLogger {
  private config: LoggerConfig
  private logStream: fs.WriteStream | null = null
  private currentLogFile: string = ''
  private currentFileSize: number = 0

  constructor() {
    this.config = { ...DEFAULT_CONFIG }
  }

  configure(config: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...config }
    
    if (this.config.enabled && !this.config.logDir) {
      // 默认日志目录（headless 友好：优先 inject/env，最后退到临时目录）
      const base = injectedDataDir || process.env.KIRO_DATA_DIR || path.join(os.tmpdir(), 'kiro-proxy')
      this.config.logDir = path.join(base, 'logs', 'proxy')
    }

    if (this.config.enabled) {
      this.initLogFile()
    } else {
      this.close()
    }
  }

  private initLogFile(): void {
    if (!this.config.logDir) return

    try {
      // 确保目录存在
      fs.mkdirSync(this.config.logDir, { recursive: true })

      // 创建新的日志文件
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      this.currentLogFile = path.join(this.config.logDir, `proxy-${timestamp}.log`)
      this.logStream = fs.createWriteStream(this.currentLogFile, { flags: 'a' })
      this.currentFileSize = 0

      this.info('Logger', 'Log file initialized', { file: this.currentLogFile })
    } catch (error) {
      console.error('[ProxyLogger] Failed to init log file:', error)
    }
  }

  private rotateIfNeeded(): void {
    if (!this.config.maxFileSize || this.currentFileSize < this.config.maxFileSize) {
      return
    }

    this.close()
    this.cleanOldLogs()
    this.initLogFile()
  }

  private cleanOldLogs(): void {
    if (!this.config.logDir || !this.config.maxFiles) return

    try {
      const files = fs.readdirSync(this.config.logDir)
        .filter(f => f.startsWith('proxy-') && f.endsWith('.log'))
        .map(f => ({
          name: f,
          path: path.join(this.config.logDir!, f),
          time: fs.statSync(path.join(this.config.logDir!, f)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time)

      // 删除超出数量限制的旧文件
      while (files.length >= this.config.maxFiles) {
        const oldest = files.pop()
        if (oldest) {
          fs.unlinkSync(oldest.path)
        }
      }
    } catch (error) {
      console.error('[ProxyLogger] Failed to clean old logs:', error)
    }
  }

  private isWriting = false
  private write(rawEntry: LogEntry): void {
    // 热路径优化：默认丢弃 DEBUG（省去 redact + JSON + proxyLogStore 的逐请求开销）。
    // 需要排查时通过 configure({ logDebug: true }) 打开。
    if (rawEntry.level === 'DEBUG' && !this.config.logDebug) return
    // 统一脱敏：message + data 里的代理账密 / token / password 等，避免明文落盘或上屏
    const entry: LogEntry = {
      ...rawEntry,
      message: redactString(rawEntry.message),
      data: rawEntry.data === undefined ? undefined : redactValue(rawEntry.data)
    }
    const line = JSON.stringify(entry) + '\n'

    if (this.config.logToConsole) {
      const prefix = `[${entry.level}][${entry.category}]`
      // 设置 flag 防止 console 拦截器重复写入 proxyLogStore
      this.isWriting = true
      if (entry.level === 'ERROR') {
        console.error(prefix, entry.message, entry.data || '')
      } else if (entry.level === 'WARN') {
        console.warn(prefix, entry.message, entry.data || '')
      } else {
        console.log(prefix, entry.message, entry.data || '')
      }
      this.isWriting = false
    }

    if (this.config.enabled && this.logStream) {
      this.logStream.write(line)
      this.currentFileSize += Buffer.byteLength(line)
      this.rotateIfNeeded()
    }

    // 同时添加到内存存储（用于 UI 显示）
    proxyLogStore.add(entry)
  }

  get _isWriting(): boolean { return this.isWriting }

  debug(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'DEBUG',
      category,
      message,
      data
    })
  }

  info(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      category,
      message,
      data
    })
  }

  warn(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'WARN',
      category,
      message,
      data
    })
  }

  error(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      category,
      message,
      data
    })
  }

  // 记录请求
  request(info: {
    path: string
    method: string
    model?: string
    accountId?: string
  }): void {
    this.info('Request', `${info.method} ${info.path}`, info)
  }

  // 记录响应
  response(info: {
    path: string
    status: number
    tokens?: number
    responseTime?: number
    error?: string
  }): void {
    if (info.error) {
      this.error('Response', `${info.path} -> ${info.status}`, info)
    } else {
      this.info('Response', `${info.path} -> ${info.status}`, info)
    }
  }

  // 记录 Token 刷新
  tokenRefresh(accountId: string, success: boolean, error?: string): void {
    if (success) {
      this.info('TokenRefresh', `Account ${accountId} refreshed successfully`)
    } else {
      this.error('TokenRefresh', `Account ${accountId} refresh failed`, { error })
    }
  }

  close(): void {
    if (this.logStream) {
      this.logStream.end()
      this.logStream = null
    }
  }

  getLogDir(): string | undefined {
    return this.config.logDir
  }
}

// 内存日志存储（用于 UI 显示）
//
// 性能修复要点：
// 1. maxLogs 从 100万 降至 5万 — 避免序列化数百 MB 大对象阻塞主进程
// 2. save() 改为异步 fs.promises.writeFile — 不再 freeze 主进程事件循环
// 3. 单次写盘原子化（in-flight guard）防止并发 writeFile 导致竞态
// 4. 写盘节流间隔从 5s 提至 30s — 大幅降低高频日志场景下的 IO 频率
// 5. 应用退出时通过 flushSaveNow() 强制写盘，防止数据丢失
class ProxyLogStore {
  private logs: LogEntry[] = []
  // 5 万条 × 平均 200 字节 ≈ 10 MB；既能覆盖常规调试需求，又把单次写盘成本控制在可接受范围内
  private maxLogs: number = 50000
  private listeners: ((entry: LogEntry) => void)[] = []
  private storePath: string = ''

  private initialized = false
  initialize(userDataPath: string): void {
    if (this.initialized) return
    this.initialized = true
    // 同时把日志目录注入给 ProxyLogger（保持 Electron 侧原有 userData 行为）
    setProxyLoggerDataDir(userDataPath)
    this.storePath = path.join(userDataPath, 'proxy-logs.json')
    this.load()
  }

  private load(): void {
    try {
      if (fs.existsSync(this.storePath)) {
        const data = fs.readFileSync(this.storePath, 'utf-8')
        const parsed = JSON.parse(data)
        // 验证并过滤有效的日志条目
        const filtered = Array.isArray(parsed) ? parsed.filter((log: LogEntry) => {
          if (!log.timestamp || isNaN(new Date(log.timestamp).getTime())) return false
          if (!log.level || !log.category) return false
          return true
        }) : []
        // 加载时也施加上限，避免旧版本遗留的超大日志文件导致首次启动卡顿
        this.logs = filtered.length > this.maxLogs ? filtered.slice(-this.maxLogs) : filtered
        console.log(`[ProxyLogStore] Loaded ${this.logs.length} valid logs`)
      }
    } catch (error) {
      console.error('[ProxyLogStore] Failed to load logs:', error)
      this.logs = []
    }
  }

  /** 异步保存日志（不阻塞主进程事件循环）。并发调用通过 in-flight 标志合并。 */
  private writeInFlight = false
  private writePending = false

  async save(): Promise<void> {
    // headless/未初始化时 storePath 为空，跳过磁盘写入（避免 ENOENT 刷屏）
    if (!this.storePath) return
    if (this.writeInFlight) {
      // 已有写盘进行中：标记 pending，让其完成后立即重写最新数据
      this.writePending = true
      return
    }
    this.writeInFlight = true
    try {
      // 拷贝引用快照（不复制数组）以保证 JSON.stringify 期间数据稳定
      const snapshot = this.logs
      await fs.promises.writeFile(this.storePath, JSON.stringify(snapshot), 'utf-8')
    } catch (error) {
      console.error('[ProxyLogStore] Failed to save logs:', error)
    } finally {
      this.writeInFlight = false
      if (this.writePending) {
        this.writePending = false
        // 用 microtask 而不是立即递归，避免栈深问题
        queueMicrotask(() => { void this.save() })
      }
    }
  }

  private saveTimer: NodeJS.Timeout | null = null

  add(entry: LogEntry): void {
    this.logs.push(entry)

    // 超过最大数量时删除最旧的
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs)
    }

    // 通知监听器（异常隔离）
    for (const listener of this.listeners) {
      try { listener(entry) } catch { /* ignore */ }
    }

    // 节流调度异步保存
    this.scheduleSave()
  }

  private scheduleSave(): void {
    if (this.saveTimer) return  // 已调度，等待 flush
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null
      void this.save()
    }, 30_000) // 30 秒批量写盘一次（异步、不阻塞）
  }

  /** 强制立即写盘（用于退出场景），保证最新数据落盘 */
  async flushSaveNow(): Promise<void> {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer)
      this.saveTimer = null
    }
    await this.save()
  }

  getAll(): LogEntry[] {
    return [...this.logs]
  }

  getLast(count: number): LogEntry[] {
    return this.logs.slice(-count)
  }

  clear(): void {
    this.logs = []
    void this.save()
  }

  count(): number {
    return this.logs.length
  }

  onLog(listener: (entry: LogEntry) => void): () => void {
    this.listeners.push(listener)
    return () => {
      const index = this.listeners.indexOf(listener)
      if (index >= 0) {
        this.listeners.splice(index, 1)
      }
    }
  }
}

export const proxyLogStore = new ProxyLogStore()

// 单例导出
export const proxyLogger = new ProxyLogger()

// 把任意错误（含 undici/fetch 的 TypeError）格式化为可诊断的字符串。
// 背景：网络层错误常见两种"信息黑洞"：
//   1) `new TypeError('fetch failed')` —— message 只有 'fetch failed'，真正原因在 .cause（ECONNRESET/ETIMEDOUT/证书等）
//   2) 非 Error 抛出物（如 {}）—— message 为空，console.error 直接打出 "{}"
// 该 helper 会一并提取 name/code/cause，避免日志里只剩 "{}" 或 "fetch failed" 而无从排查。
export function formatError(error: unknown): string {
  if (error == null) return String(error)
  if (error instanceof Error) {
    const parts: string[] = []
    parts.push(`${error.name}: ${error.message || '(empty message)'}`)
    // Node 网络错误把 errno/code 挂在错误对象上
    const code = (error as { code?: string }).code
    if (code) parts.push(`code=${code}`)
    // undici 'fetch failed' 的真实原因在 cause 里
    const cause = (error as { cause?: unknown }).cause
    if (cause !== undefined && cause !== null) {
      if (cause instanceof Error) {
        const causeCode = (cause as { code?: string }).code
        parts.push(`cause=${cause.name}: ${cause.message}${causeCode ? ` (code=${causeCode})` : ''}`)
      } else {
        parts.push(`cause=${typeof cause === 'object' ? JSON.stringify(cause) : String(cause)}`)
      }
    }
    return parts.join(' | ')
  }
  if (typeof error === 'object') {
    try {
      const json = JSON.stringify(error)
      return json === '{}' ? `[non-Error object: ${Object.prototype.toString.call(error)}]` : json
    } catch {
      return Object.prototype.toString.call(error)
    }
  }
  return String(error)
}

// 拦截主进程 console 输出，自动转发到 proxyLogStore
// 这样所有 console.log/warn/error 都能在日志页面显示
let consoleIntercepted = false
export function interceptConsole(): void {
  if (consoleIntercepted) return
  consoleIntercepted = true

  const originalLog = console.log
  const originalWarn = console.warn
  const originalError = console.error

  const parseConsoleCategory = (args: unknown[]): { category: string; message: string } => {
    const first = String(args[0] || '')
    // 匹配 [Category] 或 [INFO][Category] 格式
    const bracketMatch = first.match(/^\[(?:DEBUG|INFO|WARN|ERROR)\]?\[?([^\]]*)\]?\s*(.*)/)
    if (bracketMatch) {
      return { category: bracketMatch[1] || 'App', message: bracketMatch[2] || '' }
    }
    const simpleMatch = first.match(/^\[([^\]]+)\]\s*(.*)/)
    if (simpleMatch) {
      return { category: simpleMatch[1], message: simpleMatch[2] || '' }
    }
    return { category: 'App', message: first }
  }

  const buildEntry = (args: unknown[], level: 'INFO' | 'WARN' | 'ERROR'): LogEntry => {
    const { category, message } = parseConsoleCategory(args)
    const rest = args.slice(1)
    // data: 后续参数（对象/数组保留结构，字符串拼接）
    let data: unknown = undefined
    if (rest.length === 1) {
      data = rest[0]
    } else if (rest.length > 1) {
      // 如果全是字符串，拼成一个；否则保持数组
      const allStrings = rest.every(r => typeof r === 'string')
      data = allStrings ? (rest as string[]).join(' ') : rest
    }
    return { timestamp: new Date().toISOString(), level, category, message, data }
  }

  console.log = (...args: unknown[]) => {
    originalLog.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'INFO'))
  }

  console.warn = (...args: unknown[]) => {
    originalWarn.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'WARN'))
  }

  console.error = (...args: unknown[]) => {
    originalError.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'ERROR'))
  }
}
