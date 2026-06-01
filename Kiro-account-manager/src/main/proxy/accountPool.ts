// 多账号智能轮询管理器
// 参考 Kiro Gateway 的 Circuit Breaker + Sticky + 指数退避 + 概率重试机制
import type { ProxyAccount, AccountStats } from './types'

// 错误类型分类（决定 failover 策略）
export enum ErrorType {
  FATAL = 'fatal',           // 请求本身有问题 → 直接返回客户端，不切号
  RECOVERABLE = 'recoverable' // 账号问题 → 切换到下一个账号
}

// 根据 HTTP 状态码和错误原因分类错误
export function classifyError(statusCode: number, reason?: string): ErrorType {
  // RECOVERABLE: 配额/计费问题
  if (statusCode === 402) return ErrorType.RECOVERABLE
  // RECOVERABLE: Token 过期/无效
  if (statusCode === 403) return ErrorType.RECOVERABLE
  // RECOVERABLE: 限流
  if (statusCode === 429) return ErrorType.RECOVERABLE
  // 400: 根据原因细分
  if (statusCode === 400) {
    // 上下文超限 → 所有账号都会失败
    if (reason === 'CONTENT_LENGTH_EXCEEDS_THRESHOLD') return ErrorType.FATAL
    return ErrorType.FATAL
  }
  // 422: 请求格式错误
  if (statusCode === 422) return ErrorType.FATAL
  // 5xx: 服务端错误
  if (statusCode >= 500) return ErrorType.FATAL
  return ErrorType.FATAL
}

export interface AccountPoolConfig {
  baseCooldownMs: number      // 基础冷却时间（指数退避的基数）
  maxBackoffMultiplier: number // 最大退避倍数
  quotaResetMs: number        // 配额耗尽冷却时间
  probabilisticRetryChance: number // 概率重试几率（0-1）
}

const DEFAULT_CONFIG: AccountPoolConfig = {
  baseCooldownMs: 60000,        // 60s 基础冷却
  maxBackoffMultiplier: 1440,   // 最大 1440 倍 = 24h
  quotaResetMs: 3600000,        // 1h 配额重置
  probabilisticRetryChance: 0.1 // 10% 概率重试
}

// TEMPORARILY_SUSPENDED 自动解封时长：Kiro 的临时风控多数会在一段时间后自动解除，
// 旧逻辑把所有 suspended 视为永久（需人工 clearSuspended），导致 pool 持续缩水。
// 临时封禁超过此时长后，账号自动重新加入轮询（仍会再次试探，若仍被风控会再次标记）。
// PERMANENTLY_SUSPENDED / ACCOUNT_SUSPENDED 不受此影响，保持永久跳过。
const TEMPORARY_SUSPEND_AUTO_CLEAR_MS = 2 * 60 * 60 * 1000  // 2h

export type AccountSelectionStrategy = 'round-robin' | 'sticky'

export class AccountPool {
  private accounts: Map<string, ProxyAccount> = new Map()
  private accountStats: Map<string, AccountStats> = new Map()
  private currentIndex: number = 0
  private config: AccountPoolConfig
  // 默认 round-robin: 每次成功后指针前进 (满足负载均衡期望)
  // sticky: 一个账号成功就粘住 (保留 prompt cache 命中)
  private strategy: AccountSelectionStrategy = 'round-robin'

  constructor(config: Partial<AccountPoolConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  // 切换账号选择策略
  setStrategy(strategy: AccountSelectionStrategy): void {
    if (this.strategy !== strategy) {
      console.log(`[AccountPool] Strategy changed: ${this.strategy} → ${strategy}`)
      this.strategy = strategy
    }
  }

  getStrategy(): AccountSelectionStrategy {
    return this.strategy
  }

  // 添加账号
  // 如果传入的 account 已带 suspended 字段（启动复原场景），保留其 suspended 状态
  addAccount(account: ProxyAccount): void {
    const suspended = this.isSuspended(account)
    this.accounts.set(account.id, {
      ...account,
      isAvailable: !suspended,
      requestCount: 0,
      errorCount: 0,
      lastUsed: 0
    })
    this.accountStats.set(account.id, {
      requests: 0,
      tokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      errors: 0,
      lastUsed: 0,
      avgResponseTime: 0,
      totalResponseTime: 0
    })
    if (suspended) {
      console.warn(`[AccountPool] Added SUSPENDED account: ${account.email || account.id} (${account.suspendReason})`)
    } else {
      console.log(`[AccountPool] Added account: ${account.email || account.id}`)
    }
  }

  // 移除账号
  removeAccount(accountId: string): void {
    this.accounts.delete(accountId)
    this.accountStats.delete(accountId)
    console.log(`[AccountPool] Removed account: ${accountId}`)
  }

  // 更新账号
  updateAccount(accountId: string, updates: Partial<ProxyAccount>): void {
    const account = this.accounts.get(accountId)
    if (account) {
      this.accounts.set(accountId, { ...account, ...updates })
    }
  }

  // 获取下一个可用账号（粘滞 + 断路器 + 指数退避 + 概率重试）
  getNextAccount(excludeIds?: Set<string>): ProxyAccount | null {
    const accountList = Array.from(this.accounts.values())
    if (accountList.length === 0) {
      return null
    }

    // 单账号特殊处理：绕过断路器，直接返回（让用户看到真实 API 错误）
    if (accountList.length === 1) {
      const account = accountList[0]
      if (excludeIds?.has(account.id)) return null
      return account
    }

    const now = Date.now()
    // 从当前粘滞索引开始遍历所有账号
    const startIndex = this.currentIndex

    for (let i = 0; i < accountList.length; i++) {
      const idx = (startIndex + i) % accountList.length
      const account = accountList[idx]

      // 跳过当前请求已试过的账号
      if (excludeIds?.has(account.id)) continue

      // 检查账号是否可用（含断路器状态）
      if (this.isAccountAvailable(account, now)) {
        return this.reviveExpiredSuspension(account)
      }
    }

    // 没有可用账号：检查是否全部因配额耗尽
    const candidates = excludeIds
      ? accountList.filter(a => !excludeIds.has(a.id))
      : accountList
    const allExhausted = candidates.length > 0 && candidates.every(a => this.isQuotaExhausted(a, now))
    if (allExhausted) {
      console.log(`[AccountPool] All ${candidates.length} accounts quota exhausted, no fallback available`)
      return null
    }

    // 还有非配额原因不可用的账号，返回冷却时间最短的
    const nonExhausted = candidates.filter(a => !this.isQuotaExhausted(a, now))
    return this.getAccountWithShortestCooldown(nonExhausted, now)
  }

  // 获取特定账号
  getAccount(accountId: string): ProxyAccount | null {
    return this.accounts.get(accountId) || null
  }

  // 获取下一个可用账号（排除指定账号；支持单 ID 或 ID 集合）
  // 集合形式用于「请求级累计已试账号」，避免重试时循环命中已经失败过的账号
  getNextAvailableAccount(exclude: string | Set<string>): ProxyAccount | null {
    const excludeSet = typeof exclude === 'string' ? new Set([exclude]) : exclude
    const accountList = Array.from(this.accounts.values())
    if (accountList.length === 0) return null

    const now = Date.now()

    // 尝试找到一个可用的账号（排除指定账号）
    for (const account of accountList) {
      if (!excludeSet.has(account.id) && this.isAccountAvailable(account, now)) {
        return this.reviveExpiredSuspension(account)
      }
    }

    // 没有立即可用的账号，返回冷却时间最短的（排除指定账号）
    const otherAccounts = accountList.filter(a => !excludeSet.has(a.id))
    if (otherAccounts.length === 0) return null
    return this.getAccountWithShortestCooldown(otherAccounts, now)
  }

  // 获取所有账号
  getAllAccounts(): ProxyAccount[] {
    return Array.from(this.accounts.values())
  }

  // 检查账号是否可用（断路器 + 指数退避 + 概率重试）
  // 纯读取，不产生任何副作用——getQuotaStatus / metrics 等只读路径会调用本函数，
  // 临时封禁的"真正字段清理"由写路径（getNextAccount/getNextAvailableAccount 选中时）
  // 调 reviveExpiredSuspension 完成，不能在这里 mutate。
  private isAccountAvailable(account: ProxyAccount, now: number): boolean {
    // 检查是否被 Kiro 后端封禁（需人工解封）
    if (this.isSuspended(account)) {
      return false
    }
    // isSuspended 已返回 false 但仍残留 suspendedAt → 临时封禁已自动过期。
    // 此时 markSuspended 设置的 isAvailable=false / 旧 errorCount 都应被视为已失效，
    // 否则下面的检查会错误地把"已解封"的账号判为不可用。
    const expiredTempSuspend = typeof account.suspendedAt === 'number' && account.suspendedAt > 0

    // 检查配额是否耗尽
    if (this.isQuotaExhausted(account, now)) {
      return false
    }

    // 检查 token 是否过期
    // - 无 refreshToken 时直接判为不可用（无法刷新）
    // - 有 refreshToken 时让账号通过 —— proxyServer.getAvailableAccount 会检测
    //   isTokenExpiringSoon 并主动调用 refreshToken；若刷新失败会通过 markNeedsRefresh
    //   设置 isAvailable=false，下次循环再被本函数跳过，形成闭环
    if (account.expiresAt && account.expiresAt < now && !account.refreshToken) {
      return false
    }

    if (account.isAvailable === false && !expiredTempSuspend) {
      return false
    }

    // 断路器检查：指数退避 + 概率重试
    // 临时封禁过期的账号视为全新回归（errorCount 归零），给一次干净的探测机会
    const failures = expiredTempSuspend ? 0 : (account.errorCount || 0)
    if (failures > 0 && account.lastUsed) {
      const timeSinceFailure = now - account.lastUsed
      // 指数退避：base * 2^(failures-1)，封顶为 maxBackoffMultiplier
      const backoffMultiplier = Math.min(Math.pow(2, failures - 1), this.config.maxBackoffMultiplier)
      const effectiveCooldown = this.config.baseCooldownMs * backoffMultiplier

      if (timeSinceFailure < effectiveCooldown) {
        // 未超出冷却期，用概率重试
        if (Math.random() > this.config.probabilisticRetryChance) {
          return false
        }
        console.log(`[AccountPool] Probabilistic retry for ${account.email || account.id} (failures=${failures}, cooldown=${Math.round(effectiveCooldown / 1000)}s)`)
      }
      // else: 冷却期已过，Half-Open 状态，允许重试
    }

    return true
  }

  // 检查账号是否被长期封禁（TEMPORARILY_SUSPENDED / AccountSuspendedException 等风控触发）
  // 不同于临时 errorCount 冷却，需要人工解封或调用 clearSuspended
  //
  // 例外：reason 为 TEMPORARILY_SUSPENDED 且距今已超过 TEMPORARY_SUSPEND_AUTO_CLEAR_MS（2h）时，
  // 视为已自动解封（返回 false），账号重新参与轮询。这样无需人工干预即可让临时风控账号回归，
  // 避免 pool 因临时封禁持续缩水。若账号实际仍被风控，下次请求会再次触发 markSuspended。
  isSuspended(account: ProxyAccount): boolean {
    if (typeof account.suspendedAt !== 'number' || account.suspendedAt <= 0) return false
    if (account.suspendReason === 'TEMPORARILY_SUSPENDED') {
      const elapsed = Date.now() - account.suspendedAt
      if (elapsed >= TEMPORARY_SUSPEND_AUTO_CLEAR_MS) {
        return false  // 临时封禁已过期 → 自动解封
      }
    }
    return true
  }

  // 标记账号为被封禁状态，账号池会持续跳过该账号直到 clearSuspended（或临时封禁自动过期）
  markSuspended(accountId: string, reason: string, message?: string): boolean {
    const account = this.accounts.get(accountId)
    if (!account) return false
    // 用原始字段判断"是否已是同因封禁中"，而非 isSuspended()：
    // isSuspended() 含 2h 自动过期逻辑，临界点附近会误判为"未封禁"从而丢弃这次新信号；
    // 这里只要 suspendedAt 仍在且 reason 相同就视为已记录，但仍刷新时间戳以延长临时封禁窗口。
    const alreadySuspended = typeof account.suspendedAt === 'number' && account.suspendedAt > 0
    if (alreadySuspended && account.suspendReason === reason) {
      // 同因再次触发：刷新 suspendedAt（重置 2h 窗口），但不算"新标记"
      this.accounts.set(accountId, { ...account, suspendedAt: Date.now(), suspendMessage: message ?? account.suspendMessage })
      return false
    }
    this.accounts.set(accountId, {
      ...account,
      suspendedAt: Date.now(),
      suspendReason: reason,
      suspendMessage: message,
      isAvailable: false
    })
    console.warn(`[AccountPool] Account ${account.email || accountId} SUSPENDED (${reason})`)
    return true
  }

  // 解除账号封禁标记（供手动重置、检测到被解封，或临时封禁自动过期后调用）
  clearSuspended(accountId: string): void {
    const account = this.accounts.get(accountId)
    // 用原始字段判断而非 isSuspended()：临时封禁自动过期后 isSuspended() 已返回 false，
    // 但 suspendedAt/isAvailable 等字段仍残留，需要在此真正清除
    if (!account || typeof account.suspendedAt !== 'number' || account.suspendedAt <= 0) return
    this.accounts.set(accountId, {
      ...account,
      suspendedAt: undefined,
      suspendReason: undefined,
      suspendMessage: undefined,
      isAvailable: true,
      errorCount: 0
    })
    console.log(`[AccountPool] Account ${account.email || accountId} unsuspended`)
  }

  // 写路径专用：若账号的临时封禁已过期（isSuspended=false 但字段残留），真正清除残留字段。
  // 仅在选中账号准备使用时调用（getNextAccount/getNextAvailableAccount），保持 isAccountAvailable 纯读。
  private reviveExpiredSuspension(account: ProxyAccount | null): ProxyAccount | null {
    if (!account) return account
    if (typeof account.suspendedAt === 'number' && account.suspendedAt > 0 && !this.isSuspended(account)) {
      this.clearSuspended(account.id)
      return this.accounts.get(account.id) || account
    }
    return account
  }

  // 检查账号配额是否耗尽
  isQuotaExhausted(account: ProxyAccount, now: number = Date.now()): boolean {
    // 如果配额已重置（过了重置时间），不再视为耗尽
    if (account.quotaResetAt && account.quotaResetAt <= now) {
      return false
    }
    // 有明确的耗尽标记
    if (account.quotaExhaustedAt && account.quotaExhaustedAt > 0) {
      return true
    }
    // 有配额数据且已用尽
    if (account.quotaLimit && account.quotaLimit > 0 && (account.quotaUsed ?? 0) >= account.quotaLimit) {
      return true
    }
    return false
  }

  // 获取冷却时间最短的账号
  private getAccountWithShortestCooldown(accounts: ProxyAccount[], now: number): ProxyAccount | null {
    let bestAccount: ProxyAccount | null = null
    let shortestWait = Infinity

    for (const account of accounts) {
      const cooldownUntil = account.cooldownUntil || 0
      const wait = Math.max(0, cooldownUntil - now)
      
      if (wait < shortestWait) {
        shortestWait = wait
        bestAccount = account
      }
    }

    return bestAccount
  }

  // 记录请求成功（重置断路器 + 粘滞到当前账号）
  recordSuccess(accountId: string, tokens: number = 0): void {
    const account = this.accounts.get(accountId)
    if (account) {
      this.accounts.set(accountId, {
        ...account,
        requestCount: (account.requestCount || 0) + 1,
        errorCount: 0, // 重置断路器失败计数
        lastUsed: Date.now(),
        isAvailable: true
      })

      const accountList = Array.from(this.accounts.keys())
      const successIndex = accountList.indexOf(accountId)
      if (successIndex >= 0 && accountList.length > 0) {
        if (this.strategy === 'sticky') {
          // 粘滞: 成功后将全局索引固定在这个账号 (保留 prompt cache 命中)
          this.currentIndex = successIndex
        } else {
          // round-robin: 成功后指向下一个账号 (满足负载均衡)
          this.currentIndex = (successIndex + 1) % accountList.length
        }
      }
    }

    const stats = this.accountStats.get(accountId)
    if (stats) {
      this.accountStats.set(accountId, {
        ...stats,
        requests: stats.requests + 1,
        tokens: stats.tokens + tokens,
        lastUsed: Date.now()
      })
    }
  }

  // 记录请求失败（区分错误类型）
  recordError(accountId: string, errorType: ErrorType = ErrorType.RECOVERABLE, statusCode?: number): void {
    const account = this.accounts.get(accountId)
    if (!account) return

    const now = Date.now()
    const stats = this.accountStats.get(accountId)
    if (stats) {
      this.accountStats.set(accountId, { ...stats, errors: stats.errors + 1, lastUsed: now })
    }

    // FATAL 错误不增加失败计数（是请求的问题，不是账号的问题）
    if (errorType === ErrorType.FATAL) return

    // RECOVERABLE: 增加失败计数，断路器指数退避自动生效
    const errorCount = (account.errorCount || 0) + 1
    let quotaExhaustedAt = account.quotaExhaustedAt

    // 配额类错误额外标记耗尽
    const isQuotaError = statusCode === 402 || statusCode === 429
    if (isQuotaError) {
      quotaExhaustedAt = now
    }

    // 计算当前退避时间用于日志
    const backoffMultiplier = Math.min(Math.pow(2, errorCount - 1), this.config.maxBackoffMultiplier)
    const effectiveCooldown = this.config.baseCooldownMs * backoffMultiplier
    const cooldownStr = effectiveCooldown < 60000 ? `${Math.round(effectiveCooldown / 1000)}s`
      : effectiveCooldown < 3600000 ? `${Math.round(effectiveCooldown / 60000)}m`
      : `${Math.round(effectiveCooldown / 3600000)}h`

    console.log(`[AccountPool] Account ${account.email || accountId} failure #${errorCount}: status=${statusCode || '?'}, cooldown=${cooldownStr}`)

    this.accounts.set(accountId, {
      ...account,
      errorCount,
      quotaExhaustedAt,
      lastUsed: now
    })
  }

  // 更新账号配额信息
  updateQuota(accountId: string, used: number, limit: number, resetAt?: number): void {
    const account = this.accounts.get(accountId)
    if (!account) return

    const wasExhausted = this.isQuotaExhausted(account)
    this.accounts.set(accountId, {
      ...account,
      quotaUsed: used,
      quotaLimit: limit,
      quotaResetAt: resetAt,
      // 如果配额从耗尽恢复，清除耗尽标记
      quotaExhaustedAt: (used < limit) ? undefined : account.quotaExhaustedAt
    })

    if (!wasExhausted && used >= limit) {
      console.log(`[AccountPool] Account ${account.email || accountId} quota reached: ${used}/${limit}`)
    } else if (wasExhausted && used < limit) {
      console.log(`[AccountPool] Account ${account.email || accountId} quota recovered: ${used}/${limit}`)
    }
  }

  // 获取配额状态摘要
  getQuotaStatus(): { total: number; available: number; exhausted: number; cooldown: number } {
    const now = Date.now()
    const all = Array.from(this.accounts.values())
    let available = 0
    let exhausted = 0
    let cooldown = 0

    for (const account of all) {
      if (this.isQuotaExhausted(account, now)) {
        exhausted++
      } else if (account.cooldownUntil && account.cooldownUntil > now) {
        cooldown++
      } else if (this.isAccountAvailable(account, now)) {
        available++
      }
    }

    return { total: all.length, available, exhausted, cooldown }
  }

  // 标记账号需要刷新 Token
  markNeedsRefresh(accountId: string): void {
    const account = this.accounts.get(accountId)
    if (account) {
      this.accounts.set(accountId, {
        ...account,
        isAvailable: false
      })
    }
  }

  // 获取统计信息
  getStats(): { accounts: Map<string, AccountStats>; total: { requests: number; tokens: number; errors: number } } {
    let totalRequests = 0
    let totalTokens = 0
    let totalErrors = 0

    for (const stats of this.accountStats.values()) {
      totalRequests += stats.requests
      totalTokens += stats.tokens
      totalErrors += stats.errors
    }

    return {
      accounts: new Map(this.accountStats),
      total: {
        requests: totalRequests,
        tokens: totalTokens,
        errors: totalErrors
      }
    }
  }

  // 重置所有账号状态（含封禁标记 — 手动重置表示用户已确认可用）
  reset(): void {
    for (const [id, account] of this.accounts) {
      this.accounts.set(id, {
        ...account,
        isAvailable: true,
        errorCount: 0,
        cooldownUntil: undefined,
        quotaExhaustedAt: undefined,
        suspendedAt: undefined,
        suspendReason: undefined,
        suspendMessage: undefined
      })
    }
    this.currentIndex = 0
  }

  // 清空所有账号
  clear(): void {
    this.accounts.clear()
    this.accountStats.clear()
    this.currentIndex = 0
  }

  // 批量同步账号池（幂等 diff，替代 clear()+addAccount() 全量重建）
  //
  // 旧做法每次都 clear() 再逐个 addAccount()：
  //   1. 即使账号没变也会刷屏 console.log，导致日志被 "Added account" 淹没（曾见 45s 内 5w 条）
  //   2. 丢弃已有账号的运行时状态（断路器冷却 / 配额 / 统计 / suspended），相当于每次同步都重置健康度
  // 新做法只对真正变化的账号增删，保留既有账号的状态，并仅在有变化时汇总打印一行。
  // 返回本次同步后的账号总数。
  setAccounts(incoming: ProxyAccount[]): number {
    const nextIds = new Set(incoming.map(a => a.id))
    let added = 0
    let removed = 0
    let updated = 0

    // 删除不在最新列表中的账号
    for (const id of Array.from(this.accounts.keys())) {
      if (!nextIds.has(id)) {
        this.accounts.delete(id)
        this.accountStats.delete(id)
        removed++
      }
    }

    // 新增 / 更新
    for (const account of incoming) {
      const existing = this.accounts.get(account.id)
      if (!existing) {
        // 新账号：复用 addAccount 的初始化逻辑，但抑制其逐条日志（下方汇总打印）
        this.addAccountSilent(account)
        added++
      } else {
        // 已存在：仅刷新凭证等可变字段，保留 isAvailable / 断路器 / 计数等运行时状态
        const suspended = this.isSuspended(account)
        this.accounts.set(account.id, {
          ...existing,
          ...account,
          // suspended 状态以最新数据为准；未 suspended 时维持既有可用性（不强行复位断路器）
          isAvailable: suspended ? false : existing.isAvailable
        })
        updated++
      }
    }

    // 仅在确有变化时打印一行汇总，避免无变化的周期性同步刷屏
    if (added > 0 || removed > 0) {
      console.log(`[AccountPool] Synced accounts: +${added} -${removed} (~${updated} kept), total=${this.accounts.size}`)
    }
    return this.accounts.size
  }

  // addAccount 的静默版本：用于 setAccounts 批量同步，不打印逐条日志
  private addAccountSilent(account: ProxyAccount): void {
    const suspended = this.isSuspended(account)
    this.accounts.set(account.id, {
      ...account,
      isAvailable: !suspended,
      requestCount: 0,
      errorCount: 0,
      lastUsed: 0
    })
    this.accountStats.set(account.id, {
      requests: 0,
      tokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      errors: 0,
      lastUsed: 0,
      avgResponseTime: 0,
      totalResponseTime: 0
    })
  }

  // 获取账号数量
  get size(): number {
    return this.accounts.size
  }

  // 获取可用账号数量
  get availableCount(): number {
    const now = Date.now()
    let count = 0
    for (const account of this.accounts.values()) {
      if (this.isAccountAvailable(account, now)) {
        count++
      }
    }
    return count
  }
}
