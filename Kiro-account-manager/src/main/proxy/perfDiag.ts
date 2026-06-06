/**
 * 性能诊断（perf diagnostics）—— 默认关闭的轻量 instrumentation，用于定位「反代越优化越慢 / 偶发卡顿」。
 *
 * 设计原则：
 *  - 关闭时零成本：所有热路径调用先看 `perfDiag.enabled`（一个布尔判断）再决定是否取时间戳/记录，
 *    关闭状态下不分配对象、不调用 performance.now()、不计算 payload 体积。
 *  - 不改变任何对外行为/计费数值：只读地观测，不参与请求逻辑。
 *  - 数据出口复用现有 /metrics（Prometheus 文本）+ 每 ~30s 一行汇总日志（仅开启时）。
 *
 * 观测维度：
 *  1) event-loop delay（perf_hooks.monitorEventLoopDelay）—— 直接量化「主线程被同步任务阻塞」的程度，
 *     这正是「卡一下一下 / 并发请求一起变慢」的根因信号（某个请求的同步大开销会卡住所有在途 stream）。
 *  2) 每请求分阶段耗时（translate / clone / upstreamTTFB / TTFT / total）+ 关联 payload 体积、history 长度、附件数，
 *     用于验证「input 越大越慢」是否成立、慢在 CPU 段还是网络段。
 *  3) 具体嫌疑计数器（hard-window 阻塞刷新、后台刷新失败、trim 触发、CONTENT_LENGTH 400、save-accounts 序列化耗时）。
 */
import { monitorEventLoopDelay, performance, type IntervalHistogram } from 'perf_hooks'

const NS_PER_MS = 1_000_000

/** 单个耗时指标的累计统计（有界采样，用于 p50/p99）。 */
interface TimingStat {
  count: number
  sumMs: number
  maxMs: number
  /** 有界环形采样（用于近似分位数）；超过容量后覆盖最旧样本。 */
  samples: number[]
  sampleCap: number
  writeIdx: number
}

function newTimingStat(sampleCap = 2000): TimingStat {
  return { count: 0, sumMs: 0, maxMs: 0, samples: [], sampleCap, writeIdx: 0 }
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))
  return sorted[idx]
}

export interface PerfSnapshot {
  enabled: boolean
  uptimeMs: number
  eventLoop: { p50Ms: number; p99Ms: number; maxMs: number; meanMs: number } | null
  timings: Record<string, { count: number; meanMs: number; p50Ms: number; p99Ms: number; maxMs: number }>
  counters: Record<string, number>
}

class PerfDiagnostics {
  /** 热路径用这个布尔做零成本短路。public 只读访问。 */
  enabled = false

  private loopMonitor: IntervalHistogram | null = null
  private timings = new Map<string, TimingStat>()
  private counters = new Map<string, number>()
  private summaryTimer: ReturnType<typeof setInterval> | null = null
  private startedAt = 0

  /** 开启诊断：启动 event-loop 监视 + 周期汇总日志。幂等。 */
  start(summaryIntervalMs = 30_000): void {
    if (this.enabled) return
    this.enabled = true
    this.startedAt = Date.now()
    this.timings.clear()
    this.counters.clear()
    try {
      this.loopMonitor = monitorEventLoopDelay({ resolution: 10 })
      this.loopMonitor.enable()
    } catch (err) {
      // perf_hooks 不可用时降级：仍记录 timing/counter，只是没有 event-loop 维度
      console.warn('[PerfDiag] monitorEventLoopDelay unavailable:', err)
      this.loopMonitor = null
    }
    if (this.summaryTimer) clearInterval(this.summaryTimer)
    this.summaryTimer = setInterval(() => this.logSummary(), summaryIntervalMs)
    this.summaryTimer.unref?.()
    console.log('[PerfDiag] enabled — event-loop + per-request timing instrumentation active')
  }

  /** 关闭诊断并清理资源。幂等。 */
  stop(): void {
    if (!this.enabled) return
    this.enabled = false
    if (this.summaryTimer) { clearInterval(this.summaryTimer); this.summaryTimer = null }
    if (this.loopMonitor) { try { this.loopMonitor.disable() } catch { /* ignore */ } this.loopMonitor = null }
    console.log('[PerfDiag] disabled')
  }

  /** 计数器 +by（关闭时零成本短路）。 */
  incr(name: string, by = 1): void {
    if (!this.enabled) return
    this.counters.set(name, (this.counters.get(name) || 0) + by)
  }

  /** 记录一次耗时样本（毫秒）。关闭时零成本短路。 */
  recordTiming(name: string, ms: number): void {
    if (!this.enabled) return
    if (!(ms >= 0) || !isFinite(ms)) return
    let st = this.timings.get(name)
    if (!st) { st = newTimingStat(); this.timings.set(name, st) }
    st.count++
    st.sumMs += ms
    if (ms > st.maxMs) st.maxMs = ms
    // 有界环形采样：先填满，再覆盖最旧
    if (st.samples.length < st.sampleCap) st.samples.push(ms)
    else { st.samples[st.writeIdx] = ms; st.writeIdx = (st.writeIdx + 1) % st.sampleCap }
  }

  /** 取高精度时间戳（毫秒，亚毫秒精度）。仅在开启时进入此路径。 */
  now(): number {
    return performance.now()
  }

  /** 快照（供 /metrics 与日志汇总用）。 */
  snapshot(): PerfSnapshot {
    const timings: PerfSnapshot['timings'] = {}
    for (const [name, st] of this.timings) {
      const sorted = st.samples.slice().sort((a, b) => a - b)
      timings[name] = {
        count: st.count,
        meanMs: st.count ? st.sumMs / st.count : 0,
        p50Ms: percentile(sorted, 50),
        p99Ms: percentile(sorted, 99),
        maxMs: st.maxMs
      }
    }
    const counters: Record<string, number> = {}
    for (const [k, v] of this.counters) counters[k] = v

    let eventLoop: PerfSnapshot['eventLoop'] = null
    const lm = this.loopMonitor
    if (lm) {
      eventLoop = {
        p50Ms: lm.percentile(50) / NS_PER_MS,
        p99Ms: lm.percentile(99) / NS_PER_MS,
        maxMs: lm.max / NS_PER_MS,
        meanMs: lm.mean / NS_PER_MS
      }
    }
    return {
      enabled: this.enabled,
      uptimeMs: this.enabled ? Date.now() - this.startedAt : 0,
      eventLoop,
      timings,
      counters
    }
  }

  /** 渲染为 Prometheus 文本行（追加到现有 /metrics）。关闭时返回空。 */
  renderMetrics(): string[] {
    if (!this.enabled) return []
    const snap = this.snapshot()
    const lines: string[] = []
    if (snap.eventLoop) {
      lines.push('# HELP kiro_proxy_event_loop_delay_ms Event loop delay (ms) since diagnostics enabled')
      lines.push('# TYPE kiro_proxy_event_loop_delay_ms gauge')
      lines.push(`kiro_proxy_event_loop_delay_ms{quantile="0.5"} ${snap.eventLoop.p50Ms.toFixed(2)}`)
      lines.push(`kiro_proxy_event_loop_delay_ms{quantile="0.99"} ${snap.eventLoop.p99Ms.toFixed(2)}`)
      lines.push(`kiro_proxy_event_loop_delay_ms{quantile="max"} ${snap.eventLoop.maxMs.toFixed(2)}`)
      lines.push(`kiro_proxy_event_loop_delay_ms{quantile="mean"} ${snap.eventLoop.meanMs.toFixed(2)}`)
    }
    const tNames = Object.keys(snap.timings)
    if (tNames.length) {
      lines.push('# HELP kiro_proxy_phase_duration_ms Per-request phase duration (ms)')
      lines.push('# TYPE kiro_proxy_phase_duration_ms summary')
      for (const name of tNames) {
        const t = snap.timings[name]
        lines.push(`kiro_proxy_phase_duration_ms{phase="${name}",quantile="0.5"} ${t.p50Ms.toFixed(2)}`)
        lines.push(`kiro_proxy_phase_duration_ms{phase="${name}",quantile="0.99"} ${t.p99Ms.toFixed(2)}`)
        lines.push(`kiro_proxy_phase_duration_ms{phase="${name}",quantile="max"} ${t.maxMs.toFixed(2)}`)
        lines.push(`kiro_proxy_phase_duration_ms_count{phase="${name}"} ${t.count}`)
      }
    }
    const cNames = Object.keys(snap.counters)
    if (cNames.length) {
      lines.push('# HELP kiro_proxy_perf_events_total Diagnostic event counters')
      lines.push('# TYPE kiro_proxy_perf_events_total counter')
      for (const name of cNames) {
        lines.push(`kiro_proxy_perf_events_total{event="${name}"} ${snap.counters[name]}`)
      }
    }
    return lines
  }

  private logSummary(): void {
    if (!this.enabled) return
    const snap = this.snapshot()
    const el = snap.eventLoop
      ? `loop p50=${snap.eventLoop.p50Ms.toFixed(1)}ms p99=${snap.eventLoop.p99Ms.toFixed(1)}ms max=${snap.eventLoop.maxMs.toFixed(1)}ms`
      : 'loop n/a'
    const phases = Object.entries(snap.timings)
      .map(([n, t]) => `${n}(n=${t.count} p50=${t.p50Ms.toFixed(1)} p99=${t.p99Ms.toFixed(1)} max=${t.maxMs.toFixed(1)})`)
      .join(' ')
    const counters = Object.entries(snap.counters).map(([n, v]) => `${n}=${v}`).join(' ')
    console.log(`[PerfDiag] ${el} | ${phases || 'no-requests'}${counters ? ' | ' + counters : ''}`)
  }
}

/** 进程级单例：proxyServer 与 kiroApi（free functions）共享同一诊断器。 */
export const perfDiag = new PerfDiagnostics()

/** 计数器事件名集中常量，避免拼写漂移。 */
export const PerfEvent = {
  HardWindowBlockingRefresh: 'hard_window_blocking_refresh',
  HardWindowAwaitInflight: 'hard_window_await_inflight',
  BackgroundRefreshFailure: 'background_refresh_failure',
  TrimTriggered: 'trim_triggered',
  ContentLength400: 'content_length_400',
  SaveAccountsUnchanged: 'save_accounts_unchanged',
  SaveAccountsPersisted: 'save_accounts_persisted'
} as const

/** 耗时阶段名集中常量。 */
export const PerfPhase = {
  Translate: 'translate',
  Clone: 'clone',
  Serialize: 'serialize',
  UpstreamTTFB: 'upstream_ttfb',
  TTFT: 'ttft',
  Total: 'total',
  SaveAccountsStringify: 'save_accounts_stringify'
} as const
