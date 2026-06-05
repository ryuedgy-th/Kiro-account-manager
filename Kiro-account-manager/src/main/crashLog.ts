// 24/7 稳定性诊断：进程级崩溃 / 退出原因日志
//
// 背景：原来主进程的 console.error 只写内存里的 proxyLogStore，进程一旦"凭空消失"
// （renderer/GPU 崩溃、native 段错误、OOM、被信号杀死、OS 重启）日志根本没机会落盘，
// 导致"应用自己退出了"无从排查。
//
// 本模块做两件事：
//   1) writeCrashLog()：同步 append 到磁盘文件，崩溃路径上也能保证写入（不依赖事件循环）。
//   2) installCrashDiagnostics()：挂上 Electron/Node 的所有崩溃 & 退出事件，集中记录原因。
//
// 设计取舍：用同步 fs.appendFileSync——崩溃处理器里异步写没有保证能完成。日常每条日志
// 的同步写开销可忽略（仅崩溃/退出/少量生命周期事件，不在请求热路径上）。

import * as fs from 'fs'
import * as path from 'path'
import { app, crashReporter, powerMonitor, type BrowserWindow } from 'electron'

let logFilePath = ''
let initialized = false

const MAX_LOG_BYTES = 5 * 1024 * 1024 // 5MB 后轮转，保留一个 .old

/** 轮转：超过上限就把当前文件改名为 .old（覆盖旧的），重新开一个 */
function rotateIfNeeded(): void {
  try {
    const st = fs.statSync(logFilePath)
    if (st.size > MAX_LOG_BYTES) {
      const oldPath = `${logFilePath}.old`
      try { fs.rmSync(oldPath, { force: true }) } catch { /* ignore */ }
      fs.renameSync(logFilePath, oldPath)
    }
  } catch { /* 文件不存在等 → 忽略 */ }
}

/** 同步写一条崩溃/诊断日志（崩溃路径安全）。format: ISO\t[TAG]\tmessage */
export function writeCrashLog(tag: string, message: string, detail?: unknown): void {
  if (!logFilePath) return
  try {
    rotateIfNeeded()
    let line = `${new Date().toISOString()}\t[${tag}]\t${message}`
    if (detail !== undefined) {
      let d: string
      try {
        d = detail instanceof Error
          ? `${detail.name}: ${detail.message}\n${detail.stack || ''}`
          : typeof detail === 'string' ? detail : JSON.stringify(detail)
      } catch {
        d = String(detail)
      }
      line += `\n  ${d.replace(/\n/g, '\n  ')}`
    }
    fs.appendFileSync(logFilePath, line + '\n', 'utf8')
  } catch { /* 写日志失败不能再抛，否则递归崩溃 */ }
}

export function getCrashLogPath(): string {
  return logFilePath
}

/**
 * 安装全部崩溃 & 退出诊断。应在 app.whenReady 之前尽早调用（这样早期崩溃也能记录）。
 * getMainWindow 用于在 ready 后挂 renderer/GPU 崩溃监听（窗口创建后调用 attachWindowDiagnostics）。
 */
export function installCrashDiagnostics(): void {
  if (initialized) return
  initialized = true

  try {
    logFilePath = path.join(app.getPath('logs'), 'main-crash.log')
  } catch {
    // app.getPath('logs') 在极早期可能不可用 → 退回 userData
    try { logFilePath = path.join(app.getPath('userData'), 'main-crash.log') } catch { logFilePath = '' }
  }

  // 进程级本地崩溃转储（native 段错误的 minidump）。仅本地收集，不上传。
  try {
    crashReporter.start({ uploadToServer: false })
  } catch (e) {
    writeCrashLog('CRASH-REPORTER', 'failed to start crashReporter', e)
  }

  writeCrashLog('LIFECYCLE', `process started pid=${process.pid} electron=${process.versions.electron} node=${process.versions.node}`)

  // ---- Node 进程级 ----
  // 注意：index.ts 已有 uncaughtException/unhandledRejection 的"保活"处理器（只记 console，不退出）。
  // 这里额外做"落盘"，两者并存（同一事件多个监听器都会被调用）。
  process.on('uncaughtException', (err) => {
    writeCrashLog('UNCAUGHT-EXCEPTION', 'kept alive', err)
  })
  process.on('unhandledRejection', (reason) => {
    writeCrashLog('UNHANDLED-REJECTION', 'kept alive', reason)
  })
  process.on('warning', (w) => {
    writeCrashLog('NODE-WARNING', w.message, w.stack)
  })
  // 退出信号（被 OS / 任务管理器 / kill 杀死）
  for (const sig of ['SIGTERM', 'SIGINT', 'SIGHUP'] as const) {
    try {
      process.on(sig, () => {
        writeCrashLog('SIGNAL', `received ${sig}, exiting`)
        app.quit()
      })
    } catch { /* 某些平台不支持某信号 */ }
  }
  process.on('exit', (code) => {
    writeCrashLog('PROCESS-EXIT', `code=${code}`)
  })

  // ---- Electron app 级 ----
  app.on('before-quit', () => writeCrashLog('LIFECYCLE', 'before-quit'))
  app.on('will-quit', () => writeCrashLog('LIFECYCLE', 'will-quit'))
  app.on('quit', (_e, code) => writeCrashLog('LIFECYCLE', `quit exitCode=${code}`))
  // GPU 进程崩溃（不一定杀主进程，但常导致界面异常）
  app.on('gpu-process-crashed' as 'gpu-info-update', ((_e: unknown, killed: boolean) => {
    writeCrashLog('GPU-CRASH', `gpu process crashed killed=${killed}`)
  }) as () => void)
  // 任意 child 进程（GPU/utility/pepper 等）退出
  app.on('child-process-gone', (_e, details) => {
    writeCrashLog('CHILD-PROCESS-GONE', `type=${details.type} reason=${details.reason} exitCode=${details.exitCode} name=${details.name || ''}`)
  })

  // ---- 电源事件（macOS 睡眠会让 proxy "看起来死了"）----
  try {
    powerMonitor.on('suspend', () => writeCrashLog('POWER', 'system suspend (sleep)'))
    powerMonitor.on('resume', () => writeCrashLog('POWER', 'system resume (wake)'))
    powerMonitor.on('shutdown', () => writeCrashLog('POWER', 'system shutdown'))
  } catch { /* powerMonitor 需 ready 后才完全可用，包一层防御 */ }
}

/** 窗口创建后调用：挂 renderer 崩溃 / 无响应监听。返回的回调可在 onRendererGone 里触发重建。 */
export function attachWindowDiagnostics(
  win: BrowserWindow,
  onRendererGone?: (reason: string) => void
): void {
  const wc = win.webContents
  wc.on('render-process-gone', (_e, details) => {
    writeCrashLog('RENDERER-GONE', `reason=${details.reason} exitCode=${details.exitCode}`)
    onRendererGone?.(details.reason)
  })
  wc.on('unresponsive', () => writeCrashLog('RENDERER', 'unresponsive'))
  wc.on('responsive', () => writeCrashLog('RENDERER', 'responsive again'))
}
