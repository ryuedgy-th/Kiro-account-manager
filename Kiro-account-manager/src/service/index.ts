// Headless entrypoint: รัน Kiro reverse-proxy โดยไม่มี Electron (สำหรับ 24/7 / Windows Service)
//   env:
//     KIRO_DATA_DIR       โฟลเดอร์เก็บ cert/log (ดีฟอลต์ ~/.kiro-proxy-service)
//     KIRO_SERVICE_DATA   path ไฟล์ JSON ข้อมูล (config+accounts+stats)
//     KIRO_PROXY_PORT / KIRO_PROXY_HOST   override port/host
import { ProxyServer } from '../main/proxy/proxyServer'
import { refreshTokenByMethod } from '../main/auth/tokenRefresh'
import { proxyLogStore } from '../main/proxy/logger'
import * as fs from 'fs'
import {
  loadServiceData, mapAccounts, ServicePersist, resolveDataFile, resolveDataDir
} from './dataStore'
import type { ProxyConfig, ProxyAccount } from '../main/proxy/types'

// ---- crash guards: อย่าให้ error หลุดตัวเดียวทำ process ตาย (หัวใจของการรัน 24/7) ----
process.on('uncaughtException', (err) => {
  console.error('[service] uncaughtException (จับไว้ ไม่ปิด process):', err)
})
process.on('unhandledRejection', (reason) => {
  console.error('[service] unhandledRejection (จับไว้ ไม่ปิด process):', reason)
})

async function main(): Promise<void> {
  const dataDir = resolveDataDir()
  const dataFile = resolveDataFile()
  // ตั้ง log store ให้เขียนลง dataDir/proxy-logs.json (กัน ENOENT จาก storePath ว่าง)
  try { fs.mkdirSync(dataDir, { recursive: true }); proxyLogStore.initialize(dataDir) } catch { /* ignore */ }
  const data = loadServiceData(dataFile)
  const persist = new ServicePersist(dataFile, data)

  const accounts: ProxyAccount[] = mapAccounts(data.accountData)
  if (accounts.length === 0) {
    console.warn(`[service] ไม่มี active account ในไฟล์ข้อมูล (${dataFile}). proxy จะสตาร์ทแต่ตอบ 503 จนกว่าจะมีบัญชี`)
  }

  const config: Partial<ProxyConfig> = {
    ...(data.proxyConfig || {}),
    enabled: true,
    dataDir,
    port: process.env.KIRO_PROXY_PORT ? Number(process.env.KIRO_PROXY_PORT) : (data.proxyConfig?.port ?? 5580),
    host: process.env.KIRO_PROXY_HOST || data.proxyConfig?.host || '127.0.0.1'
  }

  const server = new ProxyServer(config, {
    onTokenRefresh: async (account) => {
      try {
        const r = await refreshTokenByMethod(
          account.refreshToken || '',
          account.clientId || '',
          account.clientSecret || '',
          account.region || 'us-east-1',
          account.authMethod,
          account.proxyUrl,
          account.machineId
        )
        if (r.success && r.accessToken) {
          return {
            success: true,
            accessToken: r.accessToken,
            refreshToken: r.refreshToken,
            expiresAt: Date.now() + (r.expiresIn || 3600) * 1000
          }
        }
        return { success: false, error: r.error || 'refresh failed' }
      } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : 'unknown' }
      }
    },
    onAccountUpdate: (account) => {
      persist.patchAccountToken(account.id, account.accessToken, account.refreshToken, account.expiresAt)
    },
    onConfigChanged: (cfg) => persist.patchConfig(cfg),
    onCreditsUpdate: (totalCredits) => persist.patchStats({ totalCredits }),
    onTokensUpdate: (inputTokens, outputTokens) => persist.patchStats({ inputTokens, outputTokens }),
    onRequestStatsUpdate: (totalRequests, successRequests, failedRequests) =>
      persist.patchStats({ totalRequests, successRequests, failedRequests }),
    onPoolEmpty: async () => {
      const fresh = mapAccounts(loadServiceData(dataFile).accountData)
      if (fresh.length > 0) server.getAccountPool().setAccounts(fresh)
    },
    onAccountSuspended: (info) =>
      console.warn(`[service] account suspended: ${info.email || info.accountId} (${info.reason})`),
    onError: (err) => console.error('[service] proxy error:', err.message),
    onStatusChange: (running, port) =>
      console.log(`[service] proxy ${running ? 'RUNNING' : 'STOPPED'} :${port}`)
  })

  // pre-seed pool + กู้ stats สะสมข้ามรีสตาร์ท
  if (accounts.length > 0) server.getAccountPool().setAccounts(accounts)
  const s = data.stats || {}
  if (s.totalCredits) server.setTotalCredits(s.totalCredits)
  if (s.inputTokens || s.outputTokens) server.setTotalTokens(s.inputTokens || 0, s.outputTokens || 0)
  if (s.totalRequests || s.successRequests || s.failedRequests)
    server.setRequestStats(s.totalRequests || 0, s.successRequests || 0, s.failedRequests || 0)

  await server.start()
  console.log(`[service] เริ่มแล้ว: http://${config.host}:${config.port} | data=${dataFile} | accounts=${accounts.length}`)

  // ---- graceful shutdown ----
  let shuttingDown = false
  const shutdown = async (sig: string): Promise<void> => {
    if (shuttingDown) return
    shuttingDown = true
    console.log(`[service] ได้รับ ${sig}, กำลังปิดอย่างนุ่มนวล...`)
    try { await server.stop() } catch (e) { console.error('[service] stop error:', e) }
    try { await persist.flush() } catch { /* ignore */ }
    process.exit(0)
  }
  process.on('SIGINT', () => void shutdown('SIGINT'))
  process.on('SIGTERM', () => void shutdown('SIGTERM'))
}

main().catch((err) => {
  console.error('[service] สตาร์ทไม่สำเร็จ:', err)
  process.exit(1)
})
