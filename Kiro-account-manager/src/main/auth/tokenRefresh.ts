// Electron-free token refresh สำหรับ headless service
// ทำซ้ำ logic จาก index.ts (refreshOidcToken/refreshSocialToken/refreshTokenByMethod)
// แต่ไม่ผูก electron / machineId / kproxy — proxy ออกนอกใช้ undici + (override → env → system → direct)
import { fetch as undiciFetch, type RequestInit as UndiciRequestInit } from 'undici'
import { getSystemProxy, safeCreateProxyAgent } from '../proxy/systemProxy'

export interface OidcRefreshResult {
  success: boolean
  accessToken?: string
  refreshToken?: string
  expiresIn?: number
  error?: string
}

const KIRO_AUTH_ENDPOINT = 'https://prod.us-east-1.auth.desktop.kiro.dev'
const KIRO_VERSION = '0.12.155'

// ============ refresh 网络韧性（与请求路径 isTransientNetworkError 同源思路）============
// 背景：批量刷新（gakBatchRefresh）会同时发起 N 个 TLS 握手；连接池冷启动时 AWS/中间层偶尔
// 在建连阶段提前关 socket（fetch failed / UND_ERR_SOCKET / ECONNRESET）。单账号手动刷新只 1 条
// 请求、无争用 → 几乎不触发；批量则每秒成簇失败。原实现无 timeout、无 retry，故失败即永久失败。
// 这里给 refresh 这一类「幂等的建连阶段请求」加：单次超时 + 对瞬时网络错误的有限重试（指数退避 + 抖动）。
const REFRESH_TIMEOUT_MS = 15_000
const REFRESH_MAX_ATTEMPTS = 3
const REFRESH_BASE_BACKOFF_MS = 400

/**
 * 判断是否为「连接阶段」的瞬时网络错误——值得重试，而非把账号判死。
 * 仅匹配网络层错误（undici socket 提前关 / 连接重置 / DNS 抖动 / 超时），
 * 绝不匹配 HTTP 状态错误（401/403/invalid_grant 等应用层拒绝由调用方据 HTTP 文本处理）。
 */
export function isTransientRefreshError(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const codes = ['UND_ERR_SOCKET', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT', 'ECONNRESET', 'EPIPE', 'ECONNREFUSED', 'ETIMEDOUT', 'EAI_AGAIN', 'ENOTFOUND']
  const cause = (error as { cause?: { code?: string; message?: string } }).cause
  if (cause?.code && codes.includes(cause.code)) return true
  const hay = `${error.message} ${cause?.message || ''}`
  return /other side closed|socket hang up|terminated|fetch failed|network|timeout|aborted|ECONNRESET|EPIPE|UND_ERR_SOCKET/i.test(hay)
}

/** 给单次 fetch 套一个 AbortController 超时——避免某条连接卡死拖垮整批刷新。 */
async function fetchWithTimeout(url: string, options: UndiciRequestInit, overrideProxyUrl: string | undefined, timeoutMs: number): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetchWithProxy(url, { ...options, signal: controller.signal }, overrideProxyUrl)
  } finally {
    clearTimeout(timer)
  }
}

/**
 * 对一次 refresh 调用做「超时 + 瞬时网络错误有限重试」。
 * - 仅瞬时网络错误重试；HTTP 错误（由 fn 内部读 response.ok 返回 success:false）不会进入这里抛错路径。
 * - 指数退避 + 抖动，避免「重试也成簇」再次自造突发。
 */
export async function withRefreshRetry<T>(fn: (timeoutMs: number) => Promise<T>): Promise<T> {
  let lastError: unknown
  for (let attempt = 1; attempt <= REFRESH_MAX_ATTEMPTS; attempt++) {
    try {
      return await fn(REFRESH_TIMEOUT_MS)
    } catch (error) {
      lastError = error
      if (attempt >= REFRESH_MAX_ATTEMPTS || !isTransientRefreshError(error)) throw error
      const backoff = REFRESH_BASE_BACKOFF_MS * 2 ** (attempt - 1) + Math.floor(Math.random() * 300)
      await new Promise((r) => setTimeout(r, backoff))
    }
  }
  throw lastError
}

function kiroUserAgent(machineId?: string): string {
  return machineId ? `aws-toolkit-vscode/KiroIDE-${KIRO_VERSION}-${machineId}` : `aws-toolkit-vscode/KiroIDE-${KIRO_VERSION}`
}

// proxy 优先级：账号绑定代理 → 环境变量 → 系统代理 → 直连
async function fetchWithProxy(url: string, options: UndiciRequestInit, overrideProxyUrl?: string): Promise<Response> {
  const candidate =
    overrideProxyUrl ||
    process.env.HTTPS_PROXY || process.env.https_proxy ||
    process.env.HTTP_PROXY || process.env.http_proxy ||
    getSystemProxy()
  const agent = candidate ? safeCreateProxyAgent(candidate) : undefined
  return await undiciFetch(url, agent ? { ...options, dispatcher: agent } : options) as unknown as Response
}

// IdC / BuilderId: OIDC refresh
export async function refreshOidcToken(
  refreshToken: string,
  clientId: string,
  clientSecret: string,
  region = 'us-east-1',
  proxyUrl?: string
): Promise<OidcRefreshResult> {
  const url = `https://oidc.${region}.amazonaws.com/token`
  try {
    const response = await withRefreshRetry((timeoutMs) => fetchWithTimeout(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clientId, clientSecret, refreshToken, grantType: 'refresh_token' })
    }, proxyUrl, timeoutMs))
    if (!response.ok) {
      return { success: false, error: `HTTP ${response.status}: ${await response.text()}` }
    }
    const data = await response.json() as { accessToken?: string; refreshToken?: string; expiresIn?: number }
    return { success: true, accessToken: data.accessToken, refreshToken: data.refreshToken || refreshToken, expiresIn: data.expiresIn }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

// Social (GitHub/Google): Kiro Auth Service refresh
export async function refreshSocialToken(refreshToken: string, proxyUrl?: string, machineId?: string): Promise<OidcRefreshResult> {
  const url = `${KIRO_AUTH_ENDPOINT}/refreshToken`
  try {
    const response = await withRefreshRetry((timeoutMs) => fetchWithTimeout(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'User-Agent': kiroUserAgent(machineId) },
      body: JSON.stringify({ refreshToken })
    }, proxyUrl, timeoutMs))
    if (!response.ok) {
      return { success: false, error: `HTTP ${response.status}: ${await response.text()}` }
    }
    const data = await response.json() as { accessToken?: string; refreshToken?: string; expiresIn?: number }
    return { success: true, accessToken: data.accessToken, refreshToken: data.refreshToken || refreshToken, expiresIn: data.expiresIn }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

// 通用：根据 authMethod 选择刷新方式
export async function refreshTokenByMethod(
  token: string,
  clientId: string,
  clientSecret: string,
  region = 'us-east-1',
  authMethod?: string,
  proxyUrl?: string,
  machineId?: string
): Promise<OidcRefreshResult> {
  if (authMethod === 'social') return refreshSocialToken(token, proxyUrl, machineId)
  return refreshOidcToken(token, clientId, clientSecret, region, proxyUrl)
}
