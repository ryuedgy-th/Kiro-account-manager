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
    const response = await fetchWithProxy(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clientId, clientSecret, refreshToken, grantType: 'refresh_token' })
    }, proxyUrl)
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
    const response = await fetchWithProxy(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'User-Agent': kiroUserAgent(machineId) },
      body: JSON.stringify({ refreshToken })
    }, proxyUrl)
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
