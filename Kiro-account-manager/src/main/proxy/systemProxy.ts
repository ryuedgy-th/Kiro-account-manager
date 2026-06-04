// 系统代理检测（Windows 注册表 / macOS scutil）+ 安全 ProxyAgent 工厂
// 含 SOCKS5/SOCKS4 代理支持（通过 socks 包 + undici Agent.connect 钩子）

import { ProxyAgent, Agent, type Dispatcher } from 'undici'
import * as tls from 'tls'

let _cachedSystemProxy: string | null = null
let _systemProxyCacheTime = 0
const SYSTEM_PROXY_CACHE_TTL = 30_000 // 30秒缓存

// ============ 出站连接池（keep-alive 复用）============
// 背景：旧实现每次 API 调用都 new ProxyAgent / new Agent，等于丢弃所有已建立的 keep-alive
// 连接，每个请求都要重做 TCP + TLS 握手（到 us-east-1 往返 ~100-300ms）。kiro-cli 走的是
// 复用连接，所以反代会显得明显更慢。这里把出站 dispatcher 池化并按身份缓存：
//   - 直连（无代理）：单例 _directAgent，跨请求复用同一连接池
//   - 代理：按 proxyUrl 缓存，保证「同一账号 ↔ 同一 IP」绑定不变（不同 proxyUrl 各自独立池）
//
// keep-alive 调参说明（贴近普通 HTTP 客户端的常见默认，不制造异常指纹）：
//   keepAliveTimeout    空闲连接保活 30s（undici 默认仅 4s，导致稍有间隔就重新握手）
//   keepAliveMaxTimeout 单连接最长存活 10min 后强制轮换（避免 NAT/LB 静默断连后用到死连接）
//   connections         每个 origin 最多 64 条连接，足够并发又不至于过量
//   pipelining          1（HTTP/1.1，不跨请求复用同一管道，避免多账号请求互相串扰/指纹异常）
const POOL_OPTS = {
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 600_000,
  connections: 64,
  pipelining: 1
} as const

let _directAgent: Agent | null = null
const _proxyAgentCache = new Map<string, Dispatcher>()

/**
 * 直连（无代理）共享 dispatcher——跨请求复用 keep-alive 连接池。
 * 不调 setGlobalDispatcher：保持全局 dispatcher 不变，避免影响注册/邮件/webTools 等其他出站逻辑，
 * 仅 Kiro API 调用显式传入此 agent。
 */
export function getDirectPoolAgent(): Agent {
  if (!_directAgent) _directAgent = new Agent(POOL_OPTS)
  return _directAgent
}

/** 释放所有池化出站连接（供反代停止/重置时清理，防止句柄泄漏）。 */
export function destroyOutboundPools(): void {
  try { _directAgent?.close().catch(() => undefined) } catch { /* ignore */ }
  _directAgent = null
  for (const agent of _proxyAgentCache.values()) {
    try { (agent as { close?: () => Promise<void> }).close?.()?.catch(() => undefined) } catch { /* ignore */ }
  }
  _proxyAgentCache.clear()
}


/**
 * 检查 URL 是否为 undici ProxyAgent 支持的协议（http / https）
 */
function isHttpLikeProxyUrl(url: string): boolean {
  try {
    const u = new URL(url)
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch {
    return false
  }
}

/**
 * 解析 Windows ProxyServer 注册表值，仅返回 undici 可用的 http(s) 代理 URL
 *
 * 可能的格式：
 *   1) "host:port"            — 单一代理，应用于所有协议
 *   2) "http=host:port;https=host:port;ftp=host:port;socks=host:port"
 *                              — 按协议分别配置
 *   3) "scheme://host:port"   — 已带 scheme（http / https / socks5 ...）
 *
 * 不支持 socks/socks4/socks5/pac 等非 http(s) 协议，遇到时返回 null（回退直连）
 */
function parseWindowsProxyServer(raw: string): string | null {
  const trimmed = raw.trim()
  if (!trimmed) return null

  // 形如 "http=host:port;https=host:port;socks=host:port" 的多协议格式
  if (trimmed.includes('=')) {
    const map = new Map<string, string>()
    for (const seg of trimmed.split(';')) {
      const eq = seg.indexOf('=')
      if (eq > 0) {
        const k = seg.slice(0, eq).trim().toLowerCase()
        const v = seg.slice(eq + 1).trim()
        if (k && v) map.set(k, v)
      }
    }
    const https = map.get('https')
    if (https) return `http://${https}`
    const http = map.get('http')
    if (http) return `http://${http}`
    return null
  }

  // 已带 scheme
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) {
    return isHttpLikeProxyUrl(trimmed) ? trimmed : null
  }

  // 裸的 host:port，按 http 处理
  return `http://${trimmed}`
}

export function getSystemProxy(): string | null {
  const now = Date.now()
  if (_systemProxyCacheTime > 0 && now - _systemProxyCacheTime < SYSTEM_PROXY_CACHE_TTL) {
    return _cachedSystemProxy
  }
  try {
    if (process.platform === 'win32') {
      const { execSync } = require('child_process')
      const result = execSync(
        'reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" /v ProxyEnable',
        { encoding: 'utf8', timeout: 3000, windowsHide: true }
      )
      if (result.includes('0x1')) {
        const serverResult = execSync(
          'reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" /v ProxyServer',
          { encoding: 'utf8', timeout: 3000, windowsHide: true }
        )
        const match = serverResult.match(/ProxyServer\s+REG_SZ\s+(.+)/)
        if (match) {
          const parsed = parseWindowsProxyServer(match[1])
          _cachedSystemProxy = parsed
          _systemProxyCacheTime = now
          return _cachedSystemProxy
        }
      }
    } else if (process.platform === 'darwin') {
      const { execSync } = require('child_process')
      const result = execSync('scutil --proxy', { encoding: 'utf8', timeout: 3000 })
      // 优先 HTTPS 代理，回退到 HTTP 代理（仅 undici 支持的 http/https 协议）
      const httpsEnabled = /HTTPSEnable\s*:\s*1/.test(result)
      if (httpsEnabled) {
        const hostMatch = result.match(/HTTPSProxy\s*:\s*(\S+)/)
        const portMatch = result.match(/HTTPSPort\s*:\s*(\d+)/)
        if (hostMatch) {
          const proxy = `http://${hostMatch[1]}${portMatch ? ':' + portMatch[1] : ''}`
          _cachedSystemProxy = proxy
          _systemProxyCacheTime = now
          return _cachedSystemProxy
        }
      }
      const httpEnabled = /HTTPEnable\s*:\s*1/.test(result)
      if (httpEnabled) {
        const hostMatch = result.match(/HTTPProxy\s*:\s*(\S+)/)
        const portMatch = result.match(/HTTPPort\s*:\s*(\d+)/)
        if (hostMatch) {
          const proxy = `http://${hostMatch[1]}${portMatch ? ':' + portMatch[1] : ''}`
          _cachedSystemProxy = proxy
          _systemProxyCacheTime = now
          return _cachedSystemProxy
        }
      }
      // macOS 仅配 SOCKS 时 undici 不支持，静默回退直连（safeCreateProxyAgent 也会兜底）
    }
  } catch { /* 检测失败静默回退直连 */ }
  _cachedSystemProxy = null
  _systemProxyCacheTime = now
  return null
}

/**
 * 安全地创建 undici Dispatcher
 *
 * 支持协议：
 *   - http: / https: → undici 原生 ProxyAgent
 *   - socks5: / socks4: → 通过 socks 包 + undici Agent 的 connect 钩子实现 SOCKS 隧道
 *
 * URL 无效或协议无法支持时返回 undefined，让调用方回退直连，
 * 而不会让异常向上传播阻塞业务流程。
 */
export function safeCreateProxyAgent(
  proxyUrl: string | null | undefined
): Dispatcher | undefined {
  if (!proxyUrl) return undefined

  // 校验 URL
  let u: URL
  try {
    u = new URL(proxyUrl)
  } catch {
    console.warn(`[Proxy] 代理 URL 无效: ${proxyUrl}`)
    return undefined
  }

  const protocol = u.protocol

  // http / https 走原生 ProxyAgent（带 keep-alive 池化参数）
  // 注意：这里是「纯工厂」——每次调用都返回新 dispatcher，不跨调用缓存。
  // 注册流程（registrar.ts）依赖这一点：每次注册用全新连接/出口 IP，避免多账号被关联。
  // 需要跨请求复用连接的是 Kiro API 服务路径，那条路径走 getCachedProxyAgent（见下）。
  if (protocol === 'http:' || protocol === 'https:') {
    try {
      return new ProxyAgent({ uri: proxyUrl, requestTls: { rejectUnauthorized: false }, ...POOL_OPTS })
    } catch (err) {
      console.warn(`[Proxy] 创建 HTTP ProxyAgent 失败，回退直连: ${proxyUrl}`, err)
      return undefined
    }
  }

  // SOCKS 走自定义 connect
  if (protocol === 'socks5:' || protocol === 'socks5h:' || protocol === 'socks4:' || protocol === 'socks4a:') {
    try {
      return createSocksDispatcher(u)
    } catch (err) {
      console.warn(`[Proxy] 创建 SOCKS Agent 失败，回退直连: ${proxyUrl}`, err)
      return undefined
    }
  }

  console.warn(`[Proxy] 忽略不支持的代理协议 (仅支持 http/https/socks5/socks4): ${proxyUrl}`)
  return undefined
}

/**
 * Kiro API 服务路径专用：按 proxyUrl 缓存并复用池化 dispatcher。
 * 与 safeCreateProxyAgent（纯工厂）区分：
 *   - 这里跨请求复用同一连接池，让「同一账号/同一 proxyUrl」的流量稳定从同一出口 IP 走，
 *     既复用 keep-alive（快），也符合「N 账号 ↔ 1 IP」分桶设计（更像真实客户端，不增加被识别风险）。
 *   - 不同 proxyUrl 各自独立缓存，绑定关系互不串扰。
 * 仅供 kiroApi.getNetworkAgent 调用；注册流程不要用本函数（需要每次新 IP）。
 */
export function getCachedProxyAgent(
  proxyUrl: string | null | undefined
): Dispatcher | undefined {
  if (!proxyUrl) return undefined
  const cached = _proxyAgentCache.get(proxyUrl)
  if (cached) return cached
  const agent = safeCreateProxyAgent(proxyUrl)
  if (agent) _proxyAgentCache.set(proxyUrl, agent)
  return agent
}

/**
 * 通过 undici Agent 的 connect 钩子实现 SOCKS5/4 隧道
 * 流程：socks.createConnection 建立 TCP 隧道 → 如目标是 https 再 TLS 升级 → 把 socket 交给 undici
 */
function createSocksDispatcher(u: URL): Agent {
  const isSocks5 = u.protocol === 'socks5:' || u.protocol === 'socks5h:'
  const type: 4 | 5 = isSocks5 ? 5 : 4
  const proxyHost = u.hostname
  const proxyPort = Number(u.port) || 1080
  const userId = u.username ? decodeURIComponent(u.username) : undefined
  const password = u.password ? decodeURIComponent(u.password) : undefined

  // undici Agent.connect callback 的类型签名是 (err: Error, socket: null) | (err: null, socket: Socket)
  // 用宽松 any 包装避免严格类型不匹配，运行时行为完全正确
  return new Agent({
    ...POOL_OPTS,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    connect: ((options: any, callback: any): void => {
      const targetHost = options.hostname || options.host || ''
      const targetPort = Number(options.port) || (options.protocol === 'https:' ? 443 : 80)

      // 动态导入 socks 库
      let SocksClient: typeof import('socks').SocksClient
      try {
        SocksClient = require('socks').SocksClient
      } catch (err) {
        callback(err as Error, null)
        return
      }

      void SocksClient.createConnection({
        proxy: { host: proxyHost, port: proxyPort, type, userId, password },
        command: 'connect',
        destination: { host: targetHost, port: targetPort }
      })
        .then(({ socket }) => {
          // HTTPS 需要 TLS 升级
          if (options.protocol === 'https:') {
            const tlsSocket = tls.connect({
              socket,
              servername: options.servername || targetHost,
              rejectUnauthorized: options.rejectUnauthorized ?? false
            })
            tlsSocket.once('secureConnect', () => callback(null, tlsSocket))
            tlsSocket.once('error', (err) => callback(err, null))
          } else {
            callback(null, socket)
          }
        })
        .catch((err: Error) => callback(err, null))
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any
  })
}
