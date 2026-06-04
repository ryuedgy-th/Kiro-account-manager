// 客户门户（Customer Portal）认证与业务逻辑
//
// 设计目标：
//   - 仅用 node:crypto 内置原语，不引入新依赖（scrypt 派生密码、HMAC 签名会话）
//   - 业务函数尽量保持纯函数（输入 config + 参数 → 输出结果/变更），便于 esbuild 单测
//   - 实际的 HTTP 路由、读 body、持久化由 proxyServer 调用这里的函数完成
//
// 安全要点：
//   - 密码用 scryptSync 派生，存 salt+hash，不存明文；校验用 timingSafeEqual 防时序旁路
//   - 会话 token = base64url(payload).hmac，HMAC-SHA256 签名 + 过期校验，无状态
//   - 客户只能看到/操作自己名下的 Key（按 customerId 严格过滤）

import * as crypto from 'crypto'
import { promisify } from 'util'
import { v4 as uuidv4 } from 'uuid'
import type { ProxyConfig, Customer, ApiKey, PortalInvite } from './types'

const SCRYPT_KEYLEN = 64
const SCRYPT_SALT_BYTES = 16
const DEFAULT_SESSION_TTL_HOURS = 24
const DEFAULT_MAX_KEYS = 5
/** 密码最大长度，防止超长输入放大 scrypt 计算成本（DoS） */
export const MAX_PASSWORD_LENGTH = 256

const scryptAsync = promisify(crypto.scrypt) as (password: string, salt: string, keylen: number) => Promise<Buffer>

// ============ 密码哈希 ============

/** 生成 salt + scrypt 哈希（均为 hex 字符串）。异步，避免阻塞事件循环。 */
export async function hashPassword(password: string): Promise<{ salt: string; hash: string }> {
  const salt = crypto.randomBytes(SCRYPT_SALT_BYTES).toString('hex')
  const derived = await scryptAsync(password, salt, SCRYPT_KEYLEN)
  return { salt, hash: derived.toString('hex') }
}

/** 校验密码（timingSafeEqual 防时序攻击；长度/格式异常时安全返回 false）。异步。 */
export async function verifyPassword(password: string, salt: string, expectedHash: string): Promise<boolean> {
  if (!password || !salt || !expectedHash) return false
  if (password.length > MAX_PASSWORD_LENGTH) return false
  let actual: Buffer
  try {
    actual = await scryptAsync(password, salt, SCRYPT_KEYLEN)
  } catch {
    return false
  }
  let expected: Buffer
  try {
    expected = Buffer.from(expectedHash, 'hex')
  } catch {
    return false
  }
  if (actual.length !== expected.length) {
    // 仍走一次比较保持时序一致
    try { crypto.timingSafeEqual(actual, actual) } catch { /* ignore */ }
    return false
  }
  try {
    return crypto.timingSafeEqual(actual, expected)
  } catch {
    return false
  }
}

// ============ 会话 token（无状态 HMAC） ============

interface SessionPayload {
  cid: string   // customer id
  exp: number   // 过期时间戳（毫秒）
}

function b64urlEncode(buf: Buffer): string {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function b64urlDecode(s: string): Buffer {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4))
  return Buffer.from(s.replace(/-/g, '+').replace(/_/g, '/') + pad, 'base64')
}

function hmac(secret: string, data: string): Buffer {
  return crypto.createHmac('sha256', secret).update(data).digest()
}

/** 签发会话 token：base64url(payload).base64url(hmac) */
export function signSession(secret: string, customerId: string, ttlHours: number, now: number): string {
  const exp = now + Math.max(1, ttlHours) * 3600 * 1000
  const payload: SessionPayload = { cid: customerId, exp }
  const payloadB64 = b64urlEncode(Buffer.from(JSON.stringify(payload), 'utf8'))
  const sig = b64urlEncode(hmac(secret, payloadB64))
  return `${payloadB64}.${sig}`
}

/** 校验会话 token，返回 customerId 或 null（签名错误/过期/格式错误均返回 null） */
export function verifySession(secret: string, token: string, now: number): string | null {
  if (!token || typeof token !== 'string') return null
  const dot = token.indexOf('.')
  if (dot <= 0) return null
  const payloadB64 = token.slice(0, dot)
  const sigB64 = token.slice(dot + 1)
  if (!payloadB64 || !sigB64) return null

  // 常数时间比较签名
  const expectedSig = hmac(secret, payloadB64)
  let providedSig: Buffer
  try {
    providedSig = b64urlDecode(sigB64)
  } catch {
    return null
  }
  if (providedSig.length !== expectedSig.length) return null
  if (!crypto.timingSafeEqual(providedSig, expectedSig)) return null

  // 解析 payload + 过期校验
  let payload: SessionPayload
  try {
    payload = JSON.parse(b64urlDecode(payloadB64).toString('utf8'))
  } catch {
    return null
  }
  if (!payload || typeof payload.cid !== 'string' || typeof payload.exp !== 'number') return null
  if (now >= payload.exp) return null
  return payload.cid
}

// ============ 客户与 Key 业务逻辑（纯函数，作用于 config 副本） ============

export function normalizeEmail(email: string): string {
  return (email || '').trim().toLowerCase()
}

export function findCustomerByEmail(config: ProxyConfig, email: string): Customer | undefined {
  const norm = normalizeEmail(email)
  return (config.customers || []).find(c => normalizeEmail(c.email) === norm)
}

export function findCustomerById(config: ProxyConfig, id: string): Customer | undefined {
  return (config.customers || []).find(c => c.id === id)
}

/** 客户名下的 Key（严格按 customerId 过滤，保证隔离） */
export function customerKeys(config: ProxyConfig, customerId: string): ApiKey[] {
  return (config.apiKeys || []).filter(k => k.customerId === customerId)
}

export function maxKeysFor(config: ProxyConfig, customer: Customer): number {
  if (typeof customer.maxKeys === 'number' && customer.maxKeys > 0) return customer.maxKeys
  if (typeof config.portalDefaultMaxKeys === 'number' && config.portalDefaultMaxKeys > 0) {
    return config.portalDefaultMaxKeys
  }
  return DEFAULT_MAX_KEYS
}

export function sessionTtlHours(config: ProxyConfig): number {
  return config.portalSessionTtlHours && config.portalSessionTtlHours > 0
    ? config.portalSessionTtlHours
    : DEFAULT_SESSION_TTL_HOURS
}

/** 生成一个新的 sk- 风格 API Key 字符串 */
export function generateApiKeyString(): string {
  return 'sk-' + crypto.randomBytes(24).toString('hex')
}

/** 创建一个新 ApiKey 对象（空用量），归属指定客户 */
export function buildApiKey(name: string, customerId: string, now: number): ApiKey {
  return {
    id: uuidv4(),
    name: name || 'key',
    key: generateApiKeyString(),
    format: 'sk',
    enabled: true,
    createdAt: now,
    customerId,
    usage: {
      totalRequests: 0,
      totalCredits: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      daily: {}
    }
  }
}

/** 创建客户对象（密码已哈希）。异步（scrypt）。 */
export async function buildCustomer(email: string, password: string, opts: { name?: string; creditBalance?: number; maxKeys?: number }, now: number): Promise<Customer> {
  const { salt, hash } = await hashPassword(password)
  return {
    id: uuidv4(),
    email: normalizeEmail(email),
    name: opts.name,
    passwordSalt: salt,
    passwordHash: hash,
    enabled: true,
    createdAt: now,
    creditBalance: opts.creditBalance ?? 0,
    totalToppedUp: opts.creditBalance && opts.creditBalance > 0 ? opts.creditBalance : 0,
    maxKeys: opts.maxKeys,
    topupHistory: opts.creditBalance && opts.creditBalance > 0
      ? [{ timestamp: now, amount: opts.creditBalance, note: 'initial', by: 'admin' }]
      : []
  }
}

/**
 * 创建 Google 登录（passwordless）客户对象。无 salt/hash，绑定 googleSub。
 * 用于 invite 首次注册：email 已由 invite 绑定并与 Google 账号校验一致。
 */
export function buildGoogleCustomer(email: string, googleSub: string, opts: { name?: string; creditBalance?: number; maxKeys?: number }, now: number): Customer {
  return {
    id: uuidv4(),
    email: normalizeEmail(email),
    name: opts.name,
    googleSub,
    enabled: true,
    createdAt: now,
    creditBalance: opts.creditBalance ?? 0,
    totalToppedUp: opts.creditBalance && opts.creditBalance > 0 ? opts.creditBalance : 0,
    maxKeys: opts.maxKeys,
    topupHistory: opts.creditBalance && opts.creditBalance > 0
      ? [{ timestamp: now, amount: opts.creditBalance, note: 'initial (invite)', by: 'admin' }]
      : []
  }
}

// ============ 邀请码（invite-only 注册） ============

/** 生成不可猜测的邀请码（base64url, 24 字节 ≈ 32 字符）。 */
export function generateInviteCode(): string {
  return b64urlEncode(crypto.randomBytes(24))
}

/** 按 code 查找邀请（精确匹配）。 */
export function findInviteByCode(config: ProxyConfig, code: string): PortalInvite | undefined {
  if (!code) return undefined
  return (config.portalInvites || []).find(i => i.code === code)
}

/**
 * 校验邀请是否可用于指定 email 注册。
 * 返回 { ok, reason }：reason 仅用于服务端日志，不要原样回传给客户端（避免泄露邀请状态）。
 */
export function validateInvite(invite: PortalInvite | undefined, email: string, now: number): { ok: boolean; reason?: string } {
  if (!invite) return { ok: false, reason: 'not_found' }
  if (invite.usedAt) return { ok: false, reason: 'already_used' }
  if (invite.expiresAt && now > invite.expiresAt) return { ok: false, reason: 'expired' }
  if (normalizeEmail(invite.email) !== normalizeEmail(email)) return { ok: false, reason: 'email_mismatch' }
  return { ok: true }
}

// ============ Google ID token 校验（无新依赖，仅用 node:crypto） ============

interface GoogleCerts { keys: Array<{ kid: string; n: string; e: string; alg?: string; kty?: string }>; fetchedAt: number }
let googleCertsCache: GoogleCerts | null = null
const GOOGLE_CERTS_URL = 'https://www.googleapis.com/oauth2/v3/certs'
const GOOGLE_CERTS_TTL_MS = 60 * 60 * 1000 // 1h（Google 轮换不频繁，缓存降低延迟与失败面）

/** 拉取并缓存 Google 公钥（JWKS）。失败时若有旧缓存则复用。 */
async function getGoogleCerts(now: number, fetchImpl: typeof fetch = fetch): Promise<GoogleCerts['keys']> {
  if (googleCertsCache && now - googleCertsCache.fetchedAt < GOOGLE_CERTS_TTL_MS) {
    return googleCertsCache.keys
  }
  try {
    const res = await fetchImpl(GOOGLE_CERTS_URL)
    if (!res.ok) throw new Error('certs http ' + res.status)
    const data = await res.json() as { keys: GoogleCerts['keys'] }
    googleCertsCache = { keys: data.keys || [], fetchedAt: now }
    return googleCertsCache.keys
  } catch (e) {
    if (googleCertsCache) return googleCertsCache.keys // 退回旧缓存，避免临时网络问题阻断登录
    throw e
  }
}

/** 把 base64url 的 JWK (n,e) 组装成 PEM 公钥，供 crypto.verify 使用。 */
function jwkToPem(n: string, e: string): crypto.KeyObject {
  return crypto.createPublicKey({
    key: { kty: 'RSA', n, e } as crypto.JsonWebKey,
    format: 'jwk'
  })
}

export interface GoogleIdentity { sub: string; email: string; emailVerified: boolean; name?: string }

/**
 * 校验 Google ID token（RS256）。验证：签名、aud=clientId、iss、exp/nbf、email_verified。
 * 通过则返回 { sub, email, ... }；任何不符返回 null（调用方据此回 401/403）。
 * fetchImpl 可注入以便单测。
 */
export async function verifyGoogleIdToken(
  token: string,
  clientId: string,
  now: number,
  fetchImpl: typeof fetch = fetch
): Promise<GoogleIdentity | null> {
  if (!token || !clientId) return null
  const parts = token.split('.')
  if (parts.length !== 3) return null
  let header: { alg?: string; kid?: string }
  let payload: Record<string, unknown>
  try {
    header = JSON.parse(b64urlDecode(parts[0]).toString('utf8'))
    payload = JSON.parse(b64urlDecode(parts[1]).toString('utf8'))
  } catch {
    return null
  }
  if (header.alg !== 'RS256' || !header.kid) return null

  // 找到匹配 kid 的公钥
  let keys: GoogleCerts['keys']
  try { keys = await getGoogleCerts(now, fetchImpl) } catch { return null }
  const jwk = keys.find(k => k.kid === header.kid)
  if (!jwk) return null

  // 验签：RS256 over `${header}.${payload}`
  let verified = false
  try {
    const pubKey = jwkToPem(jwk.n, jwk.e)
    verified = crypto.verify(
      'RSA-SHA256',
      Buffer.from(parts[0] + '.' + parts[1]),
      pubKey,
      b64urlDecode(parts[2])
    )
  } catch {
    return null
  }
  if (!verified) return null

  // 校验 claims
  const iss = payload.iss
  if (iss !== 'https://accounts.google.com' && iss !== 'accounts.google.com') return null
  if (payload.aud !== clientId) return null
  const expSec = typeof payload.exp === 'number' ? payload.exp : 0
  const nbfSec = typeof payload.nbf === 'number' ? payload.nbf : 0
  const nowSec = Math.floor(now / 1000)
  if (expSec && nowSec >= expSec) return null
  if (nbfSec && nowSec < nbfSec - 60) return null // 容忍 60s 时钟偏移

  const email = typeof payload.email === 'string' ? normalizeEmail(payload.email) : ''
  const sub = typeof payload.sub === 'string' ? payload.sub : ''
  if (!email || !sub) return null
  // email_verified 可能是 boolean 或字符串 "true"
  const ev = payload.email_verified
  const emailVerified = ev === true || ev === 'true'

  return { sub, email, emailVerified, name: typeof payload.name === 'string' ? payload.name : undefined }
}

/**
 * 对客户名下 Key 的用量做脱敏视图（不暴露完整 key 明文，只给末 4 位）。
 * 门户列表用，避免 key 全文反复在网络上传输。
 */
export function maskKey(key: string): string {
  if (!key || key.length <= 8) return '****'
  return key.slice(0, 5) + '...' + key.slice(-4)
}

// ============ 校验辅助 ============

export function isValidEmail(email: string): boolean {
  const e = normalizeEmail(email)
  return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(e)
}

export function isStrongEnoughPassword(pw: string): boolean {
  return typeof pw === 'string' && pw.length >= 8 && pw.length <= MAX_PASSWORD_LENGTH
}

// ============ 转账slip自动充值（slip2go）纯逻辑 ============
// 抽到 portal.ts 作纯函数，便于 esbuild 单测（与密码/会话逻辑同样不依赖 electron）。
// proxyServer 负责 HTTP/持久化，金额与防重复的核心判定集中在这里。

/**
 * THB → credit 换算。在「分（สตางค์）」整数域计算避免浮点漂移，floor 保证不会多给 credit。
 * rate <= 0 视为非法 → 返回 0 credit（调用方应拒绝入账）。
 */
export function bahtToCredits(bahtAmount: number, bahtPerCredit: number): number {
  if (!Number.isFinite(bahtAmount) || bahtAmount <= 0) return 0
  if (!Number.isFinite(bahtPerCredit) || bahtPerCredit <= 0) return 0
  const bahtCents = Math.round(bahtAmount * 100)
  const rateCents = Math.round(bahtPerCredit * 100)
  if (rateCents <= 0) return 0
  return Math.floor(bahtCents / rateCents)
}

/** slip2go 结果码 → 客户可读拒绝原因（不泄露内部细节）。 */
export function slipRejectReason(code: number): string {
  switch (code) {
    case 200401: return 'receiver_not_match'
    case 200402: return 'amount_not_match'
    case 200403: return 'date_not_match'
    case 200404: return 'slip_not_found'   // 银行查无此slip = 可能伪造
    case 200501: return 'duplicate_slip'
    case 200000: return 'conditions_not_asserted'
    default: return code >= 400000 ? 'verification_error' : 'rejected'
  }
}

/** 仅 200200（Slip is Valid，所有 checkCondition 通过）才允许入账。 */
export function isSlipCreditable(code: number): boolean {
  return code === 200200
}

export interface SlipReceiverMatcher {
  accountType?: string
  accountNumber?: string
  accountNameTH?: string
  accountNameEN?: string
}

/**
 * 服务端二次核对收款人是否为我方账号（不只信 slip2go 的 200200，defense in depth）。
 * slip2go 返回账号常部分脱敏（如 "xxx-x-x5366-x"），故用「数字后缀匹配」：
 * 我方账号末 4 位数字需出现在 slip 返回账号的数字串中；或姓名 TH/EN 任一部分匹配。
 * ours 为空 = 未配置 checkReceiver（运营自担）→ 返回 true。
 */
export function slipReceiverMatches(
  ours: SlipReceiverMatcher[],
  receiverAccount: string | undefined,
  receiverName: string | undefined
): boolean {
  if (!Array.isArray(ours) || ours.length === 0) return true
  const recvDigits = (receiverAccount || '').replace(/\D/g, '')
  const recvName = (receiverName || '').toLowerCase()
  for (const o of ours) {
    if (o.accountNumber) {
      const od = o.accountNumber.replace(/\D/g, '')
      if (od.length >= 4 && recvDigits.length >= 4 && recvDigits.includes(od.slice(-4))) return true
    }
    if (o.accountNameTH && recvName && recvName.includes(o.accountNameTH.toLowerCase())) return true
    if (o.accountNameEN && recvName && recvName.includes(o.accountNameEN.toLowerCase())) return true
  }
  return false
}

/**
 * slip 新鲜度判定。返回 'ok' | 'too_old' | 'future'。
 * ageMs > freshnessHours → too_old；ageMs < -300s（容忍 5 分钟时钟偏移）→ future。
 * 无法解析日期 → 'ok'（不因解析失败而误拒，金额/收款人/去重仍各自把关）。
 */
export function slipFreshness(slipDateTimeIso: string | undefined, freshnessHours: number, now: number): 'ok' | 'too_old' | 'future' {
  if (!slipDateTimeIso) return 'ok'
  const ts = Date.parse(slipDateTimeIso)
  if (!Number.isFinite(ts)) return 'ok'
  const ageMs = now - ts
  if (ageMs > Math.max(0, freshnessHours) * 3600_000) return 'too_old'
  if (ageMs < -300_000) return 'future'
  return 'ok'
}
