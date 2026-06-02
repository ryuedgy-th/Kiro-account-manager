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
import type { ProxyConfig, Customer, ApiKey } from './types'

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
