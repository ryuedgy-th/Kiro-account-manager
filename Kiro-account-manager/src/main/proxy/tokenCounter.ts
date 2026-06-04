/**
 * Token 计数工具：使用 js-tiktoken cl100k_base 编码精确计算，
 * 失败时降级到字节系数估算；并提供按模型 ID 查询 context 窗口大小，
 * 以便从 Kiro 后端的 contextUsagePercentage 反推真实 input tokens。
 */
import { getEncoding, type Tiktoken } from 'js-tiktoken'

let encoder: Tiktoken | null = null
let encoderInitFailed = false

/** 懒加载 cl100k_base 编码器（GPT-4/Claude 通用近似） */
function getEncoder(): Tiktoken | null {
  if (encoder) return encoder
  if (encoderInitFailed) return null
  try {
    encoder = getEncoding('cl100k_base')
    return encoder
  } catch (err) {
    console.warn('[TokenCounter] Failed to load cl100k_base encoder:', err)
    encoderInitFailed = true
    return null
  }
}

// 对超长文本跳过 tiktoken 的阈值（字符数）。
// tiktoken 是同步且 O(n) 的：对几 MB 的 base64（PDF/图片附件）跑 encode 会阻塞 event loop，
// 让整个反代“卡死”。超过此长度时改用字节系数估算（足够触发裁剪阈值，且不阻塞）。
const TIKTOKEN_MAX_CHARS = 200_000

// ============ tiktoken 编码结果记忆化（memoization）============
// 对话是 append-only 的：每个 turn 只在尾部追加新消息，历史 block 的内容完全不变。
// 但 promptCacheTracker.buildClaudeProfile 每次请求都会对「整段历史的每个 block」重跑 tiktoken，
// 而 encode 是同步、O(n)、且常数极大的 BPE 过程——会随对话变长而线性加重，阻塞 main 进程造成 UI 卡顿
//（这正是“cache 一多就像卡住”的根因：cache 多 ≈ 上下文长 ≈ 每个 turn 重编码的历史更多）。
// 这里用「内容 → token 数」的有界 LRU 缓存：相同内容只编码一次，后续直接命中。
// 结果对相同输入完全确定，不改变任何对外 usage 数值，只省掉重复的同步计算。
const MEMO_MIN_CHARS = 128       // 低于此长度 encode 极快，缓存键开销不划算
const MEMO_MAX_ENTRIES = 1024    // 有界，超出后按 LRU 淘汰最久未用，防止内存无限增长
const tokenMemo = new Map<string, number>()

function memoGet(text: string): number | undefined {
  const hit = tokenMemo.get(text)
  if (hit === undefined) return undefined
  // LRU 命中：删除后重新插入到末尾，标记为“最近使用”
  tokenMemo.delete(text)
  tokenMemo.set(text, hit)
  return hit
}

function memoSet(text: string, tokens: number): void {
  tokenMemo.set(text, tokens)
  if (tokenMemo.size > MEMO_MAX_ENTRIES) {
    // Map 迭代顺序 = 插入/更新顺序，首个即最久未使用，淘汰之
    const oldest = tokenMemo.keys().next().value
    if (oldest !== undefined) tokenMemo.delete(oldest)
  }
}

/** 清空 token 记忆缓存（供测试或 prompt cache 整体清理时调用）。 */
export function clearTokenMemo(): void {
  tokenMemo.clear()
}

/**
 * 使用 tiktoken cl100k_base 精确计算 token 数。
 * 兜底：UTF-8 字节数 / 3.0（针对 payload JSON 经验值，误差 ±10%）。
 * 超长文本（> TIKTOKEN_MAX_CHARS）直接走字节系数，避免同步 encode 阻塞 event loop。
 * 中等长度文本走 LRU 记忆化：append-only 对话里重复出现的历史 block 只编码一次。
 */
export function countTokens(text: string): number {
  if (!text) return 0
  // 超长内容（典型是 base64 附件）跳过 tiktoken，防止同步 encode 卡死事件循环
  if (text.length > TIKTOKEN_MAX_CHARS) {
    return Math.ceil(Buffer.byteLength(text, 'utf-8') / 3.0)
  }
  // 记忆化：相同内容（典型是 append-only 对话里反复出现的历史 block）只编码一次
  const memoable = text.length >= MEMO_MIN_CHARS
  if (memoable) {
    const cached = memoGet(text)
    if (cached !== undefined) return cached
  }
  const enc = getEncoder()
  if (enc) {
    try {
      const n = enc.encode(text).length
      if (memoable) memoSet(text, n)
      return n
    } catch (err) {
      console.warn('[TokenCounter] encode failed, using fallback:', err)
    }
  }
  return Math.ceil(Buffer.byteLength(text, 'utf-8') / 3.0)
}

// ============ 二进制附件（base64 文档 / 图片）token 估算 ============
// 关键：base64 字符串本身不能按 length/4 或 tiktoken 计 token——一个 3MB 的 PDF
// 会被误算成 ~100 万 token，导致：
//   1) 反代把 tiktoken 跑在几 MB 字符串上 → 同步阻塞 → 服务器卡死
//   2) 客户端（Claude Code）按这个虚高数字判断上下文用量 → autocompact 逻辑被打乱
// 这里按“解码后的真实字节数”做保守估算，贴近模型对 PDF/图片的实际计费量级。

// PDF/二进制文档：每 token 对应的解码后字节数（经验值，偏保守）
const DOCUMENT_BYTES_PER_TOKEN = 8
// 单个文档 token 上限：即便附件巨大，也不让单个块支配整个上下文统计
const DOCUMENT_MAX_TOKENS = 64_000
// 纯文本类文档（md/csv/txt/html）：每 token 字节数（贴近自然语言）
const TEXT_DOCUMENT_BYTES_PER_TOKEN = 4
// 单张图片的固定 token 估算（与 Anthropic 视觉 token 量级一致的保守值）
export const IMAGE_TOKEN_ESTIMATE = 1600

/** 估算 base64 字符串解码后的字节数（不实际解码，O(1)）。 */
export function base64DecodedByteLength(b64: string): number {
  if (!b64) return 0
  // 去掉可能的 data URL 前缀与空白
  const comma = b64.indexOf(',')
  const raw = (comma >= 0 && comma < 64 && b64.slice(0, comma).includes('base64')) ? b64.slice(comma + 1) : b64
  const len = raw.length
  if (len === 0) return 0
  let padding = 0
  if (raw.endsWith('==')) padding = 2
  else if (raw.endsWith('=')) padding = 1
  return Math.max(0, Math.floor(len * 3 / 4) - padding)
}

/** 文档格式是否为“纯文本类”（按字节折算更接近自然语言 token）。 */
function isTextDocumentFormat(format?: string): boolean {
  if (!format) return false
  const f = format.toLowerCase()
  return f === 'txt' || f === 'md' || f === 'markdown' || f === 'csv' || f === 'html' || f === 'htm' || f.startsWith('text')
}

/**
 * 估算一个 base64 文档块的 token 数（按解码字节数，绝不 tiktoken/按 length 计）。
 * 文本类文档用较小的字节系数，二进制（PDF 等）用较大系数并封顶。
 */
export function estimateBase64DocumentTokens(b64: string, format?: string): number {
  const bytes = base64DecodedByteLength(b64)
  if (bytes === 0) return 0
  const perToken = isTextDocumentFormat(format) ? TEXT_DOCUMENT_BYTES_PER_TOKEN : DOCUMENT_BYTES_PER_TOKEN
  const est = Math.ceil(bytes / perToken)
  return Math.min(est, DOCUMENT_MAX_TOKENS)
}

// ============ 模型 context 窗口缓存 ============
// 由 proxyServer 在 fetchKiroModels 后通过 setModelContextWindow 填充
// modelId → maxInputTokens
const modelContextWindowCache = new Map<string, number>()

export function setModelContextWindow(modelId: string, maxInputTokens: number): void {
  if (modelId && maxInputTokens > 0) {
    modelContextWindowCache.set(modelId, maxInputTokens)
  }
}

export function getModelContextWindow(modelId: string): number | undefined {
  return modelContextWindowCache.get(modelId)
}

/**
 * 归一化 model ID 用于模糊匹配：
 *   claude-sonnet-4.5                   → claudesonnet45
 *   CLAUDE_SONNET_4_5_20251001_V1_0     → claudesonnet45
 *   claude-3.7-sonnet                   → claude37sonnet
 */
function normalizeModelId(id: string): string {
  return id
    .toLowerCase()
    .replace(/[-._]/g, '')
    .replace(/\d{8}/g, '')   // 移除日期 (20251001)
    .replace(/v\d+$/g, '')    // 移除尾部版本号 (v1)
    .replace(/v\d+_\d+$/g, '') // 移除 v1_0 形式
}

/**
 * 从缓存中按模糊匹配查找 context window。
 * 例如：用户传 alias `claude-sonnet-4.5`，但 cache 里存的是 CW 内部 ID
 * `CLAUDE_SONNET_4_5_20251001_V1_0`，归一化后都是 `claudesonnet45`。
 */
function guessContextFromCache(modelId: string): number | undefined {
  if (modelContextWindowCache.size === 0) return undefined
  const queryNorm = normalizeModelId(modelId)
  if (!queryNorm) return undefined

  // 精确归一化匹配
  for (const [id, ctx] of modelContextWindowCache) {
    if (normalizeModelId(id) === queryNorm) return ctx
  }
  // 双向子串匹配（处理别名简短形式）
  for (const [id, ctx] of modelContextWindowCache) {
    const idNorm = normalizeModelId(id)
    if (idNorm.includes(queryNorm) || queryNorm.includes(idNorm)) return ctx
  }
  return undefined
}

/**
 * 根据 model ID 返回 context 窗口大小（用于 contextUsagePercentage 反推 inputTokens）。
 *
 * 优先级：
 *   1. 直接命中 cache（Kiro 真实拉取的 maxInputTokens，最准确）
 *   2. 模糊匹配 cache（处理 alias ↔ CW 内部 ID 映射）
 *   3. 关键词匹配兜底（cache 未填充时）
 */
export function getModelContextLength(modelId: string | undefined | null): number {
  if (!modelId) return 200000

  // 1. 优先用 Kiro 后端真实返回的 maxInputTokens
  const cached = modelContextWindowCache.get(modelId)
  if (cached && cached > 0) return cached

  // 2. 模糊匹配 cache（alias ↔ CW 内部 ID）
  const guessed = guessContextFromCache(modelId)
  if (guessed && guessed > 0) return guessed

  // 3. 关键词匹配兜底（首次请求 cache 未填充时使用）
  const id = modelId.toLowerCase()

  // Claude 系列（默认 200K）
  if (id.includes('claude-opus-4') || id.includes('claude-sonnet-4') || id.includes('claude-haiku-4')) return 200000
  if (id.includes('claude-3-7') || id.includes('claude-3.7')) return 200000
  if (id.includes('claude-3-5') || id.includes('claude-3.5')) return 200000
  if (id.includes('claude-3')) return 200000
  if (id.includes('claude-2.1')) return 200000
  if (id.includes('claude-2')) return 100000
  if (id.includes('claude-instant')) return 100000

  // GPT 系列
  if (id.includes('gpt-4o') || id.includes('gpt-4-turbo')) return 128000
  if (id.includes('gpt-4.1')) return 1000000
  if (id.includes('gpt-4-32k')) return 32768
  if (id.includes('gpt-4')) return 8192
  if (id.includes('gpt-3.5-turbo-16k')) return 16384
  if (id.includes('gpt-3.5')) return 4096
  if (id.includes('o1') || id.includes('o3')) return 128000

  // Gemini 系列
  if (id.includes('gemini-2.5') || id.includes('gemini-2.0') || id.includes('gemini-1.5')) return 1000000
  if (id.includes('gemini')) return 32768

  // Amazon Titan / Nova 系列
  if (id.includes('nova-pro') || id.includes('nova-lite')) return 300000
  if (id.includes('nova-micro')) return 128000
  if (id.includes('titan')) return 8000

  // CodeWhisperer/Q Developer 内部模型一般跟 Claude 看齐
  return 200000
}
