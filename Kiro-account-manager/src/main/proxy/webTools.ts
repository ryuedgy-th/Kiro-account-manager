import { fetch as undiciFetch, type RequestInit as UndiciRequestInit } from 'undici'
import { getSystemProxy, safeCreateProxyAgent } from './systemProxy'
import { proxyLogger } from './logger'

// ============================================================================
// Server-side web tools (web_search / web_fetch)
//
// Anthropic 的 web_search/web_fetch 是 server-side 工具：客户端（如 Claude Code）
// 只发送工具声明，期待"服务端"执行搜索并把结果喂回对话。Kiro/CodeWhisperer 后端
// 没有 web search 引擎（已三方确认），因此由本代理充当执行方：
//   1. 把 web 工具转成带 inputSchema 的 custom tool 让 Kiro 模型可以调用
//   2. 拦截模型发起的 web_search/web_fetch tool_use，在代理侧真正执行
//   3. 把结果作为 tool_result 喂回 Kiro，循环到模型给出最终回答
// ============================================================================

// Kiro 官方文档对 web_fetch 的限制：单页 10MB、30s 超时、最多 10 次跳转、3 次重试
const WEB_FETCH_TIMEOUT_MS = 30_000
const WEB_SEARCH_TIMEOUT_MS = 20_000
// 重试次数：2 次足够覆盖瞬时失败，避免 backoff 叠加拖慢整体响应（3 次最坏多等 ~3.5s）
const MAX_RETRIES = 2
const RETRY_BASE_DELAY_MS = 500

// 工具名（与 Anthropic 客户端发送的 name 对齐）
export const WEB_SEARCH_TOOL_NAME = 'web_search'
export const WEB_FETCH_TOOL_NAME = 'web_fetch'

// 暴露给 Kiro 模型的 custom-tool inputSchema（替代客户端无 input_schema 的 server-tool 声明）
export const WEB_SEARCH_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    query: { type: 'string', description: 'The search query to look up on the web.' }
  },
  required: ['query']
}
export const WEB_FETCH_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    url: { type: 'string', description: 'The URL whose page content should be fetched.' }
  },
  required: ['url']
}

export const WEB_SEARCH_TOOL_DESCRIPTION =
  'Search the web for current, up-to-date information. Returns relevant results with titles, URLs, and content snippets. Use this when the user asks about recent events or information that may have changed since your training data.'
export const WEB_FETCH_TOOL_DESCRIPTION =
  'Fetch and read the text content of a specific web page URL. Use this to get the full content of a page found via web_search or provided by the user.'

// 判断某个工具（按 name 或 Anthropic server-tool 的 type 前缀）是否是我们支持的 web 工具
export function isServerWebTool(name?: string, type?: string): 'web_search' | 'web_fetch' | null {
  const n = (name || '').toLowerCase()
  const t = (type || '').toLowerCase()
  if (n === WEB_SEARCH_TOOL_NAME || t.startsWith('web_search')) return 'web_search'
  if (n === WEB_FETCH_TOOL_NAME || t.startsWith('web_fetch')) return 'web_fetch'
  return null
}

export interface WebSearchResult {
  title: string
  url: string
  content: string
}
export interface WebToolConfig {
  enabled: boolean
  provider: 'tavily'
  apiKey: string
  maxRounds?: number
}

// ---- provider 抽象：当前实现 Tavily，后续可加 Brave/Serper ----
interface SearchProvider {
  search(query: string, apiKey: string, signal?: AbortSignal): Promise<{ answer?: string; results: WebSearchResult[] }>
  fetch(url: string, apiKey: string, signal?: AbortSignal): Promise<{ url: string; content: string }>
}

function getProxyDispatcher(): unknown {
  try {
    return safeCreateProxyAgent(getSystemProxy())
  } catch {
    return undefined
  }
}

async function fetchJson(url: string, options: UndiciRequestInit, timeoutMs: number, signal?: AbortSignal): Promise<unknown> {
  // 合并外部 signal 与超时 signal
  const timeoutController = new AbortController()
  const timer = setTimeout(() => timeoutController.abort(), timeoutMs)
  const onAbort = () => timeoutController.abort()
  if (signal) {
    if (signal.aborted) timeoutController.abort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }
  try {
    const agent = getProxyDispatcher()
    const opts = { ...options, signal: timeoutController.signal } as UndiciRequestInit
    if (agent) (opts as Record<string, unknown>).dispatcher = agent
    const res = (await undiciFetch(url, opts)) as unknown as Response
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`)
    }
    return await res.json()
  } finally {
    clearTimeout(timer)
    if (signal) signal.removeEventListener('abort', onAbort)
  }
}

async function withRetry<T>(label: string, fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  let lastErr: unknown
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    if (signal?.aborted) throw new Error('aborted')
    try {
      return await fn()
    } catch (err) {
      lastErr = err
      // abort 不重试
      if (signal?.aborted) throw err
      if (attempt < MAX_RETRIES - 1) {
        const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt)
        proxyLogger.warn('WebTools', `${label} attempt ${attempt + 1} failed, retrying in ${delay}ms: ${err instanceof Error ? err.message : String(err)}`)
        await new Promise(r => setTimeout(r, delay))
      }
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
}

const tavilyProvider: SearchProvider = {
  async search(query, apiKey, signal) {
    const data = (await withRetry('tavily.search', () => fetchJson(
      'https://api.tavily.com/search',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ query, max_results: 3, include_answer: true, search_depth: 'basic' })
      },
      WEB_SEARCH_TIMEOUT_MS, signal
    ), signal)) as { answer?: string; results?: Array<{ title?: string; url?: string; content?: string }> }
    return {
      answer: data.answer,
      results: (data.results || []).map(r => ({ title: r.title || '', url: r.url || '', content: r.content || '' }))
    }
  },
  async fetch(url, apiKey, signal) {
    const data = (await withRetry('tavily.extract', () => fetchJson(
      'https://api.tavily.com/extract',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ urls: [url] })
      },
      WEB_FETCH_TIMEOUT_MS, signal
    ), signal)) as { results?: Array<{ url?: string; raw_content?: string }> }
    const first = data.results?.[0]
    return { url: first?.url || url, content: first?.raw_content || '' }
  }
}

function getProvider(_provider: string): SearchProvider {
  // 目前只支持 tavily；未知 provider 也回退到 tavily 接口
  return tavilyProvider
}

// 执行 web_search，返回喂回模型的纯文本（含来源 URL）
export async function executeWebSearch(query: string, config: WebToolConfig, signal?: AbortSignal): Promise<string> {
  try {
    const { answer, results } = await getProvider(config.provider).search(query, config.apiKey, signal)
    if (!results.length && !answer) return `No web search results found for query: "${query}".`
    const lines: string[] = []
    if (answer) lines.push(`Summary: ${answer}`, '')
    results.forEach((r, i) => {
      lines.push(`[${i + 1}] ${r.title}`, `URL: ${r.url}`, r.content, '')
    })
    return lines.join('\n').trim()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    proxyLogger.error('WebTools', `web_search failed for "${query}": ${msg}`)
    return `Web search failed: ${msg}. Unable to retrieve current information for "${query}".`
  }
}

// 执行 web_fetch，返回页面文本（截断到合理长度）
const MAX_FETCH_CHARS = 100_000
export async function executeWebFetch(url: string, config: WebToolConfig, signal?: AbortSignal): Promise<string> {
  try {
    const { url: finalUrl, content } = await getProvider(config.provider).fetch(url, config.apiKey, signal)
    if (!content) return `No readable content found at ${url}.`
    const trimmed = content.length > MAX_FETCH_CHARS ? content.slice(0, MAX_FETCH_CHARS) + '\n...[truncated]' : content
    return `Content fetched from ${finalUrl}:\n\n${trimmed}`
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    proxyLogger.error('WebTools', `web_fetch failed for "${url}": ${msg}`)
    return `Web fetch failed: ${msg}. Unable to retrieve content from ${url}.`
  }
}

// 根据工具类型 + 输入执行，返回 tool_result 文本
export async function executeWebTool(
  kind: 'web_search' | 'web_fetch',
  input: Record<string, unknown>,
  config: WebToolConfig,
  signal?: AbortSignal
): Promise<string> {
  if (kind === 'web_search') {
    const query = typeof input.query === 'string' ? input.query : ''
    if (!query) return 'Web search failed: missing "query" parameter.'
    return executeWebSearch(query, config, signal)
  } else {
    const url = typeof input.url === 'string' ? input.url : ''
    if (!url) return 'Web fetch failed: missing "url" parameter.'
    return executeWebFetch(url, config, signal)
  }
}

// ============================================================================
// Structured 执行：除了喂回 Kiro 的纯文本，还保留结构化结果（title/url/content）。
// 用于把搜索结果以 Anthropic 原生 server_tool_use / web_search_tool_result content block
// 形式回传给客户端（Claude Code 据此显示 "Did N searches" 与可点击来源）。
// ============================================================================
export interface WebToolExecResult {
  // 喂回 Kiro 模型的纯文本（与 executeWebTool 一致）
  text: string
  // 结构化来源（仅 web_search 有意义；web_fetch 退化为单条 url）
  sources: Array<{ title: string; url: string; pageAge?: string }>
  // 是否出错（出错时 sources 为空，text 含错误说明）
  isError: boolean
}

export async function executeWebToolStructured(
  kind: 'web_search' | 'web_fetch',
  input: Record<string, unknown>,
  config: WebToolConfig,
  signal?: AbortSignal
): Promise<WebToolExecResult> {
  if (kind === 'web_search') {
    const query = typeof input.query === 'string' ? input.query : ''
    if (!query) return { text: 'Web search failed: missing "query" parameter.', sources: [], isError: true }
    try {
      const { answer, results } = await getProvider(config.provider).search(query, config.apiKey, signal)
      if (!results.length && !answer) {
        return { text: `No web search results found for query: "${query}".`, sources: [], isError: false }
      }
      const lines: string[] = []
      if (answer) lines.push(`Summary: ${answer}`, '')
      results.forEach((r, i) => {
        lines.push(`[${i + 1}] ${r.title}`, `URL: ${r.url}`, r.content, '')
      })
      return {
        text: lines.join('\n').trim(),
        sources: results.map(r => ({ title: r.title || r.url, url: r.url })),
        isError: false
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      proxyLogger.error('WebTools', `web_search failed for "${query}": ${msg}`)
      return { text: `Web search failed: ${msg}. Unable to retrieve current information for "${query}".`, sources: [], isError: true }
    }
  } else {
    const url = typeof input.url === 'string' ? input.url : ''
    if (!url) return { text: 'Web fetch failed: missing "url" parameter.', sources: [], isError: true }
    try {
      const { url: finalUrl, content } = await getProvider(config.provider).fetch(url, config.apiKey, signal)
      if (!content) return { text: `No readable content found at ${url}.`, sources: [{ title: url, url }], isError: false }
      const trimmed = content.length > MAX_FETCH_CHARS ? content.slice(0, MAX_FETCH_CHARS) + '\n...[truncated]' : content
      return {
        text: `Content fetched from ${finalUrl}:\n\n${trimmed}`,
        sources: [{ title: finalUrl, url: finalUrl }],
        isError: false
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      proxyLogger.error('WebTools', `web_fetch failed for "${url}": ${msg}`)
      return { text: `Web fetch failed: ${msg}. Unable to retrieve content from ${url}.`, sources: [], isError: true }
    }
  }
}
