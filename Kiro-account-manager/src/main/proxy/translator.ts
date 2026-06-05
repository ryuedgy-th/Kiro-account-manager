// OpenAI/Claude 格式与 Kiro 格式转换器
import { v4 as uuidv4 } from 'uuid'
import type {
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAITool,
  OpenAIChatResponse,
  OpenAIStreamChunk,
  OpenAIResponsesRequest,
  OpenAIResponsesResponse,
  OpenAIResponseContentPart,
  OpenAIResponseOutputItem,
  ClaudeRequest,
  ClaudeMessage,
  ClaudeResponse,
  ClaudeStreamEvent,
  ClaudeContentBlock,
  KiroPayload,
  KiroHistoryMessage,
  KiroToolWrapper,
  KiroToolResult,
  KiroImage,
  KiroDocument,
  KiroToolUse,
  KiroUserInputMessage,
  KiroCachePoint,
  KiroReasoningContent,
  KiroUsage
} from './types'
import { buildKiroPayload, mapModelId } from './kiroApi'
import { ToolNameRegistry } from './toolNameRegistry'
import { proxyLogger } from './logger'
import {
  isServerWebTool,
  WEB_SEARCH_INPUT_SCHEMA, WEB_FETCH_INPUT_SCHEMA,
  WEB_SEARCH_TOOL_DESCRIPTION, WEB_FETCH_TOOL_DESCRIPTION,
  WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME
} from './webTools'

const KIRO_CACHE_POINT: KiroCachePoint = { type: 'default' }

// 模型能力注册表：以后端 /v1/models 返回的真实 schema 为准（由 proxyServer 在拉取/缓存模型时同步）。
// 这是判断「该模型是否接受 additionalModelRequestFields / 接受哪些 effort 值」的唯一权威来源——
// 后端实测：claude-sonnet-4.6 的 effort 枚举为 [low,medium,high,max]（无 xhigh），
// claude-haiku-4.5 / deepseek / minimax / glm / qwen 完全不接受 additionalModelRequestFields。
// 早期版本用「Claude 4+ 一律支持」的硬编码猜测，会对不支持的模型或非法 effort 值强行下发，导致后端
// 返回 400（"additionalModelRequestFields is not supported for this model" 或
// "does not have a value in the enumeration [...]"）。
interface ModelThinkingCapability {
  supportsThinking: boolean
  thinkingEfforts: string[] // 后端允许的 effort 枚举；为空表示该模型不接受 effort
}

const modelCapabilityRegistry = new Map<string, ModelThinkingCapability>()

function capabilityKey(modelId: string): string {
  return modelId.trim().toLowerCase()
}

// 由 proxyServer 在每次成功拉取模型列表后调用，将后端真实能力同步进来。
export function setModelThinkingCapability(
  modelId: string,
  capability: ModelThinkingCapability
): void {
  if (!modelId) return
  modelCapabilityRegistry.set(capabilityKey(modelId), {
    supportsThinking: capability.supportsThinking,
    thinkingEfforts: capability.thinkingEfforts ?? []
  })
}

export function clearModelThinkingCapabilities(): void {
  modelCapabilityRegistry.clear()
}

function lookupCapability(modelId: string): ModelThinkingCapability | undefined {
  return modelCapabilityRegistry.get(capabilityKey(modelId))
}

// 后备启发式：仅在注册表尚未填充（首个请求早于首次模型拉取）时使用，保守判断模型族是否「可能」支持。
// 注意：这只决定是否「尝试」下发字段，真正的 effort 合法性始终以注册表为准（注册表为空时仅下发安全枚举）。
function heuristicSupportsThinking(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  if (!lower.includes('claude')) return false
  if (lower.includes('claude-3-') || lower.includes('claude-3.')) return false
  if (lower === 'auto') return false
  return true
}

// 注册表未填充时允许下发的 effort 安全集合：取所有支持 thinking 的 Claude 模型 effort 枚举的交集
// （sonnet=[low,medium,high,max]、opus=[low,medium,high,xhigh,max]）。
// 不含 xhigh——它仅 opus 接受，对 sonnet 下发会触发 enumeration 400。模型列表拉取后改以真实枚举为准。
const SAFE_THINKING_EFFORTS = new Set(['low', 'medium', 'high', 'max'])

// 从 Claude 请求推导 effort 档位（供 additionalModelRequestFields.output_config.effort 下发）。
//
// 背景（经抓包 + 真机验证）：标准 Claude Code 按 Anthropic API 规范用 thinking.budget_tokens
// 表达推理强度，并不发送 Kiro 私有的 output_config.effort。若只读 output_config.effort，
// 标准 Claude Code 的「思考预算」永远落空——Opus 始终跑默认 effort，客户付了 budget 却拿不到对应算力。
// 因此这里做映射：显式 effort 优先，否则把 budget_tokens 折算成档位。
// 阈值与 ProxyServer.deriveEffortLevel（用量统计用）保持一致，确保「下发的 effort」与「dashboard 显示的 effort」口径相同。
// 注：仅决定「请求哪个档位」；该档位是否被真正下发，仍由 buildAdditionalModelRequestFields 按后端枚举校验。
export function deriveClaudeEffort(request: {
  output_config?: { effort?: string }
  thinking?: { type?: string; budget_tokens?: number }
}): string | undefined {
  // 1. 显式 effort（Kiro 私有字段，若客户端已知 Kiro 协议会直接发）→ 直接采用
  const explicit = request.output_config?.effort
  if (typeof explicit === 'string' && explicit.trim()) return explicit.trim().toLowerCase()

  // 2. 标准 Claude Code：从 thinking.budget_tokens 折算
  const thinking = request.thinking
  if (!thinking || thinking.type === 'disabled') return undefined
  const budget = typeof thinking.budget_tokens === 'number' ? thinking.budget_tokens : undefined
  // enabled/adaptive 但未给 budget → 视为中档（开启了思考但未指定强度）
  if (budget === undefined) {
    return thinking.type === 'enabled' || thinking.type === 'adaptive' ? 'medium' : undefined
  }
  if (budget <= 0) return undefined
  if (budget < 6000) return 'low'
  if (budget < 16000) return 'medium'
  if (budget < 32000) return 'high'
  return 'max'
}

// 统一构建 additionalModelRequestFields（thinking + reasoning effort）
// effort 走 additionalModelRequestFields.output_config.effort（与 /v1/models 暴露的 schema 对齐，
// 见 proxyServer.ts extractThinkingEfforts：output_config 与 thinking 同级，位于 additionalModelRequestFields 下）。
// 一律以后端真实能力注册表为准：
//   - 注册表显示该模型不支持 thinking → 返回 undefined（不下发任何字段）
//   - 客户端请求的 effort 不在后端枚举内 → 丢弃该 effort（不会因非法值触发 400）
//   - 注册表尚未填充 → 退回保守启发式判断模型族，effort 只下发安全集合内的值（防止 xhigh 误发 sonnet）
function buildAdditionalModelRequestFields(
  modelId: string,
  thinking?: { type: string } | undefined,
  effort?: string | undefined
): Record<string, unknown> | undefined {
  const capability = lookupCapability(modelId)
  const normalizedEffort = effort?.trim()

  // 注册表已知该模型：以真实能力为准
  if (capability) {
    if (!capability.supportsThinking) return undefined
    const fields: Record<string, unknown> = {}
    if (thinking && thinking.type !== 'disabled') {
      fields.thinking = { type: 'adaptive' }
    }
    // 仅在 effort 落在后端允许枚举内时下发，否则静默丢弃（防止 enumeration 400）
    if (normalizedEffort && capability.thinkingEfforts.includes(normalizedEffort)) {
      fields.output_config = { effort: normalizedEffort }
    }
    return Object.keys(fields).length > 0 ? fields : undefined
  }

  // 注册表未知（尚未拉取模型）：退回保守启发式，effort 仅下发安全集合内的值
  if (!heuristicSupportsThinking(modelId)) return undefined
  const fields: Record<string, unknown> = {}
  if (thinking && thinking.type !== 'disabled') {
    fields.thinking = { type: 'adaptive' }
  }
  if (normalizedEffort && SAFE_THINKING_EFFORTS.has(normalizedEffort)) {
    fields.output_config = { effort: normalizedEffort }
  }
  return Object.keys(fields).length > 0 ? fields : undefined
}

function toKiroCachePoint(cacheControl?: { type: string }): KiroCachePoint | undefined {
  if (!cacheControl) return undefined
  if (cacheControl.type !== 'ephemeral') {
    throw new Error(`Unsupported cache_control type: ${cacheControl.type}`)
  }
  return KIRO_CACHE_POINT
}

function mergeCachePoint(
  first?: KiroCachePoint,
  second?: KiroCachePoint
): KiroCachePoint | undefined {
  return first || second
}

export function responsesToOpenAIChat(request: OpenAIResponsesRequest): OpenAIChatRequest {
  if (!request || typeof request !== 'object') {
    throw new Error('Responses request body must be an object')
  }
  if (!request.model) {
    throw new Error('Responses request requires model')
  }
  if (request.input === undefined) {
    throw new Error('Responses request requires input')
  }

  const messages: OpenAIMessage[] = []
  if (request.instructions) {
    messages.push({ role: 'system', content: request.instructions })
  }
  if (typeof request.input === 'string') {
    messages.push({ role: 'user', content: request.input })
  } else {
    if (!Array.isArray(request.input)) {
      throw new Error('Responses input must be a string or an array')
    }
    for (const item of request.input) {
      const itemType = item.type as string | undefined
      if (itemType === 'function_call_output') {
        if (!item.call_id) {
          throw new Error('function_call_output requires call_id')
        }
        if (item.output === undefined) {
          throw new Error('function_call_output requires output')
        }
        messages.push({
          role: 'tool',
          content: item.output,
          tool_call_id: item.call_id
        })
      } else if (itemType === 'function_call') {
        if (!item.call_id) {
          throw new Error('function_call requires call_id')
        }
        if (!item.name) {
          throw new Error('function_call requires name')
        }
        if (item.arguments === undefined) {
          throw new Error('function_call requires arguments')
        }
        messages.push({
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: item.call_id,
            type: 'function',
            function: {
              name: item.name,
              arguments: item.arguments
            }
          }]
        })
      } else {
        if (itemType !== undefined && itemType !== 'message') {
          throw new Error(`Unsupported responses input item type: ${itemType}`)
        }
        if (item.content === undefined) {
          throw new Error('message input item requires content')
        }
        messages.push({
          role: item.role === 'assistant' ? 'assistant' : item.role === 'system' ? 'system' : 'user',
          content: convertResponseInputContent(item.content)
        })
      }
    }
  }

  const chatRequest: OpenAIChatRequest = {
    model: request.model,
    messages
  }
  if (request.temperature !== undefined) chatRequest.temperature = request.temperature
  if (request.top_p !== undefined) chatRequest.top_p = request.top_p
  if (request.max_output_tokens !== undefined) chatRequest.max_tokens = request.max_output_tokens
  if (request.stream !== undefined) chatRequest.stream = request.stream
  if (request.tools !== undefined) chatRequest.tools = request.tools
  const toolChoice = convertResponseToolChoice(request.tool_choice)
  if (toolChoice !== undefined) chatRequest.tool_choice = toolChoice
  if (request.previous_response_id !== undefined) chatRequest.conversation_id = request.previous_response_id
  // Responses API 的 reasoning.effort 映射到 chat 的 reasoning_effort，最终透传到 Kiro
  if (request.reasoning?.effort) chatRequest.reasoning_effort = request.reasoning.effort
  if (request.metadata !== undefined) chatRequest.metadata = request.metadata
  if (request.kiro_context !== undefined) chatRequest.kiro_context = request.kiro_context
  return chatRequest
}

function convertResponseInputContent(content: string | OpenAIResponseContentPart[] | undefined): OpenAIMessage['content'] {
  if (typeof content === 'string') return content
  if (content === undefined) return ''
  if (!Array.isArray(content)) {
    throw new Error('message content must be a string or an array')
  }
  return content.map(part => {
    const partType = part.type as string
    if (partType === 'input_image') {
      if (!part.image_url) {
        throw new Error('input_image requires image_url')
      }
      return { type: 'image_url', image_url: { url: part.image_url } }
    }
    if (partType === 'input_file') {
      if (!part.file_data) {
        throw new Error('input_file requires file_data')
      }
      return {
        type: 'file',
        file: {
          file_data: part.file_data,
          ...(part.filename !== undefined ? { filename: part.filename } : {})
        }
      }
    }
    if (partType !== 'input_text' && partType !== 'output_text') {
      throw new Error(`Unsupported responses content part type: ${partType}`)
    }
    if (part.text === undefined) {
      throw new Error(`${partType} requires text`)
    }
    return { type: 'text', text: part.text }
  })
}

function convertResponseToolChoice(toolChoice: OpenAIResponsesRequest['tool_choice']): OpenAIChatRequest['tool_choice'] {
  if (!toolChoice || typeof toolChoice === 'string') return toolChoice
  if (toolChoice.type === 'none' || toolChoice.type === 'auto') return toolChoice.type
  if (toolChoice.type === 'function' && toolChoice.name) {
    return { type: 'function', function: { name: toolChoice.name } }
  }
  if (toolChoice.function?.name) return { type: 'function', function: { name: toolChoice.function.name } }
  throw new Error('Unsupported responses tool_choice')
}

export function openAIChatToResponsesResponse(
  response: OpenAIChatResponse,
  previousResponseId?: string
): OpenAIResponsesResponse {
  const output: OpenAIResponseOutputItem[] = response.choices.flatMap<OpenAIResponseOutputItem>(choice => {
    if (choice.message.tool_calls?.length) {
      return choice.message.tool_calls.map(toolCall => ({
        type: 'function_call' as const,
        id: `fc_${uuidv4()}`,
        call_id: toolCall.id,
        name: toolCall.function.name,
        arguments: toolCall.function.arguments
      }))
    }
    return [{
      type: 'message' as const,
      id: `msg_${uuidv4()}`,
      role: 'assistant' as const,
      content: [{ type: 'output_text' as const, text: choice.message.content || '' }]
    }]
  })

  const usage: OpenAIResponsesResponse['usage'] = {
    input_tokens: response.usage.prompt_tokens,
    output_tokens: response.usage.completion_tokens,
    total_tokens: response.usage.total_tokens
  }
  const cachedTokens = response.usage.prompt_tokens_details?.cached_tokens
  if (cachedTokens !== undefined) {
    usage.input_tokens_details = { cached_tokens: cachedTokens }
  }
  const reasoningTokens = response.usage.completion_tokens_details?.reasoning_tokens
  if (reasoningTokens !== undefined) {
    usage.output_tokens_details = { reasoning_tokens: reasoningTokens }
  }

  const responsesResponse: OpenAIResponsesResponse = {
    id: `resp_${uuidv4()}`,
    object: 'response',
    created_at: response.created,
    model: response.model,
    output,
    usage
  }
  if (previousResponseId !== undefined) {
    responsesResponse.previous_response_id = previousResponseId
  }
  return responsesResponse
}

// ============ OpenAI -> Kiro 转换 ============

export function openaiToKiro(
  request: OpenAIChatRequest,
  profileArn?: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry()
): KiroPayload {
  const modelId = mapModelId(request.model)
  const origin = 'AI_EDITOR'

  // 提取系统提示
  let systemPrompt = ''
  let systemCachePoint: KiroCachePoint | undefined
  const nonSystemMessages: OpenAIMessage[] = []

  for (const msg of request.messages) {
    if (msg.role === 'system') {
      systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(msg.cache_control))
      if (typeof msg.content === 'string') {
        systemPrompt += (systemPrompt ? '\n' : '') + msg.content
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(part.cache_control))
          if (part.type === 'text' && part.text) {
            systemPrompt += (systemPrompt ? '\n' : '') + part.text
          }
        }
      }
    } else {
      nonSystemMessages.push(msg)
    }
  }

  // 时间戳改注入到 currentMessage 末尾（见下方 finalContent），而非 system prompt 头部。
  // system prompt 是 prompt-cache 的 prefix；每轮变化的时间戳放在头部会让上游缓存永不命中
  // （In 随上下文线性增长）。放到本轮新消息末尾则不影响稳定 prefix，缓存可命中。
  const timestampContext = `[Context: Current time is ${new Date().toISOString()}]`

  // 构建历史消息（参考 Proxycast 实现）
  const history: KiroHistoryMessage[] = []
  const toolResults: KiroToolResult[] = []
  let currentContent = ''
  let currentCachePoint: KiroCachePoint | undefined
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  for (let i = 0; i < nonSystemMessages.length; i++) {
    const msg = nonSystemMessages[i]
    const isLast = i === nonSystemMessages.length - 1

    if (msg.role === 'user') {
      const { content: userContent, images: userImages, documents: userDocuments, cachePoint } = extractOpenAIContent(msg)
      
      const mergedContent = userContent || 'Continue'
      const messageCachePoint = cachePoint
      
      if (isLast) {
        currentContent = mergedContent
        currentCachePoint = messageCachePoint
        images.push(...userImages)
        documents.push(...userDocuments)
      } else {
        history.push({
          userInputMessage: {
            content: mergedContent,
            modelId,
            origin,
            images: userImages.length > 0 ? userImages : undefined,
            documents: userDocuments.length > 0 ? userDocuments : undefined,
            ...(messageCachePoint ? { cachePoint: messageCachePoint } : {})
          }
        })
      }
    } else if (msg.role === 'assistant') {
      // Kiro API 要求 content 非空
      // 注意: 故意不读取 msg.reasoning_content (history 中不传给 Kiro)
      // Kiro 后端 schema 仅在响应输出中支持 assistantResponseMessage.reasoningContent，
      // 在请求 history 中传入此字段会触发 400 "Improperly formed request"
      let assistantContent = typeof msg.content === 'string' ? msg.content : ''
      if (!assistantContent.trim() && msg.tool_calls && msg.tool_calls.length > 0) {
        assistantContent = ' '
      } else if (!assistantContent.trim()) {
        assistantContent = 'I understand.'
      }
      const toolUses: KiroToolUse[] = []

      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          if (tc.type === 'function') {
            let input = {}
            try {
              input = JSON.parse(tc.function.arguments)
            } catch { /* ignore */ }
            toolUses.push({
              toolUseId: tc.id,
              name: toolNameRegistry.toKiroName(tc.function.name),
              input
            })
          }
        }
      }

      history.push({
        assistantResponseMessage: {
          content: assistantContent,
          toolUses: toolUses.length > 0 ? toolUses : undefined
        }
      })
    } else if (msg.role === 'tool') {
      // Tool result - 收集到待处理列表
      if (msg.tool_call_id) {
        let rawText = ''
        let extractedImageCount = 0
        // content 是数组时（部分客户端把图像/多模态结果挂在这里）：
        // 提取所有 text 块拼接为文本；image_url 块提取到外层 images，避免被 JSON.stringify 序列化丢失
        if (Array.isArray(msg.content)) {
          const textParts: string[] = []
          for (const part of msg.content) {
            if (part.type === 'text' && typeof part.text === 'string') {
              textParts.push(part.text)
            } else if (part.type === 'image_url' && part.image_url?.url) {
              const img = parseImageUrl(part.image_url.url)
              if (img) { images.push(img); extractedImageCount++ }
            }
          }
          rawText = textParts.join('')
          if (!rawText && extractedImageCount === 0) {
            // 退化：把不识别的结构 stringify 让模型至少看到原始结构
            rawText = JSON.stringify(msg.content)
          }
          if (extractedImageCount > 0) {
            rawText = (rawText ? rawText + '\n\n' : '') +
              `[Tool returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message]`
          }
        } else {
          rawText = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
        }
        toolResults.push({
          toolUseId: msg.tool_call_id,
          content: [{ text: rawText || '(no output)' }],
          status: 'success'
        })
      }
      
      // 检查下一条消息：如果不是 tool 消息或已到末尾，将收集的 toolResults 添加为 user 消息
      const nextMsg = nonSystemMessages[i + 1]
      const shouldFlush = !nextMsg || nextMsg.role !== 'tool'
      
      if (shouldFlush && toolResults.length > 0 && !isLast) {
        // 将 toolResults 作为 user 消息添加到 history
        history.push({
          userInputMessage: {
            content: 'Tool results provided.',
            modelId,
            origin,
            userInputMessageContext: {
              toolResults: [...toolResults]
            }
          }
        })
        // 清空已处理的 toolResults
        toolResults.length = 0
      }
    }
  }

  // 如果最后一条是 assistant 消息，自动发送 Continue（参考 Proxycast）
  if (history.length > 0 && history[history.length - 1].assistantResponseMessage && !currentContent) {
    currentContent = 'Continue.'
  }

  // 如果没有当前内容但有工具结果（最后一轮的），保留它们传给 currentMessage
  if (!currentContent && toolResults.length > 0) {
    currentContent = 'Tool results provided.'
  }

  // System prompt 以 Kiro 官方方式注入：作为 Human/AI pair 插入到 history 头部
  if (systemPrompt) {
    const systemMessages: KiroHistoryMessage[] = [
      {
        userInputMessage: {
          content: systemPrompt,
          userInputMessageContext: {},
          origin,
          ...(systemCachePoint ? { cachePoint: systemCachePoint } : {})
        }
      },
      {
        assistantResponseMessage: {
          content: 'I will follow these instructions.'
        }
      }
    ]
    history.unshift(...systemMessages)
  }
  // 时间戳追加到本轮消息末尾（不污染 system/history prefix，保住 prompt-cache 命中）。
  const finalContent = `${currentContent || 'Continue.'}\n\n${timestampContext}`

  // 转换工具定义
  const kiroTools = convertOpenAITools(request.tools, toolNameRegistry)

  // OpenAI 兼容请求的 thinking + reasoning_effort 映射到 Kiro additionalModelRequestFields
  // 仅对支持 thinking 的模型传递（Claude 4+ 系列）
  const additionalModelRequestFields = buildAdditionalModelRequestFields(
    modelId,
    request.thinking,
    request.reasoning_effort
  )

  return buildKiroPayload(
    finalContent,
    modelId,
    origin,
    history,
    kiroTools,
    toolResults,
    images,
    profileArn,
    {
      maxTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p
    },
    {
      cachePoint: currentCachePoint,
      documents,
      conversationId: request.conversation_id,
      context: request.kiro_context
    },
    additionalModelRequestFields,
    request.model // clientModelId：保留 [1m] 等后缀，供裁剪预算按正确 context window 计算
  )
}

function extractOpenAIContent(msg: OpenAIMessage): { content: string; images: KiroImage[]; documents: KiroDocument[]; cachePoint?: KiroCachePoint } {
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  let content = ''
  let cachePoint = toKiroCachePoint(msg.cache_control)

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      cachePoint = mergeCachePoint(cachePoint, toKiroCachePoint(part.cache_control))
      if (part.type === 'text' && part.text) {
        content += part.text
      } else if (part.type === 'image_url' && part.image_url?.url) {
        const image = parseImageUrl(part.image_url.url)
        if (image) {
          images.push(image)
        }
      } else if (part.type === 'file' || part.type === 'document') {
        if (part.file?.file_data) {
          const name = part.file.filename || part.name
          if (!name) {
            throw new Error(`${part.type} requires filename or name`)
          }
          documents.push(parseOpenAIFileData(part.file.file_data, name))
        } else if (part.source) {
          if (!part.name) {
            throw new Error(`${part.type} requires name`)
          }
          documents.push(parseClaudeDocumentSource(part.source, part.name))
        } else {
          throw new Error(`${part.type} requires file_data or source`)
        }
      }
    }
  }

  return { content, images, documents, cachePoint }
}

// 解析图像 URL（支持 data URL 和 HTTP URL）
function parseImageUrl(url: string): KiroImage | null {
  if (url.startsWith('data:')) {
    // 解析 data URL: data:image/png;base64,xxxxx
    const match = url.match(/^data:image\/(\w+);base64,(.+)$/)
    if (match) {
      return {
        format: normalizeImageFormat(match[1]),
        source: { bytes: match[2] }
      }
    }
  }
  return null
}

function parseOpenAIFileData(fileData: string, name: string): KiroDocument {
  const dataUrlMatch = fileData.match(/^data:([^;]+);base64,(.+)$/)
  if (dataUrlMatch) {
    return {
      format: normalizeDocumentFormat(dataUrlMatch[1], name),
      name,
      source: { bytes: dataUrlMatch[2] }
    }
  }

  return {
    format: normalizeDocumentFormat(undefined, name),
    name,
    source: { bytes: fileData }
  }
}

function parseClaudeDocumentSource(source: NonNullable<ClaudeContentBlock['source']>, name: string): KiroDocument {
  if (source.type === 'base64') {
    return {
      format: normalizeDocumentFormat(source.media_type, name),
      name,
      source: { bytes: source.data }
    }
  }
  if (source.type === 'text') {
    return {
      format: normalizeDocumentFormat(source.media_type, name),
      name,
      source: { bytes: Buffer.from(source.data, 'utf8').toString('base64') }
    }
  }
  throw new Error(`Unsupported document source type: ${source.type}`)
}

// 标准化图像格式
function normalizeImageFormat(format: string): string {
  const lower = format.toLowerCase()
  const formatMap: Record<string, string> = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
    'gif': 'gif',
    'webp': 'webp'
  }
  const normalized = formatMap[lower]
  if (!normalized) {
    throw new Error(`Unsupported image format: ${format}`)
  }
  return normalized
}

function normalizeDocumentFormat(mediaType: string | undefined, name: string): string {
  const lowerMediaType = mediaType?.toLowerCase()
  if (lowerMediaType === 'application/pdf') return 'pdf'
  if (lowerMediaType === 'text/markdown') return 'md'
  if (lowerMediaType === 'text/csv') return 'csv'
  if (lowerMediaType === 'text/html') return 'html'
  if (lowerMediaType?.startsWith('text/')) return 'txt'
  const extension = name.split('.').pop()?.toLowerCase()
  if (extension === 'pdf') return 'pdf'
  if (extension === 'md' || extension === 'markdown') return 'md'
  if (extension === 'csv') return 'csv'
  if (extension === 'html' || extension === 'htm') return 'html'
  return 'txt'
}


// Kiro API 工具描述最大长度
const KIRO_MAX_TOOL_DESC_LEN = 10237 // 留出 "..." 的空间

// Anthropic 的 server-side / Anthropic-defined 工具（web_search / computer / text_editor / bash / memory / tool_search 等）
// 用 { type, name } 声明、不带 input_schema，需由供应商在服务端执行或走 native InvokeModel + anthropic_beta。
// 本代理经 CodeWhisperer GenerateAssistantResponse（Amazon Q 专有协议）转发，其 toolSpecification 只有
// { name, description, inputSchema }，没有声明 server 工具的 type 字段，后端也没有对应 handler；
// 传入空 inputSchema 会被以 "inputSchema is empty" 整条请求 400 拒绝。
// 注：AWS 文档明确 "web_search_20250305 server tool is not supported on Amazon Bedrock"；
// 其它 Anthropic-defined 工具即便 Bedrock native 支持，也无法经此 CodeWhisperer 通道执行。
// 故在转换前过滤掉缺少可用 JSON schema 的工具，保留普通 custom 工具正常工作。
// 注意：无参工具的合法 schema 是 { type: 'object' }（可无 properties），不应被误删——
// 仅当 schema 为 undefined/null/非对象/空对象时才判定为不受支持的 server 工具。
function hasUsableInputSchema(inputSchema: unknown): boolean {
  if (inputSchema === null || typeof inputSchema !== 'object') return false
  // 空对象 {} 视为不可用（server 工具常表现为无 schema）；{ type: 'object' } 等有键的对象视为可用
  return Object.keys(inputSchema as Record<string, unknown>).length > 0
}

// OpenAI function 工具一律是 custom 工具（无 server 工具概念），parameters 可选：
// 无参函数是合法用例。但 Bedrock 不接受空 inputSchema，故将缺失/空的 schema 规范化为合法的空对象 schema。
function normalizeCustomToolSchema(schema: unknown): Record<string, unknown> {
  return hasUsableInputSchema(schema) ? (schema as Record<string, unknown>) : { type: 'object', properties: {} }
}

function convertOpenAITools(
  tools: OpenAITool[] | undefined,
  toolNameRegistry: ToolNameRegistry
): KiroToolWrapper[] {
  if (!tools) return []

  return tools.flatMap(tool => {
    let description = tool.function.description || `Tool: ${tool.function.name}`
    // 截断过长的描述
    if (description.length > KIRO_MAX_TOOL_DESC_LEN) {
      description = description.substring(0, KIRO_MAX_TOOL_DESC_LEN) + '...'
    }
    const kiroTool: KiroToolWrapper = {
      toolSpecification: {
        name: shortenToolName(tool.function.name, toolNameRegistry),
        description,
        inputSchema: { json: normalizeCustomToolSchema(tool.function.parameters) }
      }
    }
    const cachePoint = toKiroCachePoint(tool.cache_control)
    return cachePoint ? [kiroTool, { cachePoint }] : [kiroTool]
  })
}

function shortenToolName(name: string, toolNameRegistry: ToolNameRegistry): string {
  return toolNameRegistry.toKiroName(name)
}

// ============ Kiro -> OpenAI 转换 ============

export function kiroToOpenaiResponse(
  content: string,
  toolUses: KiroToolUse[],
  usage: KiroUsage,
  model: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
): OpenAIChatResponse {
  const restoredToolUses = toolNameRegistry.restoreToolUses(toolUses)
  const openaiUsage: OpenAIChatResponse['usage'] = {
    prompt_tokens: usage.inputTokens,
    completion_tokens: usage.outputTokens,
    total_tokens: usage.inputTokens + usage.outputTokens
  }
  if (usage.cacheReadTokens) {
    openaiUsage.prompt_tokens_details = {
      cached_tokens: usage.cacheReadTokens
    }
  }
  if (usage.reasoningTokens) {
    openaiUsage.completion_tokens_details = {
      reasoning_tokens: usage.reasoningTokens
    }
  }
  const response: OpenAIChatResponse = {
    id: `chatcmpl-${uuidv4()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: (restoredToolUses.length > 0 || !content?.trim()) ? null : content,
        ...(reasoningContent?.text ? { reasoning_content: reasoningContent.text } : {}),
        tool_calls: restoredToolUses.length > 0 ? restoredToolUses.map(tu => ({
          id: tu.toolUseId,
          type: 'function' as const,
          function: {
            name: tu.name,
            arguments: JSON.stringify(tu.input)
          }
        })) : undefined
      },
      finish_reason: restoredToolUses.length > 0 ? 'tool_calls' : 'stop'
    }],
    usage: openaiUsage
  }

  return response
}

export interface OpenAIUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  prompt_tokens_details?: {
    cached_tokens?: number
  }
  completion_tokens_details?: {
    reasoning_tokens?: number
  }
}

export function createOpenaiStreamChunk(
  id: string,
  model: string,
  delta: { role?: 'assistant'; content?: string; reasoning_content?: string; tool_calls?: { index: number; id?: string; type?: 'function'; function?: { name?: string; arguments?: string } }[] },
  finishReason: 'stop' | 'tool_calls' | null = null,
  usage?: OpenAIUsage
): OpenAIStreamChunk & { usage?: OpenAIUsage } {
  const chunk: OpenAIStreamChunk & { usage?: OpenAIUsage } = {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta: delta as OpenAIStreamChunk['choices'][0]['delta'],
      finish_reason: finishReason
    }]
  }
  if (usage) {
    chunk.usage = usage
  }
  return chunk
}

// ============ Claude -> Kiro 转换 ============

export function claudeToKiro(
  request: ClaudeRequest,
  profileArn?: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  webToolsEnabled = false
): KiroPayload {
  const modelId = mapModelId(request.model)
  const origin = 'AI_EDITOR'

  // 提取系统提示
  let systemPrompt = ''
  let systemCachePoint: KiroCachePoint | undefined
  if (typeof request.system === 'string') {
    systemPrompt = request.system
  } else if (Array.isArray(request.system)) {
    systemPrompt = request.system.map(b => {
      systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(b.cache_control))
      return b.text
    }).join('\n')
  }

  // 时间戳改注入到 currentMessage 末尾（见下方 finalContent），而非 system prompt 头部。
  // 原因：system prompt 是 history 的首块，构成 prompt-cache 的 prefix。把每轮都变化的
  // 时间戳放在 prefix 头部会让上游缓存（cachePoint）每轮 prefix 都不同 → 永不命中 →
  // 每轮按全量 input 计费（In 随上下文线性增长）。时间戳放到「本轮新消息」末尾不影响
  // 稳定的 system/history/tools prefix，缓存可命中，同时模型仍能看到当前时间。
  const timestampContext = `[Context: Current time is ${new Date().toISOString()}]`

  // 构建历史消息 - Kiro API 要求严格的 user -> assistant 交替
  const history: KiroHistoryMessage[] = []
  let currentToolResults: KiroToolResult[] = []  // 只保存最后一条消息的 toolResults
  let currentContent = ''
  let currentCachePoint: KiroCachePoint | undefined
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []

  // 临时存储，用于合并连续的同类型消息
  let pendingUserContent = ''
  let pendingUserImages: KiroImage[] = []
  let pendingUserDocuments: KiroDocument[] = []
  let pendingToolResults: KiroToolResult[] = []
  let pendingUserCachePoint: KiroCachePoint | undefined

  for (let i = 0; i < request.messages.length; i++) {
    const msg = request.messages[i]
    const isLast = i === request.messages.length - 1

    if (msg.role === 'user') {
      const { content: userContent, images: userImages, documents: userDocuments, toolResults: userToolResults, cachePoint: userCachePoint } = extractClaudeContent(msg)

      if (isLast) {
        // 最后一条消息：合并之前的 pending 内容，toolResults 放入 currentMessage
        currentContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
        images.push(...pendingUserImages, ...userImages)
        documents.push(...pendingUserDocuments, ...userDocuments)
        currentToolResults = [...pendingToolResults, ...userToolResults]
        currentCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
        pendingUserContent = ''
        pendingUserImages = []
        pendingUserDocuments = []
        pendingToolResults = []
        pendingUserCachePoint = undefined
      } else {
        // 非最后一条：检查下一条是否是 assistant
        const nextMsg = request.messages[i + 1]
        if (nextMsg && nextMsg.role === 'assistant') {
          // 下一条是 assistant，可以安全添加到 history
          const finalUserContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
          const finalUserImages = [...pendingUserImages, ...userImages]
          const finalUserDocuments = [...pendingUserDocuments, ...userDocuments]
          const finalToolResults = [...pendingToolResults, ...userToolResults]
          const finalCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
          
          if (finalUserContent.trim() || finalUserImages.length > 0 || finalUserDocuments.length > 0 || finalToolResults.length > 0) {
            const userInputMessage: KiroUserInputMessage = {
              content: finalUserContent || (finalToolResults.length > 0 ? 'Tool results provided.' : 'Continue'),
              modelId,
              origin,
              images: finalUserImages.length > 0 ? finalUserImages : undefined,
              documents: finalUserDocuments.length > 0 ? finalUserDocuments : undefined,
              ...(finalCachePoint ? { cachePoint: finalCachePoint } : {})
            }
            // 如果有 toolResults，放入 userInputMessageContext
            if (finalToolResults.length > 0) {
              userInputMessage.userInputMessageContext = {
                toolResults: finalToolResults
              }
            }
            history.push({ userInputMessage })
          }
          pendingUserContent = ''
          pendingUserImages = []
          pendingUserDocuments = []
          pendingToolResults = []
          pendingUserCachePoint = undefined
        } else {
          // 下一条不是 assistant（可能是连续 user 或结束），累积内容
          pendingUserContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
          pendingUserImages.push(...userImages)
          pendingUserDocuments.push(...userDocuments)
          pendingToolResults.push(...userToolResults)
          pendingUserCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
        }
      }
    } else if (msg.role === 'assistant') {
      // 注意: 故意丢弃 reasoningContent (history 中不传给 Kiro)
      // Kiro 后端 schema 仅在响应输出中支持 assistantResponseMessage.reasoningContent，
      // 在请求 history 中传入此字段会触发 400 "Improperly formed request"
      // 当前消息的 thinking 开关由 additionalModelRequestFields.thinking 控制
      const { content: assistantContent, toolUses } = extractClaudeAssistantContent(msg, toolNameRegistry)

      // 如果有 pending 的 user 内容但还没添加到 history，先添加
      if (pendingUserContent.trim() || pendingUserImages.length > 0 || pendingUserDocuments.length > 0 || pendingToolResults.length > 0) {
        const userInputMessage: KiroUserInputMessage = {
          content: pendingUserContent || (pendingToolResults.length > 0 ? 'Tool results provided.' : 'Continue'),
          modelId,
          origin,
          images: pendingUserImages.length > 0 ? pendingUserImages : undefined,
          documents: pendingUserDocuments.length > 0 ? pendingUserDocuments : undefined,
          ...(pendingUserCachePoint ? { cachePoint: pendingUserCachePoint } : {})
        }
        if (pendingToolResults.length > 0) {
          userInputMessage.userInputMessageContext = {
            toolResults: pendingToolResults
          }
        }
        history.push({ userInputMessage })
        pendingUserContent = ''
        pendingUserImages = []
        pendingUserDocuments = []
        pendingToolResults = []
        pendingUserCachePoint = undefined
      }

      const assistantResponseMessage = {
        content: assistantContent,
        ...(toolUses.length > 0 ? { toolUses } : {})
      }
      history.push({ assistantResponseMessage })
    }
  }

  // 处理剩余的 pending 内容（如果最后几条都是 user 且不是 isLast）
  if (pendingUserContent.trim() || pendingUserImages.length > 0 || pendingUserDocuments.length > 0 || pendingToolResults.length > 0) {
    currentContent = pendingUserContent + (currentContent ? '\n' + currentContent : '')
    images.unshift(...pendingUserImages)
    documents.unshift(...pendingUserDocuments)
    currentToolResults = [...pendingToolResults, ...currentToolResults]
    currentCachePoint = mergeCachePoint(pendingUserCachePoint, currentCachePoint)
  }

  // 确保 history 以 user 开始（Kiro API 要求）
  // 如果 history 以 assistant 开始，在前面插入一个空的 user 消息
  if (history.length > 0 && history[0].assistantResponseMessage) {
    history.unshift({
      userInputMessage: {
        content: 'Begin conversation',
        modelId,
        origin
      }
    })
  }

  // 构建最终内容
  // System prompt 以 Kiro 官方方式注入：作为 Human/AI pair 插入到 history 头部
  // 官方 Kiro IDE: [Human(systemPrompt, forcedRole), AI("I will follow these instructions.", forcedRole)]
  if (systemPrompt) {
    const systemMessages: KiroHistoryMessage[] = [
      {
        userInputMessage: {
          content: systemPrompt,
          userInputMessageContext: {},
          origin,
          ...(systemCachePoint ? { cachePoint: systemCachePoint } : {})
        }
      },
      {
        assistantResponseMessage: {
          content: 'I will follow these instructions.'
        }
      }
    ]
    history.unshift(...systemMessages)
  }
  // 时间戳追加到本轮消息末尾（不污染 system/history prefix，保住 prompt-cache 命中）。
  const baseContent = currentContent || (currentToolResults.length > 0 ? 'Tool results provided.' : 'Continue')
  const finalContent = `${baseContent}\n\n${timestampContext}`

  // 转换工具定义
  const kiroTools = convertClaudeTools(request.tools, toolNameRegistry, webToolsEnabled)

  // 仅传 effort，不传 thinking 字段到 additionalModelRequestFields。
  //
  // 原因（已实测验证）：一旦 payload 里出现 additionalModelRequestFields.thinking，
  // Kiro 后端会把推理内容以「加密」形式返回（只给 signature，没有可读 thinking 文本），
  // 客户付了 reasoning token 的费用却看不到思考过程。反之，不传 thinking 字段时，
  // 后端会回传明文 thinking（客户可见），且 effort 仍正常生效（high effort 实测推理量更大）。
  //
  // 多轮 + signature 续传不受影响：历史里的 thinking block 走 reasoningContent 通道
  // （见本文件 messageToKiro / kiroToClaudeResponse），与此字段无关，已实测 200 通过。
  // 注意：仅改 Claude path；OpenAI path 维持原状。
  const additionalModelRequestFields = buildAdditionalModelRequestFields(
    modelId,
    undefined,
    deriveClaudeEffort(request)
  )

  return buildKiroPayload(
    finalContent,
    modelId,
    origin,
    history,
    kiroTools,
    currentToolResults,
    images,
    profileArn,
    {
      maxTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p
    },
    {
      cachePoint: currentCachePoint,
      documents,
      conversationId: request.conversation_id,
      context: request.kiro_context
    },
    additionalModelRequestFields,
    request.model // clientModelId：保留 [1m] 等后缀，供裁剪预算按正确 context window 计算
  )
}

function extractClaudeContent(msg: ClaudeMessage): { content: string; images: KiroImage[]; documents: KiroDocument[]; toolResults: KiroToolResult[]; cachePoint?: KiroCachePoint } {
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  const toolResults: KiroToolResult[] = []
  let content = ''
  let cachePoint = toKiroCachePoint(msg.cache_control)

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      cachePoint = mergeCachePoint(cachePoint, toKiroCachePoint(block.cache_control))
      if (block.type === 'text' && block.text) {
        content += block.text
      } else if (block.type === 'image' && block.source?.type === 'base64') {
        const mediaTypeParts = block.source.media_type.split('/')
        const imageFormat = mediaTypeParts[1]
        if (mediaTypeParts[0] !== 'image' || !imageFormat) {
          throw new Error(`Unsupported image media_type: ${block.source.media_type}`)
        }
        images.push({
          format: normalizeImageFormat(imageFormat),
          source: { bytes: block.source.data }
        })
      } else if (block.type === 'document' && block.source) {
        if (!block.name) {
          throw new Error('document requires name')
        }
        documents.push(parseClaudeDocumentSource(block.source, block.name))
      } else if (block.type === 'tool_result' && block.tool_use_id) {
        let resultContent = ''
        // Kiro tool_result.content 只支持 text，但用户层 images 可以承载图片。
        // 把内嵌 image block 提取到外层 images，避免「读取本地图片」这类场景图像内容被静默丢弃。
        let extractedImageCount = 0
        if (typeof block.content === 'string') {
          resultContent = block.content || '(empty)'
        } else if (Array.isArray(block.content)) {
          const textParts: string[] = []
          for (const b of block.content) {
            if (b.type === 'text') {
              textParts.push(b.text || '')
            } else if (b.type === 'image' && b.source?.type === 'base64' && b.source.data) {
              const mediaTypeParts = (b.source.media_type || '').split('/')
              const imageFormat = mediaTypeParts[1]
              if (mediaTypeParts[0] === 'image' && imageFormat) {
                try {
                  images.push({
                    format: normalizeImageFormat(imageFormat),
                    source: { bytes: b.source.data }
                  })
                  extractedImageCount++
                } catch {
                  // 不支持的格式：跳过但不抛错（保留旧行为，避免整轮失败）
                }
              }
            }
          }
          resultContent = textParts.join('')
          if (!resultContent) {
            resultContent = extractedImageCount > 0
              ? `(tool returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message)`
              : '(no text output)'
          } else if (extractedImageCount > 0) {
            // 既有文本又有图片：在文本末尾提示模型有附图
            resultContent += `\n\n[Tool also returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message]`
          }
        } else if (block.content === undefined || block.content === null) {
          resultContent = '(no output)'
        } else {
          resultContent = String(block.content) || '(empty)'
        }
        toolResults.push({
          toolUseId: block.tool_use_id,
          content: [{ text: resultContent }],
          status: 'success'
        })
      }
    }
  }

  return { content, images, documents, toolResults, cachePoint }
}

function extractClaudeAssistantContent(
  msg: ClaudeMessage,
  toolNameRegistry: ToolNameRegistry
): { content: string; toolUses: KiroToolUse[]; reasoningContent?: KiroReasoningContent } {
  const toolUses: KiroToolUse[] = []
  let content = ''
  let thinking = ''
  let signature: string | undefined
  let redactedContent: string | undefined

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block.type === 'text' && block.text) {
        content += block.text
      } else if (block.type === 'thinking' && block.thinking) {
        thinking += block.thinking
        signature = block.signature || signature
      } else if (block.type === 'redacted_thinking' && block.data) {
        // redacted_thinking 是加密的思考内容，原样保留
        redactedContent = (redactedContent || '') + block.data
      } else if (block.type === 'tool_use' && block.id && block.name) {
        if (!block.input || typeof block.input !== 'object' || Array.isArray(block.input)) {
          throw new Error(`tool_use requires object input: ${block.name}`)
        }
        toolUses.push({
          toolUseId: block.id,
          name: toolNameRegistry.toKiroName(block.name),
          input: block.input as Record<string, unknown>
        })
      } else if (block.type === 'server_tool_use' || block.type === 'web_search_tool_result') {
        // 多轮场景：客户端会把上一轮我们合成的 server_tool_use / web_search_tool_result
        // 原样回传进 history。这两类是 Anthropic server-tool 专有 block，由代理侧执行并已完成，
        // Kiro/CodeWhisperer 后端不认识它们（且 server_tool_use 不在 tools 声明里，作为 toolUse 回传
        // 会因缺少配对的 tool schema / tool_result 触发 400）。因此显式丢弃，不转成 Kiro toolUse。
        // 真正的回答文本与引用已在同一条 assistant message 的独立 text block 中，照常累加。
        continue
      }
    }
  }

  // Kiro API 要求 content 非空
  if (!content.trim() && toolUses.length > 0) {
    content = ' '
  }

  if (thinking || redactedContent) {
    const reasoningContent: KiroReasoningContent = {}
    if (thinking) {
      reasoningContent.reasoningText = signature ? { text: thinking, signature } : { text: thinking }
    }
    if (redactedContent) {
      reasoningContent.redactedContent = redactedContent
    }
    return { content, toolUses, reasoningContent }
  }

  return { content, toolUses }
}

function convertClaudeTools(
  tools: { name: string; description: string; input_schema: unknown; cache_control?: { type: string }; type?: string }[] | undefined,
  toolNameRegistry: ToolNameRegistry,
  webToolsEnabled = false
): KiroToolWrapper[] {
  if (!tools) return []

  return tools.flatMap(tool => {
    // web_search / web_fetch 是 server-side 工具，客户端不带 input_schema。
    // 若代理启用了 web 工具执行，则转成带 schema 的 custom tool 让 Kiro 模型可调用（代理侧拦截执行）；
    // 否则与其它 server 工具一样丢弃，避免空 inputSchema 触发 Bedrock 400。
    const webKind = isServerWebTool(tool.name, tool.type)
    if (webKind) {
      if (!webToolsEnabled) {
        proxyLogger.warn('Translator', `Dropping web tool "${tool.name}": web search not configured (no provider/apiKey)`)
        return []
      }
      const isSearch = webKind === 'web_search'
      const kiroWebTool: KiroToolWrapper = {
        toolSpecification: {
          name: shortenToolName(isSearch ? WEB_SEARCH_TOOL_NAME : WEB_FETCH_TOOL_NAME, toolNameRegistry),
          description: isSearch ? WEB_SEARCH_TOOL_DESCRIPTION : WEB_FETCH_TOOL_DESCRIPTION,
          inputSchema: { json: isSearch ? WEB_SEARCH_INPUT_SCHEMA : WEB_FETCH_INPUT_SCHEMA }
        }
      }
      return [kiroWebTool]
    }
    // 过滤掉其它 server-side 工具（computer/bash/text_editor 等），缺少可用 inputSchema 会导致 Bedrock 整条 400
    if (!hasUsableInputSchema(tool.input_schema)) {
      proxyLogger.warn('Translator', `Dropping unsupported server-side tool "${tool.name}"${tool.type ? ` (type: ${tool.type})` : ''}: Kiro/CodeWhisperer backend only supports custom tools with inputSchema`)
      return []
    }
    let description = tool.description || `Tool: ${tool.name}`
    // 截断过长的描述
    if (description.length > KIRO_MAX_TOOL_DESC_LEN) {
      description = description.substring(0, KIRO_MAX_TOOL_DESC_LEN) + '...'
    }
    const kiroTool: KiroToolWrapper = {
      toolSpecification: {
        name: shortenToolName(tool.name, toolNameRegistry),
        description,
        inputSchema: { json: tool.input_schema }
      }
    }
    const cachePoint = toKiroCachePoint(tool.cache_control)
    return cachePoint ? [kiroTool, { cachePoint }] : [kiroTool]
  })
}

// ============ Kiro -> Claude 转换 ============

export function kiroToClaudeResponse(
  content: string,
  toolUses: KiroToolUse[],
  usage: KiroUsage,
  model: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
): ClaudeResponse {
  const contentBlocks: ClaudeContentBlock[] = []
  const restoredToolUses = toolNameRegistry.restoreToolUses(toolUses)

  if (reasoningContent?.text) {
    contentBlocks.push(reasoningContent.signature ? {
      type: 'thinking',
      thinking: reasoningContent.text,
      signature: reasoningContent.signature
    } : {
      type: 'thinking',
      thinking: reasoningContent.text
    })
  }
  if (reasoningContent?.redactedContent) {
    contentBlocks.push({
      type: 'redacted_thinking',
      data: reasoningContent.redactedContent
    })
  }

  // 仅在有实际文本内容时添加 text block
  if (content && content.trim()) {
    contentBlocks.push({ type: 'text', text: content })
  }

  for (const tu of restoredToolUses) {
    contentBlocks.push({
      type: 'tool_use',
      id: tu.toolUseId,
      name: tu.name,
      input: tu.input
    })
  }

  const claudeUsage: ClaudeResponse['usage'] = {
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens
  }
  if (usage.cacheWriteTokens) {
    claudeUsage.cache_creation_input_tokens = usage.cacheWriteTokens
  }
  if (usage.cacheReadTokens) {
    claudeUsage.cache_read_input_tokens = usage.cacheReadTokens
  }

  const response: ClaudeResponse = {
    id: `msg_${uuidv4()}`,
    type: 'message',
    role: 'assistant',
    content: contentBlocks,
    model,
    stop_reason: restoredToolUses.length > 0 ? 'tool_use' : 'end_turn',
    stop_sequence: null,
    usage: claudeUsage
  }
  return response
}

export function createClaudeStreamEvent(
  type: ClaudeStreamEvent['type'],
  data?: Partial<ClaudeStreamEvent>
): ClaudeStreamEvent {
  return { type, ...data } as ClaudeStreamEvent
}
