#!/usr/bin/env tsx
/**
 * Maxplus Effort Parameter Test
 * ทดสอบว่า effort parameter มีผลจริงหรือไม่
 */

interface TestResult {
  effort: string
  inputTokens: number
  outputTokens: number
  responseTimeMs: number
  answer: string
  thinkingContent?: string
  error?: string
}

// โจทย์ทดสอบ - เลือกโจทย์ที่ต้องใช้ reasoning
const TEST_PROMPTS = {
  logic: `แก้ปริศนานี้:
มีชายสามคนข้ามแม่น้ำพร้อมกับไก่ สุนัขจิ้งจอก และข้าวสาลี
เรือรับได้แค่คนคนเดียวกับสิ่งของชิ้นเดียว
ห้ามทิ้งไก่กับสุนัขจิ้งจอกไว้ด้วยกัน และห้ามทิ้งสุนัขจิ้งจอกกับข้าวสาลีไว้ด้วยกัน
จะข้ามแม่น้ำได้อย่างไร? อธิบายทีละขั้นตอน`,

  math: `แก้โจทย์นี้:
ถ้า x^3 - 6x^2 + 11x - 6 = 0
จงหาคำตอบทั้งหมดของ x และอธิบายวิธีแก้อย่างละเอียด`,

  code: `หา bug ในโค้ดนี้:
\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
\`\`\`
อธิบายว่า bug อยู่ตรงไหนและจะแก้อย่างไร`
}

const EFFORT_LEVELS = ['low', 'medium', 'high', 'max']

async function testEffortLevel(
  apiKey: string,
  baseUrl: string,
  model: string,
  prompt: string,
  effort: string
): Promise<TestResult> {
  const startTime = Date.now()

  try {
    const response = await fetch(`${baseUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: model,
        max_tokens: 4096,
        messages: [{ role: 'user', content: prompt }],
        thinking: { type: 'enabled' },
        output_config: { effort: effort }
      })
    })

    const responseTimeMs = Date.now() - startTime
    const data = await response.json()

    if (!response.ok) {
      return {
        effort,
        inputTokens: 0,
        outputTokens: 0,
        responseTimeMs,
        answer: '',
        error: `HTTP ${response.status}: ${JSON.stringify(data)}`
      }
    }

    // Extract thinking and text content
    const thinkingBlock = data.content?.find((b: any) => b.type === 'thinking')
    const textBlock = data.content?.find((b: any) => b.type === 'text')

    return {
      effort,
      inputTokens: data.usage?.input_tokens || 0,
      outputTokens: data.usage?.output_tokens || 0,
      responseTimeMs,
      answer: textBlock?.text || '',
      thinkingContent: thinkingBlock?.thinking || undefined
    }
  } catch (error) {
    return {
      effort,
      inputTokens: 0,
      outputTokens: 0,
      responseTimeMs: Date.now() - startTime,
      answer: '',
      error: error instanceof Error ? error.message : String(error)
    }
  }
}

async function runTest(
  apiKey: string,
  baseUrl: string,
  model: string,
  promptType: keyof typeof TEST_PROMPTS
) {
  const prompt = TEST_PROMPTS[promptType]
  console.log(`\n${'='.repeat(80)}`)
  console.log(`Testing: ${promptType.toUpperCase()}`)
  console.log(`${'='.repeat(80)}`)
  console.log(`Prompt: ${prompt.substring(0, 100)}...`)
  console.log(`Model: ${model}`)
  console.log(`\nTesting effort levels: ${EFFORT_LEVELS.join(', ')}`)
  console.log(`${'='.repeat(80)}\n`)

  const results: TestResult[] = []

  for (const effort of EFFORT_LEVELS) {
    console.log(`Testing effort="${effort}"...`)
    const result = await testEffortLevel(apiKey, baseUrl, model, prompt, effort)
    results.push(result)

    if (result.error) {
      console.log(`  ❌ Error: ${result.error}`)
    } else {
      console.log(`  ⏱️  Time: ${result.responseTimeMs}ms`)
      console.log(`  📊 Tokens: ${result.inputTokens} in / ${result.outputTokens} out`)
      if (result.thinkingContent) {
        console.log(`  🧠 Thinking: ${result.thinkingContent.length} chars`)
      }
    }
    console.log('')

    // Wait a bit between requests to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000))
  }

  return results
}

function analyzeResults(results: TestResult[]) {
  console.log(`\n${'='.repeat(80)}`)
  console.log(`ANALYSIS`)
  console.log(`${'='.repeat(80)}\n`)

  const validResults = results.filter(r => !r.error)

  if (validResults.length < 2) {
    console.log('❌ Not enough valid results to analyze')
    return
  }

  // Compare tokens
  console.log('📊 Token Usage Comparison:')
  console.log('┌──────────┬─────────────┬──────────────┬────────────┐')
  console.log('│  Effort  │ Input Tokens│ Output Tokens│ Time (ms)  │')
  console.log('├──────────┼─────────────┼──────────────┼────────────┤')
  validResults.forEach(r => {
    console.log(`│ ${r.effort.padEnd(8)} │ ${String(r.inputTokens).padStart(11)} │ ${String(r.outputTokens).padStart(13)} │ ${String(r.responseTimeMs).padStart(10)} │`)
  })
  console.log('└──────────┴─────────────┴──────────────┴────────────┘\n')

  // Calculate variations
  const outputTokens = validResults.map(r => r.outputTokens)
  const responseTimes = validResults.map(r => r.responseTimeMs)
  const thinkingLengths = validResults.map(r => r.thinkingContent?.length || 0)

  const maxOutput = Math.max(...outputTokens)
  const minOutput = Math.min(...outputTokens)
  const maxTime = Math.max(...responseTimes)
  const minTime = Math.min(...responseTimes)
  const maxThinking = Math.max(...thinkingLengths)
  const minThinking = Math.min(...thinkingLengths)

  const outputVariation = ((maxOutput - minOutput) / minOutput * 100).toFixed(1)
  const timeVariation = ((maxTime - minTime) / minTime * 100).toFixed(1)
  const thinkingVariation = minThinking > 0 ? ((maxThinking - minThinking) / minThinking * 100).toFixed(1) : 'N/A'

  console.log('📈 Variation Analysis:')
  console.log(`   Output Tokens: ${minOutput} → ${maxOutput} (+${outputVariation}%)`)
  console.log(`   Response Time: ${minTime}ms → ${maxTime}ms (+${timeVariation}%)`)
  console.log(`   Thinking Length: ${minThinking} → ${maxThinking} chars (+${thinkingVariation}%)`)
  console.log('')

  // Verdict
  console.log('🎯 Verdict:')
  const hasTokenImpact = parseFloat(outputVariation) > 10
  const hasTimeImpact = parseFloat(timeVariation) > 20
  const hasThinkingImpact = thinkingVariation !== 'N/A' && parseFloat(thinkingVariation) > 10

  if (hasTokenImpact && hasTimeImpact) {
    console.log('   ✅ effort parameter HAS REAL IMPACT')
    console.log('      - Output tokens increase with effort')
    console.log('      - Response time increases with effort')
    if (hasThinkingImpact) {
      console.log('      - Thinking content increases with effort')
    }
  } else if (hasTokenImpact || hasTimeImpact) {
    console.log('   ⚠️  effort parameter has PARTIAL IMPACT')
    console.log(`      - Token impact: ${hasTokenImpact ? 'YES' : 'NO'}`)
    console.log(`      - Time impact: ${hasTimeImpact ? 'YES' : 'NO'}`)
    console.log(`      - Thinking impact: ${hasThinkingImpact ? 'YES' : 'NO'}`)
  } else {
    console.log('   ❌ effort parameter has NO SIGNIFICANT IMPACT')
    console.log('      - Tokens stay roughly the same')
    console.log('      - Response time stays roughly the same')
    console.log('      → May be a passthrough parameter with no backend effect')
  }
  console.log('')
}

async function main() {
  // Get configuration from environment or command line
  const apiKey = process.env.MAXPLUS_API_KEY || process.argv[2]
  const baseUrl = process.env.MAXPLUS_BASE_URL || process.argv[3] || 'https://api.maxplus.com'
  const model = process.env.MAXPLUS_MODEL || process.argv[4] || 'claude-sonnet-4.6'
  const testType = (process.argv[5] || 'logic') as keyof typeof TEST_PROMPTS

  if (!apiKey) {
    console.error('Error: API key required')
    console.error('Usage: tsx test-effort-impact.ts <API_KEY> [BASE_URL] [MODEL] [TEST_TYPE]')
    console.error('Or set environment variables: MAXPLUS_API_KEY, MAXPLUS_BASE_URL, MAXPLUS_MODEL')
    console.error('')
    console.error('TEST_TYPE options: logic, math, code')
    process.exit(1)
  }

  console.log('Maxplus Effort Impact Test')
  console.log(`API Endpoint: ${baseUrl}`)
  console.log(`Model: ${model}`)
  console.log(`Test Type: ${testType}`)

  const results = await runTest(apiKey, baseUrl, model, testType)
  analyzeResults(results)

  // Save detailed results
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const filename = `effort-test-${testType}-${timestamp}.json`
  await Bun.write(filename, JSON.stringify(results, null, 2))
  console.log(`\n📝 Detailed results saved to: ${filename}`)
}

main().catch(console.error)
