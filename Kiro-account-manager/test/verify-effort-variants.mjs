// Standalone harness: replicate splitEffortSuffix + expandEffortVariants logic verbatim,
// feed adversarial inputs, assert. Run: node test/verify-effort-variants.mjs

let pass = 0, fail = 0
function ok(cond, msg) { if (cond) { pass++ } else { fail++; console.error('FAIL:', msg) } }
function eq(a, b, msg) { ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`) }

// --- verbatim copies from proxyServer.ts ---
function parseClaudeFamilyVersion(id) {
  const m = id.toLowerCase().match(/^claude-(opus|sonnet|haiku)-(\d+)(?:[.-](\d{1,2})(?=$|[^\d]))?/)
  if (!m) return null
  const major = m[2]; const minor = m[3]
  const versionLabel = minor !== undefined ? `${major}.${minor}` : major
  return { family: m[1], version: parseFloat(versionLabel), versionLabel }
}
function splitEffortSuffix(modelId) {
  const m = modelId.match(/^(.+)-(low|medium|high|xhigh|max)$/i)
  if (!m) return { baseId: modelId }
  const base = m[1]
  if (!parseClaudeFamilyVersion(base)) return { baseId: modelId }
  return { baseId: base, effort: m[2].toLowerCase() }
}
const EFFORT_VARIANT_SUFFIXES = ['low', 'medium', 'high', 'xhigh', 'max']
function expandEffortVariants(models) {
  const EFFORT_ORDER = EFFORT_VARIANT_SUFFIXES
  const out = []
  for (const m of models) {
    out.push(m)
    if (!parseClaudeFamilyVersion(m.id)) continue
    const efforts = (m.thinkingEfforts || []).filter(e => EFFORT_ORDER.includes(e.toLowerCase()))
    if (efforts.length === 0) continue
    const ordered = EFFORT_ORDER.filter(e => efforts.some(x => x.toLowerCase() === e))
    for (const eff of ordered) {
      const cap = eff.charAt(0).toUpperCase() + eff.slice(1)
      out.push({ ...m, id: `${m.id}-${eff}`, name: `${m.name} (${cap})`, model_name: `${m.model_name || m.name} (${cap})`, description: `${m.description} — effort: ${eff}`, reasoning: true, root: `${m.id}-${eff}` })
    }
  }
  return out
}

// --- splitEffortSuffix ---
console.log('# splitEffortSuffix')
eq(splitEffortSuffix('claude-opus-4.8-max'), { baseId: 'claude-opus-4.8', effort: 'max' }, 'opus-max splits')
eq(splitEffortSuffix('claude-opus-4.8-high'), { baseId: 'claude-opus-4.8', effort: 'high' }, 'opus-high splits')
eq(splitEffortSuffix('claude-sonnet-4.6-low'), { baseId: 'claude-sonnet-4.6', effort: 'low' }, 'sonnet-low splits')
eq(splitEffortSuffix('CLAUDE-OPUS-4.8-MAX'), { baseId: 'CLAUDE-OPUS-4.8', effort: 'max' }, 'case-insensitive suffix')
// base id must remain valid Claude family
eq(splitEffortSuffix('claude-opus-4.8'), { baseId: 'claude-opus-4.8' }, 'plain opus untouched')
eq(splitEffortSuffix('claude-opus-4-8'), { baseId: 'claude-opus-4-8' }, 'dash version not an effort suffix')
// third-party / non-claude ending in effort word → NOT split
eq(splitEffortSuffix('minimax-m2-max'), { baseId: 'minimax-m2-max' }, 'minimax-max not split (not claude)')
eq(splitEffortSuffix('foo-bar-high'), { baseId: 'foo-bar-high' }, 'foo-high not split')
eq(splitEffortSuffix('deepseek-3.2'), { baseId: 'deepseek-3.2' }, 'deepseek untouched')
eq(splitEffortSuffix('auto'), { baseId: 'auto' }, 'auto untouched')
eq(splitEffortSuffix('gpt-4'), { baseId: 'gpt-4' }, 'gpt-4 untouched')
// edge: effort word but base not claude
eq(splitEffortSuffix('max'), { baseId: 'max' }, 'bare max untouched')
eq(splitEffortSuffix('claude-3.7-sonnet-max'), { baseId: 'claude-3.7-sonnet-max' }, 'claude-3.x (legacy id form) not parsed as family → not split')

// --- expandEffortVariants ---
console.log('# expandEffortVariants')
const opus = { id: 'claude-opus-4.8', name: 'Opus 4.8', model_name: 'Opus 4.8', description: 'X', thinkingEfforts: ['low','medium','high','xhigh','max'], capabilities: {} }
const sonnet = { id: 'claude-sonnet-4.6', name: 'Sonnet 4.6', model_name: 'Sonnet 4.6', description: 'Y', thinkingEfforts: ['low','medium','high','max'], capabilities: {} }
const auto = { id: 'auto', name: 'Auto', description: 'Z', thinkingEfforts: [], capabilities: {} }
const haikuNoEnum = { id: 'claude-haiku-4.5', name: 'Haiku 4.5', description: 'H', thinkingEfforts: [], capabilities: {} }
const deepseek = { id: 'deepseek-3.2', name: 'DS', description: 'D', thinkingEfforts: ['low'], capabilities: {} } // non-claude → no expand

const out = expandEffortVariants([opus, sonnet, auto, haikuNoEnum, deepseek])
const ids = out.map(m => m.id)
// opus → base + 5 variants
eq(ids.filter(i => i.startsWith('claude-opus-4.8')).sort(), ['claude-opus-4.8','claude-opus-4.8-high','claude-opus-4.8-low','claude-opus-4.8-max','claude-opus-4.8-medium','claude-opus-4.8-xhigh'].sort(), 'opus base+5')
// sonnet → base + 4 (no xhigh)
ok(!ids.includes('claude-sonnet-4.6-xhigh'), 'sonnet has no xhigh variant')
eq(ids.filter(i => i.startsWith('claude-sonnet-4.6')).length, 5, 'sonnet base+4')
// auto / haiku-no-enum / deepseek → no variants (only themselves)
eq(ids.filter(i => i.startsWith('auto')), ['auto'], 'auto no variants')
eq(ids.filter(i => i.startsWith('claude-haiku')), ['claude-haiku-4.5'], 'haiku no enum → no variants')
eq(ids.filter(i => i.startsWith('deepseek')), ['deepseek-3.2'], 'deepseek (non-claude) no variants')
// variant ordering follows EFFORT_ORDER (low, medium, high, xhigh, max)
const opusVarOrder = out.filter(m => m.id.startsWith('claude-opus-4.8-')).map(m => m.id.replace('claude-opus-4.8-',''))
eq(opusVarOrder, ['low','medium','high','xhigh','max'], 'opus variant order = EFFORT_ORDER')
// variant naming + reasoning flag
const opusMax = out.find(m => m.id === 'claude-opus-4.8-max')
eq(opusMax.name, 'Opus 4.8 (Max)', 'variant display name')
eq(opusMax.reasoning, true, 'variant reasoning=true')
eq(opusMax.root, 'claude-opus-4.8-max', 'variant root updated')
// base entry preserved unchanged
const opusBase = out.find(m => m.id === 'claude-opus-4.8')
eq(opusBase.name, 'Opus 4.8', 'base name unchanged')

// round-trip: every generated variant id must split back to its base + effort
console.log('# round-trip')
for (const m of out) {
  if (m.id.match(/-(low|medium|high|xhigh|max)$/)) {
    const r = splitEffortSuffix(m.id)
    ok(r.effort && parseClaudeFamilyVersion(r.baseId), `round-trip ${m.id} → base=${r.baseId} effort=${r.effort}`)
  }
}

console.log(`\n${pass} passed, ${fail} failed (logic harness)`)

// === LIVE: boot real ProxyServer, exercise the REAL compiled methods (TS `private` is
// compile-time only → callable at runtime). Proves the config gate is wired on the real
// object, not just this file's logic copy. ===
import { build } from 'esbuild'
import { rawPlugin } from '../scripts/esbuild-plugins.mjs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const tmp = mkdtempSync(join(tmpdir(), 'effortvar-'))
process.on('exit', () => { try { rmSync(tmp, { recursive: true, force: true }) } catch { /* ignore */ } })
const stubElectron = {
  name: 'stub-electron',
  setup(b) {
    b.onResolve({ filter: /^electron$/ }, () => ({ path: 'electron', namespace: 'stub' }))
    b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
      contents: 'export const app = { getPath: () => "/tmp", on: () => {}, whenReady: () => Promise.resolve(), getName: () => "t", getVersion: () => "0" }; export const safeStorage = { isEncryptionAvailable: () => false };',
      loader: 'js'
    }))
  }
}
const outfile = join(tmp, 'proxyServer.cjs')
await build({ entryPoints: [join(here, '../src/main/proxy/proxyServer.ts')], outfile, bundle: true, format: 'cjs', platform: 'node', plugins: [stubElectron, rawPlugin], logLevel: 'silent' })
const { ProxyServer } = createRequire(import.meta.url)(outfile)

console.log('\n# LIVE real-instance config gate')
const sampleModels = [
  { id: 'claude-opus-4.8', name: 'Opus 4.8', model_name: 'Opus 4.8', description: 'X', thinkingEfforts: ['low','medium','high','xhigh','max'], capabilities: {} },
  { id: 'auto', name: 'Auto', description: 'Z', thinkingEfforts: [], capabilities: {} }
]

// OFF: resolveEffortVariant must NOT split (feature gate), expandEffortVariants only via handleModels gate
{
  const srv = new ProxyServer({ host: '127.0.0.1', port: 0, effortVariantsExposed: false }, {})
  eq(srv.resolveEffortVariant('claude-opus-4.8-max'), { baseId: 'claude-opus-4.8-max' }, 'OFF: resolveEffortVariant does NOT split')
  // Cursor names carry a user-chosen effort ("max thinking") + thinking marker. Even with the
  // variant gate OFF, a Cursor name must still resolve to {base, effort} (wasReversed=true),
  // else the user's explicit effort is silently dropped. Real strings (family-first, dash
  // version, -thinking mid-token) confirmed from a live Cursor request.
  eq(srv.resolveEffortVariant('claude-opus-4-8-thinking-max'), { baseId: 'claude-opus-4.8', effort: 'max' }, 'OFF: Cursor real Opus string still splits effort')
  eq(srv.resolveEffortVariant('claude-sonnet-4-6-thinking-max'), { baseId: 'claude-sonnet-4.6', effort: 'max' }, 'OFF: Cursor real Sonnet string still splits effort')
  eq(srv.resolveEffortVariant('claude-opus-4-8-thinking'), { baseId: 'claude-opus-4.8' }, 'OFF: Cursor name w/o effort → base only, no leak')
  eq(srv.resolveEffortVariant('claude-4.6-sonnet-max-thinking'), { baseId: 'claude-sonnet-4.6', effort: 'max' }, 'OFF: Cursor version-first name still splits effort')
  // expandEffortVariants itself has no gate (the gate is in handleModels) — but resolve gate is the request-path guard
}
// ON: resolveEffortVariant splits; expandEffortVariants adds entries
{
  const srv = new ProxyServer({ host: '127.0.0.1', port: 0, effortVariantsExposed: true }, {})
  eq(srv.resolveEffortVariant('claude-opus-4.8-max'), { baseId: 'claude-opus-4.8', effort: 'max' }, 'ON: resolveEffortVariant splits')
  eq(srv.resolveEffortVariant('deepseek-3.2'), { baseId: 'deepseek-3.2' }, 'ON: non-claude untouched')
  const expanded = srv.expandEffortVariants(sampleModels)
  ok(expanded.some(m => m.id === 'claude-opus-4.8-max'), 'ON: expand adds opus-max')
  ok(expanded.filter(m => m.id.startsWith('auto')).length === 1, 'ON: auto gets no variant')
  ok(expanded.find(m => m.id === 'claude-opus-4.8-max').reasoning === true, 'ON: variant reasoning=true on real instance')
}

console.log(`\n${pass} passed, ${fail} failed (total, pre-HTTP)`)

// === LIVE HTTP: hit /v1/models over real HTTP, inject a modelCache so getAvailableModels
// returns a Claude model whose schema yields a real effort enum. Proves end-to-end:
// OFF → response has NO -max/-high ids (no always-add regression); ON → variants present. ===
import http from 'node:http'
function httpGet(port, path) {
  return new Promise((resolve, reject) => {
    const r = http.request({ host: '127.0.0.1', port, path, method: 'GET' }, (res) => {
      let buf = ''; res.on('data', d => buf += d)
      res.on('end', () => { let j = null; try { j = JSON.parse(buf) } catch {} resolve({ status: res.statusCode, json: j }) })
    })
    r.on('error', reject); r.end()
  })
}
// KiroModel whose additionalModelRequestFieldsSchema makes extractThinkingEfforts return [low,medium,high,xhigh,max]
const kiroModel = {
  modelId: 'claude-opus-4.8', modelName: 'Opus 4.8', description: 'opus',
  supportedInputTypes: ['TEXT', 'IMAGE'],
  tokenLimits: { maxInputTokens: 200000, maxOutputTokens: 64000 },
  additionalModelRequestFieldsSchema: {
    properties: {
      thinking: { properties: { type: { enum: ['adaptive', 'disabled'] } } },
      output_config: { properties: { effort: { enum: ['low', 'medium', 'high', 'xhigh', 'max'] } } }
    }
  }
}
async function bootWithModels(cfg) {
  const ow = console.warn, ol = console.log; console.warn = () => {}; console.log = () => {}
  const srv = new ProxyServer({ host: '127.0.0.1', port: 0, ...cfg }, {})
  await srv.start()
  console.warn = ow; console.log = ol
  srv.modelCache = { models: [kiroModel], timestamp: Date.now() }
  return { srv, port: srv.server.address().port }
}

console.log('\n# LIVE HTTP /v1/models')
{
  const { srv, port } = await bootWithModels({ effortVariantsExposed: false })
  try {
    const r = await httpGet(port, '/v1/models')
    const ids = (r.json?.data || []).map(m => m.id)
    ok(r.status === 200, `OFF: /v1/models 200 (got ${r.status})`)
    ok(ids.includes('claude-opus-4.8'), 'OFF: base opus present')
    ok(!ids.some(i => /-(low|medium|high|xhigh|max)$/.test(i)), `OFF: NO effort-variant ids (regression guard) — got ${JSON.stringify(ids)}`)
  } finally { await srv.stop(0) }
}
{
  const { srv, port } = await bootWithModels({ effortVariantsExposed: true })
  try {
    const r = await httpGet(port, '/v1/models')
    const ids = (r.json?.data || []).map(m => m.id)
    ok(r.status === 200, `ON: /v1/models 200 (got ${r.status})`)
    ok(ids.includes('claude-opus-4.8'), 'ON: base opus still present')
    ok(ids.includes('claude-opus-4.8-max'), `ON: opus-max variant present — got ${JSON.stringify(ids)}`)
    ok(ids.includes('claude-opus-4.8-high'), 'ON: opus-high variant present')
    ok(ids.includes('claude-opus-4.8-xhigh'), 'ON: opus-xhigh variant present (opus enum)')
  } finally { await srv.stop(0) }
}

console.log(`\n${pass} passed, ${fail} failed (total)`)
process.exit(fail ? 1 : 0)


