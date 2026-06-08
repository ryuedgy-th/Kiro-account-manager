// Verify mapModelId handles model-ID variants Kiro's backend accepts, especially
// the date-snapshot suffix that broke Claude Code subagents.
//
// Root cause (2026-06): Claude Code subagents default to model id
// `claude-haiku-4-5-20251001`. The proxy normalized dashes→dots but kept the
// trailing -YYYYMMDD date, producing `claude-haiku-4.5-20251001`, which Kiro
// rejects with `INVALID_MODEL_ID`. The fix strips the date snapshot before
// passthrough. The main session model `claude-opus-4-8[1m]` was never the
// culprit — proxy maps it to `claude-opus-4.8`, which Kiro accepts.
//
// Run: node test/verify-model-mapping.mjs
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const tmp = mkdtempSync(join(tmpdir(), 'modelmap-'))
process.on('exit', () => { try { rmSync(tmp, { recursive: true, force: true }) } catch { /* ignore */ } })

const stubElectron = {
  name: 'stub-electron',
  setup(b) {
    b.onResolve({ filter: /^electron$/ }, () => ({ path: 'electron', namespace: 'stub' }))
    b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
      contents: 'export const app = { getPath: () => "/tmp", on: () => {}, whenReady: () => Promise.resolve() }; export default {};',
      loader: 'js'
    }))
  }
}

const outfile = join(tmp, 'kiroApi.cjs')
await build({
  entryPoints: [join(here, '../src/main/proxy/kiroApi.ts')],
  outfile, bundle: true, format: 'cjs', platform: 'node',
  plugins: [stubElectron], logLevel: 'silent'
})
const { mapModelId, canonicalizeModelId } = createRequire(import.meta.url)(outfile)

let pass = 0, fail = 0
function check(name, cond) {
  if (cond) { console.log(`  ok   - ${name}`); pass++ }
  else { console.log(`  FAIL - ${name}`); fail++ }
}

console.log('mapModelId — date-snapshot suffix (the subagent bug):')
check('claude-haiku-4-5-20251001 → claude-haiku-4.5', mapModelId('claude-haiku-4-5-20251001') === 'claude-haiku-4.5')
check('claude-haiku-4-5-20251001[1m] → claude-haiku-4.5', mapModelId('claude-haiku-4-5-20251001[1m]') === 'claude-haiku-4.5')
check('claude-opus-4-5-20250101 → claude-opus-4.5', mapModelId('claude-opus-4-5-20250101') === 'claude-opus-4.5')
check('claude-sonnet-4-5-20250929 → claude-sonnet-4.5', mapModelId('claude-sonnet-4-5-20250929') === 'claude-sonnet-4.5')

console.log('mapModelId — capability suffix [1m] (never the culprit):')
check('claude-opus-4-8[1m] → claude-opus-4.8', mapModelId('claude-opus-4-8[1m]') === 'claude-opus-4.8')
check('claude-sonnet-4-5[1m] → claude-sonnet-4.5', mapModelId('claude-sonnet-4-5[1m]') === 'claude-sonnet-4.5')

console.log('mapModelId — no regression on existing cases:')
check('claude-haiku-4-5 → claude-haiku-4.5', mapModelId('claude-haiku-4-5') === 'claude-haiku-4.5')
check('claude-sonnet-4-20250514 → claude-sonnet-4 (explicit map kept)', mapModelId('claude-sonnet-4-20250514') === 'claude-sonnet-4')
check('claude-sonnet-4 → claude-sonnet-4', mapModelId('claude-sonnet-4') === 'claude-sonnet-4')
check('gpt-4o alias → claude-sonnet-4.5', mapModelId('gpt-4o') === 'claude-sonnet-4.5')
check('deepseek-3.2 passthrough', mapModelId('deepseek-3.2') === 'deepseek-3.2')
check('auto passthrough', mapModelId('auto') === 'auto')
check('empty → default', mapModelId('') === 'claude-sonnet-4.5')

// canonicalizeModelId powers the Model Allowlist (isModelAllowed compares canonical forms).
// Bug (2026-06): it stripped [1m] + dash→dot but NOT the -YYYYMMDD date snapshot, while
// mapModelId DID strip it. So a subagent/workflow request `claude-haiku-4-5-20251001`
// canonicalized to `claude-haiku-4.5-20251001` and never matched the allowlist entry
// `claude-haiku-4.5` → 403. Main session model (no date) passed, masking the bug.
console.log('\ncanonicalizeModelId — allowlist matching (the subagent 403 bug):')
const canonEq = (a, b) => canonicalizeModelId(a) === canonicalizeModelId(b)
check('date-suffix request matches dated-stripped allowlist entry',
  canonEq('claude-haiku-4-5-20251001', 'claude-haiku-4.5'))
check('date + [1m] suffix still matches',
  canonEq('claude-haiku-4-5-20251001[1m]', 'claude-haiku-4.5'))
check('opus date-suffix matches canonical',
  canonEq('claude-opus-4-5-20250101', 'claude-opus-4.5'))
check('sonnet date-suffix matches canonical',
  canonEq('claude-sonnet-4-5-20250929', 'claude-sonnet-4.5'))
check('claude-opus-4-8[1m] matches claude-opus-4.8 (original [1m] case still works)',
  canonEq('claude-opus-4-8[1m]', 'claude-opus-4.8'))
check('dash and dot forms unify', canonEq('claude-haiku-4-5', 'claude-haiku-4.5'))

console.log('canonicalizeModelId — must NOT over-strip / collide:')
// 8-digit date strip is anchored to Claude family only — don't mangle non-Claude ids
check('non-Claude id with trailing digits untouched',
  canonicalizeModelId('qwen3-coder-next') === 'qwen3-coder-next')
check('distinct models stay distinct',
  canonicalizeModelId('claude-opus-4.8') !== canonicalizeModelId('claude-opus-4.5'))
check('haiku 4.5 ≠ sonnet 4.5 after canonicalize',
  canonicalizeModelId('claude-haiku-4-5-20251001') !== canonicalizeModelId('claude-sonnet-4-5-20250929'))
check('case-insensitive', canonEq('CLAUDE-OPUS-4-8', 'claude-opus-4.8'))
check('empty string → empty (no crash)', canonicalizeModelId('') === '')

// Cursor sends reversed names: claude-{version}-{family}[-{effort}]-thinking
// (e.g. claude-4.6-sonnet-max-thinking). Our parsers assumed claude-{family}-{version},
// so every reversed name failed to parse → mapModelId fell back to default (silent
// downgrade to Sonnet 4.5) or leaked -max-thinking to the backend → 400 INVALID_MODEL_ID.
// mapModelId/canonicalizeModelId must reorder + strip effort/thinking as a safety net.
console.log('\nmapModelId — Cursor reversed naming (the Opus-fails/Sonnet-downgrades bug):')
check('claude-4.6-sonnet-max-thinking → claude-sonnet-4.6', mapModelId('claude-4.6-sonnet-max-thinking') === 'claude-sonnet-4.6')
check('claude-4.8-opus-xhigh-thinking → claude-opus-4.8', mapModelId('claude-4.8-opus-xhigh-thinking') === 'claude-opus-4.8')
check('claude-4.8-opus (no effort/thinking) → claude-opus-4.8', mapModelId('claude-4.8-opus') === 'claude-opus-4.8')
check('claude-4.5-haiku → claude-haiku-4.5', mapModelId('claude-4.5-haiku') === 'claude-haiku-4.5')
check('claude-4.6-sonnet-thinking (no effort) → claude-sonnet-4.6', mapModelId('claude-4.6-sonnet-thinking') === 'claude-sonnet-4.6')
check('dash-version reversed claude-4-8-opus → claude-opus-4.8', mapModelId('claude-4-8-opus') === 'claude-opus-4.8')

console.log('mapModelId — reversed naming must NOT touch legacy 3.x aliases:')
check('claude-3-opus stays mapped (NOT reordered)', mapModelId('claude-3-opus') === 'claude-sonnet-4.5')
check('claude-3-5-sonnet stays mapped', mapModelId('claude-3-5-sonnet') === 'claude-sonnet-4.5')

console.log('canonicalizeModelId — reversed names match canonical allowlist entries:')
check('reversed sonnet matches canonical', canonEq('claude-4.6-sonnet-max-thinking', 'claude-sonnet-4.6'))
check('reversed opus matches canonical', canonEq('claude-4.8-opus-xhigh-thinking', 'claude-opus-4.8'))
check('reversed haiku matches canonical', canonEq('claude-4.5-haiku', 'claude-haiku-4.5'))
check('reversed opus ≠ reversed sonnet (no collision)',
  canonicalizeModelId('claude-4.8-opus') !== canonicalizeModelId('claude-4.6-sonnet'))

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
