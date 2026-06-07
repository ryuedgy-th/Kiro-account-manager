// Verify the credit-exhaustion gate returns NON-retryable billing errors, not 429.
//
// Regression for the "topped up but client keeps retrying forever" report:
//   - Customer prepaid balance <= 0  → reason 'Insufficient credit balance' → caller maps 402
//   - Standalone key over creditsLimit → reason 'Credits limit exceeded'     → caller maps 402
//   - Customer-bound key must IGNORE per-key creditsLimit (Gate A); only prepaid
//     creditBalance (Gate B) gates it, so an admin/slip top-up actually unblocks it.
//   - Disabled customer → reason 'Account disabled' → caller maps 403
//   - getAnthropicErrorType(402) → 'billing_error' (terminal, not rate_limit_error)
//
// validateApiKey is private but reachable on the instance from the bundled CJS.
// We drive it with a minimal mock IncomingMessage carrying the x-api-key header.
//
// Run: node test/verify-credit-gate.mjs
import { build } from 'esbuild'
import { rawPlugin } from '../scripts/esbuild-plugins.mjs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const tmp = mkdtempSync(join(tmpdir(), 'creditgate-'))
process.on('exit', () => { try { rmSync(tmp, { recursive: true, force: true }) } catch { /* ignore */ } })

const stubElectron = {
  name: 'stub-electron',
  setup(b) {
    b.onResolve({ filter: /^electron$/ }, () => ({ path: 'electron', namespace: 'stub' }))
    b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
      contents: 'export const app = { getPath: () => "/tmp", on: () => {}, whenReady: () => Promise.resolve(), getName: () => "test", getVersion: () => "0" }; export const safeStorage = { isEncryptionAvailable: () => false };',
      loader: 'js'
    }))
  }
}

const outfile = join(tmp, 'proxyServer.cjs')
await build({
  entryPoints: [join(here, '../src/main/proxy/proxyServer.ts')],
  outfile, bundle: true, format: 'cjs', platform: 'node',
  plugins: [stubElectron, rawPlugin], logLevel: 'silent'
})
const require = createRequire(import.meta.url)
const { ProxyServer } = require(outfile)

let failed = 0
const ok = (cond, label) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} - ${label}`); if (!cond) failed++ }

console.log('credit-exhaustion gate → non-retryable billing status:\n')

const reqWith = (key) => ({ headers: { 'x-api-key': key } })

// Mirror the status-mapping the request handler applies to validateApiKey's reason.
// (Kept in sync with proxyServer.ts around the validateApiKey call site.)
function statusFor(reason) {
  const isBilling = reason === 'Credits limit exceeded' || reason === 'Insufficient credit balance'
  const isDisabled = reason === 'Account disabled'
  return isBilling ? 402 : isDisabled ? 403 : 401
}

function makeServer({ balance = 1000, enabled = true, creditsLimit, totalCredits = 0, standalone = false } = {}) {
  // standalone = no owning customer (legacy key, gated by per-key creditsLimit).
  // Note: can't express this via `customerId: undefined` in the caller — a default
  // param would override undefined back to 'cust-1'. Use an explicit flag.
  const customerId = standalone ? undefined : 'cust-1'
  const apiKey = {
    id: 'key-1', name: 'k', key: 'sk-1', format: 'sk', enabled: true, createdAt: 0,
    ...(customerId ? { customerId } : {}),
    ...(creditsLimit !== undefined ? { creditsLimit } : {}),
    usage: { totalRequests: 0, totalCredits, totalInputTokens: 0, totalOutputTokens: 0, daily: {} }
  }
  const customers = customerId ? [{ id: customerId, creditBalance: balance, enabled, name: 'c' }] : []
  const server = new ProxyServer({ apiKeys: [apiKey], customers }, {})
  return { server, apiKey, customers }
}

// === 1. Prepaid balance exhausted → 402 (not 429) ===
{
  const { server } = makeServer({ balance: 0 })
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === false, 'balance 0 → rejected')
  ok(r.reason === 'Insufficient credit balance', `reason = Insufficient credit balance (got ${r.reason})`)
  ok(statusFor(r.reason) === 402, `→ HTTP 402, never 429 (got ${statusFor(r.reason)})`)
}

// === 2. Negative balance (slight overage from last request) → 402 ===
{
  const { server } = makeServer({ balance: -5 })
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === false && statusFor(r.reason) === 402, 'negative balance → 402')
}

// === 3. Top-up restores access: balance > 0 → valid ===
{
  const { server, customers } = makeServer({ balance: 0 })
  ok(server.validateApiKey(reqWith('sk-1')).valid === false, 'pre-topup: rejected')
  customers[0].creditBalance = 100 // simulate topupCustomer mutating the same config object
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === true, 'post-topup: same key now valid (no stale cache)')
}

// === 4. Gate A bug fix: customer-bound key IGNORES per-key creditsLimit ===
//   Previously a customer key over creditsLimit stayed blocked even after top-up
//   (top-up doesn't reset usage). Now creditBalance alone gates customer keys.
{
  const { server } = makeServer({ balance: 500, creditsLimit: 10, totalCredits: 9999 })
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === true, 'customer key over per-key creditsLimit but with balance → valid (Gate A skipped)')
}

// === 5. Standalone key (no customer) still honors creditsLimit → 402 ===
{
  const { server } = makeServer({ standalone: true, creditsLimit: 10, totalCredits: 10 })
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === false, 'standalone key at creditsLimit → rejected')
  ok(r.reason === 'Credits limit exceeded', `reason = Credits limit exceeded (got ${r.reason})`)
  ok(statusFor(r.reason) === 402, `→ HTTP 402 (got ${statusFor(r.reason)})`)
}

// === 6. Disabled customer → 403 (billing top-up won't unblock) ===
{
  const { server } = makeServer({ balance: 500, enabled: false })
  const r = server.validateApiKey(reqWith('sk-1'))
  ok(r.valid === false, 'disabled customer → rejected')
  ok(r.reason === 'Account disabled', `reason = Account disabled (got ${r.reason})`)
  ok(statusFor(r.reason) === 403, `→ HTTP 403 (got ${statusFor(r.reason)})`)
}

// === 7. Anthropic error type for 402 is terminal (billing_error, not rate_limit_error) ===
{
  const { server } = makeServer()
  const t = server.getAnthropicErrorType(402)
  ok(t === 'billing_error', `getAnthropicErrorType(402) = billing_error (got ${t})`)
  ok(server.getAnthropicErrorType(429) === 'rate_limit_error', '429 still rate_limit_error (real rate limits unchanged)')
}

console.log(`\n${failed === 0 ? 'All' : failed + ' of'} credit-gate checks ${failed === 0 ? 'passed.' : 'FAILED.'}`)
process.exit(failed === 0 ? 0 : 1)
