// Verify token-refresh resilience helpers (src/main/auth/tokenRefresh.ts):
//   isTransientRefreshError() — classifies network-layer errors worth retrying
//                               (must NOT match HTTP 401/403/invalid_grant app errors)
//   withRefreshRetry()        — retries transient errors with backoff; non-transient
//                               and exhausted attempts rethrow. Fixes batch-refresh
//                               "fetch failed" storm where N parallel TLS handshakes
//                               reset sockets and the no-retry path failed permanently.
//
// tokenRefresh.ts imports ./proxy/systemProxy which pulls electron — stub it out.
// Run: node test/verify-refresh-retry.mjs
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const tmp = mkdtempSync(join(tmpdir(), 'refresh-retry-'))
process.on('exit', () => { try { rmSync(tmp, { recursive: true, force: true }) } catch { /* ignore */ } })

const stubElectron = {
  name: 'stub-electron',
  setup(b) {
    b.onResolve({ filter: /^electron$/ }, () => ({ path: 'electron', namespace: 'stub' }))
    b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
      contents: 'export const app = { getPath: () => "/tmp", on: () => {}, whenReady: () => Promise.resolve(), getName: () => "t", getVersion: () => "0" };',
      loader: 'js'
    }))
  }
}

const outfile = join(tmp, 'tokenRefresh.cjs')
await build({
  entryPoints: [join(here, '../src/main/auth/tokenRefresh.ts')],
  outfile, bundle: true, format: 'cjs', platform: 'node',
  plugins: [stubElectron], logLevel: 'silent'
})
const require = createRequire(import.meta.url)
const { isTransientRefreshError, withRefreshRetry } = require(outfile)

let pass = 0, fail = 0
const ok = (cond, msg) => { if (cond) { pass++; console.log(`  ok   - ${msg}`) } else { fail++; console.log(`  FAIL - ${msg}`) } }

// helper to mimic undici's error shape (TypeError fetch failed + cause with code)
const withCause = (msg, causeMsg, code) => {
  const e = new Error(msg)
  e.cause = code ? { message: causeMsg, code } : { message: causeMsg }
  return e
}

console.log('isTransientRefreshError classification:\n')

// --- the exact production errors from kiro-logs-2026-06-08 (all "Refresh error: {}" = fetch failed) ---
ok(isTransientRefreshError(withCause('fetch failed', 'other side closed', 'UND_ERR_SOCKET')),
   'UND_ERR_SOCKET "other side closed" → transient (batch storm signature)')
ok(isTransientRefreshError(withCause('fetch failed', 'read ECONNRESET', 'ECONNRESET')),
   'ECONNRESET → transient')
ok(isTransientRefreshError(withCause('fetch failed', 'connect ETIMEDOUT', 'ETIMEDOUT')),
   'ETIMEDOUT → transient')
ok(isTransientRefreshError(withCause('fetch failed', 'getaddrinfo EAI_AGAIN', 'EAI_AGAIN')),
   'EAI_AGAIN (DNS jitter) → transient')
ok(isTransientRefreshError(withCause('Connect Timeout Error', 'connect timeout', 'UND_ERR_CONNECT_TIMEOUT')),
   'UND_ERR_CONNECT_TIMEOUT → transient')
ok(isTransientRefreshError(new Error('The operation was aborted')),
   'AbortController timeout ("aborted") → transient (our own timeout fires)')
ok(isTransientRefreshError(new Error('TypeError: fetch failed')),
   'bare "fetch failed" message → transient')

// --- must NOT retry these: HTTP/app/auth errors are returned as {success:false} not thrown,
//     but guard anyway so a thrown app error never burns retries ---
ok(!isTransientRefreshError(new Error('HTTP 403: The bearer token included in the request is invalid.')),
   'HTTP 403 invalid token → NOT transient (real auth failure, do not retry)')
ok(!isTransientRefreshError(new Error('HTTP 400: invalid_grant')),
   'invalid_grant → NOT transient (refresh token revoked, retry is futile)')
ok(!isTransientRefreshError(new Error('HTTP 401: Unauthorized')),
   'HTTP 401 → NOT transient')
ok(!isTransientRefreshError(undefined), 'undefined → false (no crash)')
ok(!isTransientRefreshError('a string'), 'non-Error → false')

console.log('\nwithRefreshRetry behavior:\n')

// transient error on first attempt, then success → should retry and resolve
{
  let calls = 0
  const result = await withRefreshRetry(async () => {
    calls++
    if (calls === 1) throw withCause('fetch failed', 'other side closed', 'UND_ERR_SOCKET')
    return 'ok'
  })
  ok(result === 'ok' && calls === 2, `transient then success → retried once, resolved (calls=${calls})`)
}

// non-transient error → must throw immediately, no retry
{
  let calls = 0
  let threw = false
  try {
    await withRefreshRetry(async () => {
      calls++
      throw new Error('HTTP 403: invalid token')
    })
  } catch { threw = true }
  ok(threw && calls === 1, `non-transient → threw immediately, no retry (calls=${calls})`)
}

// always-transient → exhausts attempts then throws (3 attempts total)
{
  let calls = 0
  let threw = false
  try {
    await withRefreshRetry(async () => {
      calls++
      throw withCause('fetch failed', 'read ECONNRESET', 'ECONNRESET')
    })
  } catch { threw = true }
  ok(threw && calls === 3, `always-transient → exhausted 3 attempts then threw (calls=${calls})`)
}

// success on first attempt → no retry, timeoutMs is passed to fn
{
  let calls = 0
  let sawTimeout = 0
  const result = await withRefreshRetry(async (timeoutMs) => {
    calls++
    sawTimeout = timeoutMs
    return 42
  })
  ok(result === 42 && calls === 1, `first-attempt success → no retry (calls=${calls})`)
  ok(typeof sawTimeout === 'number' && sawTimeout > 0, `fn receives a positive timeoutMs (${sawTimeout})`)
}

console.log(`\n${fail === 0 ? 'All' : pass + '/' + (pass + fail)} refresh-retry checks passed.`)
process.exit(fail === 0 ? 0 : 1)
