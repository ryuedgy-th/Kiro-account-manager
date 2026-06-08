// Verify document-block name handling in translator (src/main/proxy/translator.ts).
//
// Bug: a Claude/OpenAI document (PDF etc.) without a name/title/filename used to
// throw → the whole request 500'd with "document requires name". Anthropic's native
// document block uses an OPTIONAL `title`, OpenAI file parts use an optional
// `filename` — clients (e.g. Claude Code attaching a PDF) often send neither.
// Kiro's backend requires document.name, so the fix synthesizes a placeholder
// (document-<seq>.<ext>) from the media_type instead of failing.
//
// Bundles translator.ts via esbuild (electron stubbed) — same pattern as verify-effort.mjs.
// Run: node test/verify-document-name.mjs
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { rmSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const translatorEntry = join(here, '..', 'src', 'main', 'proxy', 'translator.ts')
const tmp = mkdtempSync(join(tmpdir(), 'doc-name-'))
process.on('exit', () => { try { rmSync(tmp, { recursive: true, force: true }) } catch { /* ignore */ } })
const outfile = join(tmp, 'translator.cjs')

await build({
  entryPoints: [translatorEntry], bundle: true, format: 'cjs', platform: 'node',
  outfile, logLevel: 'error',
  plugins: [{
    name: 'stub-electron',
    setup(b) {
      b.onResolve({ filter: /^electron($|\/)/ }, () => ({ path: 'electron-stub', namespace: 'stub' }))
      b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({ contents: 'module.exports = { app: {}, ipcMain: {} };', loader: 'js' }))
    }
  }]
})
const require = createRequire(import.meta.url)
const { openaiToKiro, claudeToKiro } = require(outfile)

let pass = 0, fail = 0
const ok = (cond, msg) => { if (cond) { pass++; console.log(`  ok   - ${msg}`) } else { fail++; console.log(`  FAIL - ${msg}`) } }

// documents land on the LAST message's currentMessage.userInputMessage, or on
// history[].userInputMessage for earlier ones. Pull the single doc out wherever it is.
const firstDoc = (payload) => {
  const cur = payload?.conversationState?.currentMessage?.userInputMessage?.documents
  if (cur?.length) return cur[0]
  const hist = payload?.conversationState?.history || []
  for (const h of hist) {
    const d = h?.userInputMessage?.documents
    if (d?.length) return d[0]
  }
  return undefined
}

const PDF_B64 = Buffer.from('%PDF-1.4 fake').toString('base64')

console.log('Claude path — document without name/title:\n')

// 1) Anthropic-native doc with NO name and NO title (the exact 500 trigger) → must not throw
{
  let threw = null
  let doc
  try {
    const p = claudeToKiro({
      model: 'claude-sonnet-4-5',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'summarize this' },
          { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: PDF_B64 } }
        ]
      }]
    })
    doc = firstDoc(p)
  } catch (e) { threw = e }
  ok(threw === null, `nameless PDF document does NOT throw (was: 500 "document requires name")${threw ? ' :: ' + threw.message : ''}`)
  ok(doc && typeof doc.name === 'string' && doc.name.length > 0, `synthesized a non-empty name (${doc?.name})`)
  ok(doc && doc.format === 'pdf', `format derived from media_type = pdf (${doc?.format})`)
  ok(doc && doc.name.endsWith('.pdf'), `synthesized name carries .pdf extension (${doc?.name})`)
}

// 2) Anthropic-native doc with title (spec field) → title used as the name
{
  const p = claudeToKiro({
    model: 'claude-sonnet-4-5',
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'hi' },
        { type: 'document', title: 'report.pdf', source: { type: 'base64', media_type: 'application/pdf', data: PDF_B64 } }
      ]
    }]
  })
  const doc = firstDoc(p)
  ok(doc && doc.name === 'report.pdf', `title used as document name when present (${doc?.name})`)
}

// 3) explicit name still wins
{
  const p = claudeToKiro({
    model: 'claude-sonnet-4-5',
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'hi' },
        { type: 'document', name: 'explicit.pdf', source: { type: 'base64', media_type: 'application/pdf', data: PDF_B64 } }
      ]
    }]
  })
  const doc = firstDoc(p)
  ok(doc && doc.name === 'explicit.pdf', `explicit name preserved (${doc?.name})`)
}

// 4) text/* document with no name → format falls back to txt, name synthesized
{
  const txtB64 = Buffer.from('hello world').toString('base64')
  let threw = null, doc
  try {
    const p = claudeToKiro({
      model: 'claude-sonnet-4-5',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'read' },
          { type: 'document', source: { type: 'base64', media_type: 'text/plain', data: txtB64 } }
        ]
      }]
    })
    doc = firstDoc(p)
  } catch (e) { threw = e }
  ok(threw === null && doc && doc.format === 'txt', `nameless text/plain doc → txt format, no throw (${doc?.format})`)
}

console.log('\nOpenAI path — file/document without filename/name:\n')

// 5) OpenAI file part with file_data data-URL but NO filename → synthesized from data-URL media type
{
  let threw = null, doc
  try {
    const p = openaiToKiro({
      model: 'claude-sonnet-4-5',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'summarize' },
          { type: 'file', file: { file_data: `data:application/pdf;base64,${PDF_B64}` } }
        ]
      }]
    })
    doc = firstDoc(p)
  } catch (e) { threw = e }
  ok(threw === null, `OpenAI file_data without filename does NOT throw${threw ? ' :: ' + threw.message : ''}`)
  ok(doc && doc.format === 'pdf' && doc.name.endsWith('.pdf'), `OpenAI file_data → pdf format + synthesized name (${doc?.name})`)
}

// 6) OpenAI file part filename still wins
{
  const p = openaiToKiro({
    model: 'claude-sonnet-4-5',
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'x' },
        { type: 'file', file: { filename: 'mydoc.pdf', file_data: `data:application/pdf;base64,${PDF_B64}` } }
      ]
    }]
  })
  const doc = firstDoc(p)
  ok(doc && doc.name === 'mydoc.pdf', `OpenAI filename preserved (${doc?.name})`)
}

console.log(`\n${fail === 0 ? 'All' : pass + '/' + (pass + fail)} document-name checks passed.`)
process.exit(fail === 0 ? 0 : 1)
