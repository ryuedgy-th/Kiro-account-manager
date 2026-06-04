// แปลงข้อมูลจาก electron-store เดิม (kiro-accounts.json) → kiro-service-data.json
// ใช้กับ headless service โดย "ไม่ต้อง build GUI ใหม่"
//   node scripts/export-service-data.mjs [--store <path>] [--out <path>]
// ถอดรหัสตรงตาม conf v15 (electron-store v11): aes-256-cbc, IV=16 ไบต์แรก, ':' คั่น,
//   key = pbkdf2(encryptionKey, IV, 10000, 32, sha512)
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const ENCRYPTION_KEY = 'kiro-account-manager-secret-key' // ตรงกับ initStore() ใน src/main/index.ts
const STORE_NAME = 'kiro-accounts.json'

function arg(name) {
  const i = process.argv.indexOf(name)
  return i >= 0 ? process.argv[i + 1] : undefined
}

function storeCandidates() {
  const home = os.homedir()
  const names = ['Kiro Account Manager', 'kiro-account-manager'] // packaged (productName) / dev (name)
  const bases = []
  if (process.platform === 'win32') {
    const appData = process.env.APPDATA || path.join(home, 'AppData', 'Roaming')
    for (const n of names) bases.push(path.join(appData, n))
  } else if (process.platform === 'darwin') {
    for (const n of names) bases.push(path.join(home, 'Library', 'Application Support', n))
  } else {
    const cfg = process.env.XDG_CONFIG_HOME || path.join(home, '.config')
    for (const n of names) bases.push(path.join(cfg, n))
  }
  return bases.map(b => path.join(b, STORE_NAME))
}

function decryptStore(buf) {
  // ไฟล์ไม่เข้ารหัส (เผื่อกรณีไม่มี encryptionKey)
  try { return JSON.parse(buf.toString('utf8')) } catch { /* เข้ารหัสอยู่ */ }
  const iv = buf.subarray(0, 16)
  const ciphertext = buf.subarray(17) // ข้าม IV(16) + ':'(1)
  for (const salt of [iv, Buffer.from(iv.toString())]) { // primary (Buffer) → legacy (iv.toString())
    try {
      const password = crypto.pbkdf2Sync(ENCRYPTION_KEY, salt, 10000, 32, 'sha512')
      const decipher = crypto.createDecipheriv('aes-256-cbc', password, iv)
      const out = Buffer.concat([decipher.update(ciphertext), decipher.final()])
      return JSON.parse(out.toString('utf8'))
    } catch { /* ลองตัวถัดไป */ }
  }
  throw new Error('ถอดรหัส store ไม่สำเร็จ (encryptionKey/รูปแบบไฟล์ไม่ตรง)')
}

const storePath = arg('--store') || storeCandidates().find(p => fs.existsSync(p))
if (!storePath || !fs.existsSync(storePath)) {
  console.error('ไม่พบไฟล์ store. ลองชี้ด้วย --store <path>. ตำแหน่งที่ลองหา:')
  for (const p of storeCandidates()) console.error('  -', p)
  process.exit(1)
}

const store = decryptStore(fs.readFileSync(storePath))
const num = (k) => (typeof store[k] === 'number' ? store[k] : 0)
const serviceData = {
  proxyConfig: store.proxyConfig || null,
  accountData: store.accountData || { accounts: {} },
  stats: {
    totalCredits: num('proxyTotalCredits'),
    inputTokens: num('proxyInputTokens'),
    outputTokens: num('proxyOutputTokens'),
    totalRequests: num('proxyTotalRequests'),
    successRequests: num('proxySuccessRequests'),
    failedRequests: num('proxyFailedRequests')
  }
}

const accountCount = Object.keys(serviceData.accountData.accounts || {}).length
const outPath = arg('--out') || path.join(process.cwd(), STORE_NAME.replace('kiro-accounts', 'kiro-service-data'))
fs.writeFileSync(outPath, JSON.stringify(serviceData, null, 2), 'utf-8')
console.log(`OK: เขียน ${outPath}`)
console.log(`   accounts=${accountCount}, proxyConfig=${serviceData.proxyConfig ? 'yes' : 'no'}, from=${storePath}`)
