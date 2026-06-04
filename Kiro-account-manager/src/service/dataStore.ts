// Headless data layer: โหลด/บันทึก proxyConfig + accounts + stats จากไฟล์ JSON เดียว
// path: env KIRO_SERVICE_DATA หรือ <dataDir>/kiro-service-data.json
import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'
import type { ProxyConfig, ProxyAccount } from '../main/proxy/types'

export interface ServiceStats {
  totalCredits?: number
  inputTokens?: number
  outputTokens?: number
  totalRequests?: number
  successRequests?: number
  failedRequests?: number
}

// รูปแบบ accountData ตรงกับ store ภายในแอป (เพื่อให้ map เดิมใช้ได้)
interface RawAccount {
  id: string
  email?: string
  status?: string
  idp?: string
  machineId?: string
  profileArn?: string
  credentials?: {
    accessToken?: string
    refreshToken?: string
    clientId?: string
    clientSecret?: string
    region?: string
    authMethod?: string
    provider?: string
    expiresAt?: number
  }
}
export interface AccountData {
  accounts?: Record<string, RawAccount>
  accountProxyBindings?: Record<string, string>
  proxyPool?: Record<string, { url?: string; enabled?: boolean; status?: string }>
}
export interface ServiceData {
  proxyConfig?: Partial<ProxyConfig>
  accountData?: AccountData
  stats?: ServiceStats
}

export function resolveDataDir(): string {
  return process.env.KIRO_DATA_DIR || path.join(os.homedir(), '.kiro-proxy-service')
}

export function resolveDataFile(): string {
  return process.env.KIRO_SERVICE_DATA || path.join(resolveDataDir(), 'kiro-service-data.json')
}

export function loadServiceData(file: string): ServiceData {
  try {
    if (fs.existsSync(file)) {
      return JSON.parse(fs.readFileSync(file, 'utf-8')) as ServiceData
    }
  } catch (err) {
    console.error(`[data] โหลด ${file} ไม่สำเร็จ:`, err)
  }
  return {}
}

// map accountData → ProxyAccount[] (logic เดียวกับ onPoolEmpty ใน index.ts)
export function mapAccounts(accountData?: AccountData): ProxyAccount[] {
  if (!accountData?.accounts) return []
  const bindings = accountData.accountProxyBindings || {}
  const proxyPool = accountData.proxyPool || {}
  const buildProxyUrl = (accountId: string): string | undefined => {
    const proxyId = bindings[accountId]
    if (!proxyId) return undefined
    const p = proxyPool[proxyId]
    if (!p || !p.enabled || p.status === 'dead') return undefined
    return p.url
  }
  return Object.values(accountData.accounts)
    .filter((acc) => acc.status === 'active' && acc.credentials?.accessToken)
    .map((acc) => ({
      id: acc.id,
      email: acc.email,
      accessToken: acc.credentials!.accessToken!,
      refreshToken: acc.credentials?.refreshToken,
      profileArn: acc.profileArn,
      expiresAt: acc.credentials?.expiresAt,
      machineId: acc.machineId,
      clientId: acc.credentials?.clientId,
      clientSecret: acc.credentials?.clientSecret,
      region: acc.credentials?.region || 'us-east-1',
      authMethod: acc.credentials?.authMethod,
      provider: acc.credentials?.provider || acc.idp,
      proxyUrl: buildProxyUrl(acc.id)
    })) as ProxyAccount[]
}

// บันทึกกลับแบบ debounce (กัน IO ถี่จาก callback usage/credit)
export class ServicePersist {
  private data: ServiceData
  private timer: NodeJS.Timeout | null = null
  constructor(private file: string, initial: ServiceData) {
    this.data = initial
  }
  patchConfig(config: Partial<ProxyConfig>): void { this.data.proxyConfig = config; this.schedule() }
  patchStats(partial: ServiceStats): void { this.data.stats = { ...this.data.stats, ...partial }; this.schedule() }
  // อัปเดต token ของ account กลับเข้าไฟล์ (กันหมดอายุข้ามรีสตาร์ท)
  patchAccountToken(id: string, accessToken?: string, refreshToken?: string, expiresAt?: number): void {
    const acc = this.data.accountData?.accounts?.[id]
    if (!acc) return
    acc.credentials = { ...acc.credentials, accessToken, refreshToken, expiresAt }
    this.schedule()
  }
  private schedule(): void {
    if (this.timer) return
    this.timer = setTimeout(() => { this.timer = null; void this.flush() }, 1000)
  }
  async flush(): Promise<void> {
    try {
      await fs.promises.mkdir(path.dirname(this.file), { recursive: true })
      await fs.promises.writeFile(this.file, JSON.stringify(this.data, null, 2), 'utf-8')
    } catch (err) {
      console.error('[data] เขียนไฟล์ไม่สำเร็จ:', err)
    }
  }
}
