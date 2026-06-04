import { useState, useEffect, useCallback } from 'react'
import type { ReactElement } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { ScanLine, Plus, Trash2, RefreshCw } from 'lucide-react'
import { useAccountsStore } from '@/store/accounts'

// 与 preload SlipTopupConfigView 对齐（apiSecret 读取时为 '' 或 '***'，写入时空串=不修改）
type ReceiverAccount = {
  accountType?: string
  accountNumber?: string
  accountNameTH?: string
  accountNameEN?: string
}
type SlipConfig = {
  enabled: boolean
  apiSecret: string
  receiverAccounts: ReceiverAccount[]
  minAmountThb?: number
  maxAmountThb?: number
  freshnessHours?: number
  dailyMaxSubmitsPerCustomer?: number
  perMinuteMaxSubmitsPerCustomer?: number
}

const DEFAULTS: SlipConfig = {
  enabled: false,
  apiSecret: '',
  receiverAccounts: [],
  minAmountThb: 1,
  maxAmountThb: 0,
  freshnessHours: 48,
  dailyMaxSubmitsPerCustomer: 20,
  perMinuteMaxSubmitsPerCustomer: 5
}

export function SlipTopupSettings(): ReactElement {
  const { language } = useAccountsStore()
  const isEn = language === 'en'
  const t = (en: string, zh: string): string => (isEn ? en : zh)

  const [cfg, setCfg] = useState<SlipConfig>(DEFAULTS)
  const [hasSecret, setHasSecret] = useState(false)   // 服务端已存在 apiSecret（显示 ***）
  const [secretDraft, setSecretDraft] = useState('')  // 新密钥输入；空=不修改
  const [accounts, setAccounts] = useState<ReceiverAccount[]>([])
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [records, setRecords] = useState<Array<{ verifiedAt: number; bahtAmount: number; creditsAdded: number; status: string; rejectReason?: string; senderName?: string }>>([])
  const [showRecords, setShowRecords] = useState(false)

  const flash = useCallback((msg: string) => {
    setNotice(msg)
    setTimeout(() => setNotice(null), 3000)
  }, [])

  const load = useCallback(async () => {
    try {
      const res = await window.api.proxySlipTopupGetConfig()
      if (res.success && res.config) {
        const c = res.config
        setCfg({ ...DEFAULTS, ...c, apiSecret: '' })
        setHasSecret(c.apiSecret === '***')
        setAccounts(Array.isArray(c.receiverAccounts) ? c.receiverAccounts : [])
        setDraft({
          minAmountThb: String(c.minAmountThb ?? DEFAULTS.minAmountThb),
          maxAmountThb: String(c.maxAmountThb ?? DEFAULTS.maxAmountThb),
          freshnessHours: String(c.freshnessHours ?? DEFAULTS.freshnessHours),
          dailyMaxSubmitsPerCustomer: String(c.dailyMaxSubmitsPerCustomer ?? DEFAULTS.dailyMaxSubmitsPerCustomer),
          perMinuteMaxSubmitsPerCustomer: String(c.perMinuteMaxSubmitsPerCustomer ?? DEFAULTS.perMinuteMaxSubmitsPerCustomer)
        })
      }
    } catch (e) {
      console.error('Failed to load slip topup config:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // 组装并保存整份配置（apiSecret 空串=保留服务端旧值）
  const persist = useCallback(async (overrides: Partial<SlipConfig>) => {
    setSaving(true)
    setError(null)
    try {
      const num = (k: string, fallback: number): number => {
        const n = Number(draft[k])
        return Number.isFinite(n) && n >= 0 ? n : fallback
      }
      const payload: Partial<SlipConfig> = {
        enabled: cfg.enabled,
        receiverAccounts: accounts.filter(a => a.accountNumber || a.accountNameTH || a.accountNameEN),
        minAmountThb: num('minAmountThb', DEFAULTS.minAmountThb!),
        maxAmountThb: num('maxAmountThb', 0),
        freshnessHours: num('freshnessHours', DEFAULTS.freshnessHours!),
        dailyMaxSubmitsPerCustomer: num('dailyMaxSubmitsPerCustomer', DEFAULTS.dailyMaxSubmitsPerCustomer!),
        perMinuteMaxSubmitsPerCustomer: num('perMinuteMaxSubmitsPerCustomer', DEFAULTS.perMinuteMaxSubmitsPerCustomer!),
        ...overrides
      }
      // 仅当用户输入了新密钥才发送（空串 = 不修改）
      if (secretDraft.trim()) payload.apiSecret = secretDraft.trim()
      const res = await window.api.proxySlipTopupSetConfig(payload)
      if (res.success && res.config) {
        const c = res.config
        setCfg({ ...DEFAULTS, ...c, apiSecret: '' })
        setHasSecret(c.apiSecret === '***')
        setAccounts(Array.isArray(c.receiverAccounts) ? c.receiverAccounts : [])
        setSecretDraft('')
        flash(t('Saved', '已保存'))
      } else {
        setError(res.error || t('Save failed', '保存失败'))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : t('Save failed', '保存失败'))
    } finally {
      setSaving(false)
    }
  }, [cfg.enabled, accounts, draft, secretDraft, flash, t])

  const toggleEnabled = useCallback((checked: boolean) => {
    // 开启前要求至少有密钥（已存在或正在输入）
    if (checked && !hasSecret && !secretDraft.trim()) {
      setError(t('Set the slip2go API secret before enabling', '启用前请先填写 slip2go API secret'))
      return
    }
    setCfg(prev => ({ ...prev, enabled: checked }))
    persist({ enabled: checked })
  }, [hasSecret, secretDraft, persist, t])

  const setDraftField = (key: string, val: string): void => setDraft(prev => ({ ...prev, [key]: val }))

  const numField = (key: string, label: string, step: string, placeholder: string): ReactElement => (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        step={step}
        value={draft[key] ?? ''}
        onChange={(e) => setDraftField(key, e.target.value)}
        onBlur={() => persist({})}
        disabled={saving}
        placeholder={placeholder}
        className="h-9"
      />
    </div>
  )

  const updateAccount = (idx: number, field: keyof ReceiverAccount, val: string): void => {
    setAccounts(prev => prev.map((a, i) => (i === idx ? { ...a, [field]: val } : a)))
  }
  const addAccount = (): void => setAccounts(prev => [...prev, { accountNumber: '', accountNameTH: '' }])
  const removeAccount = (idx: number): void => setAccounts(prev => prev.filter((_, i) => i !== idx))

  const loadRecords = useCallback(async () => {
    try {
      const res = await window.api.proxySlipTopupRecords(50)
      if (res.success && res.records) {
        setRecords(res.records.map(r => ({
          verifiedAt: r.verifiedAt, bahtAmount: r.bahtAmount, creditsAdded: r.creditsAdded,
          status: r.status, rejectReason: r.rejectReason, senderName: r.senderName
        })))
        setShowRecords(true)
      }
    } catch (e) {
      console.error('Failed to load slip records:', e)
    }
  }, [])

  return (
    <Card className="w-full mb-4">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ScanLine className="h-5 w-5" />
            <CardTitle>{t('Slip auto top-up (slip2go)', '转账slip自动充值（slip2go）')}</CardTitle>
          </div>
          <Switch checked={cfg.enabled} onCheckedChange={toggleEnabled} disabled={saving || loading} />
        </div>
        <CardDescription>
          {t('Customers transfer to your bank, submit the slip QR, and slip2go verifies it against the bank to auto-credit their balance.',
             '客户转账到你的银行账户后提交slip二维码，slip2go 与银行核验通过后自动入账。')}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {notice && <div className="text-sm rounded-md px-3 py-2 bg-emerald-500/10 text-emerald-600">{notice}</div>}
        {error && <div className="text-sm rounded-md px-3 py-2 bg-red-500/10 text-red-600">{error}</div>}

        {loading ? (
          <p className="text-sm text-muted-foreground py-2">{t('Loading...', '加载中…')}</p>
        ) : (
          <>
            {/* API secret */}
            <div className="space-y-1">
              <Label className="text-xs">
                {t('slip2go API Secret', 'slip2go API Secret')}
                {hasSecret && <span className="ml-2 text-emerald-600">{t('(set)', '（已设置）')}</span>}
              </Label>
              <Input
                type="password"
                value={secretDraft}
                onChange={(e) => setSecretDraft(e.target.value)}
                onBlur={() => { if (secretDraft.trim()) persist({}) }}
                disabled={saving}
                placeholder={hasSecret ? t('Leave blank to keep current', '留空表示不修改') : 'Bearer secret…'}
                className="h-9 font-mono"
                autoComplete="off"
              />
              <p className="text-[11px] text-muted-foreground">
                {t('Stored locally in the main process; never sent to the customer portal.',
                   '仅保存在本地主进程，绝不下发到客户门户。')}
              </p>
            </div>

            {/* Receiver accounts */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">{t('Receiver accounts (our bank / PromptPay)', '收款账号（我方银行/พร้อมเพย์）')}</Label>
                <Button variant="ghost" size="sm" className="h-7" onClick={addAccount} disabled={saving}>
                  <Plus className="h-3.5 w-3.5 mr-1" />{t('Add', '添加')}
                </Button>
              </div>
              {accounts.length === 0 && (
                <p className="text-[11px] text-muted-foreground">
                  {t('Add at least one receiver account to verify the slip was paid to you.',
                     '至少添加一个收款账号，用于核验slip确实转入我方。')}
                </p>
              )}
              {accounts.map((a, i) => (
                <div key={i} className="rounded-lg border p-2 space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <Input className="h-8 text-xs" placeholder={t('Account / phone no.', '账号/手机号')}
                      value={a.accountNumber ?? ''} onChange={(e) => updateAccount(i, 'accountNumber', e.target.value)}
                      onBlur={() => persist({})} disabled={saving} />
                    <Input className="h-8 text-xs" placeholder={t('Type code (e.g. 01004)', '类型码（如 01004）')}
                      value={a.accountType ?? ''} onChange={(e) => updateAccount(i, 'accountType', e.target.value)}
                      onBlur={() => persist({})} disabled={saving} />
                    <Input className="h-8 text-xs" placeholder={t('Name (TH)', '姓名（泰）')}
                      value={a.accountNameTH ?? ''} onChange={(e) => updateAccount(i, 'accountNameTH', e.target.value)}
                      onBlur={() => persist({})} disabled={saving} />
                    <Input className="h-8 text-xs" placeholder={t('Name (EN)', '姓名（英）')}
                      value={a.accountNameEN ?? ''} onChange={(e) => updateAccount(i, 'accountNameEN', e.target.value)}
                      onBlur={() => persist({})} disabled={saving} />
                  </div>
                  <div className="flex justify-end">
                    <Button variant="ghost" size="sm" className="h-7 text-red-600" onClick={() => { removeAccount(i); setTimeout(() => persist({}), 0) }} disabled={saving}>
                      <Trash2 className="h-3.5 w-3.5 mr-1" />{t('Remove', '删除')}
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {/* Limits */}
            <div className="grid grid-cols-2 gap-3">
              {numField('minAmountThb', t('Min amount (฿)', '最低金额（฿）'), '1', '1')}
              {numField('maxAmountThb', t('Max amount (฿, 0=none)', '最高金额（฿，0=不限）'), '1', '0')}
              {numField('freshnessHours', t('Slip freshness (hours)', 'slip 有效期（小时）'), '1', '48')}
              {numField('perMinuteMaxSubmitsPerCustomer', t('Max submits / min', '每分钟提交上限'), '1', '5')}
              {numField('dailyMaxSubmitsPerCustomer', t('Max submits / day', '每日提交上限'), '1', '20')}
            </div>

            {/* Records */}
            <div className="pt-2 border-t">
              <Button variant="ghost" size="sm" className="h-7" onClick={loadRecords}>
                <RefreshCw className="h-3.5 w-3.5 mr-1" />{t('Load recent slip records', '加载最近slip流水')}
              </Button>
              {showRecords && (
                records.length === 0 ? (
                  <p className="text-xs text-muted-foreground py-2">{t('No records yet.', '暂无流水。')}</p>
                ) : (
                  <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
                    {records.map((r, i) => (
                      <div key={i} className="flex items-center justify-between text-xs rounded border px-2 py-1.5">
                        <span className="text-muted-foreground">{new Date(r.verifiedAt).toLocaleString(isEn ? 'en-US' : 'th-TH', { dateStyle: 'short', timeStyle: 'short' })}</span>
                        <span className="tabular-nums">฿{r.bahtAmount.toLocaleString()}</span>
                        {r.status === 'settled'
                          ? <span className="text-emerald-600 font-medium">+{r.creditsAdded.toLocaleString()} cr</span>
                          : <span className="text-red-600">{r.rejectReason || 'rejected'}</span>}
                      </div>
                    ))}
                  </div>
                )
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
