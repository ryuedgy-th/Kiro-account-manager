import { useState, useEffect, useCallback } from 'react'
import type { ReactElement } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Coins, TrendingUp } from 'lucide-react'
import { useAccountsStore } from '@/store/accounts'

interface PricingConfig {
  enabled?: boolean
  bahtPerCredit?: number
  costPerCredit?: number
  usdToBaht?: number
  gatewayFeePct?: number
  kiroRetailUsdPerCredit?: number
  modelMarkup?: Record<string, number>
}

const DEFAULTS: Required<Omit<PricingConfig, 'modelMarkup'>> = {
  enabled: false,
  bahtPerCredit: 0.42,
  costPerCredit: 0.16,
  usdToBaht: 36,
  gatewayFeePct: 0,
  kiroRetailUsdPerCredit: 0.02
}

export function PricingSettings() {
  const { language } = useAccountsStore()
  const isEn = language === 'en'
  const t = (en: string, zh: string): string => (isEn ? en : zh)

  const [pricing, setPricing] = useState<PricingConfig>(DEFAULTS)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)

  // 草稿值：输入框按字符串保存，失焦/保存时再解析，避免边打边校验干扰输入
  const [draft, setDraft] = useState<Record<string, string>>({})

  const loadPricing = useCallback(async () => {
    try {
      const status = await window.api.proxyGetStatus()
      const cfg = (status?.config || {}) as { pricing?: PricingConfig }
      const p = { ...DEFAULTS, ...(cfg.pricing || {}) }
      setPricing(p)
      setDraft({
        bahtPerCredit: String(p.bahtPerCredit ?? DEFAULTS.bahtPerCredit),
        costPerCredit: String(p.costPerCredit ?? DEFAULTS.costPerCredit),
        usdToBaht: String(p.usdToBaht ?? DEFAULTS.usdToBaht),
        gatewayFeePct: String(p.gatewayFeePct ?? DEFAULTS.gatewayFeePct),
        kiroRetailUsdPerCredit: String(p.kiroRetailUsdPerCredit ?? DEFAULTS.kiroRetailUsdPerCredit)
      })
    } catch (e) {
      console.error('Failed to load pricing:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadPricing() }, [loadPricing])

  const flash = useCallback((msg: string) => {
    setNotice(msg)
    setTimeout(() => setNotice(null), 3000)
  }, [])

  // 把完整 pricing 对象写回（shallow merge，主进程会原样替换整个 pricing）
  const persist = useCallback(async (next: PricingConfig) => {
    setSaving(true)
    try {
      await window.api.proxyUpdateConfig({ pricing: next })
      setPricing(next)
      flash(t('Pricing saved', '定价已保存'))
    } catch (e) {
      console.error('Failed to save pricing:', e)
    } finally {
      setSaving(false)
    }
  }, [flash, t])

  const toggleEnabled = useCallback((checked: boolean) => {
    persist({ ...pricing, enabled: checked })
  }, [pricing, persist])

  // 解析草稿里的数字字段并保存。非法/空值回退到现值。
  const commitField = useCallback((key: keyof PricingConfig) => {
    const raw = draft[key as string]
    const num = Number(raw)
    if (!Number.isFinite(num) || num < 0) {
      // 还原草稿到当前生效值
      setDraft(prev => ({ ...prev, [key as string]: String(pricing[key] ?? 0) }))
      return
    }
    if (num === pricing[key]) return
    persist({ ...pricing, [key]: num })
  }, [draft, pricing, persist])

  const setDraftField = (key: string, val: string): void =>
    setDraft(prev => ({ ...prev, [key]: val }))

  // ===== 毛利预估（基于当前输入） =====
  const sell = pricing.bahtPerCredit || 0
  const cost = pricing.costPerCredit || 0
  const gw = (pricing.gatewayFeePct || 0) / 100
  const netPerCredit = sell * (1 - gw)
  const grossMarginPct = cost > 0 ? ((netPerCredit - cost) / cost) * 100 : 0
  // เทียบ "ถูกกว่าเรียก Anthropic API ตรง ๆ" — แกนเดียวกับที่ลูกค้าเห็นในพอร์ทัล
  // ราคาเรา/1M input (Opus) = 6.4 credit × ฿/credit ; Anthropic Opus input = $15/1M
  const usd2baht = pricing.usdToBaht || 36
  const ourInputBaht = 6.4 * sell
  const anthropicInputBaht = 15 * usd2baht
  const cheaperThanAnthropicPct =
    anthropicInputBaht > 0 && ourInputBaht < anthropicInputBaht
      ? (1 - ourInputBaht / anthropicInputBaht) * 100
      : 0

  const numField = (
    key: keyof PricingConfig,
    label: string,
    step: string,
    placeholder: string
  ): ReactElement => (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        step={step}
        value={draft[key as string] ?? ''}
        onChange={(e) => setDraftField(key as string, e.target.value)}
        onBlur={() => commitField(key)}
        onKeyDown={(e) => { if (e.key === 'Enter') commitField(key) }}
        disabled={saving || !pricing.enabled}
        placeholder={placeholder}
        className="h-9"
      />
    </div>
  )

  return (
    <Card className="w-full mb-4">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Coins className="h-5 w-5" />
            <CardTitle>{t('Resale Pricing (THB)', '转售定价（泰铢）')}</CardTitle>
          </div>
          <Switch checked={!!pricing.enabled} onCheckedChange={toggleEnabled} disabled={saving || loading} />
        </div>
        <CardDescription>
          {t('Charge customers in baht per Kiro credit. When off, billing stays raw credits (no markup).',
             '按 Kiro credit 以泰铢向客户计费。关闭时沿用原始 credit 计费（不加价）。')}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {notice && (
          <div className="text-sm rounded-md px-3 py-2 bg-emerald-500/10 text-emerald-600">{notice}</div>
        )}

        {loading ? (
          <p className="text-sm text-muted-foreground py-2">{t('Loading...', '加载中…')}</p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3">
              {numField('bahtPerCredit', t('Sell price (฿/credit)', '售价（฿/credit）'), '0.01', '0.42')}
              {numField('costPerCredit', t('Your cost (฿/credit)', '成本（฿/credit）'), '0.01', '0.16')}
              {numField('gatewayFeePct', t('Payment gateway fee (%)', '支付网关费（%）'), '0.1', '0')}
              {numField('usdToBaht', t('USD → THB rate', '美元兑泰铢汇率'), '0.1', '36')}
            </div>

            {/* 毛利预估 */}
            <div className="rounded-lg border p-3 space-y-2">
              <div className="flex items-center gap-1.5 text-sm font-medium">
                <TrendingUp className="h-4 w-4" />
                {t('Margin preview', '毛利预估')}
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className={`text-lg font-bold ${grossMarginPct >= 0 ? 'text-emerald-600' : 'text-destructive'}`}>
                    {grossMarginPct >= 0 ? '+' : ''}{grossMarginPct.toFixed(0)}%
                  </div>
                  <div className="text-[11px] text-muted-foreground">{t('Net margin/credit', '每 credit 净利')}</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-foreground">฿{netPerCredit.toFixed(3)}</div>
                  <div className="text-[11px] text-muted-foreground">{t('Net revenue/credit', '每 credit 净收')}</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-blue-600">
                    {cheaperThanAnthropicPct > 0 ? `${cheaperThanAnthropicPct.toFixed(1)}%` : '–'}
                  </div>
                  <div className="text-[11px] text-muted-foreground">{t('Cheaper than Anthropic', '比 Anthropic 便宜')}</div>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground">
                {t(`After ${(pricing.gatewayFeePct || 0)}% gateway fee. "Cheaper than Anthropic" is what customers see in the portal (Opus input basis).`,
                   `已扣 ${(pricing.gatewayFeePct || 0)}% 网关费。"比 Anthropic 便宜"即客户在门户看到的数字（以 Opus input 为基准）。`)}
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
