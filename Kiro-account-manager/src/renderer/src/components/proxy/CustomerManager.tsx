import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import {
  Users, Plus, Trash2, Copy, Check, RefreshCw, Wallet,
  Power, KeyRound, ExternalLink, AlertTriangle, X, Mail, Ticket
} from 'lucide-react'
import { useAccountsStore } from '@/store/accounts'

interface CustomerView {
  id: string
  email: string
  name?: string
  enabled: boolean
  createdAt: number
  lastLoginAt?: number
  creditBalance: number
  totalToppedUp: number
  keyCount: number
  maxKeys: number
}

interface InviteView {
  code: string
  email: string
  name?: string
  creditBalance: number
  maxKeys?: number
  createdAt: number
  expiresAt?: number
  usedAt?: number
  usedByCustomerId?: string
}

export function CustomerManager() {
  const { language } = useAccountsStore()
  const isEn = language === 'en'
  const t = (en: string, zh: string): string => (isEn ? en : zh)

  const [customers, setCustomers] = useState<CustomerView[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // 门户开关 + 访问地址
  const [portalEnabled, setPortalEnabled] = useState(false)
  const [portalUrl, setPortalUrl] = useState('')
  const [hostIsLocal, setHostIsLocal] = useState(true)
  const [copied, setCopied] = useState(false)

  // 新建客户表单
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [initialCredit, setInitialCredit] = useState('')

  // Google 登录配置
  const [googleEnabled, setGoogleEnabled] = useState(false)
  const [googleClientId, setGoogleClientId] = useState('')
  const [clientIdDraft, setClientIdDraft] = useState('')

  // 邀请码
  const [invites, setInvites] = useState<InviteView[]>([])
  const [invEmail, setInvEmail] = useState('')
  const [invName, setInvName] = useState('')
  const [invCredit, setInvCredit] = useState('')
  const [invExpiry, setInvExpiry] = useState('')
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  // 操作弹窗（替代被 Electron 屏蔽的 window.prompt/confirm）
  // kind 区分充值 / 重置密码 / 删除三种操作，复用同一套模态外壳
  const [modal, setModal] = useState<
    | { kind: 'topup'; customer: CustomerView }
    | { kind: 'password'; customer: CustomerView }
    | { kind: 'delete'; customer: CustomerView }
    | null
  >(null)
  const [modalInput, setModalInput] = useState('')
  const [modalNote, setModalNote] = useState('')

  const flash = useCallback((msg: string, isError = false) => {
    if (isError) { setError(msg); setNotice(null) } else { setNotice(msg); setError(null) }
    setTimeout(() => { setError(null); setNotice(null) }, 4000)
  }, [])

  const loadStatus = useCallback(async () => {
    try {
      const status = await window.api.proxyGetStatus()
      const cfg = (status?.config || {}) as { port?: number; host?: string; tls?: { enabled?: boolean }; portalEnabled?: boolean; portalGoogleEnabled?: boolean; googleClientId?: string }
      setPortalEnabled(!!cfg.portalEnabled)
      setGoogleEnabled(!!cfg.portalGoogleEnabled)
      setGoogleClientId(cfg.googleClientId || '')
      setClientIdDraft(cfg.googleClientId || '')
      const host = cfg.host || '127.0.0.1'
      const isLocal = host === '127.0.0.1' || host === '::1' || host === 'localhost'
      setHostIsLocal(isLocal)
      const displayHost = host === '0.0.0.0' ? 'localhost' : host
      const scheme = cfg.tls?.enabled ? 'https' : 'http'
      setPortalUrl(`${scheme}://${displayHost}:${cfg.port ?? 5580}/portal`)
    } catch (e) {
      console.error('Failed to load proxy status:', e)
    }
  }, [])

  const loadInvites = useCallback(async () => {
    try {
      const result = await window.api.proxyListInvites()
      if (result.success && result.invites) setInvites(result.invites)
    } catch (e) {
      console.error('Failed to load invites:', e)
    }
  }, [])

  const loadCustomers = useCallback(async () => {
    try {
      const result = await window.api.proxyListCustomers()
      if (result.success && result.customers) {
        setCustomers(result.customers)
      } else if (!result.success) {
        flash(result.error || t('Failed to load customers', '加载客户失败'), true)
      }
    } catch (e) {
      console.error('Failed to load customers:', e)
    } finally {
      setLoading(false)
    }
  }, [flash, t])

  useEffect(() => {
    loadStatus()
    loadCustomers()
    loadInvites()
  }, [loadStatus, loadCustomers, loadInvites])

  const togglePortal = useCallback(async (next: boolean) => {
    setBusy(true)
    try {
      const result = await window.api.proxyPortalSetEnabled(next)
      if (result.success) {
        setPortalEnabled(!!result.portalEnabled)
        flash(next
          ? t('Portal enabled', '门户已启用')
          : t('Portal disabled', '门户已关闭'))
        if (result.needsRestart) {
          flash(t('Restart proxy to apply host/port changes', '需重启代理以应用 host/port 变更'), true)
        }
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [flash, t])

  const copyUrl = useCallback(() => {
    navigator.clipboard.writeText(portalUrl).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }, [portalUrl])

  const createCustomer = useCallback(async () => {
    if (!email.trim() || !password) {
      flash(t('Email and password are required', '邮箱和密码为必填'), true)
      return
    }
    if (password.length < 8) {
      flash(t('Password must be at least 8 characters', '密码至少 8 位'), true)
      return
    }
    setBusy(true)
    try {
      const creditNum = initialCredit.trim() ? Number(initialCredit) : 0
      const result = await window.api.proxyCreateCustomer({
        email: email.trim(),
        password,
        name: name.trim() || undefined,
        creditBalance: Number.isFinite(creditNum) ? creditNum : 0
      })
      if (result.success) {
        setEmail(''); setPassword(''); setName(''); setInitialCredit('')
        flash(t('Customer created', '客户已创建'))
        await loadCustomers()
      } else {
        flash(result.error || t('Failed to create customer', '创建客户失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [email, password, name, initialCredit, flash, t, loadCustomers])

  // 切换 Google 登录开关（共用 proxy-update-config 白名单）
  const toggleGoogle = useCallback(async (next: boolean) => {
    if (next && !googleClientId.trim()) {
      flash(t('Set the Google Client ID first', '请先填写 Google Client ID'), true)
      return
    }
    setBusy(true)
    try {
      const result = await window.api.proxyUpdateConfig({ portalGoogleEnabled: next })
      if (result.success) {
        setGoogleEnabled(next)
        flash(next ? t('Google login enabled', 'Google 登录已启用') : t('Google login disabled', 'Google 登录已关闭'))
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [googleClientId, flash, t])

  const saveClientId = useCallback(async () => {
    setBusy(true)
    try {
      const result = await window.api.proxyUpdateConfig({ googleClientId: clientIdDraft.trim() })
      if (result.success) {
        setGoogleClientId(clientIdDraft.trim())
        flash(t('Google Client ID saved', 'Google Client ID 已保存'))
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [clientIdDraft, flash, t])

  const createInvite = useCallback(async () => {
    if (!invEmail.trim()) {
      flash(t('Email is required for an invite', '邀请需填写邮箱'), true)
      return
    }
    setBusy(true)
    try {
      const creditNum = invCredit.trim() ? Number(invCredit) : 0
      const expiryNum = invExpiry.trim() ? Number(invExpiry) : undefined
      const result = await window.api.proxyCreateInvite({
        email: invEmail.trim(),
        name: invName.trim() || undefined,
        creditBalance: Number.isFinite(creditNum) ? creditNum : 0,
        expiresInDays: expiryNum && Number.isFinite(expiryNum) ? expiryNum : undefined
      })
      if (result.success) {
        setInvEmail(''); setInvName(''); setInvCredit(''); setInvExpiry('')
        flash(t('Invite created — copy the link to send', '邀请已创建 — 复制链接发送'))
        await loadInvites()
      } else {
        flash(result.error || t('Failed to create invite', '创建邀请失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [invEmail, invName, invCredit, invExpiry, flash, t, loadInvites])

  // 邀请链接：portal 地址 + ?invite=code，客户打开后用 Google 登录自动带上 code
  const inviteLink = useCallback((code: string): string => {
    return `${portalUrl}?invite=${encodeURIComponent(code)}`
  }, [portalUrl])

  const copyInviteLink = useCallback((code: string) => {
    navigator.clipboard.writeText(inviteLink(code)).then(() => {
      setCopiedCode(code)
      setTimeout(() => setCopiedCode(null), 1500)
    })
  }, [inviteLink])

  const revokeInvite = useCallback(async (code: string) => {
    setBusy(true)
    try {
      const result = await window.api.proxyRevokeInvite(code)
      if (result.success) {
        flash(t('Invite revoked', '邀请已撤销'))
        await loadInvites()
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [flash, t, loadInvites])

  const topup = useCallback((c: CustomerView) => {
    setModalInput('100')
    setModalNote('')
    setModal({ kind: 'topup', customer: c })
  }, [])

  const submitTopup = useCallback(async () => {
    if (!modal || modal.kind !== 'topup') return
    const amount = Number(modalInput)
    if (!Number.isFinite(amount) || amount === 0) {
      flash(t('Amount must be a non-zero number', '金额必须是非零数字'), true)
      return
    }
    setBusy(true)
    try {
      const result = await window.api.proxyTopupCustomer(modal.customer.id, amount, modalNote.trim() || undefined)
      if (result.success) {
        flash(t(`New balance: ${result.creditBalance}`, `最新余额：${result.creditBalance}`))
        setModal(null)
        await loadCustomers()
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [modal, modalInput, modalNote, flash, t, loadCustomers])

  const toggleEnabled = useCallback(async (c: CustomerView) => {
    setBusy(true)
    try {
      const result = await window.api.proxySetCustomerEnabled(c.id, !c.enabled)
      if (result.success) {
        flash(c.enabled
          ? t('Customer disabled', '客户已停用')
          : t('Customer enabled', '客户已启用'))
        await loadCustomers()
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [flash, t, loadCustomers])

  const resetPassword = useCallback((c: CustomerView) => {
    setModalInput('')
    setModal({ kind: 'password', customer: c })
  }, [])

  const submitResetPassword = useCallback(async () => {
    if (!modal || modal.kind !== 'password') return
    if (modalInput.length < 8) {
      flash(t('Password must be at least 8 characters', '密码至少 8 位'), true)
      return
    }
    setBusy(true)
    try {
      const result = await window.api.proxyResetCustomerPassword(modal.customer.id, modalInput)
      if (result.success) {
        flash(t('Password reset', '密码已重置'))
        setModal(null)
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [modal, modalInput, flash, t])

  const removeCustomer = useCallback((c: CustomerView) => {
    setModal({ kind: 'delete', customer: c })
  }, [])

  const submitDelete = useCallback(async () => {
    if (!modal || modal.kind !== 'delete') return
    setBusy(true)
    try {
      const result = await window.api.proxyDeleteCustomer(modal.customer.id)
      if (result.success) {
        flash(t(`Customer deleted (${result.revokedKeys} keys revoked)`, `客户已删除（吊销 ${result.revokedKeys} 个 Key）`))
        setModal(null)
        await loadCustomers()
      } else {
        flash(result.error || t('Failed', '失败'), true)
      }
    } finally {
      setBusy(false)
    }
  }, [modal, flash, t, loadCustomers])

  const fmtDate = (ts?: number): string => {
    if (!ts) return '-'
    try { return new Date(ts).toLocaleString(isEn ? 'en-US' : 'zh-CN') } catch { return '-' }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            <CardTitle>{t('Customer Portal', '客户门户')}</CardTitle>
          </div>
          <Button variant="ghost" size="sm" onClick={() => { loadStatus(); loadCustomers() }} disabled={busy}>
            <RefreshCw className="h-4 w-4 mr-1" />
            {t('Refresh', '刷新')}
          </Button>
        </div>
        <CardDescription>
          {t('Let customers log in, self-create API keys, and bill by Kiro credit.',
             '让客户自助登录、创建 API Key，并按 Kiro credit 计费。')}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {(error || notice) && (
          <div className={`text-sm rounded-md px-3 py-2 ${error ? 'bg-destructive/10 text-destructive' : 'bg-emerald-500/10 text-emerald-600'}`}>
            {error || notice}
          </div>
        )}

        {/* 门户开关 + 访问地址 */}
        <div className="rounded-lg border p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">{t('Enable customer portal', '启用客户门户')}</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                {t('Exposes /portal login page and customer self-service API.',
                   '开启 /portal 登录页与客户自助 API。')}
              </p>
            </div>
            <Switch checked={portalEnabled} onCheckedChange={togglePortal} disabled={busy} />
          </div>

          {portalEnabled && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <code className="flex-1 text-xs bg-muted rounded px-2 py-1.5 truncate">{portalUrl}</code>
                <Button variant="outline" size="sm" onClick={copyUrl}>
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
                <Button variant="outline" size="sm" onClick={() => window.api.openExternal(portalUrl)}>
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
              {hostIsLocal && (
                <div className="flex items-start gap-1.5 text-xs text-amber-600">
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>
                    {t('Bound to localhost — external customers must reach this through your tunnel domain, not this URL.',
                       '绑定在本机 — 外部客户需通过你的 tunnel 域名访问，而非此地址。')}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Google 登录配置 */}
        <div className="rounded-lg border p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">{t('Google login (invite-only)', 'Google 登录（仅限邀请）')}</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                {t('Customers sign in with Google. New accounts need an invite bound to their email.',
                   '客户用 Google 登录。新账号需使用与邮箱绑定的邀请码。')}
              </p>
            </div>
            <Switch checked={googleEnabled} onCheckedChange={toggleGoogle} disabled={busy || !googleClientId.trim()} />
          </div>
          <div>
            <Label className="text-xs">{t('Google OAuth Client ID', 'Google OAuth Client ID')}</Label>
            <div className="flex items-center gap-2 mt-1">
              <Input
                placeholder="xxxxxxxx.apps.googleusercontent.com"
                value={clientIdDraft}
                onChange={(e) => setClientIdDraft(e.target.value)}
              />
              <Button
                variant="outline"
                size="sm"
                onClick={saveClientId}
                disabled={busy || clientIdDraft.trim() === googleClientId}
              >
                {t('Save', '保存')}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-1.5">
              {t('Create an OAuth 2.0 Client ID (type: Web) in Google Cloud Console. Add your portal origin to Authorized JavaScript origins.',
                 '在 Google Cloud Console 创建 OAuth 2.0 客户端 ID（类型：Web），并把门户域名加入"已获授权的 JavaScript 来源"。')}
            </p>
          </div>
        </div>

        {/* 邀请码 */}
        <div className="rounded-lg border p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Ticket className="h-4 w-4 text-muted-foreground" />
            <Label className="text-sm font-medium">{t('Invites', '邀请码')}</Label>
          </div>
          <p className="text-xs text-muted-foreground">
            {t('An invite binds to one email. The customer logs in with that same Google account to register.',
               '一个邀请绑定一个邮箱。客户用同一 Google 账号登录即可完成注册。')}
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Input
              placeholder={t('Email to invite', '受邀邮箱')}
              value={invEmail}
              onChange={(e) => setInvEmail(e.target.value)}
              type="email"
            />
            <Input
              placeholder={t('Display name (optional)', '显示名（可选）')}
              value={invName}
              onChange={(e) => setInvName(e.target.value)}
            />
            <Input
              placeholder={t('Initial credit (optional)', '初始 credit（可选）')}
              value={invCredit}
              onChange={(e) => setInvCredit(e.target.value)}
              type="number"
            />
            <Input
              placeholder={t('Expires in days (optional)', '有效天数（可选）')}
              value={invExpiry}
              onChange={(e) => setInvExpiry(e.target.value)}
              type="number"
            />
          </div>
          <Button onClick={createInvite} disabled={busy} className="w-full" variant="outline">
            <Mail className="h-4 w-4 mr-1" />
            {t('Create invite', '创建邀请')}
          </Button>

          {invites.length > 0 && (
            <div className="space-y-2 pt-1">
              {invites.map((inv) => {
                const used = !!inv.usedAt
                const expired = !used && !!inv.expiresAt && inv.expiresAt < Date.now()
                return (
                  <div key={inv.code} className={`rounded-md border p-2.5 ${used || expired ? 'opacity-60' : ''}`}>
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium truncate">{inv.email}</span>
                          {used && <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{t('used', '已使用')}</span>}
                          {expired && <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600">{t('expired', '已过期')}</span>}
                          {!used && !expired && <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600">{t('pending', '待使用')}</span>}
                        </div>
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-1 text-xs text-muted-foreground">
                          {inv.creditBalance > 0 && <span>{t('Credit', '初始 credit')}: {inv.creditBalance}</span>}
                          <span>{t('Created', '创建')}: {fmtDate(inv.createdAt)}</span>
                          {inv.expiresAt && <span>{t('Expires', '过期')}: {fmtDate(inv.expiresAt)}</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        {!used && !expired && (
                          <Button variant="ghost" size="icon" className="h-7 w-7" title={t('Copy invite link', '复制邀请链接')} onClick={() => copyInviteLink(inv.code)} disabled={busy}>
                            {copiedCode === inv.code ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                          </Button>
                        )}
                        {!used && (
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" title={t('Revoke invite', '撤销邀请')} onClick={() => revokeInvite(inv.code)} disabled={busy}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* 新建客户 */}
        <div className="rounded-lg border p-4 space-y-3">
          <Label className="text-sm font-medium">{t('Create customer', '创建客户')}</Label>
          <p className="text-xs text-muted-foreground">
            {t('Manual password account. For Google login, send an invite instead.',
               '手动创建密码账号。若用 Google 登录，请改用邀请。')}
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Input
              placeholder={t('Email (login)', '邮箱（登录账号）')}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              type="email"
            />
            <Input
              placeholder={t('Password (min 8)', '密码（至少 8 位）')}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              type="password"
            />
            <Input
              placeholder={t('Display name (optional)', '显示名（可选）')}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Input
              placeholder={t('Initial credit (optional)', '初始 credit（可选）')}
              value={initialCredit}
              onChange={(e) => setInitialCredit(e.target.value)}
              type="number"
            />
          </div>
          <Button onClick={createCustomer} disabled={busy} className="w-full">
            <Plus className="h-4 w-4 mr-1" />
            {t('Create customer', '创建客户')}
          </Button>
        </div>

        {/* 客户列表 */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('Customers', '客户')} ({customers.length})
          </Label>
          {loading ? (
            <p className="text-sm text-muted-foreground py-4 text-center">{t('Loading...', '加载中…')}</p>
          ) : customers.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">{t('No customers yet.', '暂无客户。')}</p>
          ) : (
            <div className="space-y-2">
              {customers.map((c) => (
                <div key={c.id} className={`rounded-lg border p-3 ${!c.enabled ? 'opacity-60' : ''}`}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm truncate">{c.email}</span>
                        {c.name && <span className="text-xs text-muted-foreground truncate">({c.name})</span>}
                        {!c.enabled && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-destructive/15 text-destructive">
                            {t('disabled', '已停用')}
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-1.5 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Wallet className="h-3 w-3" />
                          {t('Balance', '余额')}: <span className={`font-medium ${c.creditBalance <= 0 ? 'text-destructive' : 'text-foreground'}`}>{c.creditBalance}</span>
                        </span>
                        <span>{t('Keys', 'Key')}: {c.keyCount}/{c.maxKeys}</span>
                        <span>{t('Topped up', '累计充值')}: {c.totalToppedUp}</span>
                        <span>{t('Created', '创建')}: {fmtDate(c.createdAt)}</span>
                        <span>{t('Last login', '上次登录')}: {fmtDate(c.lastLoginAt)}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Button variant="ghost" size="icon" className="h-7 w-7" title={t('Top up credit', '充值')} onClick={() => topup(c)} disabled={busy}>
                        <Wallet className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-7 w-7" title={c.enabled ? t('Disable', '停用') : t('Enable', '启用')} onClick={() => toggleEnabled(c)} disabled={busy}>
                        <Power className={`h-3.5 w-3.5 ${c.enabled ? 'text-emerald-600' : 'text-muted-foreground'}`} />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-7 w-7" title={t('Reset password', '重置密码')} onClick={() => resetPassword(c)} disabled={busy}>
                        <KeyRound className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" title={t('Delete', '删除')} onClick={() => removeCustomer(c)} disabled={busy}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>

      {/* 操作弹窗：充值/扣减、重置密码、删除（替代 Electron 屏蔽的 window.prompt/confirm） */}
      {modal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => !busy && setModal(null)} />
          <div className="relative w-full max-w-sm mx-4 rounded-lg border bg-background p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">
                {modal.kind === 'topup' && t('Top up / deduct credit', '充值 / 扣减 credit')}
                {modal.kind === 'password' && t('Reset password', '重置密码')}
                {modal.kind === 'delete' && t('Delete customer', '删除客户')}
              </h3>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setModal(null)} disabled={busy}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground break-all">{modal.customer.email}</p>

            {modal.kind === 'topup' && (
              <div className="space-y-3">
                <div>
                  <Label className="text-xs">{t('Amount (negative to deduct)', '金额（负数为扣减）')}</Label>
                  <Input
                    type="number"
                    value={modalInput}
                    onChange={(e) => setModalInput(e.target.value)}
                    autoFocus
                    onKeyDown={(e) => { if (e.key === 'Enter') submitTopup() }}
                  />
                </div>
                <div>
                  <Label className="text-xs">{t('Note (optional)', '备注（可选）')}</Label>
                  <Input value={modalNote} onChange={(e) => setModalNote(e.target.value)} />
                </div>
                <div className="text-xs text-muted-foreground">
                  {t('Current balance', '当前余额')}: <span className="font-medium text-foreground">{modal.customer.creditBalance}</span>
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" size="sm" onClick={() => setModal(null)} disabled={busy}>{t('Cancel', '取消')}</Button>
                  <Button size="sm" onClick={submitTopup} disabled={busy}>{t('Confirm', '确认')}</Button>
                </div>
              </div>
            )}

            {modal.kind === 'password' && (
              <div className="space-y-3">
                <div>
                  <Label className="text-xs">{t('New password (min 8 chars)', '新密码（至少 8 位）')}</Label>
                  <Input
                    type="password"
                    value={modalInput}
                    onChange={(e) => setModalInput(e.target.value)}
                    autoFocus
                    onKeyDown={(e) => { if (e.key === 'Enter') submitResetPassword() }}
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" size="sm" onClick={() => setModal(null)} disabled={busy}>{t('Cancel', '取消')}</Button>
                  <Button size="sm" onClick={submitResetPassword} disabled={busy}>{t('Confirm', '确认')}</Button>
                </div>
              </div>
            )}

            {modal.kind === 'delete' && (
              <div className="space-y-3">
                <div className="flex items-start gap-1.5 text-xs text-destructive">
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>
                    {t(`This revokes all ${modal.customer.keyCount} of their API keys and cannot be undone.`,
                       `将吊销其名下全部 ${modal.customer.keyCount} 个 API Key，且不可恢复。`)}
                  </span>
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" size="sm" onClick={() => setModal(null)} disabled={busy}>{t('Cancel', '取消')}</Button>
                  <Button variant="destructive" size="sm" onClick={submitDelete} disabled={busy}>{t('Delete', '删除')}</Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}
