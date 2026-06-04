# Headless Proxy Service

รัน Kiro reverse-proxy แบบ 24/7 โดยไม่ต้องเปิด Electron GUI

## Build & Run

```bash
npm run build:service     # bundle → out/service/index.cjs (ไม่รวม electron)
npm run service           # node out/service/index.cjs
```

## Environment variables

| ตัวแปร | ความหมาย | ดีฟอลต์ |
|---|---|---|
| `KIRO_DATA_DIR` | โฟลเดอร์เก็บ cert/log | `~/.kiro-proxy-service` |
| `KIRO_SERVICE_DATA` | path ไฟล์ JSON (config+accounts+stats) | `<KIRO_DATA_DIR>/kiro-service-data.json` |
| `KIRO_PROXY_PORT` | override port | `5580` |
| `KIRO_PROXY_HOST` | override host | `127.0.0.1` |

## ไฟล์ข้อมูล (kiro-service-data.json)

```jsonc
{
  "proxyConfig": { "port": 5580, "apiKeys": [], "enableMultiAccount": true, "accountSelectionStrategy": "round-robin", "sessionAffinityEnabled": true },
  "accountData": {
    "accounts": {
      "<id>": { "id": "<id>", "email": "...", "status": "active",
        "credentials": { "accessToken": "...", "refreshToken": "...", "clientId": "...", "clientSecret": "...", "region": "us-east-1", "authMethod": "social|...", "provider": "..." },
        "machineId": "...", "profileArn": "..." }
    },
    "accountProxyBindings": {}, "proxyPool": {}
  },
  "stats": {}
}
```
> รูปแบบ `accountData` ตรงกับ store ภายในแอป — export จากแอป (JSON + Include Credentials) แล้ววางในไฟล์นี้ได้
> token/credit/usage จะถูกเขียนกลับไฟล์นี้อัตโนมัติ (รอดข้ามรีสตาร์ท)

## รันเป็น Windows Service (NSSM)

```powershell
nssm install KiroProxy "C:\Program Files\nodejs\node.exe" "C:\path\to\out\service\index.cjs"
nssm set KiroProxy AppEnvironmentExtra KIRO_DATA_DIR=C:\KiroProxy KIRO_PROXY_PORT=5580
nssm set KiroProxy AppDirectory C:\path\to\repo
nssm start KiroProxy
```

NSSM จะรีสตาร์ทอัตโนมัติเมื่อ process ตาย และสตาร์ทเองหลังบูต (ไม่ต้อง login desktop session เหมือน Electron)
