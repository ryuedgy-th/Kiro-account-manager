# ติดตั้ง Kiro headless proxy เป็น Windows Service ด้วย NSSM
# ต้องมี NSSM ก่อน (winget install nssm หรือ choco install nssm)
# ใช้ (รัน PowerShell as Administrator ในโฟลเดอร์แอป):
#   .\scripts\install-service.ps1 -DataFile C:\KiroProxy\kiro-service-data.json -Port 5581
param(
  [Parameter(Mandatory=$true)][string]$DataFile,
  [string]$DataDir = "C:\KiroProxy",
  [int]$Port = 5581,
  [string]$ProxyHost = "127.0.0.1",
  [string]$ServiceName = "KiroProxy",
  [string]$Nssm = "nssm"
)
$ErrorActionPreference = "Stop"
$app  = (Resolve-Path "$PSScriptRoot\..").Path
$node = (Get-Command node).Source
$cjs  = Join-Path $app "out\service\index.cjs"

if (-not (Test-Path $cjs)) { throw "ยังไม่ได้ build: รัน 'npm run build:service' ก่อน (ไม่พบ $cjs)" }

Write-Host "[nssm] installing service '$ServiceName'..." -ForegroundColor Cyan
& $Nssm install $ServiceName $node $cjs
& $Nssm set $ServiceName AppDirectory $app
& $Nssm set $ServiceName AppEnvironmentExtra `
    "KIRO_SERVICE_DATA=$DataFile" `
    "KIRO_DATA_DIR=$DataDir" `
    "KIRO_PROXY_PORT=$Port" `
    "KIRO_PROXY_HOST=$ProxyHost"
& $Nssm set $ServiceName AppStdout (Join-Path $DataDir "service.log")
& $Nssm set $ServiceName AppStderr (Join-Path $DataDir "service.err.log")
& $Nssm set $ServiceName Start SERVICE_AUTO_START
& $Nssm start $ServiceName
Write-Host "[nssm] '$ServiceName' started on http://${ProxyHost}:${Port}" -ForegroundColor Green
Write-Host "จัดการ: nssm restart/stop/remove $ServiceName  | log: $DataDir\service.log"
