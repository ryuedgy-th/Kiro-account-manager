# One-shot deploy + run Kiro headless proxy service (Windows)
# ใช้: เปิด PowerShell ในโฟลเดอร์แอป (ที่มี package.json) แล้วรัน:
#   .\scripts\deploy-service.ps1 -DataFile C:\path\to\kiro-service-data.json -Port 5581
param(
  [string]$DataFile = "$PSScriptRoot\..\kiro-service-data.json",
  [string]$DataDir  = "$env:USERPROFILE\.kiro-proxy-service",
  [int]$Port = 5581,
  [string]$ProxyHost = "127.0.0.1",
  [switch]$SkipPull
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

if (-not $SkipPull) {
  Write-Host "[deploy] git pull..." -ForegroundColor Cyan
  git pull
}
Write-Host "[deploy] npm install..." -ForegroundColor Cyan
npm install
Write-Host "[deploy] build service bundle..." -ForegroundColor Cyan
npm run build:service

if (-not (Test-Path $DataFile)) {
  Write-Host "[deploy] ไม่พบ $DataFile — แปลงจาก electron-store เดิมอัตโนมัติ..." -ForegroundColor Yellow
  node "scripts\export-service-data.mjs" --out $DataFile
}
if (-not (Test-Path $DataFile)) {
  Write-Warning "ยังไม่มีไฟล์ข้อมูล: $DataFile"
  Write-Warning "ลองรันเอง: node scripts\export-service-data.mjs --store <path ของ kiro-accounts.json> --out $DataFile"
  exit 1
}

$env:KIRO_SERVICE_DATA = (Resolve-Path $DataFile).Path
$env:KIRO_DATA_DIR     = $DataDir
$env:KIRO_PROXY_PORT   = "$Port"
$env:KIRO_PROXY_HOST   = $ProxyHost
Write-Host "[deploy] starting service on http://${ProxyHost}:${Port} (data=$($env:KIRO_SERVICE_DATA))" -ForegroundColor Green
node "out\service\index.cjs"
