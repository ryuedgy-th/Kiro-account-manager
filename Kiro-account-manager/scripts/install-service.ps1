# Install Kiro headless proxy as a Windows Service via NSSM
# Requires NSSM (winget install nssm  OR  choco install nssm)
# Usage (run PowerShell as Administrator, in the app folder):
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

if (-not (Test-Path $cjs)) { throw "Not built yet. Run 'npm run build:service' first (missing $cjs)" }
if (-not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

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
Write-Host "Manage: nssm restart/stop/remove $ServiceName  | logs: $DataDir\service.log"
