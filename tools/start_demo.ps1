# Start LiveTalking + a Cloudflare quick tunnel (no account needed) and print the public test URL.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File tools\start_demo.ps1
# Stop with:          powershell -ExecutionPolicy Bypass -File tools\stop_demo.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$cloudflared = 'D:\tools\cloudflared\cloudflared.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null

if (Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8010 is already in use. Run tools\stop_demo.ps1 first.'
}

$app = Start-Process -FilePath $python -WorkingDirectory $root `
    -ArgumentList 'app.py', '--transport', 'webrtc', '--model', 'wav2lip', '--avatar_id', 'wav2lip256_avatar1' `
    -RedirectStandardOutput (Join-Path $logDir 'app.out.log') -RedirectStandardError (Join-Path $logDir 'app.err.log') `
    -WindowStyle Hidden -PassThru
Set-Content (Join-Path $logDir 'app.pid') $app.Id

Write-Host 'Waiting for LiveTalking on :8010 (model load can take a minute)...'
$deadline = (Get-Date).AddMinutes(5)
while (-not (Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue)) {
    if ($app.HasExited) { throw "LiveTalking exited (code $($app.ExitCode)); see $logDir\app.err.log" }
    if ((Get-Date) -gt $deadline) { throw 'LiveTalking did not open :8010 within 5 minutes' }
    Start-Sleep -Seconds 2
}

$tunnelLog = Join-Path $logDir 'tunnel.err.log'
$tunnel = Start-Process -FilePath $cloudflared -ArgumentList 'tunnel', '--no-autoupdate', '--url', 'http://localhost:8010' `
    -RedirectStandardOutput (Join-Path $logDir 'tunnel.out.log') -RedirectStandardError $tunnelLog `
    -WindowStyle Hidden -PassThru
Set-Content (Join-Path $logDir 'tunnel.pid') $tunnel.Id

$url = $null
$deadline = (Get-Date).AddSeconds(60)
while (-not $url) {
    if ($tunnel.HasExited) { throw "cloudflared exited (code $($tunnel.ExitCode)); see $tunnelLog" }
    if ((Get-Date) -gt $deadline) { throw "No tunnel URL within 60 s; see $tunnelLog" }
    Start-Sleep -Seconds 2
    $m = Select-String -Path $tunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $url = $m.Matches[0].Value }
}
Write-Host ''
Write-Host "Test URL: $url/dashboard.html"
Write-Host 'The address changes every time the tunnel restarts.'
