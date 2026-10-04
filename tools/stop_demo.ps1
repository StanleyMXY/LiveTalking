# Stop the processes started by start_demo.ps1 (tunnel first, then LiveTalking).
$logDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'logs'
foreach ($name in 'tunnel', 'app') {
    $pidFile = Join-Path $logDir "$name.pid"
    if (Test-Path $pidFile) {
        $procId = [int](Get-Content $pidFile)
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile
        Write-Host "stopped $name (pid $procId)"
    }
}
