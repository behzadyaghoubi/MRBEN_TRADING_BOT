param(
  [ValidateSet("paper","live")]$Mode="paper",
  [string]$Symbol="XAUUSD.PRO",
  [switch]$AB=$true
)

# Safety check - ensure emergency brake is in place
if ($Mode -eq "live") {
    Write-Host "⚠️  LIVE MODE - Safety checks required" -ForegroundColor Yellow

    # Check if halt.flag exists
    if (Test-Path ".\halt.flag") {
        Write-Host "❌ Emergency brake is active. Remove halt.flag before live trading." -ForegroundColor Red
        exit 1
    }

    # Confirm live trading
    $confirm = Read-Host "Are you sure you want to start LIVE trading? (type 'YES' to confirm)"
    if ($confirm -ne "YES") {
        Write-Host "Live trading cancelled." -ForegroundColor Yellow
        exit 0
    }

    Write-Host "✅ Live trading confirmed. Starting system..." -ForegroundColor Green
} else {
    # Paper mode - create emergency brake
    New-Item -ItemType File -Force .\halt.flag | Out-Null
    Write-Host "✅ Emergency brake activated for paper mode" -ForegroundColor Green
}

# Start PRO main system
$abFlag = if ($AB.IsPresent) { "--ab=on" } else { "--ab=off" }
Write-Host "🚀 Starting MRBEN PRO system..." -ForegroundColor Cyan
Write-Host "Mode: $Mode, Symbol: $Symbol, A/B: $abFlag" -ForegroundColor White

python mrben\main.py start --mode=$Mode --symbol $Symbol --track=pro $abFlag

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ MRBEN PRO system started successfully!" -ForegroundColor Green
    Write-Host "📊 Monitor metrics at: http://127.0.0.1:8765/metrics" -ForegroundColor Cyan
    Write-Host "📝 Check logs at: .\logs\mrben.log" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to start MRBEN PRO system" -ForegroundColor Red
    exit 1
}
