# MR BEN - Canary Mode Activation Script
# Activates PRO mode with comprehensive monitoring

Write-Host "🚀 MR BEN Canary Mode Activation" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Step 1: Verify emergency brake is active
Write-Host "`n🔒 Verifying Emergency Brake..." -ForegroundColor Yellow
if (Test-Path "halt.flag") {
    Write-Host "✅ Emergency brake active (halt.flag exists)" -ForegroundColor Green
} else {
    Write-Host "⚠️  Emergency brake not found - creating..." -ForegroundColor Yellow
    echo "" > halt.flag
    Write-Host "✅ Emergency brake created" -ForegroundColor Green
}

# Step 2: Set PRO Mode Environment Variables
Write-Host "`n⚙️ Setting PRO Mode Environment Variables..." -ForegroundColor Yellow
$env:MRBEN__RISK__BASE_R_PCT = "0.05"
$env:MRBEN__GATES__EXPOSURE_MAX_POSITIONS = "1"
$env:MRBEN__GATES__DAILY_LOSS_PCT = "0.8"
$env:MRBEN__CONFIDENCE__THRESHOLD__MIN = "0.62"
$env:MRBEN__STRATEGY__PRICE_ACTION__ENABLED = "true"
$env:MRBEN__STRATEGY__ML_FILTER__ENABLED = "true"
$env:MRBEN__STRATEGY__LSTM_FILTER__ENABLED = "true"
$env:MRBEN__CONFIDENCE__DYNAMIC__ENABLED = "true"
$env:MRBEN__SESSION__ENABLED = "true"

# Optional: Enable SLS strategy (uncomment if desired)
# $env:MRBEN__STRATEGY__CORE = "sls"

Write-Host "✅ Environment variables set for PRO mode" -ForegroundColor Green

# Step 3: Verify Configuration
Write-Host "`n🔍 Verifying Configuration..." -ForegroundColor Yellow
try {
    $dryRunResult = python mrben/app.py --config mrben/config/config.yaml --dry-run 2>&1
    if ($dryRunResult -match "config_loaded") {
        Write-Host "✅ Configuration verification successful" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Configuration verification incomplete - check logs" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Configuration verification failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 4: Start Shadow A/B Testing in Paper Mode
Write-Host "`n🧪 Starting Shadow A/B Testing (Paper Mode)..." -ForegroundColor Yellow
Write-Host "📝 This will run for 10-15 minutes to verify both control and pro tracks" -ForegroundColor Cyan

try {
    # Start MRBEN in background
    $job = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python mrben/main.py start --mode=paper --symbol XAUUSD.PRO --track=pro
    }
    
    Write-Host "✅ Shadow A/B testing started in background" -ForegroundColor Green
    Write-Host "⏰ Wait 10-15 minutes for system to stabilize" -ForegroundColor Cyan
    
} catch {
    Write-Host "❌ Failed to start shadow testing: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Start Monitoring
Write-Host "`n📊 Starting Monitoring Systems..." -ForegroundColor Yellow

# Start metrics monitor in new PowerShell window
Write-Host "   Starting Metrics Monitor..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$PWD'; .\watch_metrics.ps1" -WindowStyle Normal

# Start health watchdog in new PowerShell window
Write-Host "   Starting Health Watchdog..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$PWD'; python health_watchdog.py" -WindowStyle Normal

Write-Host "✅ Monitoring systems started" -ForegroundColor Green

# Step 6: Instructions
Write-Host "`n📋 Monitoring Instructions:" -ForegroundColor Cyan
Write-Host "1. Watch the Metrics Monitor window for real-time data" -ForegroundColor White
Write-Host "2. Monitor the Health Watchdog for automatic safety checks" -ForegroundColor White
Write-Host "3. Look for track='control' and track='pro' labels in metrics" -ForegroundColor White
Write-Host "4. Verify no 'legacy mode' or 'SMA_Only' in logs" -ForegroundColor White
Write-Host "5. Check for PA, ML, LSTM decision logs" -ForegroundColor White

Write-Host "`n🎯 What to Look For:" -ForegroundColor Cyan
Write-Host "✅ Both control and pro tracks generating signals" -ForegroundColor Green
Write-Host "✅ No legacy mode logs" -ForegroundColor Green
Write-Host "✅ Ensemble decisions (PA + ML + LSTM) active" -ForegroundColor Green
Write-Host "✅ Risk gates functioning properly" -ForegroundColor Green
Write-Host "✅ Metrics collection working" -ForegroundColor Green

# Step 7: Emergency Instructions
Write-Host "`n🚨 Emergency Instructions:" -ForegroundColor Red
Write-Host "To stop immediately: New-Item -ItemType File -Name 'halt.flag' -Force" -ForegroundColor White
Write-Host "To check status: python mrben/main.py status" -ForegroundColor White
Write-Host "To check health: python mrben/main.py health" -ForegroundColor White

# Step 8: Go Live Instructions
Write-Host "`n🚀 Go Live Instructions (After 10-15 min verification):" -ForegroundColor Green
Write-Host "1. Verify both tracks are working correctly" -ForegroundColor White
Write-Host "2. Check metrics show healthy operation" -ForegroundColor White
Write-Host "3. Remove emergency brake: Remove-Item 'halt.flag'" -ForegroundColor White
Write-Host "4. Start live trading: python mrben/main.py start --mode=live --symbol XAUUSD.PRO --track=pro" -ForegroundColor White

Write-Host "`n🎯 Canary Mode Activation Complete!" -ForegroundColor Green
Write-Host "Next: Monitor for 10-15 minutes, then decide to go live" -ForegroundColor Cyan
Write-Host "`nPress any key to continue monitoring..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Keep script running to maintain environment variables
Write-Host "`n🔄 Keeping environment variables active..." -ForegroundColor Cyan
Write-Host "Close this window when ready to go live" -ForegroundColor Yellow

while ($true) {
    Start-Sleep -Seconds 30
    Write-Host "Environment variables still active... Close window to stop" -ForegroundColor DarkGray
}
