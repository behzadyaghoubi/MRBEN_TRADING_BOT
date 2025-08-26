# PHASE1_SNAPSHOT.PS1 - A/B Testing Report Generator
# Generates comprehensive reports for Phase 1 verification

Write-Host "📊 PHASE1 SNAPSHOT REPORT" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Get system status
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
$systemStatus = if ($pythonProcesses) { "Running" } else { "Stopped" }
Write-Host "System Status: $systemStatus" -ForegroundColor $(if ($systemStatus -eq "Running") { "Green" } else { "Red" })

# Check logs for Ensemble labels
$logPath = ".\logs\mrben.log"
Write-Host "`n📝 Logs (Ensemble lines):" -ForegroundColor Yellow

if (Test-Path $logPath) {
    $ensembleLines = Get-Content $logPath -Tail 600 | Select-String -Pattern '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]|legacy|SMA_Only' | Select-Object -Last 12
    
    if ($ensembleLines) {
        foreach ($line in $ensembleLines) {
            $color = if ($line -match 'ERROR|Exception') { "Red" } 
                    elseif ($line -match 'legacy|SMA_Only') { "Yellow" }
                    else { "Green" }
            Write-Host $line -ForegroundColor $color
        }
    } else {
        Write-Host "No ensemble activity found in recent logs" -ForegroundColor Yellow
    }
} else {
    Write-Host "Log file not found at: $logPath" -ForegroundColor Red
}

# Check metrics
Write-Host "`n📊 Metrics (key lines):" -ForegroundColor Yellow
try {
    $metrics = Invoke-WebRequest -Uri "http://127.0.0.1:8765/metrics" -UseBasicParsing -ErrorAction Stop
    $metricsContent = $metrics.Content
    
    $keyMetrics = ($metricsContent -split "`n") | Select-String -Pattern '^mrben_(trades|blocks|decision_score|confidence_dyn|drawdown_pct).*' | Select-Object -First 10
    
    if ($keyMetrics) {
        foreach ($metric in $keyMetrics) {
            Write-Host $metric -ForegroundColor Green
        }
    } else {
        Write-Host "No key metrics found" -ForegroundColor Yellow
    }
    
    # Check A/B tracks
    $controlTrack = $metricsContent -match 'track="control"'
    $proTrack = $metricsContent -match 'track="pro"'
    
    Write-Host "`n🔍 A/B Tracks Analysis:" -ForegroundColor Cyan
    Write-Host "Control track seen?: $(if ($controlTrack) { 'Yes' } else { 'No' })" -ForegroundColor $(if ($controlTrack) { "Green" } else { "Red" })
    Write-Host "Pro track seen?: $(if ($proTrack) { 'Yes' } else { 'No' })" -ForegroundColor $(if ($proTrack) { "Green" } else { "Red" })
    
} catch {
    Write-Host "Metrics endpoint not accessible: $($_.Exception.Message)" -ForegroundColor Red
}

# Check for errors
Write-Host "`n🚨 Errors (recent):" -ForegroundColor Red
if (Test-Path $logPath) {
    $errors = Get-Content $logPath -Tail 300 | Select-String -Pattern '10030|10018|ERROR|Exception' | Select-Object -Last 5
    
    if ($errors) {
        foreach ($error in $errors) {
            Write-Host $error -ForegroundColor Red
        }
    } else {
        Write-Host "No recent errors found" -ForegroundColor Green
    }
} else {
    Write-Host "Log file not accessible for error checking" -ForegroundColor Yellow
}

# Summary
Write-Host "`n📋 PHASE1 SUMMARY:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan

$ensembleActive = if (Test-Path $logPath) { 
    (Get-Content $logPath -Tail 600 | Select-String -Quiet '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]')
} else { $false }

$abTracksActive = $controlTrack -and $proTrack
$noLegacyMode = if (Test-Path $logPath) { 
    -not (Get-Content $logPath -Tail 600 | Select-String -Quiet 'legacy|SMA_Only')
} else { $true }

Write-Host "✅ Ensemble Strategy Active: $(if ($ensembleActive) { 'Yes' } else { 'No' })" -ForegroundColor $(if ($ensembleActive) { "Green" } else { "Red" })
Write-Host "✅ A/B Tracks Active: $(if ($abTracksActive) { 'Yes' } else { 'No' })" -ForegroundColor $(if ($abTracksActive) { "Green" } else { "Red" })
Write-Host "✅ No Legacy Mode: $(if ($noLegacyMode) { 'Yes' } else { 'No' })" -ForegroundColor $(if ($noLegacyMode) { "Green" } else { "Red" })


if ($ensembleActive -and $abTracksActive -and $noLegacyMode) {
    Write-Host "`n🎉 PHASE1 VERIFICATION PASSED!" -ForegroundColor Green
    Write-Host "System is ready for Phase 2 (Canary Live)" -ForegroundColor Cyan
} else {
    Write-Host "`n⚠️ PHASE1 VERIFICATION INCOMPLETE" -ForegroundColor Yellow
    Write-Host "Check the issues above before proceeding to Phase 2" -ForegroundColor Red
}
