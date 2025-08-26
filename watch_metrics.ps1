# MR BEN - Metrics Watch Script
# Monitors key metrics every 5 seconds with alerts

param(
    [string]$Url = "http://127.0.0.1:8765/metrics", 
    [int]$Interval = 5
)

function Get-Metric([string]$name, [string]$content) {
    $line = ($content -split "`n") | Where-Object { $_ -match "^$name\s" }
    if ($line) { 
        return [double](($line -split "\s+")[-1]) 
    } else { 
        return $null 
    }
}

function Get-Counter([string]$regex, [string]$content) {
    $lines = ($content -split "`n") | Where-Object { $_ -match $regex }
    return $lines | Select-Object -First 8
}

Write-Host "🚀 MR BEN Metrics Monitor Starting..." -ForegroundColor Green
Write-Host "URL: $Url" -ForegroundColor Cyan
Write-Host "Interval: ${Interval}s" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

while ($true) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        $c = $resp.Content

        # Get key metrics
        $equity = Get-Metric "mrben_equity" $c
        $dd     = Get-Metric "mrben_drawdown_pct" $c
        $score  = Get-Metric "mrben_decision_score" $c
        $conf   = Get-Metric "mrben_confidence_dyn" $c
        $exposure = Get-Metric "mrben_exposure_positions" $c

        # Get trade counters
        $openedPro = Get-Counter '^mrben_trades_opened_total\{.*track="pro".*\}' $c
        $closedPro = Get-Counter '^mrben_trades_closed_total\{.*track="pro".*\}' $c
        $openedControl = Get-Counter '^mrben_trades_opened_total\{.*track="control".*\}' $c
        $closedControl = Get-Counter '^mrben_trades_closed_total\{.*track="control".*\}' $c
        
        # Get block counters
        $blocks = Get-Counter '^mrben_blocks_total\{' $c

        # Clear screen and display
        Clear-Host
        Write-Host ("MR BEN – Metrics Watch @ {0}" -f (Get-Date)) -ForegroundColor Cyan
        Write-Host ("=" * 60) -ForegroundColor DarkGray
        
        # Key metrics
        Write-Host ("💰 Equity: {0:N2}   📉 DD%: {1:N2}   🎯 Score: {2:N2}   🧠 DynConf: {3:N2}   📊 Exposure: {4}" -f $equity,$dd,$score,$conf,$exposure) -ForegroundColor White
        
        # PRO track trades
        Write-Host "`n🚀 PRO Track Trades:" -ForegroundColor Green
        if ($openedPro) {
            Write-Host "   Opened:" -ForegroundColor DarkGreen
            $openedPro | ForEach-Object { Write-Host "     $_" -ForegroundColor Green }
        } else {
            Write-Host "   Opened: None yet" -ForegroundColor DarkGray
        }
        
        if ($closedPro) {
            Write-Host "   Closed:" -ForegroundColor DarkGreen
            $closedPro | ForEach-Object { Write-Host "     $_" -ForegroundColor Green }
        } else {
            Write-Host "   Closed: None yet" -ForegroundColor DarkGray
        }
        
        # Control track trades
        Write-Host "`n🔬 Control Track Trades:" -ForegroundColor Blue
        if ($openedControl) {
            Write-Host "   Opened:" -ForegroundColor DarkBlue
            $openedControl | ForEach-Object { Write-Host "     $_" -ForegroundColor Blue }
        } else {
            Write-Host "   Opened: None yet" -ForegroundColor DarkGray
        }
        
        if ($closedControl) {
            Write-Host "   Closed:" -ForegroundColor DarkBlue
            $closedControl | ForEach-Object { Write-Host "     $_" -ForegroundColor Blue }
        } else {
            Write-Host "   Closed: None yet" -ForegroundColor DarkGray
        }
        
        # Blocks
        Write-Host "`n🚫 Blocks (Top 8):" -ForegroundColor Red
        if ($blocks) {
            $blocks | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
        } else {
            Write-Host "   None yet" -ForegroundColor DarkGray
        }
        
        # Alerts
        if ($dd -gt 1.5) { 
            Write-Host "`n⚠️  WARNING: Drawdown > 1.5% — Consider HALT" -ForegroundColor Yellow 
        }
        if ($exposure -gt 1) { 
            Write-Host "`n⚠️  WARNING: Multiple positions open" -ForegroundColor Yellow 
        }
        if ($score -lt 0.6) { 
            Write-Host "`n⚠️  WARNING: Low decision score" -ForegroundColor Yellow 
        }
        
        Write-Host "`n" + ("=" * 60) -ForegroundColor DarkGray
        Write-Host "Next update in ${Interval}s... (Ctrl+C to stop)" -ForegroundColor DarkGray
        
    } catch {
        Write-Host "❌ Cannot reach $Url" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Make sure MRBEN is running and metrics port is accessible" -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds $Interval
}
