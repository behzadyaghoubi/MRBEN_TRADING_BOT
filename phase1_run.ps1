param(
  [string]$Symbol = "XAUUSD.PRO",
  [string]$Url = "http://127.0.0.1:8765/metrics",
  [int]$WaitMinutes = 12,
  [string]$LogPath = ".\logs\mrben.log"
)

function Wait-Metrics($Url, $timeoutSec=90){
  $t0 = Get-Date
  while((New-TimeSpan -Start $t0 -End (Get-Date)).TotalSeconds -lt $timeoutSec){
    try{
      $c = (Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2).Content
      if($c -match '^# HELP') { return $true }
    } catch {}
    Start-Sleep 2
  }
  return $false
}

function Get-MetricsLines($Url){
  try{
    $c = (Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3).Content
    $a = @()
    $a += ($c -split "`n") | Select-String -Pattern '^mrben_(trades_opened_total|trades_closed_total|blocks_total).*track=' | Select-Object -First 6
    $a += ($c -split "`n") | Select-String -Pattern '^mrben_(decision_score|confidence_dyn|drawdown_pct)\s' | Select-Object -First 4
    return $a
  } catch { return @("metrics unreachable") }
}

# --- Step 0: Safety brake on
ni halt.flag -Force | Out-Null

# --- Step 1: Start A/B (paper)
$pyArgs = "mrben\main.py start --mode=paper --symbol $Symbol --track=pro --ab=on"
$proc = Start-Process -FilePath "python" -ArgumentList $pyArgs -PassThru
Start-Sleep 3

# --- Step 2: Metrics up?
if(-not (Wait-Metrics $Url 120)){
  Write-Host "âŒ Metrics not accessible on $Url (port 8765 busy?)" -ForegroundColor Red
  Write-Host "Tip: netstat -ano | findstr :8765  â†’ taskkill /PID <pid> /F"
}

# --- Step 3: Poll logs/metrics up to $WaitMinutes
$deadline = (Get-Date).AddMinutes($WaitMinutes)
$seenEnsemble = $false; $seenControl=$false; $seenPro=$false
while(Get-Date -lt $deadline){
  $met = Get-MetricsLines $Url
  $seenControl = ($met | Out-String) -match 'track="control"'
  $seenPro     = ($met | Out-String) -match 'track="pro"'

  $seenEnsemble = (Test-Path $LogPath) -and `
    (Get-Content $LogPath -Tail 400 | Select-String -Quiet '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]')

  if($seenEnsemble -and $seenControl -and $seenPro){ break }
  Start-Sleep 5
}

# --- Step 4: Build PHASE1_REPORT
$report = @()
$report += "PHASE1_REPORT"
$report += "System Status: " + ($(if($proc.HasExited){"Stopped"}else{"Running"}))
$report += "Logs (5-10 lines):"
if(Test-Path $LogPath){
  $report += (Get-Content $LogPath -Tail 400 | Select-String -Pattern '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]|legacy|SMA_Only' | Select-Object -Last 10 | ForEach-Object {$_.Line})
}else{
  $report += "log file not found: $LogPath"
}
$report += ""
$report += "Metrics (6-10 lines):"
$report += (Get-MetricsLines $Url | ForEach-Object { $_.ToString() })
$report += ""
$report += "A/B Tracks seen?: " + ($(if($seenControl -and $seenPro){"Yes"}else{"No"}) + " (control & pro)")
$report += "Ensemble labels seen?: " + ($(if($seenEnsemble){"Yes"}else{"No"}) + " ([PA/ML/LSTM/CONF/VOTE])")
$report += "Errors:"
if(Test-Path $LogPath){
  $errs = Get-Content $LogPath -Tail 300 | Select-String -Pattern '10030|10018|error|ERROR' | Select-Object -Last 5
  if($errs){ $report += ($errs | ForEach-Object {$_.Line}) } else { $report += "(none)" }
} else { $report += "(log not found)" }

$report -join "`r`n" | Tee-Object -FilePath .\PHASE1_REPORT.txt
Write-Host "`n--- PHASE1_REPORT saved to PHASE1_REPORT.txt ---`n" -ForegroundColor Cyan
