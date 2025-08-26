# -------- PHASE1_MANUAL_REPORT.ps1 --------
$ErrorActionPreference = 'SilentlyContinue'
$URL = "http://127.0.0.1:8765/metrics"
$LOG = ".\logs\mrben.log"

$report = @()
$report += "PHASE1_REPORT"

# System Status
$report += ""
$report += "System Status:"
try { $status = (python mrben\main.py status) } catch { $status = $_.ToString() }
$report += ($status | Out-String).Trim()

# Metrics (port 8765)
$report += ""
$report += "Metrics (key lines):"
try {
  $m = (Invoke-WebRequest -UseBasicParsing -Uri $URL -TimeoutSec 3).Content
  $report += (($m -split "`n") | Select-String '^mrben_(trades_opened_total|trades_closed_total|blocks_total).*track=' | Select-Object -First 6 | ForEach { $_.Line })
  $report += (($m -split "`n") | Select-String '^mrben_(decision_score|confidence_dyn|drawdown_pct)\s'        | Select-Object -First 4 | ForEach { $_.Line })
} catch { $report += "metrics unreachable at $URL" }

# A/B tracks seen?
$ab = ""
if ($m) {
  $hasControl = ($m -match 'track="control"')
  $hasPro     = ($m -match 'track="pro"')
  $ab = "A/B Tracks seen?: " + ($(if($hasControl -and $hasPro){"Yes"}else{"No"}) + " (control & pro)")
} else { $ab = "A/B Tracks seen?: Unknown (metrics unreachable)" }
$report += ""
$report += $ab

# Logs (ensemble labels)
$report += ""
$report += "Logs (5-10 lines):"
if (Test-Path $LOG) {
  $lines = Get-Content $LOG -Tail 400
  $last  = $lines | Select-String '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]|legacy|SMA_Only' | Select-Object -Last 10
  if ($last) { $report += ($last | ForEach { $_.Line }) } else { $report += "(no recent ensemble tags found)" }
} else { $report += "log file not found: $LOG" }

# Ensemble labels seen?
$seenEns = $false
if (Test-Path $LOG) { $seenEns = (Get-Content $LOG -Tail 400 | Select-String -Quiet '\[PA\]|\[ML\]|\[LSTM\]|\[CONF\]|\[VOTE\]') }
$report += ""
$report += "Ensemble labels seen?: " + ($(if($seenEns){"Yes"}else{"No"}) + " ([PA/ML/LSTM/CONF/VOTE])")

# Errors
$report += ""
$report += "Errors:"
if (Test-Path $LOG) {
  $errs = Get-Content $LOG -Tail 300 | Select-String '10030|10018|error|ERROR' | Select-Object -Last 5
  if ($errs) { $report += ($errs | ForEach { $_.Line }) } else { $report += "(none)" }
} else { $report += "(log not found)" }

$txt = $report -join "`r`n"
$txt | Tee-Object -FilePath .\PHASE1_REPORT.txt
Write-Host "`n--- PHASE1_REPORT saved to PHASE1_REPORT.txt ---`n" -ForegroundColor Cyan
# -------- end --------
