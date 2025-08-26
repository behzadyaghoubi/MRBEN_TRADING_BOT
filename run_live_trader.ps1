Write-Host "Starting live trader..." -ForegroundColor Green
try {
    & python live_trader_clean.py
} catch {
    Write-Host "Error running live trader: $_" -ForegroundColor Red
}
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
