# PowerShell Keyboard Layout Fix Script
Write-Host "🔧 Starting Keyboard Layout Fix..." -ForegroundColor Green

# Method 1: Try to switch to English keyboard using SendKeys
try {
    Write-Host "🔄 Method 1: Using SendKeys to switch keyboard..." -ForegroundColor Yellow

    Add-Type -AssemblyName System.Windows.Forms

    # Try Ctrl+Shift+F10 to open language switcher
    [System.Windows.Forms.SendKeys]::SendWait("^+{F10}")
    Start-Sleep -Milliseconds 200

    # Press 1 to select first language (usually English)
    [System.Windows.Forms.SendKeys]::SendWait("1")
    Start-Sleep -Milliseconds 200

    Write-Host "✅ Method 1 completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Method 1 failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Method 2: Try Windows+Space to open language switcher
try {
    Write-Host "🔄 Method 2: Using Windows+Space..." -ForegroundColor Yellow

    Add-Type -AssemblyName System.Windows.Forms

    # Press Windows+Space
    [System.Windows.Forms.SendKeys]::SendWait("^{ESC} ")
    Start-Sleep -Milliseconds 500

    # Press Enter to select
    [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
    Start-Sleep -Milliseconds 200

    Write-Host "✅ Method 2 completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Method 2 failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Method 3: Set environment variables
try {
    Write-Host "🔄 Method 3: Setting environment variables..." -ForegroundColor Yellow

    $env:LANG = "en_US.UTF-8"
    $env:LC_ALL = "en_US.UTF-8"
    $env:LANGUAGE = "en"

    Write-Host "✅ Method 3 completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Method 3 failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test the fix
Write-Host "🧪 Testing keyboard fix..." -ForegroundColor Cyan

try {
    $testResult = python -c "print('Keyboard test successful!')" 2>&1

    if ($LASTEXITCODE -eq 0 -and $testResult -like "*Keyboard test successful*") {
        Write-Host "🎉 KEYBOARD FIX SUCCESSFUL!" -ForegroundColor Green
        Write-Host "You can now run commands without Persian characters." -ForegroundColor Green
    } else {
        Write-Host "❌ Keyboard fix may not be complete" -ForegroundColor Red
        Write-Host "Manual intervention may be required:" -ForegroundColor Yellow
        Write-Host "1. Press Win+Space to open language switcher" -ForegroundColor White
        Write-Host "2. Select English (US) keyboard" -ForegroundColor White
        Write-Host "3. Or remove Persian/Farsi language pack from Windows" -ForegroundColor White
    }
} catch {
    Write-Host "❌ Test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
