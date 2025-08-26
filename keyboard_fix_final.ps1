# Keyboard Layout Fix Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Keyboard Layout Fix Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Fixing keyboard layout..." -ForegroundColor Green
Write-Host ""

# Set English US as default keyboard layout
Write-Host "Setting English US as default keyboard layout..." -ForegroundColor Yellow
Set-ItemProperty -Path "HKCU:\Keyboard Layout\Preload" -Name "1" -Value "00000409" -Force

# Set system locale
Write-Host "Setting system locale..." -ForegroundColor Yellow
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Nls\Language" -Name "InstallLanguage" -Value "0409" -Force
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Nls\Language" -Name "Default" -Value "0409" -Force

# Set user locale
Write-Host "Setting user locale..." -ForegroundColor Yellow
Set-ItemProperty -Path "HKCU:\Control Panel\International" -Name "Locale" -Value "00000409" -Force

# Remove Persian/Farsi if exists
Write-Host "Removing Persian/Farsi language..." -ForegroundColor Yellow
try {
    $languages = Get-WinUserLanguageList
    $languages = $languages | Where-Object {$_.LanguageTag -ne "fa-IR"}
    Set-WinUserLanguageList $languages -Force
} catch {
    Write-Host "Warning: Could not remove Persian language (may not be installed)" -ForegroundColor Yellow
}

# Set environment variables
Write-Host "Setting environment variables..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("LANG", "en_US.UTF-8", "Machine")
[Environment]::SetEnvironmentVariable("LC_ALL", "en_US.UTF-8", "Machine")
[Environment]::SetEnvironmentVariable("LC_CTYPE", "en_US.UTF-8", "Machine")
[Environment]::SetEnvironmentVariable("INPUT_METHOD", "default", "Machine")

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Keyboard fix completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Please restart your computer for changes to take effect." -ForegroundColor Yellow
Write-Host ""
Write-Host "After restart, test with: python --version" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to continue"
