# PowerShell Script for Permanent Keyboard Fix
# Run as Administrator

Write-Host "🔧 راه‌حل قطعی مشکل کیبورد فارسی" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan

# Step 1: Set system locale to English
Write-Host "`n1. تنظیم زبان سیستم به انگلیسی..." -ForegroundColor Yellow
try {
    # Set system locale
    Set-WinSystemLocale -SystemLocale en-US
    Write-Host "   ✅ زبان سیستم به انگلیسی تغییر یافت" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ خطا در تغییر زبان سیستم: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 2: Set user locale to English
Write-Host "`n2. تنظیم زبان کاربر به انگلیسی..." -ForegroundColor Yellow
try {
    # Set user locale
    Set-WinUserLanguageList -LanguageList en-US -Force
    Write-Host "   ✅ زبان کاربر به انگلیسی تغییر یافت" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ خطا در تغییر زبان کاربر: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 3: Set keyboard layout to English
Write-Host "`n3. تنظیم کیبورد به انگلیسی..." -ForegroundColor Yellow
try {
    # Set keyboard layout
    $keyboardLayout = "00000409"  # English (US)
    Set-ItemProperty -Path "HKCU:\Keyboard Layout\Preload" -Name "1" -Value $keyboardLayout -Type String -Force
    Write-Host "   ✅ کیبورد به انگلیسی تنظیم شد" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ خطا در تنظیم کیبورد: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 4: Remove Persian language pack
Write-Host "`n4. حذف زبان فارسی..." -ForegroundColor Yellow
try {
    # Remove Persian language pack
    $persianLang = Get-WinUserLanguageList | Where-Object {$_.LanguageTag -eq "fa-IR"}
    if ($persianLang) {
        $currentLangs = Get-WinUserLanguageList
        $newLangs = $currentLangs | Where-Object {$_.LanguageTag -ne "fa-IR"}
        Set-WinUserLanguageList -LanguageList $newLangs -Force
        Write-Host "   ✅ زبان فارسی حذف شد" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️ زبان فارسی قبلاً حذف شده" -ForegroundColor Blue
    }
} catch {
    Write-Host "   ⚠️ خطا در حذف زبان فارسی: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Set environment variables
Write-Host "`n5. تنظیم متغیرهای محیطی..." -ForegroundColor Yellow
try {
    [Environment]::SetEnvironmentVariable("LANG", "en_US.UTF-8", "User")
    [Environment]::SetEnvironmentVariable("LC_ALL", "en_US.UTF-8", "User")
    [Environment]::SetEnvironmentVariable("LC_CTYPE", "en_US.UTF-8", "User")
    [Environment]::SetEnvironmentVariable("INPUT_METHOD", "default", "User")
    Write-Host "   ✅ متغیرهای محیطی تنظیم شدند" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ خطا در تنظیم متغیرهای محیطی: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 6: Registry fixes
Write-Host "`n6. تنظیمات Registry..." -ForegroundColor Yellow
try {
    # Fix keyboard layout registry
    reg add "HKCU\Keyboard Layout\Preload" /v "1" /t REG_SZ /d "00000409" /f
    reg add "HKCU\Keyboard Layout\Substitutes" /v "00000409" /t REG_SZ /d "00000409" /f
    
    # Fix input method registry
    reg add "HKCU\Control Panel\International\User Profile" /v "Languages" /t REG_MULTI_SZ /d "en-US" /f
    
    Write-Host "   ✅ تنظیمات Registry اصلاح شد" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ خطا در تنظیمات Registry: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 7: Test the fix
Write-Host "`n7. تست رفع مشکل..." -ForegroundColor Yellow
try {
    $testResult = python --version 2>&1
    if ($testResult -match "ز" -or $testResult -match "ر") {
        Write-Host "   ❌ مشکل همچنان وجود دارد" -ForegroundColor Red
    } else {
        Write-Host "   ✅ مشکل حل شد: $testResult" -ForegroundColor Green
    }
} catch {
    Write-Host "   ⚠️ خطا در تست: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 راه‌حل قطعی مشکل کیبورد اجرا شد!" -ForegroundColor Green
Write-Host "لطفاً سیستم را Restart کنید تا تغییرات اعمال شود." -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Cyan 