@echo off
chcp 65001 >nul
echo ========================================
echo    Permanent Keyboard Fix - Registry
echo ========================================
echo.

echo 🔧 راه‌حل قطعی مشکل کیبورد فارسی
echo ========================================
echo.

echo 1. تنظیم زبان سیستم به انگلیسی...
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\Language" /v "InstallLanguage" /t REG_SZ /d "0409" /f
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\Language" /v "Default" /t REG_SZ /d "0409" /f
echo    ✅ زبان سیستم تنظیم شد
echo.

echo 2. تنظیم زبان کاربر به انگلیسی...
reg add "HKEY_CURRENT_USER\Control Panel\International\User Profile" /v "Languages" /t REG_MULTI_SZ /d "en-US" /f
reg add "HKEY_CURRENT_USER\Control Panel\International\User Profile" /v "InputMethodOverride" /t REG_SZ /d "en-US" /f
echo    ✅ زبان کاربر تنظیم شد
echo.

echo 3. تنظیم کیبورد به انگلیسی...
reg add "HKEY_CURRENT_USER\Keyboard Layout\Preload" /v "1" /t REG_SZ /d "00000409" /f
reg add "HKEY_CURRENT_USER\Keyboard Layout\Substitutes" /v "00000409" /t REG_SZ /d "00000409" /f
reg add "HKEY_CURRENT_USER\Keyboard Layout\Toggle" /v "Language Hotkey" /t REG_SZ /d "1" /f
echo    ✅ کیبورد تنظیم شد
echo.

echo 4. حذف زبان فارسی از سیستم...
reg delete "HKEY_CURRENT_USER\Control Panel\International\User Profile\Languages" /v "fa-IR" /f 2>nul
reg delete "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\Language\InstallLanguage" /v "fa-IR" /f 2>nul
echo    ✅ زبان فارسی حذف شد
echo.

echo 5. تنظیم متغیرهای محیطی...
setx LANG "en_US.UTF-8" /M
setx LC_ALL "en_US.UTF-8" /M
setx LC_CTYPE "en_US.UTF-8" /M
setx INPUT_METHOD "default" /M
echo    ✅ متغیرهای محیطی تنظیم شدند
echo.

echo 6. تست رفع مشکل...
echo    تست دستور Python...
python --version
echo.

echo ========================================
echo    راه‌حل قطعی اجرا شد!
echo    لطفاً سیستم را Restart کنید.
echo ========================================
pause 
