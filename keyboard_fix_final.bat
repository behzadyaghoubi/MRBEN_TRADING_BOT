@echo off
echo ========================================
echo    Keyboard Layout Fix Script
echo ========================================
echo.

echo Fixing keyboard layout...
echo.

REM Set English US as default keyboard layout
reg add "HKEY_CURRENT_USER\Keyboard Layout\Preload" /v "1" /t REG_SZ /d "00000409" /f

REM Set system locale
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\Language" /v "InstallLanguage" /t REG_SZ /d "0409" /f
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Nls\Language" /v "Default" /t REG_SZ /d "0409" /f

REM Set user locale
reg add "HKEY_CURRENT_USER\Control Panel\International" /v "Locale" /t REG_SZ /d "00000409" /f

REM Set environment variables
setx LANG "en_US.UTF-8" /M
setx LC_ALL "en_US.UTF-8" /M
setx LC_CTYPE "en_US.UTF-8" /M
setx INPUT_METHOD "default" /M

echo.
echo ========================================
echo    Keyboard fix completed!
echo ========================================
echo.
echo Please restart your computer for changes to take effect.
echo.
echo After restart, test with: python --version
echo.
pause
