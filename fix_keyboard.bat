@echo off
chcp 65001 >nul
echo Starting keyboard fix...
python keyboard_fix.py
if %errorlevel% equ 0 (
    echo Keyboard fix completed successfully!
) else (
    echo Keyboard fix failed. Manual intervention may be required.
)
pause
