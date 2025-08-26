@echo off
chcp 65001
echo Testing MR BEN AI System...
echo Current directory: %CD%
echo Python version:
python --version
echo.
echo Testing system components:
python test_system_update.py
echo.
echo Test completed!
pause 