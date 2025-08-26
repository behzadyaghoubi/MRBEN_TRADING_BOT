@echo off
chcp 65001 >nul
echo ========================================
echo    MR BEN Keyboard Fix - Quick Solution
echo ========================================
echo.

echo Testing Python command...
python --version
echo.

echo Testing synthetic dataset analysis...
python test_synthetic.py
echo.

echo Testing signal distribution...
python check_synthetic_distribution.py
echo.

echo ========================================
echo    All tests completed!
echo ========================================
pause
