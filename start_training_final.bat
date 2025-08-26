@echo off
chcp 65001 >nul
echo ========================================
echo Starting LSTM Training - Final Solution
echo ========================================
echo.
echo This script will train the LSTM model with real market data
echo No keyboard issues - completely automated
echo.
echo Press any key to start training...
pause >nul

python final_training_solution.py

echo.
echo Training completed!
echo Press any key to exit...
pause >nul
