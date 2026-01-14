# force_reset.bat - Windows batch script to forcefully delete
@echo off
echo Attempting forceful delete...

cd /d "%~dp0"

rmdir /s /q "data\vector_stores\financial_data" 2>nul
rmdir /s /q "data\vector_stores\financial_data_v2" 2>nul

if exist "data\vector_stores\financial_data" (
    echo [FAILED] financial_data still exists
) else (
    echo [OK] financial_data deleted
)

if exist "data\vector_stores\financial_data_v2" (
    echo [FAILED] financial_data_v2 still exists  
) else (
    echo [OK] financial_data_v2 deleted
)

echo.
echo If files still exist:
echo 1. Restart your computer
echo 2. Run this script again
echo 3. Then run: python seed_aapl.py
pause
