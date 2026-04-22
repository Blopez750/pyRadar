@echo off
REM setup.bat — Windows bootstrap for Stingray X-Band radar project
REM Usage: double-click or run from Command Prompt

echo === Stingray Setup (Windows) ===
echo.

python "%~dp0setup.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Setup failed. Make sure Python 3.10+ is installed and on your PATH.
    pause
    exit /b 1
)

echo.
pause
