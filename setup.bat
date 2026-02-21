@echo off
echo ========================================
echo   PDF Chatbot - Setup Script
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found
python --version

REM Install requirements
echo.
echo [2/4] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Check .env file
echo.
echo [3/4] Checking configuration...
if not exist .env (
    echo Creating .env file from example...
    copy .env.example .env >nul
    echo.
    echo IMPORTANT: Please edit .env file and add your OpenAI API key!
    echo Then run this script again.
    pause
    exit /b 1
)

REM Run tests
echo.
echo [4/4] Running system tests...
python test_system.py
if errorlevel 1 (
    echo.
    echo Setup incomplete. Please fix the errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Initialize database: python initialize_db.py
echo   2. Start server: python run.py
echo   OR
echo   Use start.bat for quick launch
echo.
pause



