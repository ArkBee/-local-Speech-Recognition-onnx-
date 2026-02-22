@echo off
cd /d "%~dp0"
echo === Voice Recognition Setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo Found: %%i
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Installing dependencies...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. Run: run.bat
echo   2. Models will download automatically on first launch
echo   3. For Groq features (punctuation, translation, cloud STT):
echo      - Get a free API key at https://console.groq.com
echo      - Add it in the app: Groq tab ^> Add key
echo.
pause
