@echo off
cd /d "%~dp0"
echo === Voice Recognition Setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found!
    echo.
    echo [1] Install Python automatically (winget)
    echo [2] I'll install it manually
    echo.
    choice /c 12 /m "Choose"
    if errorlevel 2 (
        echo.
        echo Download Python 3.11 from https://www.python.org/downloads/
        echo IMPORTANT: check "Add Python to PATH" during installation!
        echo.
        echo After installing Python, run setup.bat again.
        pause
        exit /b 1
    )
    echo.
    echo Installing Python 3.11 via winget...
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo Automatic installation failed.
        echo Download manually: https://www.python.org/downloads/
        echo IMPORTANT: check "Add Python to PATH"!
        pause
        exit /b 1
    )
    echo.
    echo Python installed! Restarting setup to pick up new PATH...
    echo.
    :: Refresh PATH and restart
    start "" cmd /c "%~f0"
    exit /b 0
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
echo   2. Models will download automatically on first launch (~500 MB)
echo   3. For Groq features (punctuation, translation, cloud STT):
echo      - Get a free API key at https://console.groq.com
echo      - Add it in the app: Groq tab ^> Add key
echo.
pause
