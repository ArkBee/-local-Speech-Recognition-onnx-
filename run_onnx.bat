@echo off
chcp 65001 >nul

IF NOT EXIST .\venv (
    echo Error: Folder .\venv not found.
    echo Please run 'python -m venv venv' to create the virtual environment.
    pause
    exit /b 1
) ELSE (
    echo Folder .\venv found.
)

echo Activating virtual environment...
call .\venv\Scripts\activate

echo Environment activated successfully
if exist requirements.txt (
    echo Ensuring base requirements are installed...
    pip install -r requirements.txt
)

echo Installing ONNX runtime dependencies...
pip install -r requirements_onnx.txt

echo Starting start_onnx.py...
python start_onnx.py

if %errorlevel% neq 0 (
    echo Error running start_onnx.py
    pause
    exit /b %errorlevel%
)

echo Program completed
pause
