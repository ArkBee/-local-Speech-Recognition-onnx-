@echo off
chcp 65001 >nul

IF NOT EXIST .venv (
    echo Error: Folder .venv not found.
    echo Please run 'python -m venv venv' to create the virtual environment.
    pause
    exit /b 1
) ELSE (
    echo Folder .\venv found.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Environment activated successfully
echo Installing PyTorch with CUDA 11.8 (2.5.1 stack)...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

echo Installing other requirements...
pip install -r requirements.txt

echo Starting start.py...
python start.py

if %errorlevel% neq 0 (
    echo Error running start.py
    pause
    exit /b %errorlevel%
)

echo Program completed
pause
