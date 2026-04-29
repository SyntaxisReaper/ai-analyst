@echo off
echo.
echo  Excel AI Agent — Starting backend...
echo.
cd /d "%~dp0backend"
if not exist ".env" (
    echo  [!] .env not found. Copying from .env.example...
    copy .env.example .env
    echo  [!] Edit backend\.env and add your API key, then re-run this script.
    pause
    exit /b 1
)
python app.py
pause
