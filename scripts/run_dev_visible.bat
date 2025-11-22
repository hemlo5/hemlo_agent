@echo off
REM Dev-mode launcher with visible Chromium windows for Playwright flows.

REM Change to repo root (this script is in scripts\)
cd /d "%~dp0.."

REM Backend app directory
cd apps\backend

REM Create virtualenv if it does not exist
if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

pip install -r requirements.txt

REM Install Chromium browser for Playwright
python -m playwright install chromium

REM Make proofs directory if it does not already exist
if not exist "proofs" (
    mkdir proofs
)

REM Configure Redis & Playwright for dev-visible mode
set REDIS_URL=redis://localhost:6379/0
set PLAYWRIGHT_HEADLESS=false
set PLAYWRIGHT_VIEWPORT_WIDTH=1920
set PLAYWRIGHT_VIEWPORT_HEIGHT=1080
set PLAYWRIGHT_SLOW_MO_MS=800

REM Start FastAPI backend with auto-reload
start "backend" cmd /k "cd /d %CD% && call .venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

REM Start Celery worker (solo pool is safest on Windows)
start "worker" cmd /k "cd /d %CD% && call .venv\Scripts\activate.bat && celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo"

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║  DEV MODE WITH VISIBLE BROWSER ACTIVE                ║
echo ║  → Backend: http://localhost:8000                    ║
echo ║  → Proofs folder: apps\backend\proofs                ║
echo ║  → Chrome will pop up and you can watch every click  ║
echo ╚══════════════════════════════════════════════════════╝
echo.
pause
