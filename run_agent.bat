@echo off
REM Helper script to run Hemlo Super Agent using the backend virtual environment

if not exist "apps\backend\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at apps\backend\.venv
    echo Please run: docker compose --profile dev up --build
    echo Or set up the venv manually in apps/backend.
    exit /b 1
)

apps\backend\.venv\Scripts\python hemlo_super_agent.py %*
