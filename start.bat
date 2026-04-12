@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=backend"

if /I "%MODE%"=="backend" goto :backend
if /I "%MODE%"=="frontend" goto :frontend
if /I "%MODE%"=="all" goto :all

echo Usage: %~nx0 [backend ^| frontend ^| all]
echo   backend   - FastAPI on http://0.0.0.0:8000 (default)
echo   frontend  - Next.js dev on http://localhost:3000
echo   all       - opens two windows: backend + frontend
exit /b 1

:backend
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found in PATH. Install Python 3.11+ and retry.
  exit /b 1
)
echo Starting AceClaw backend (repo root: %cd%)...
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
exit /b %ERRORLEVEL%

:frontend
where npm >nul 2>&1
if errorlevel 1 (
  echo ERROR: npm not found in PATH. Install Node.js and retry.
  exit /b 1
)
cd frontend
echo Starting AceClaw frontend...
call npm run dev
exit /b %ERRORLEVEL%

:all
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found in PATH.
  exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
  echo ERROR: npm not found in PATH.
  exit /b 1
)
echo Launching backend and frontend in separate windows...
start "AceClaw Backend" cmd /k pushd "%~dp0" ^&^& python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
start "AceClaw Frontend" cmd /k pushd "%~dp0" ^&^& cd frontend ^&^& npm run dev
echo Done. Close each window to stop that process.
exit /b 0
