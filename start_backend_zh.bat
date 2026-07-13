@echo off
setlocal EnableDelayedExpansion

REM Start the Chinese-data FastAPI backend for local/LAN testing.
REM Run from Anaconda Prompt, CMD, or double-click from the project root.

cd /d "%~dp0"

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    call conda activate medrag
) else (
    echo [WARN] conda was not found in PATH. Continuing with the current Python environment.
)

set "PYTHON_EXE="
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 set "PYTHON_EXE=python"
if "%PYTHON_EXE%"=="" if exist "D:\anaconda\envs\medrag\python.exe" set "PYTHON_EXE=D:\anaconda\envs\medrag\python.exe"
if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found. Please run this script from Anaconda Prompt after activating medrag.
    exit /b 1
)

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /I "%%A"=="LLM_API_KEY" if not defined LLM_API_KEY set "LLM_API_KEY=%%B"
        if /I "%%A"=="LLM_BASE_URL" if not defined LLM_BASE_URL set "LLM_BASE_URL=%%B"
        if /I "%%A"=="LLM_CHAT_MODEL" if not defined LLM_CHAT_MODEL set "LLM_CHAT_MODEL=%%B"
        if /I "%%A"=="FORCE_MOCK_LLM" if not defined FORCE_MOCK_LLM set "FORCE_MOCK_LLM=%%B"
    )
)

set USE_ZH_DATA=true
set USE_EXTERNAL_VECTOR=true
set EXTERNAL_VECTOR_BASE_DIR=.\vector_db_zh
set EXTERNAL_VECTOR_SOURCES=medcase_reasoning,open_patients
if "%FORCE_MOCK_LLM%"=="" set FORCE_MOCK_LLM=false
set PYTHONDONTWRITEBYTECODE=1

if "%BACKEND_HOST%"=="" set BACKEND_HOST=0.0.0.0
if "%BACKEND_PORT%"=="" set BACKEND_PORT=8000

echo [INFO] Project: %CD%
echo [INFO] PYTHON_EXE=%PYTHON_EXE%
echo [INFO] USE_ZH_DATA=%USE_ZH_DATA%
echo [INFO] USE_EXTERNAL_VECTOR=%USE_EXTERNAL_VECTOR%
echo [INFO] EXTERNAL_VECTOR_BASE_DIR=%EXTERNAL_VECTOR_BASE_DIR%
echo [INFO] FORCE_MOCK_LLM=%FORCE_MOCK_LLM%
echo [INFO] LLM_BASE_URL=%LLM_BASE_URL%
echo [INFO] LLM_CHAT_MODEL=%LLM_CHAT_MODEL%
if defined LLM_API_KEY (
    echo [INFO] LLM_API_KEY is set.
) else (
    echo [WARN] LLM_API_KEY is empty. LLM calls will fall back to mock output.
)
echo [INFO] PYTHONDONTWRITEBYTECODE=%PYTHONDONTWRITEBYTECODE%
echo [INFO] Starting backend at http://%BACKEND_HOST%:%BACKEND_PORT%

"%PYTHON_EXE%" -m uvicorn backend.app.main:app --host %BACKEND_HOST% --port %BACKEND_PORT% --reload

endlocal

