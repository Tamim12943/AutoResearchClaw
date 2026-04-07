@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Start script from repository root
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.." >nul
set "REPO_ROOT=%CD%"

echo [1/7] Preparing Python environment...
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    goto :fail
  )
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate virtual environment.
  goto :fail
)
python -m pip install --upgrade pip >nul
python -m pip install -e .
if errorlevel 1 (
  echo Failed to install AutoResearchClaw.
  goto :fail
)

echo [2/7] Preparing config...
if not exist "config.arc.yaml" (
  copy /Y "config.gemma-local.example.yaml" "config.arc.yaml" >nul
  echo Created config.arc.yaml from config.gemma-local.example.yaml
) else (
  echo Using existing config.arc.yaml
)

echo [3/7] Setting AMD compatibility env...
set "HSA_OVERRIDE_GFX_VERSION=10.3.0"

echo [4/7] Ensuring Ollama is available...
where ollama >nul 2>&1
if errorlevel 1 (
  echo Ollama was not found on PATH. Install from https://ollama.com/download
  goto :fail
)

curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
  echo Starting Ollama server...
  start "" /B ollama serve >nul 2>&1
  timeout /t 4 /nobreak >nul
)

echo [5/7] Pulling local models (Gemma)...
ollama pull gemma3:12b
if errorlevel 1 (
  echo Failed to pull gemma3:12b
  goto :fail
)
ollama pull gemma3:4b
if errorlevel 1 (
  echo Failed to pull gemma3:4b
  goto :fail
)

echo [6/7] Ensuring Docker images (optional but recommended)...
where docker >nul 2>&1
if errorlevel 1 (
  echo Docker not found. Skipping Docker image setup.
) else (
  docker image inspect researchclaw/experiment:latest >nul 2>&1
  if errorlevel 1 (
    echo Building researchclaw/experiment:latest ...
    docker build -t researchclaw/experiment:latest researchclaw/docker
  )

  docker image inspect researchclaw/experiment:rocm >nul 2>&1
  if errorlevel 1 (
    echo Building researchclaw/experiment:rocm ...
    docker build -f researchclaw/docker/Dockerfile.rocm -t researchclaw/experiment:rocm researchclaw/docker
  )
)

set "TOPIC=%~1"
if "%TOPIC%"=="" (
  set /p TOPIC=Enter research topic: 
)
if "%TOPIC%"=="" (
  echo No topic provided.
  goto :fail
)

echo [7/7] Running pipeline...
researchclaw run --config config.arc.yaml --topic "%TOPIC%" --auto-approve
if errorlevel 1 goto :fail

echo.
echo Pipeline completed.
popd >nul
exit /b 0

:fail
echo.
echo Startup failed. Fix errors above and rerun:
echo   %~nx0 "your topic"
popd >nul
exit /b 1
