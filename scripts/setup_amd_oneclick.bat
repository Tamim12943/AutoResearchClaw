@echo off
setlocal ENABLEDELAYEDEXPANSION

REM One-click AMD ROCm setup + run launcher for ResearchClaw (Windows + Docker Desktop + WSL2)
REM Usage:
REM   scripts\setup_amd_oneclick.bat "Your research query here" "D:\ResearchClawWorkspace"

set "RESEARCH_QUERY=%~1"
if "%RESEARCH_QUERY%"=="" set "RESEARCH_QUERY=Explore robust LLM reasoning with constrained decoding"

set "TARGET_DIR=%~2"
if "%TARGET_DIR%"=="" set "TARGET_DIR=%CD%\amd_researchclaw_workspace"

set "REPO_ROOT=%~dp0.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

echo [1/8] Preparing workspace: %TARGET_DIR%
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
if not exist "%TARGET_DIR%\docs\kb\questions" mkdir "%TARGET_DIR%\docs\kb\questions"
if not exist "%TARGET_DIR%\docs\kb\literature" mkdir "%TARGET_DIR%\docs\kb\literature"
if not exist "%TARGET_DIR%\docs\kb\experiments" mkdir "%TARGET_DIR%\docs\kb\experiments"
if not exist "%TARGET_DIR%\docs\kb\findings" mkdir "%TARGET_DIR%\docs\kb\findings"
if not exist "%TARGET_DIR%\docs\kb\decisions" mkdir "%TARGET_DIR%\docs\kb\decisions"
if not exist "%TARGET_DIR%\docs\kb\reviews" mkdir "%TARGET_DIR%\docs\kb\reviews"

echo [2/8] Creating venv + installing package
pushd "%REPO_ROOT%"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e .[dev]
if errorlevel 1 (
  echo ERROR: Failed to install dependencies.
  popd
  exit /b 1
)

echo [3/8] Writing config.arc.yaml into workspace
copy /Y "%REPO_ROOT%\config.arc.amd.example.yaml" "%TARGET_DIR%\config.arc.yaml" >nul
if errorlevel 1 (
  echo ERROR: Failed to write config.arc.yaml
  popd
  exit /b 1
)

echo [4/8] Building AMD ROCm Docker image
docker build -f "%REPO_ROOT%\researchclaw\docker\Dockerfile.rocm" -t researchclaw/experiment:rocm "%REPO_ROOT%\researchclaw\docker"
if errorlevel 1 (
  echo ERROR: Failed building researchclaw/experiment:rocm
  popd
  exit /b 1
)

echo [5/8] Pulling Ollama models (if ollama is available)
where ollama >nul 2>nul
if not errorlevel 1 (
  ollama pull qwen2.5:14b
  ollama pull qwen3.5:4b
)

echo [6/8] Reminder: ensure "ollama serve" is running in another terminal.
echo [7/8] Running research pipeline in dedicated workspace
pushd "%TARGET_DIR%"
python -m researchclaw.cli run --config "%TARGET_DIR%\config.arc.yaml" --topic "%RESEARCH_QUERY%" --auto-approve
set "RUN_EXIT=%ERRORLEVEL%"
popd

echo [8/8] Done. Workspace: %TARGET_DIR%
popd
exit /b %RUN_EXIT%
