@echo off
chcp 65001 >nul
setlocal
title ProjectFisheries - API Service
cd /d "%~dp0"

rem ===========================================================================
rem  API service launcher (double-click).
rem  Serves BOTH the HTTP API and the web front-end (mounted at /app).
rem  Optional argument: port, default 8000.
rem  To let other machines connect, change HOST below to 0.0.0.0.
rem
rem  KEEP THIS FILE PURE ASCII -- see the note in src/fisheries/launcher.py.
rem ===========================================================================

set "HOST=127.0.0.1"
rem To serve other machines change the line above to:  set "HOST=0.0.0.0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "PY=%CD%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"
set "FISHERIES_API_HOST=%HOST%"
set "FISHERIES_API_PORT=%PORT%"

if not exist "%PY%" goto :nopython

"%PY%" -m fisheries.launcher api-banner --host "%HOST%" --port "%PORT%"

"%PY%" -m fisheries.api
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto :failed
"%PY%" -m fisheries.launcher api-stopped
goto :done

:failed
"%PY%" -m fisheries.launcher api-failed --port "%PORT%" --rc "%RC%"

:done
if not "%PF_NO_PAUSE%"=="1" pause
endlocal
exit /b %RC%

:nopython
echo.
echo   [ERROR] Python virtual environment not found:
echo           %PY%
echo.
echo   Create it once (see README.md section 1):
echo       python -m venv .venv
echo       .venv\Scripts\python.exe -m pip install -e ".[dev,rag,api]"
echo   Then double-click this file again.
echo.
if not "%PF_NO_PAUSE%"=="1" pause
endlocal
exit /b 1