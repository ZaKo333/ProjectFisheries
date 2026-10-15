@echo off
chcp 65001 >nul
setlocal
title ProjectFisheries - Web Demo
cd /d "%~dp0"

rem ===========================================================================
rem  Web demo launcher (double-click).  Optional argument: port, default 7860.
rem
rem  KEEP THIS FILE PURE ASCII.  cmd.exe decodes a batch file with the console
rem  code page in chunks; as soon as the file holds a non-ASCII byte the offsets
rem  drift and cmd executes the tail of a Chinese line as a command, printing
rem      '<fragment>' is not recognized as an internal or external command
rem  to stderr.  Every Chinese message therefore comes from
rem      python -m fisheries.launcher ...
rem  See src/fisheries/launcher.py and tests/unit/test_launchers.py.
rem ===========================================================================

set "PY=%CD%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"
set "FISHERIES_WEB_HOST=127.0.0.1"
set "FISHERIES_WEB_OPEN_BROWSER=1"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=7860"
set "FISHERIES_WEB_PORT=%PORT%"

if not exist "%PY%" goto :nopython

"%PY%" -m fisheries.launcher webui-banner --port "%PORT%"

"%PY%" -m fisheries.webui.app
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto :failed
"%PY%" -m fisheries.launcher webui-stopped
goto :done

:failed
"%PY%" -m fisheries.launcher webui-failed --port "%PORT%" --rc "%RC%"

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
echo       .venv\Scripts\python.exe -m pip install -e ".[dev,rag,webui]"
echo   Then double-click this file again.
echo.
if not "%PF_NO_PAUSE%"=="1" pause
endlocal
exit /b 1