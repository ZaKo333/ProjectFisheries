@echo off
chcp 65001 >nul
setlocal
title ProjectFisheries - Self Check
cd /d "%~dp0"

rem ===========================================================================
rem  Self check: all unit tests + all demo scripts + knowledge base index.
rem  Double-click before a demo / defence, to confirm the environment is sane.
rem
rem  KEEP THIS FILE PURE ASCII -- see the note in src/fisheries/launcher.py.
rem  All Chinese output comes from scripts/selftest.py.
rem ===========================================================================

set "PY=%CD%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" goto :nopython

"%PY%" "scripts\selftest.py"
set "RC=%ERRORLEVEL%"

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