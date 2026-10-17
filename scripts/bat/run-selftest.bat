@echo off
chcp 65001 >nul
setlocal
title ProjectFisheries - Self Check
cd /d "%~dp0..\.."

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

if exist "%PY%" goto :launch

rem First run: there is no virtual environment yet.  Build it once with the
rem same script the one-click entry point uses, then continue launching.
echo.
echo   [INFO] .venv not found -- running the one-click setup first (once only).
echo.
set "PF_NO_PAUSE=1"
call "%~dp0setup.bat"
set "PF_NO_PAUSE="
if not exist "%PY%" goto :nopython

:launch

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
echo   The automatic setup did not finish.  Re-run it by double-clicking:
echo       scripts\bat\setup.bat
echo   or see README.md section 3.2 for the manual steps.
echo.
if not "%PF_NO_PAUSE%"=="1" pause
endlocal
exit /b 1