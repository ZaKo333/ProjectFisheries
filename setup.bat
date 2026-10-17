@echo off
chcp 65001 >nul
setlocal
title ProjectFisheries - Environment Setup
cd /d "%~dp0"

rem ===========================================================================
rem  KEEP THIS FILE PURE ASCII AND CRLF.
rem
rem  cmd.exe decodes a .bat in chunks using the console code page.  A non-ASCII
rem  byte can overrun into the next line and swallow the following ASCII bytes
rem  (quotes included), which makes the batch file die with nonsense errors
rem  such as "'python' is not recognized".  Every Chinese message therefore
rem  lives in scripts/setup_env.py, which prints through the UTF-8 console.
rem
rem  Usage:  double-click this file.
rem          setup.bat --mirror tsinghua     use a PyPI mirror
rem          setup.bat --skip-kb             skip the vector store
rem          setup.bat --recreate            rebuild .venv from scratch
rem ===========================================================================

set "PY="
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"
rem Unbuffered: with the output piped (Tee-Object, redirect to a log) Python
rem block-buffers stdout, so every progress line lands after the child output
rem and the whole run reads out of order.
set "PYTHONUNBUFFERED=1"

rem The venv does not exist yet (that is what we are creating), so look for any
rem usable base interpreter: the py launcher first, then plain python.
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto use_py

python -c "import sys" >nul 2>&1
if not errorlevel 1 goto use_python

python3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto use_python3

goto no_python

:use_py
set "PY=py -3"
goto run

:use_python
set "PY=python"
goto run

:use_python3
set "PY=python3"
goto run

:run
%PY% "scripts\setup_env.py" %*
set "RC=%ERRORLEVEL%"
goto done

:no_python
echo.
echo   [ERROR] No Python 3 found on PATH.
echo.
echo   1. Download Python 3.10 or newer:
echo        https://www.python.org/downloads/
echo   2. During installation, tick "Add python.exe to PATH".
echo   3. Double-click this file again.
echo.
set "RC=1"

:done
if not "%PF_NO_PAUSE%"=="1" pause
endlocal & exit /b %RC%
