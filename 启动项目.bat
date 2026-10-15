@echo off
rem Top-level entry point: start the API service + the web front-end (/app).
rem Content must stay pure ASCII.
cd /d "%~dp0"
if not exist "run-api.bat" goto :missing
call "run-api.bat" %*
set "RC=%ERRORLEVEL%"
exit /b %RC%

:missing
echo.
echo   [ERROR] run-api.bat not found in:
echo           %CD%
echo.
if not "%PF_NO_PAUSE%"=="1" pause
exit /b 1
