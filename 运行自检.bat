@echo off
rem Chinese-named convenience entry point.  Content must stay pure ASCII.
call "%~dp0scripts\bat\run-selftest.bat" %*
set "RC=%ERRORLEVEL%"
exit /b %RC%