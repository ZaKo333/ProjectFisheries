@echo off
rem Chinese-named convenience entry point.  Content must stay pure ASCII
rem (cmd.exe mis-parses batch files that contain non-ASCII bytes), so the real
rem launcher lives in run-demo.bat next to this file.
call "%~dp0run-demo.bat" %*
set "RC=%ERRORLEVEL%"
exit /b %RC%