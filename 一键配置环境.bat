@echo off
rem Chinese-named entry point.  The real work lives in setup.bat, which must
rem stay pure ASCII; this file only forwards its arguments.
call "%~dp0scripts\bat\setup.bat" %*
exit /b %ERRORLEVEL%
