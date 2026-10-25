@echo off
call "%~dp0run.cmd" --client-only --host 127.0.0.1 %*
exit /b %errorlevel%
