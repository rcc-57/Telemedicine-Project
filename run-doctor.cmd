@echo off
call "%~dp0run.cmd" --bind 127.0.0.1 --host 127.0.0.1 %*
exit /b %errorlevel%
