@echo off
cd /d %~dp0
start "SAQR API" cmd /k run.bat
timeout /t 25 /nobreak >nul
start "SAQR Tunnel" cmd /k tunnel.bat
echo SAQR is starting. Link: https://zeppelin-diaphragm-pantomime.ngrok-free.dev
