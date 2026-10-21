@echo off
chcp 65001 >nul
cd /d %~dp0
call venv\Scripts\activate
set SAQR_MODEL_DIR=%~dp0saqr_v22_final
echo Open http://localhost:8000 in the browser to test.  Keep this window open.
uvicorn app:app --host 0.0.0.0 --port 8000
pause
