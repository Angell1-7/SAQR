@echo off
chcp 65001 >nul
cd /d %~dp0
echo === 1) Python environment ===
python -m venv venv || (echo Python not found - install Python 3.11 from python.org & pause & exit /b)
call venv\Scripts\activate
python -m pip install --upgrade pip
echo === 2) PyTorch (NVIDIA GPU version) ===
pip install torch --index-url https://download.pytorch.org/whl/cu124
echo === 3) Other libraries ===
pip install -r requirements.txt
echo === 4) Checks ===
python -c "import torch;print('GPU available:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')"
where ffmpeg >nul 2>nul && echo ffmpeg: OK || echo ffmpeg: NOT FOUND - run: winget install Gyan.FFmpeg  then reopen this window
if exist saqr_v22_final\model.safetensors (echo model: OK) else (echo model: NOT FOUND - put the saqr_v22_final folder next to this file)
pause
