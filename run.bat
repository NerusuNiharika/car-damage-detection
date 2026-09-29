@echo off
REM Installs packages on first run, then starts the app.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python app.py
pause
