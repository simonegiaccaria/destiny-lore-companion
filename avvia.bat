@echo off
rem Double-click: Ollama + server. From a terminal: avvia.bat eval --repeat 3 --temperature 0.2
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\start.py %*
pause