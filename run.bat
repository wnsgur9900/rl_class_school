@echo off
chcp 65001 > nul
cd /d "%~dp0"
uv run python -X utf8 src/run.py
pause
