@echo off
chcp 65001 >nul
cd /d "%~dp0"
node "%~dp0scripts\stop-kun.mjs"
timeout /t 2 /nobreak >nul
