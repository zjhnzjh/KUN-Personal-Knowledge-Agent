@echo off
chcp 65001 >nul
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required. Install Node.js 22 or later.
  pause
  exit /b 1
)
node "%~dp0scripts\launch-kun.mjs"
if errorlevel 1 (
  echo.
  echo KUN startup failed. Check .run\kun.err.log.
  pause
)
