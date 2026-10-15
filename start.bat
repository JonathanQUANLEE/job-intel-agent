@echo off
REM ============================================================
REM  Agent 岗位情报站 · 一键启动
REM  双击本文件即可：自动检查环境 -> 装依赖 -> 起服务 -> 打开浏览器
REM ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo   ============================================
echo    Agent 岗位情报站 · 启动中
echo   ============================================
echo.

REM ---- 1. 找 Python ----
where python >nul 2>nul
if errorlevel 1 (
  echo   [x] 没找到 Python。请先安装 Python 3.10+ 并勾选 "Add to PATH"。
  echo       https://www.python.org/downloads/
  pause
  exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do echo   [1/4] %%v 已就绪

REM ---- 2. 建虚拟环境（第一次运行才建） ----
if not exist ".venv\Scripts\python.exe" (
  echo   [2/4] 首次运行，创建虚拟环境...
  python -m venv .venv
  if errorlevel 1 ( echo   [x] 创建虚拟环境失败 & pause & exit /b 1 )
) else (
  echo   [2/4] 虚拟环境已存在
)

REM ---- 3. 装依赖 ----
echo   [3/4] 检查依赖...
.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 ( echo   [x] 依赖安装失败，请检查网络 & pause & exit /b 1 )

REM ---- 4. 起服务 ----
echo   [4/4] 启动服务...
echo.
echo   浏览器将自动打开 http://127.0.0.1:8000
echo   关闭本窗口即停止服务。
echo.

start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m api

pause
