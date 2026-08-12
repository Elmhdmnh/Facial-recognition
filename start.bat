@echo off
title 人脸识别系统
cd /d "%~dp0"

echo ============================================
echo   人脸识别系统
echo ============================================
echo.

rem ---- 第1步:查找 Python ----
set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [错误] 未找到 Python。
        echo 请先到 https://www.python.org/downloads/ 安装,
        echo 安装时记得勾选 "Add python.exe to PATH"。
        pause
        exit /b 1
    )
    set "PY=py"
)
echo [OK] 找到 Python

rem ---- 第2步:检查依赖,缺哪个自动装哪个 ----
%PY% -c "import insightface" 2>nul || (echo [安装] insightface... & %PY% -m pip install insightface -q)
%PY% -c "import onnxruntime" 2>nul || (echo [安装] onnxruntime... & %PY% -m pip install onnxruntime -q)
%PY% -c "import cv2"         2>nul || (echo [安装] opencv-python... & %PY% -m pip install opencv-python -q)
%PY% -c "import numpy"       2>nul || (echo [安装] numpy... & %PY% -m pip install numpy -q)
%PY% -c "import PIL"         2>nul || (echo [安装] pillow... & %PY% -m pip install pillow -q)
echo [OK] 依赖检查完成

rem ---- 第3步:启动程序 ----
echo.
echo 正在启动程序(首次运行会自动下载人脸模型,约200MB)...
echo 提示:程序运行期间请保持本窗口开启,关闭程序后本窗口会显示结果。
echo.
set "PYTHONIOENCODING=utf-8"
%PY% face_gui.py

echo.
echo 程序已退出。
echo 如果刚才没有正常打开界面,请把 error.log 文件内容发给开发者。
pause