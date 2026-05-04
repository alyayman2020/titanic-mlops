@echo off
REM ══════════════════════════════════════════════════════════════════
REM  setup_windows.bat — Titanic MLOps Lab 3 Setup for Windows
REM
REM  Fixes the "python not found / use py instead" issue from
REM  App Execution Aliases in Windows Settings.
REM
REM  Run this ONCE from your project root in PowerShell or CMD:
REM       .\setup_windows.bat
REM ══════════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║   Titanic MLOps — Windows Environment Setup         ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM ── Step 1: Detect Python ────────────────────────────────────────
echo [1/6] Detecting Python installation...

python --version >nul 2>&1
if %errorlevel% == 0 (
    echo       python.exe found ✓
    set PYTHON=python
    goto :python_found
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo       python.exe alias is OFF in App Execution Aliases.
    echo       Using py.exe launcher instead ✓
    set PYTHON=py
    goto :python_found
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo       python3.exe found ✓
    set PYTHON=python3
    goto :python_found
)

echo [ERROR] Python not found at all!
echo         Please install Python 3.11 from https://python.org
echo         OR enable python.exe in:
echo         Settings ^> Apps ^> Advanced App Settings ^> App Execution Aliases
pause
exit /b 1

:python_found
%PYTHON% --version
echo.

REM ── Step 2: Fix App Execution Aliases (instructions) ─────────────
echo [2/6] NOTE about App Execution Aliases...
echo       Your python.exe / python3.exe aliases may be OFF in Windows.
echo       To fix permanently (optional):
echo         Settings ^> Apps ^> Advanced App Settings ^> App Execution Aliases
echo         Turn ON: python.exe and python3.exe
echo         (from PythonSoftwareFoundation.PythonManager)
echo       For now, this script uses: %PYTHON%
echo.

REM ── Step 3: Check/Install uv ─────────────────────────────────────
echo [3/6] Checking uv package manager...
uv --version >nul 2>&1
if %errorlevel% == 0 (
    echo       uv found ✓
) else (
    echo       uv not found — installing via pip...
    %PYTHON% -m pip install uv --quiet
    echo       uv installed ✓
)
echo.

REM ── Step 4: Create virtual environment ───────────────────────────
echo [4/6] Creating virtual environment (.venv)...
if exist .venv (
    echo       .venv already exists, skipping creation.
) else (
    uv venv --python 3.11
    echo       .venv created ✓
)
echo.

REM ── Step 5: Install dependencies ─────────────────────────────────
echo [5/6] Installing project dependencies...
call .venv\Scripts\activate.bat

uv pip install -e ".[dev]" --quiet
echo       Core dependencies installed ✓

REM ── Step 6: Install PyTorch (CUDA 12.1 for RTX 4050) ─────────────
echo [6/6] Installing PyTorch with CUDA 12.1 support (RTX 4050)...
echo       This may take a few minutes (large download ~2GB)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if %errorlevel% == 0 (
    echo       PyTorch CUDA 12.1 installed ✓
) else (
    echo       CUDA wheel failed — installing CPU-only PyTorch...
    pip install torch torchvision --quiet
    echo       PyTorch (CPU) installed ✓
)

pip install pytorch-tabnet --quiet
echo       pytorch-tabnet installed ✓

echo.
echo ══════════════════════════════════════════════════════
echo   Setup complete! ✓
echo.
echo   To activate your environment in VS Code:
echo     1. Press Ctrl+Shift+P
echo     2. Type: Python: Select Interpreter
echo     3. Choose: .venv\Scripts\python.exe
echo.
echo   To run the pipeline:
echo     %PYTHON% trainer.py
echo     or: .venv\Scripts\python.exe trainer.py
echo ══════════════════════════════════════════════════════
echo.
pause
