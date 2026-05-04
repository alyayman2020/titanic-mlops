# ══════════════════════════════════════════════════════════════════
#  setup_windows.ps1 — Titanic MLOps Lab 3 Setup (PowerShell)
#
#  Run from project root:
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#      .\setup_windows.ps1
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Titanic MLOps — Windows Environment Setup         ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Detect Python ─────────────────────────────────────────
Write-Host "[1/6] Detecting Python..." -ForegroundColor Yellow

$PYTHON = $null

foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PYTHON = $cmd
            Write-Host "      Found: $cmd -> $ver" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $PYTHON) {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix option 1 (recommended — permanent):" -ForegroundColor Yellow
    Write-Host "  Settings > Apps > Advanced App Settings > App Execution Aliases"
    Write-Host "  Turn ON: python.exe and python3.exe"
    Write-Host ""
    Write-Host "Fix option 2: Install Python from https://python.org"
    Write-Host "  Make sure to check 'Add python.exe to PATH' during install"
    exit 1
}

# ── Step 2: App Execution Aliases reminder ────────────────────────
Write-Host ""
Write-Host "[2/6] Windows App Execution Aliases info..." -ForegroundColor Yellow
Write-Host "      Using: '$PYTHON' command" -ForegroundColor Green
Write-Host "      To permanently enable 'python' command:" -ForegroundColor Gray
Write-Host "        Settings > Apps > Advanced App Settings > App Execution Aliases" -ForegroundColor Gray
Write-Host "        Turn ON python.exe and python3.exe" -ForegroundColor Gray

# ── Step 3: Install uv ────────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Checking uv..." -ForegroundColor Yellow

try {
    $uvVer = & uv --version 2>&1
    Write-Host "      uv found: $uvVer" -ForegroundColor Green
} catch {
    Write-Host "      Installing uv..." -ForegroundColor Gray
    & $PYTHON -m pip install uv --quiet
    Write-Host "      uv installed ✓" -ForegroundColor Green
}

# ── Step 4: Virtual environment ───────────────────────────────────
Write-Host ""
Write-Host "[4/6] Setting up virtual environment..." -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host "      .venv already exists" -ForegroundColor Green
} else {
    & uv venv --python 3.11
    Write-Host "      .venv created ✓" -ForegroundColor Green
}

# Activate
& .\.venv\Scripts\Activate.ps1

# ── Step 5: Core dependencies ─────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Installing dependencies..." -ForegroundColor Yellow
& uv pip install -e ".[dev]" --quiet
Write-Host "      Core dependencies installed ✓" -ForegroundColor Green

# ── Step 6: PyTorch for RTX 4050 (CUDA 12.1) ─────────────────────
Write-Host ""
Write-Host "[6/6] Installing PyTorch (CUDA 12.1 for RTX 4050)..." -ForegroundColor Yellow
Write-Host "      Downloading ~2GB — please wait..." -ForegroundColor Gray

try {
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      PyTorch CUDA 12.1 installed ✓" -ForegroundColor Green
    } else { throw }
} catch {
    Write-Host "      CUDA wheel failed — using CPU-only PyTorch..." -ForegroundColor DarkYellow
    & pip install torch torchvision --quiet
    Write-Host "      PyTorch (CPU) installed ✓" -ForegroundColor Green
}

& pip install pytorch-tabnet --quiet
Write-Host "      pytorch-tabnet installed ✓" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup complete! ✓" -ForegroundColor Green
Write-Host ""
Write-Host "  VS Code — Select interpreter:" -ForegroundColor Yellow
Write-Host "    Ctrl+Shift+P > Python: Select Interpreter"
Write-Host "    Choose: .\.venv\Scripts\python.exe"
Write-Host ""
Write-Host "  Run the pipeline:" -ForegroundColor Yellow
Write-Host "    $PYTHON trainer.py"
Write-Host "    or: .\.venv\Scripts\python.exe trainer.py"
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan
