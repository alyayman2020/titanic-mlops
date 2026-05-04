# setup_dagshub_dvc.ps1
# Run once to connect DVC to DagsHub and push data
# Usage: .\setup_dagshub_dvc.ps1

Write-Host ""
Write-Host "DagsHub + DVC Setup" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Install dependencies ──────────────────────────────────
Write-Host "[1/6] Installing dagshub and python-dotenv..." -ForegroundColor Yellow
python -m pip install dagshub python-dotenv dvc dvc-s3 --quiet
Write-Host "      Done ✓" -ForegroundColor Green

# ── Step 2: Verify .env exists ────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Checking .env file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "      ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "      Create it with your DAGSHUB_TOKEN and DAGSHUB_USERNAME"
    exit 1
}
Write-Host "      .env found ✓" -ForegroundColor Green

# ── Step 3: DVC remote config (already in .dvc/config) ────────────
Write-Host ""
Write-Host "[3/6] DVC remote config..." -ForegroundColor Yellow
Write-Host "      Remote already configured in .dvc/config ✓" -ForegroundColor Green
dvc remote list

# ── Step 4: Track data files with DVC ─────────────────────────────
Write-Host ""
Write-Host "[4/6] Adding data files to DVC tracking..." -ForegroundColor Yellow
dvc add data/raw/train.csv
dvc add data/raw/test.csv
Write-Host "      Data files tracked ✓" -ForegroundColor Green

# ── Step 5: Push data to DagsHub storage ──────────────────────────
Write-Host ""
Write-Host "[5/6] Pushing data to DagsHub remote storage..." -ForegroundColor Yellow
Write-Host "      This uploads train.csv and test.csv to DagsHub S3..."
dvc push
Write-Host "      Data pushed ✓" -ForegroundColor Green

# ── Step 6: Git commit the .dvc files ─────────────────────────────
Write-Host ""
Write-Host "[6/6] Git: staging DVC pointer files..." -ForegroundColor Yellow
git add data/raw/train.csv.dvc
git add data/raw/test.csv.dvc
git add .dvc/config
git add .dvc/.gitignore
git add .gitignore
git add .env.example  2>$null
Write-Host "      Staged ✓ (commit manually after review)" -ForegroundColor Green

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "DVC + DagsHub setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. git commit -m 'feat: add DVC tracking + DagsHub remote'"
Write-Host "  2. git push origin main"
Write-Host "  3. python trainer.py  (now logs to DagsHub MLflow)"
Write-Host ""
Write-Host "View your experiments at:" -ForegroundColor Yellow
Write-Host "  https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments"
Write-Host "=======================================" -ForegroundColor Cyan
