.PHONY: help install train stage1 stage2 predict format lint test clean mlflow-ui

# ── Windows "py vs python" fix ────────────────────────────────────
# Detects which Python launcher is available. On Windows with App Execution
# Aliases disabled, 'python' fails — we fall back to 'py' automatically.
PYTHON := $(shell python --version > /dev/null 2>&1 && echo python || (py --version > /dev/null 2>&1 && echo py || echo python3))
UV     := uv

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────
install: ## Install all dependencies via uv
	$(UV) sync

install-dev: ## Install with dev extras
	$(UV) sync --extra dev

install-torch-gpu: ## Install PyTorch with CUDA 12.1 (RTX 4050)
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install pytorch-tabnet

# ── Pipeline ──────────────────────────────────────────────────────
train: ## Run full pipeline (Stage 1 + Stage 2 + Ensemble)
	$(PYTHON) trainer.py

stage1: ## Run Stage 1 only (base models)
	$(PYTHON) trainer.py pipeline.run_stage2=false pipeline.run_ensemble=false

stage2: ## Run Stage 2 only (Optuna tuning)
	$(PYTHON) trainer.py pipeline.run_stage1=false pipeline.run_ensemble=false

ensemble: ## Run Stage 2 + Ensemble only
	$(PYTHON) trainer.py pipeline.run_stage1=false

quick: ## Quick test run (5 trials, 2 models only)
	$(PYTHON) trainer.py optuna.n_trials=5 \
		models.stage1=[logistic_regression,random_forest] \
		models.stage2=[logistic_regression,random_forest] \
		pipeline.run_ensemble=false

predict: ## Run inference on test set using production model
	$(PYTHON) predict.py --input data/raw/test.csv

predict-local: ## Run inference from a local pkl file
	$(PYTHON) predict.py --input data/raw/test.csv --source local \
		--local models/titanic/xgboost_tuned.pkl

# ── MLflow ────────────────────────────────────────────────────────
mlflow-ui: ## Launch MLflow UI at localhost:5000
	mlflow ui --backend-store-uri mlruns --port 5000

# ── DVC ───────────────────────────────────────────────────────────
dvc-repro: ## Reproduce DVC pipeline
	dvc repro

dvc-push: ## Push data/models to remote DVC storage
	dvc push

dvc-pull: ## Pull data/models from remote DVC storage
	dvc pull

# ── Code quality ──────────────────────────────────────────────────
format: ## Auto-format with ruff + black
	ruff check --fix src/ tests/ trainer.py predict.py
	black src/ tests/ trainer.py predict.py

lint: ## Lint only (no writes — CI safe)
	ruff check src/ tests/ trainer.py predict.py
	black --check src/ tests/ trainer.py predict.py

# ── Tests ─────────────────────────────────────────────────────────
test: ## Run pytest with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Cleanup ───────────────────────────────────────────────────────
clean: ## Remove build artifacts, __pycache__, .hydra outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hydra      -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"             -delete
	find . -name ".coverage"         -delete
	rm -rf .pytest_cache dist build *.egg-info
	@echo "Clean ✓"
