# 🚢 Titanic MLOps: Data Versoning with DVC + MLflow Tracking + DagsHub + Optuna Tuning

> **ITI Data Science Track — MLOps Module**
> A production-grade, multi-model classification pipeline with full MLflow experiment tracking,
> DagsHub model registry, Optuna hyperparameter tuning, and ensemble methods.

[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-orange?logo=data:image/png;base64,iVBORw0KGgo=)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?logo=mlflow)](https://dagshub.com/aly.ayman.2018/titanic-mlops.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioned-945DD6?logo=dvc)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![GPU](https://img.shields.io/badge/GPU-RTX%204050-76B900?logo=nvidia)](https://nvidia.com)

---

## 🎯 Lab Objectives

- ✅ Track all models using **MLflow** integrated into the DVC training pipeline
- ✅ Register the best model to **DagsHub Model Registry** and promote to **Production**
- ✅ Load the production model and generate predictions on the test set
- ✅ Train **10 classification models** across 2 stages (base + Optuna tuning)
- ✅ Build **Voting** and **Stacking** ensemble classifiers from the best tuned models
- ✅ Version data with **DVC** backed by DagsHub S3-compatible storage
- ✅ Log system metrics (CPU, RAM, GPU) per run

---

## 📁 Project Structure

```
titanic-mlops/
├── conf/
│   ├── config.yaml              # Hydra main config (pipeline control, MLflow, Optuna)
│   ├── data/titanic.yaml        # Data paths, split config, CatBoost raw columns
│   └── model/models.yaml        # All 10 models: base_params + Optuna search spaces
├── data/
│   ├── raw/                     # train.csv + test.csv (tracked by DVC)
│   ├── interim/                 # Train/test splits
│   └── processed/               # Processed outputs
├── models/titanic/              # Saved .pkl artifacts (tracked by DVC)
│   ├── stage1__*.pkl            # Base model artifacts
│   ├── stage2__*_tuned.pkl      # Optuna-tuned model artifacts
│   └── ensemble__*.pkl          # Ensemble model artifacts
├── reports/titanic/             # Classification reports + predictions.csv
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py        # StandardPreprocessor + CatBoostPreprocessor
│   ├── training/
│   │   ├── model_factory.py     # Builds all 10 models, GPU-aware (RTX 4050)
│   │   ├── trainer.py           # Stage 1 base + Stage 2 Optuna logic
│   │   └── registry.py          # MLflow model registry helper
│   ├── ensemble/
│   │   └── ensemble.py          # VotingClassifier + StackingClassifier
│   ├── evaluation/
│   │   └── metrics.py           # All 6 metrics + classification_report
│   └── utils/
│       ├── logger.py            # Loguru logger
│       └── system_metrics.py    # CPU/RAM/GPU info for MLflow
├── tests/                       # Pytest tests for preprocessors and metrics
├── trainer.py                   # 🚀 Main entry point (Hydra + DagsHub)
├── predict.py                   # Inference from registry or local pkl
├── dvc.yaml                     # DVC pipeline stages
├── pyproject.toml               # All dependencies
└── Makefile                     # Developer shortcuts
```

---

## 🏗️ Architecture

### Two-Stage Training Pipeline

```
Stage 1 (Base)          Stage 2 (Optuna)         Ensemble
──────────────          ────────────────         ────────
Fixed params      →     5 trials × 10 models  →  VotingClassifier
MLflow logged           TPE Sampler               StackingClassifier
Models saved            Best params selected      Logged to DagsHub
```

### Preprocessing: Two Paths

| Path | Used By | What it does |
|---|---|---|
| `StandardPreprocessor` | All models except CatBoost | Full feature engineering + ordinal encoding |
| `CatBoostPreprocessor` | CatBoost only | Light clean, keeps Sex/Embarked as raw strings |

### GPU Acceleration
Models that leverage the **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**:
- XGBoost (`device=cuda`)
- LightGBM (`device=gpu`)
- CatBoost (`task_type=GPU`)
- TabNet (native PyTorch CUDA)

---

## 📊 Experiment Results

### Stage 1 — Base Models (Fixed Parameters)

| Model | Test AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **TabNet** | **0.8609** | 80% | 0.72 | 0.78 | 0.67 |
| **CatBoost** | **0.8591** | 81% | 0.73 | 0.82 | 0.65 |
| **Logistic Regression** | **0.8589** | 82% | 0.76 | 0.79 | 0.72 |
| SVM | 0.8436 | 83% | 0.76 | 0.82 | 0.71 |
| Random Forest | 0.8457 | 83% | 0.76 | 0.83 | 0.70 |
| KNN | 0.8409 | 83% | 0.77 | 0.80 | 0.74 |
| XGBoost | 0.8430 | 83% | 0.78 | 0.77 | 0.78 |
| AdaBoost | 0.8383 | 77% | 0.69 | 0.70 | 0.68 |
| LightGBM | 0.8305 | 81% | 0.75 | 0.75 | 0.75 |
| MLP | 0.8187 | 80% | 0.73 | 0.77 | 0.70 |

### Stage 2 — Optuna Tuned Models (5 Trials, TPE Sampler)

| Model | Best CV AUC | Test AUC | Test Accuracy |
|---|---|---|---|
| 🥇 **LightGBM** | **0.8927** | 0.8370 | 82% |
| 🥈 **XGBoost** | **0.8893** | 0.8465 | 84% |
| 🥉 **Random Forest** | **0.8811** | 0.8456 | 83% |
| MLP | 0.8728 | 0.8523 | 83% |
| CatBoost | 0.8763 | 0.8513 | 79% |
| AdaBoost | 0.8753 | 0.8479 | 81% |
| Logistic Regression | 0.8664 | 0.8585 | 82% |
| KNN | 0.8613 | 0.8470 | 81% |
| SVM | 0.8522 | 0.8424 | 72% |
| TabNet | 0.7995 | 0.7685 | 73% |

### Ensemble Results

| Ensemble | CV AUC | Test AUC | Test Accuracy | Members |
|---|---|---|---|---|
| **Voting (Soft)** | **0.8919** | **0.8603** | 86% | RF, XGB, LGBM, LR |
| **Stacking** | **0.8901** | **0.8584** | 86% | RF, XGB, LGBM, LR, SVM + LR meta |

> 🏆 **Best model registered to Production:** `titanic-best-model` v1
> The **Voting Classifier** achieved the highest test AUC (0.8603), matching Stage 1 TabNet while being more robust across all metrics.

### Key Findings
- **Tuning helps CV scores significantly** (LightGBM: 0.8305 → 0.8927 CV) but test AUC improvement is modest — classic small-dataset behavior
- **Ensembles provide the best balance** of accuracy, recall, and generalization
- **CatBoost on raw categoricals** performs competitively without any encoding
- **TabNet underperforms on tuning** with only 5 trials — needs more epochs and trials to shine

---

## ⚙️ Setup & Reproduction

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1 (optional — CPU fallback works)
- [uv](https://docs.astral.sh/uv/) package manager
- DagsHub account

### 1. Clone and setup
```bash
git clone https://github.com/alyayman2020/titanic-mlops.git
cd titanic-mlops

# Create venv and install dependencies
uv venv --python 3.11 --seed
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate            # Linux/Mac

uv sync

# Install PyTorch for GPU (RTX 4050 / CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-tabnet
```

### 2. Configure credentials
```bash
# Copy and fill in your DagsHub token
cp .env.example .env
# Edit .env with your DAGSHUB_TOKEN and DAGSHUB_USERNAME
```

### 3. Pull data from DVC
```bash
dvc pull
```

### 4. Run the full pipeline
```bash
# Full run: Stage 1 + Stage 2 + Ensemble (recommended)
python trainer.py

# Stage 1 only (base models, ~5 min)
python trainer.py pipeline.run_stage2=false pipeline.run_ensemble=false

# Stage 2 + Ensemble only (~20 min)
python trainer.py pipeline.run_stage1=false

# Quick test (2 models, 5 trials)
python trainer.py optuna.n_trials=5 \
    models.stage1=[logistic_regression,random_forest] \
    models.stage2=[logistic_regression,random_forest] \
    pipeline.run_ensemble=false
```

### 5. Run inference
```bash
# From local tuned model
python predict.py --input data/raw/test.csv --source local \
    --local models/titanic/stage2__lightgbm_tuned.pkl

# From DagsHub Production registry
python predict.py --input data/raw/test.csv
```

---

## 📈 MLflow Tracking

Every run logs:

| Category | What's tracked |
|---|---|
| **Metrics** | Accuracy, Recall, Precision, F1, ROC-AUC, Log Loss (train + CV + test) |
| **Params** | All hyperparameters (base and Optuna-sampled) |
| **Artifacts** | `.pkl` model file + MLflow model flavor |
| **System** | CPU count, RAM total/available, GPU name, VRAM total/free |
| **Tags** | Stage, model name, trial number, artifact path |

View experiments: **[DagsHub MLflow UI](https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments)**

---

## 🗂️ Data Versioning (DVC)

Data is versioned with DVC and stored on DagsHub S3-compatible storage:

```bash
dvc pull          # Pull data from DagsHub
dvc push          # Push new data versions
dvc repro         # Reproduce the full pipeline
```

Remote: `https://dagshub.com/aly.ayman.2018/titanic-mlops.s3`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Hydra** | Config management + CLI overrides |
| **MLflow** | Experiment tracking + model registry |
| **DagsHub** | Remote MLflow server + model registry + DVC storage |
| **DVC** | Data + model versioning |
| **Optuna** | Hyperparameter optimization (TPE Sampler) |
| **scikit-learn** | LR, RF, AdaBoost, KNN, SVM, MLP, Voting, Stacking |
| **XGBoost** | Gradient boosting (CUDA) |
| **LightGBM** | Gradient boosting (GPU) |
| **CatBoost** | Gradient boosting (GPU, native categoricals) |
| **TabNet** | Attention-based tabular model (PyTorch) |
| **Loguru** | Structured logging |
| **Rich** | Terminal tables + progress |
| **uv** | Fast Python package management |

---

## 📋 Make Commands

```bash
make train          # Run full pipeline
make stage1         # Run Stage 1 only
make stage2         # Run Stage 2 only
make ensemble       # Run Stage 2 + Ensemble
make quick          # Quick test (5 trials, 2 models)
make predict        # Inference from production model
make mlflow-ui      # Launch local MLflow UI
make dvc-push       # Push data/models to DagsHub
make format         # Auto-format with ruff + black
make lint           # Lint (CI-safe)
make test           # Run pytest with coverage
```

---

## 🔗 Links

- **DagsHub Repo:** https://dagshub.com/aly.ayman.2018/titanic-mlops
- **MLflow Experiments:** https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments
- **Model Registry:** https://dagshub.com/aly.ayman.2018/titanic-mlops/models
- **GitHub:** https://github.com/alyayman2020/titanic-mlops
- **Reference Repo:** [Ezzaldin97/ITI-MLOps](https://github.com/Ezzaldin97/ITI-MLOps/tree/not-configured-pipeline)
- **Dataset:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

---

## 👨‍💻 Author

**Aly Ayman** — Data Science Trainee, ITI Data Science Track 2026
- GitHub: [@alyayman2020](https://github.com/alyayman2020)
- LinkedIn: [linkedin.com/in/alyayman](https://linkedin.com/in/alyayman)
