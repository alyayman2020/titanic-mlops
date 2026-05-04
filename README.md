<div align="center">

# 🚢 Titanic Survival Prediction — End-to-End MLOps Pipeline

**A production-grade machine learning system built with modern MLOps practices:**
experiment tracking, model registry, hyperparameter optimization, ensemble methods, and REST API serving.

[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments%20133-orange?logo=data:image/svg+xml;base64,PHN2Zy8+)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?logo=mlflow)](https://dagshub.com/aly.ayman.2018/titanic-mlops.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6?logo=dvc)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker)](https://hub.docker.com/r/alyayman25/titanic-mlops-api)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

[🔬 Live Experiments](https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments) •
[📦 Model Registry](https://dagshub.com/aly.ayman.2018/titanic-mlops/models) •
[🐳 Docker Image](https://hub.docker.com/r/alyayman25/titanic-mlops-api) •
[📊 DagsHub Repo](https://dagshub.com/aly.ayman.2018/titanic-mlops)

</div>

---

## 📌 Project Overview

This project implements a **complete MLOps lifecycle** for the Titanic survival classification problem — from raw data to a containerized production API. It demonstrates how modern ML engineering teams manage the full model development cycle: reproducible experiments, versioned artifacts, automated hyperparameter search, and scalable serving infrastructure.

**Key achievements:**
- Trained and tracked **10 classification models** across 133 MLflow experiments on DagsHub
- Achieved **86% accuracy and 0.860 ROC-AUC** with a soft-voting ensemble
- Reduced hyperparameter search time using **Optuna TPE sampling** with MLflow nested run logging
- Built a **production FastAPI** inference server supporting single and batch predictions
- Containerized and shipped to **DockerHub** (`alyayman25/titanic-mlops-api`)
- Versioned all data and model artifacts with **DVC** backed by DagsHub S3 storage

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Layer                                 │
│  Raw CSV → DVC versioned → DagsHub S3 remote storage             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     Training Pipeline                             │
│                                                                   │
│  Hydra Config Management                                          │
│       │                                                           │
│       ├── Stage 1: Base models (10 models, fixed params)          │
│       │       └── MLflow logs metrics, params, artifacts          │
│       │                                                           │
│       ├── Stage 2: Optuna HPO (5 trials × 10 models, TPE)        │
│       │       └── Nested MLflow runs per trial                    │
│       │                                                           │
│       └── Stage 3: Ensemble (Voting + Stacking)                  │
│               └── Best model → DagsHub Model Registry            │
└────────────────────────────┬─────────────────────────────────────┘
                             │ titanic-best-model / Production
┌────────────────────────────▼─────────────────────────────────────┐
│                   Inference Layer                                  │
│                                                                   │
│  FastAPI Server                                                   │
│    POST /predict   → single or batch predictions                  │
│    GET  /health    → model status + source                        │
│    GET  /model/info → metadata                                    │
│                                                                   │
│  Model loading: DagsHub registry → local pkl fallback             │
│  Packaged as Docker image → DockerHub                             │
└──────────────────────────────────────────────────────────────────┘
```

### Two Preprocessing Paths

| Path | Models | Strategy |
|---|---|---|
| `StandardPreprocessor` | LR, RF, XGB, LGBM, AdaBoost, KNN, SVM, MLP, TabNet | Full feature engineering + ordinal encoding |
| `CatBoostPreprocessor` | CatBoost only | Minimal cleaning; Sex/Embarked passed as raw strings for native categorical handling |

### GPU Utilization (NVIDIA RTX 4050 — 6GB VRAM)

| Model | GPU Backend |
|---|---|
| XGBoost | `device=cuda` |
| LightGBM | `device=gpu` |
| CatBoost | `task_type=GPU` |
| TabNet | PyTorch CUDA |

---

## 📊 Results

### Model Comparison — Stage 1 vs Stage 2

| Model | Stage 1 AUC | Stage 2 CV AUC | Stage 2 Test AUC | Test Accuracy |
|---|---|---|---|---|
| LightGBM | 0.8305 | **0.8927** 🥇 | 0.8370 | 82% |
| XGBoost | 0.8430 | **0.8893** 🥈 | 0.8465 | 84% |
| Random Forest | 0.8457 | **0.8811** 🥉 | 0.8456 | 83% |
| CatBoost | 0.8591 | 0.8763 | 0.8513 | 79% |
| AdaBoost | 0.8383 | 0.8753 | 0.8479 | 81% |
| MLP | 0.8187 | 0.8728 | 0.8523 | 83% |
| Logistic Regression | 0.8589 | 0.8664 | 0.8585 | 82% |
| KNN | 0.8409 | 0.8613 | 0.8470 | 81% |
| SVM | 0.8436 | 0.8522 | 0.8424 | 72% |
| TabNet | 0.8609 | 0.7995 | 0.7685 | 73% |

### Ensemble Performance

| Method | Test AUC | Accuracy | F1 | Members |
|---|---|---|---|---|
| **Soft Voting** | **0.8603** | **86%** | **0.81** | RF, XGB, LGBM, LR |
| Stacking | 0.8584 | 86% | 0.80 | RF, XGB, LGBM, LR, SVM + LR meta |

> **Insight:** Ensembling boosted test AUC by ~0.5% over the best single model while improving accuracy to 86% — the expected benefit of combining diverse models' decision boundaries.

---

## 🌐 REST API

The inference server accepts **single or batch requests** through a single endpoint.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Single or batch survival prediction |
| `GET` | `/health` | Server health + model status |
| `GET` | `/model/info` | Model metadata |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — Batch Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {"Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": "S"},
      {"Pclass": 3, "Sex": "male",   "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
    ]
  }'
```

```json
{
  "predictions": [
    {"passenger_index": 0, "survived": 1, "probability": 0.9200, "confidence": "High"},
    {"passenger_index": 1, "survived": 0, "probability": 0.1300, "confidence": "High"}
  ],
  "total_passengers": 2,
  "survived_count": 1,
  "model_name": "titanic-best-model",
  "model_source": "dagshub_registry",
  "processing_time_ms": 105.4
}
```

---

## 📁 Repository Structure

```
titanic-mlops/
├── api/
│   ├── main.py              # FastAPI application + endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── model_loader.py      # Registry loader with local fallback
├── conf/
│   ├── config.yaml          # Hydra pipeline configuration
│   ├── data/titanic.yaml    # Data paths and split config
│   └── model/models.yaml    # Model hyperparameter search spaces
├── src/
│   ├── preprocessing/       # Feature engineering (2 paths)
│   ├── training/            # Model factory, trainer, registry
│   ├── ensemble/            # Voting + Stacking classifiers
│   ├── evaluation/          # Metrics (Accuracy, F1, AUC, Log Loss...)
│   └── utils/               # Logger, system metrics (CPU/GPU)
├── tests/                   # Unit tests (preprocessors, metrics)
├── bruno/                   # API test collection (5 requests)
├── trainer.py               # Pipeline entry point
├── server.py                # API server entry point
├── predict.py               # CLI inference
├── dockerfile.api           # Slim production Docker image
├── requirements-api.txt     # API-only dependencies
├── .dockerignore            # 63KB build context (vs 6.44GB without)
└── dvc.yaml                 # Reproducible pipeline stages
```

---

## ⚙️ Quickstart

### Prerequisites
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA 12.1 (optional — CPU fallback works)
- DagsHub account with access token

### 1. Clone and install
```bash
git clone https://github.com/alyayman2020/titanic-mlops.git
cd titanic-mlops

uv venv --python 3.11 --seed
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
uv sync

# GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-tabnet
```

### 2. Configure credentials
```bash
cp .env.example .env
# Set DAGSHUB_TOKEN and DAGSHUB_USERNAME in .env
```

### 3. Pull versioned data and models
```bash
dvc pull
```

### 4. Run the pipeline
```bash
# Full pipeline: Stage 1 + Stage 2 + Ensemble (~30 min with GPU)
python trainer.py

# Stage 1 only — base models, fast (~5 min)
python trainer.py pipeline.run_stage2=false pipeline.run_ensemble=false

# Override any config from CLI
python trainer.py optuna.n_trials=20 models.stage2=[xgboost,lightgbm]
```

### 5. Start the API server
```bash
python server.py
# → http://localhost:8000/docs
```

### 6. Run with Docker
```bash
docker pull alyayman25/titanic-mlops-api:latest

docker run -p 8000:8000 \
  -e DAGSHUB_TOKEN=your_token \
  -e DAGSHUB_USERNAME=aly.ayman.2018 \
  -v $(pwd)/models:/app/models \
  alyayman25/titanic-mlops-api:latest
```

---

## 📈 MLflow Experiment Tracking

133 runs tracked across all stages. Each run logs:

| Category | What's tracked |
|---|---|
| **Metrics** | Accuracy, Recall, Precision, F1-Score, ROC-AUC, Log Loss |
| **Parameters** | All hyperparameters (base and Optuna-sampled) |
| **Artifacts** | `.pkl` model file + MLflow model flavor (sklearn/catboost/pyfunc) |
| **System info** | CPU cores, RAM, GPU model, VRAM usage |
| **Tags** | Pipeline stage, model name, trial number, artifact path |

View all experiments: **https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments**

---

## 🗂️ Data & Model Versioning

All artifacts are versioned with DVC and stored on DagsHub S3-compatible storage:

```bash
dvc pull    # Reproduce exact data + models from any commit
dvc push    # Push new artifacts after training
dvc repro   # Re-run pipeline if deps change
dvc status  # Check what's changed
```

**Tracked artifacts:**

| Path | Size | Description |
|---|---|---|
| `data/raw/train.csv` | 61 KB | Training data (712 passengers) |
| `data/raw/test.csv` | 29 KB | Test data (418 passengers) |
| `models/titanic/` | 22.5 MB | 22 pkl files (10 base + 10 tuned + 2 ensemble) |
| `reports/titanic/predictions.csv` | 11 KB | Final test predictions |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Pipeline orchestration** | Hydra, DVC |
| **Experiment tracking** | MLflow, DagsHub |
| **HPO** | Optuna (TPE Sampler) |
| **ML frameworks** | scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (TabNet) |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Containerization** | Docker, DockerHub |
| **API testing** | Bruno |
| **Package management** | uv |
| **Logging** | Loguru |
| **Code quality** | Ruff, Black, isort |
| **Testing** | Pytest |

---

## 🔗 Project Links

| | Link |
|---|---|
| 📊 DagsHub | https://dagshub.com/aly.ayman.2018/titanic-mlops |
| 🔬 MLflow Experiments | https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments |
| 📦 Model Registry | https://dagshub.com/aly.ayman.2018/titanic-mlops/models |
| 💻 GitHub | https://github.com/alyayman2020/titanic-mlops |
| 🐳 DockerHub | https://hub.docker.com/r/alyayman25/titanic-mlops-api |

---

## 👨‍💻 Author

**Aly Ayman** — Data Scientist & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-alyayman2020-181717?logo=github&logoColor=white)](https://github.com/alyayman2020)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-alyayman-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/alyayman)
