<div align="center">

# 🚢 Titanic Survival Prediction — End-to-End MLOps Pipeline

**A production-grade machine learning system demonstrating the complete MLOps lifecycle:**
experiment tracking, model registry, hyperparameter optimization, ensemble methods, REST API serving, and batch orchestration.

[![DagsHub](https://img.shields.io/badge/DagsHub-133%20Experiments-orange)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)](https://dagshub.com/aly.ayman.2018/titanic-mlops.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6)](https://dagshub.com/aly.ayman.2018/titanic-mlops)
[![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker)](https://hub.docker.com/r/alyayman25/titanic-mlops-api)
[![Prefect](https://img.shields.io/badge/Prefect-Orchestrated-00A36C)](https://app.prefect.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

[🔬 Experiments](https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments) •
[📦 Model Registry](https://dagshub.com/aly.ayman.2018/titanic-mlops/models) •
[🐳 Docker Image](https://hub.docker.com/r/alyayman25/titanic-mlops-api) •
[📊 DagsHub](https://dagshub.com/aly.ayman.2018/titanic-mlops)

</div>

---

## 📌 Project Overview

This project implements a **complete MLOps lifecycle** for Titanic survival classification — from raw data ingestion to a containerized production API and automated batch prediction pipeline. It demonstrates how modern ML engineering teams manage the full model development and deployment cycle with reproducibility, observability, and scalability at every stage.

**Key achievements:**
- Trained and tracked **10 classification models** across **133 MLflow experiments** on DagsHub
- Achieved **86% accuracy and 0.860 ROC-AUC** with a soft-voting ensemble
- Automated hyperparameter search using **Optuna TPE sampling** with nested MLflow run logging
- Built a **FastAPI inference server** supporting single and batch predictions, containerized to DockerHub
- Orchestrated a **Prefect batch pipeline** that extracts data from MotherDuck, loads the Production model from DagsHub, predicts survival for all 418 test passengers, and writes results back to MotherDuck

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
│   Raw CSV → DVC versioned → DagsHub S3 remote storage               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      Training Pipeline                               │
│                                                                      │
│   Hydra Config Management                                            │
│        │                                                             │
│        ├── Stage 1: 10 base models (fixed params)                   │
│        │       └── MLflow logs metrics, params, model artifacts      │
│        │                                                             │
│        ├── Stage 2: Optuna HPO (5 trials × 10 models, TPE)          │
│        │       └── Nested MLflow runs per trial                      │
│        │                                                             │
│        └── Stage 3: Ensemble (Voting + Stacking)                    │
│                └── Best model → DagsHub Model Registry / Production │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
┌─────────────▼────────────┐      ┌──────────────▼──────────────────┐
│   Online Serving (API)   │      │   Batch Serving (Prefect)        │
│                          │      │                                  │
│  FastAPI Server          │      │  MotherDuck (test_data table)    │
│   POST /predict          │      │        │                         │
│   GET  /health           │      │        ▼                         │
│   GET  /model/info       │      │  Extract → Transform → Predict   │
│                          │      │        │                         │
│  Loads from DagsHub      │      │        ▼                         │
│  Production registry     │      │  Save → MotherDuck               │
│                          │      │  (predictions table)             │
│  Docker → DockerHub      │      │                                  │
│  alyayman25/titanic-api  │      │  Monitored via Prefect Cloud     │
└──────────────────────────┘      └──────────────────────────────────┘
```

---

## 📊 Results

### Stage 1 — Base Models (Fixed Parameters)

| Model | Test AUC | Accuracy | F1 |
|---|---|---|---|
| **TabNet** | **0.8609** | 80% | 0.72 |
| **CatBoost** | **0.8591** | 81% | 0.73 |
| **Logistic Regression** | **0.8589** | 82% | 0.76 |
| SVM | 0.8436 | 83% | 0.76 |
| Random Forest | 0.8457 | 83% | 0.76 |
| KNN | 0.8409 | 83% | 0.77 |
| XGBoost | 0.8430 | 83% | 0.78 |
| AdaBoost | 0.8383 | 77% | 0.69 |
| LightGBM | 0.8305 | 81% | 0.75 |
| MLP | 0.8187 | 80% | 0.73 |

### Stage 2 — Optuna Tuned (5 Trials, TPE Sampler)

| Model | Best CV AUC | Test AUC | Accuracy |
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

| Method | Test AUC | Accuracy | F1 | Members |
|---|---|---|---|---|
| 🏆 **Soft Voting** | **0.8603** | **86%** | **0.81** | RF, XGB, LGBM, LR |
| Stacking | 0.8584 | 86% | 0.80 | RF, XGB, LGBM, LR, SVM + LR meta |

### Batch Pipeline Results
- **418 passengers** processed in a single Prefect flow run
- **139 predicted to survive** (33.25% survival rate)
- Full predictions saved to `titanic.predictions` table in MotherDuck
- Flow monitored and logged in Prefect Cloud

---

## 📁 Repository Structure

```
titanic-mlops/
├── api/
│   ├── main.py              # FastAPI application + endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── model_loader.py      # DagsHub registry + local fallback
├── batch/
│   ├── flow.py              # Prefect flow (7 tasks: ETL + predict)
│   ├── motherduck.py        # MotherDuck connection + queries
│   └── predictor.py         # Model loading + batch inference
├── conf/
│   ├── config.yaml          # Hydra pipeline config
│   ├── data/titanic.yaml    # Data paths + feature config
│   └── model/models.yaml    # 10 models with search spaces
├── src/
│   ├── preprocessing/       # StandardPreprocessor + CatBoostPreprocessor
│   ├── training/            # Model factory, trainer, registry
│   ├── ensemble/            # Voting + Stacking classifiers
│   ├── evaluation/          # 6 metrics + classification report
│   └── utils/               # Logger + system metrics (CPU/GPU)
├── scripts/
│   └── load_test_data.py    # One-time MotherDuck data loader
├── tests/                   # Pytest unit tests
├── bruno/                   # API test collection (5 requests)
├── trainer.py               # Training pipeline entry point
├── server.py                # FastAPI server entry point
├── predict.py               # CLI inference script
├── dockerfile.api           # Slim Docker image (~800MB)
├── requirements-api.txt     # API-only dependencies (15 packages)
├── requirements-batch.txt   # Batch pipeline dependencies
├── .dockerignore            # 63KB build context
└── dvc.yaml                 # Reproducible pipeline stages
```

---

## 🌐 API Reference

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
    {"passenger_index": 0, "survived": 1, "probability": 0.92, "confidence": "High"},
    {"passenger_index": 1, "survived": 0, "probability": 0.13, "confidence": "High"}
  ],
  "total_passengers": 2,
  "survived_count": 1,
  "model_name": "titanic-best-model",
  "model_source": "dagshub_registry",
  "processing_time_ms": 105.4
}
```

---

## ⚙️ Quickstart

### Prerequisites
- Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker
- NVIDIA GPU with CUDA 12.1 (optional — CPU fallback works)
- DagsHub, MotherDuck, and Prefect accounts

### 1. Clone and install
```bash
git clone https://github.com/alyayman2020/titanic-mlops.git
cd titanic-mlops

uv venv --python 3.11 --seed
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\Activate.ps1         # Windows

uv sync

# GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-tabnet
```

### 2. Configure credentials
```bash
cp .env.example .env
# Fill in: DAGSHUB_TOKEN, DAGSHUB_USERNAME
```

### 3. Pull versioned data and models
```bash
dvc pull
```

### 4. Run the training pipeline
```bash
python trainer.py                    # Full pipeline (~30 min with GPU)
python trainer.py pipeline.run_stage2=false pipeline.run_ensemble=false  # Stage 1 only
python trainer.py pipeline.run_stage1=false   # Stage 2 + Ensemble only
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

### 7. Run the batch pipeline
```bash
# Install batch dependencies
pip install -r requirements-batch.txt

# Load test data into MotherDuck (one time)
python -m scripts.load_test_data

# Run the Prefect flow
python -m batch.flow
```

---

## 📈 MLflow Tracking

133 runs tracked across all stages. Each run logs:

| Category | What's tracked |
|---|---|
| **Metrics** | Accuracy, Recall, Precision, F1, ROC-AUC, Log Loss |
| **Parameters** | All hyperparameters (base + Optuna-sampled) |
| **Artifacts** | `.pkl` file + MLflow model flavor (sklearn/catboost/pyfunc) |
| **System** | CPU cores, RAM, GPU model, VRAM usage per run |
| **Tags** | Pipeline stage, model name, trial number |

---

## 🗂️ Data & Model Versioning

```bash
dvc pull      # Reproduce exact data + models from any commit
dvc push      # Push new artifacts after training
dvc repro     # Re-run pipeline if deps change
dvc status    # Check what's changed
```

| Path | Size | Description |
|---|---|---|
| `data/raw/train.csv` | 61 KB | 712 training passengers |
| `data/raw/test.csv` | 29 KB | 418 test passengers |
| `models/titanic/` | 22.5 MB | 22 pkl files (base + tuned + ensemble) |
| `reports/titanic/predictions.csv` | 11 KB | CLI predictions output |

---

## 🔄 Batch Pipeline (Prefect + MotherDuck)

The Prefect flow runs 7 tasks in sequence:

```
Connect to MotherDuck
      ↓
Extract test_data (418 rows)
      ↓
Transform (validate dtypes, add missing cols)
      ↓
Load model from DagsHub Production registry
      ↓
Predict survival (139/418 = 33.25%)
      ↓
Save to titanic.predictions table
      ↓
Generate markdown summary artifact
```

**Query results in MotherDuck:**
```sql
SELECT PassengerId, Sex, Pclass, Survived_pred, Survival_prob, Confidence
FROM titanic.predictions
ORDER BY Survival_prob DESC
LIMIT 10;
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Config management** | Hydra |
| **Data versioning** | DVC + DagsHub S3 |
| **Experiment tracking** | MLflow + DagsHub |
| **HPO** | Optuna (TPE Sampler) |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch TabNet |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Containerization** | Docker, DockerHub |
| **Batch orchestration** | Prefect Cloud |
| **Cloud database** | MotherDuck (DuckDB) |
| **API testing** | Bruno |
| **Package management** | uv |

---

## 🔗 Project Links

| | Link |
|---|---|
| 📊 DagsHub | https://dagshub.com/aly.ayman.2018/titanic-mlops |
| 🔬 MLflow Experiments | https://dagshub.com/aly.ayman.2018/titanic-mlops/experiments |
| 📦 Model Registry | https://dagshub.com/aly.ayman.2018/titanic-mlops/models |
| 💻 GitHub | https://github.com/alyayman2020/titanic-mlops |
| 🐳 DockerHub | https://hub.docker.com/r/alyayman25/titanic-mlops-api |
| ⚡ Prefect Cloud | https://app.prefect.io |
| 🦆 MotherDuck | https://app.motherduck.com |

---

## 👨‍💻 Author

**Aly Ayman** — Data Scientist & ML Engineer

Pharmacy background turned ML practitioner, specializing in production ML systems, Arabic NLP, and healthcare AI in the MENA market.

[![GitHub](https://img.shields.io/badge/GitHub-alyayman2020-181717?logo=github&logoColor=white)](https://github.com/alyayman2020)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-alyayman-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/alyayman)
