# 🚢 Titanic MLOps — Lab 0: Training Pipeline

> **ITI Data Science Track — MLOps Module | Lab 0**
> A production-grade, automated ML training pipeline for the Titanic survival
> prediction task, built following MLOps best practices.

---

## 🎯 Lab Objectives

This lab covers:

- ✅ Cloning the reference `not-configured-pipeline` branch as a starting point
- ✅ Creating a GitHub repository for the lab
- ✅ Downloading the Titanic dataset from Kaggle
- ✅ Building a training pipeline with **at least 2 scikit-learn models**
- ✅ Automating all steps from data loading → preprocessing → model saving
- ✅ Creating a **model pipeline** with a custom sklearn transformer/encoder
- ✅ Formatting and linting with **ruff**, **black**, and **isort**
- ✅ Pushing the final code to GitHub

---

## 📁 Project Structure

```
titanic-mlops/
├── conf/                        # Configuration files
│   ├── config.yaml              # Hydra-style config reference
│   └── pipeline/                # Pipeline stage configs (placeholder)
├── data/
│   ├── raw/                     # ← Place train.csv here (from Kaggle)
│   ├── external/                # Third-party data sources
│   ├── interim/                 # Intermediate transformed data
│   └── processed/               # Processed train/test datasets
├── models/
│   └── titanic/                 # Saved model artifacts (.pkl)
├── notebooks/                   # Jupyter notebooks
├── reports/
│   └── titanic/                 # Evaluation reports
├── src/
│   ├── fake/                    # Custom estimator placeholder
│   │   └── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── preprocess.py        # TitanicPreprocessor (custom sklearn transformer)
│   │   └── train.py             # RF + AdaBoost pipelines, evaluation, persistence
│   ├── __init__.py
│   └── logger.py                # Loguru-based logger
├── tests/
│   ├── conftest.py              # Shared fixtures (synthetic Titanic data)
│   ├── test_preprocess.py       # 16 tests for TitanicPreprocessor
│   └── test_train.py            # 18 tests for pipelines, training, save/load
├── logs/                        # Auto-generated training logs
├── trainer.py                   # 🚀 Main entry point
├── params.yaml                  # All hyperparameters and paths
├── pyproject.toml               # Project config: dependencies + tool settings
├── Makefile                     # Developer shortcuts
└── .python-version              # Python version pin for uv
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager

### 1. Clone the repo
```bash
git clone https://github.com/alyayman2020/titanic-mlops.git
cd titanic-mlops
```

### 2. Create virtual environment & install dependencies
```bash
uv venv --python 3.11
source .venv/bin/activate        # Unix / macOS
# .venv\Scripts\activate         # Windows

uv sync
# or: pip install -e ".[dev]"
```

### 3. Download the Titanic dataset
1. Go to [kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
2. Download `train.csv`
3. Place it at `data/raw/train.csv`

---

## 🚀 Running the Pipeline

```bash
python trainer.py
# or
make train
```

**What happens:**
1. Loads `params.yaml` configuration
2. Reads `data/raw/train.csv`
3. Stratified 80/20 train/test split
4. Cross-validates both pipelines (10-fold, on training data)
5. Fits both pipelines on the full training set
6. Evaluates on held-out test set (Accuracy, Precision, Recall, F1)
7. Saves artifacts to `models/titanic/random_forest.pkl` and `models/titanic/adaboost.pkl`
8. Prints a side-by-side comparison and declares the winner by F1

---

## 🏗️ Architecture

### Preprocessing Pipeline (`src/training/preprocess.py`)

`TitanicPreprocessor` is a custom `sklearn` transformer (`BaseEstimator` + `TransformerMixin`) that:

| Step | Operation |
|------|-----------|
| 1 | Extract `Initial` (title) from `Name` — Mr, Mrs, Miss, Master, Other |
| 2 | Impute missing `Age` using per-title median (learned during `fit()`) |
| 3 | Fill missing `Embarked` with mode (learned during `fit()`) |
| 4 | Fill missing `Fare` with median (learned during `fit()`) |
| 5 | Engineer `Age_band`, `Family_Size`, `Alone`, `Fare_cat` features |
| 6 | Encode `Sex`, `Embarked`, `Initial` as ordinal integers |
| 7 | Drop `Name`, `Age`, `Ticket`, `Fare`, `Cabin`, `PassengerId` |

All statistics are learned **only from training data** during `fit()`, preventing data leakage.

### Model Pipelines (`src/training/train.py`)

```
TitanicPreprocessor  →  RandomForestClassifier
TitanicPreprocessor  →  AdaBoostClassifier
```

Each saved `.pkl` artifact contains the **full pipeline** (preprocessor + classifier), so inference only needs `pipeline.predict(raw_df)`.

---

## 🧹 Code Quality

All code is formatted with **black** (style) and **ruff** (linting + import sorting):

```bash
make format   # ruff --fix + black  (auto-fix everything)
make lint     # ruff check + black --check  (CI-safe, no writes)
```

**Tool configuration** lives in `pyproject.toml`:
- `[tool.black]` — line length 99, Python 3.11
- `[tool.ruff]` — rules E, F, I, W; import sorting via rule I
- `[tool.isort]` — profile=black (for IDE compatibility)

---

## 🧪 Tests

```bash
make test
# or: pytest tests/ -v --cov=src
```

**34 tests** covering:
- `TitanicPreprocessor`: fit statistics, transform correctness, edge cases (all-null columns)
- Pipeline builders: correct step names and classifier types
- Training: fit, predict, classes learned
- Evaluation: metric keys and value ranges
- Save/load: file creation, roundtrip predictions, error handling

---

## ⚙️ Configuration (`params.yaml`)

```yaml
data:
  raw_path: data/raw/train.csv
  test_size: 0.2
  random_state: 42

model:
  model_name: titanic
  output_dir: models/titanic

models:
  random_forest:
    n_estimators: 200
    max_depth: 8
    random_state: 42
  adaboost:
    n_estimators: 100
    learning_rate: 0.1
    random_state: 42
```

---

## 📋 Make Commands

```bash
make help               # List all commands
make requirements       # Install dependencies (uv sync)
make create_environment # Create .venv with Python 3.11
make train              # Run the full training pipeline
make format             # Auto-format code
make lint               # Lint without modifying files
make test               # Run pytest with coverage
make clean              # Remove __pycache__ and build artifacts
```

---

## 🔗 References

- Reference repo: [Ezzaldin97/ITI-MLOps (not-configured-pipeline)](https://github.com/Ezzaldin97/ITI-MLOps/tree/not-configured-pipeline)
- Tools: [Ruff](https://docs.astral.sh/ruff/) · [Black](https://black.readthedocs.io/) · [isort](https://pycqa.github.io/isort/)
- Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
