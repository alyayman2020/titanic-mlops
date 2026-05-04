"""
Core training engine.

Stage 1 — train each model with fixed base_params, log to MLflow.
Stage 2 — Optuna study per model, log each trial as a child run.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    summarise_cv_scores,
)
from src.preprocessing.preprocess import CatBoostPreprocessor, StandardPreprocessor
from src.training.model_factory import build_model, get_tabnet_fit_params
from src.utils.logger import get_logger
from src.utils.system_metrics import RuntimeTimer, get_system_info

log = get_logger("trainer")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.data.raw_train)
    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_standard_pipeline(model_name: str, params: dict, cfg: DictConfig) -> Pipeline:
    """Preprocessor → (optional scaler) → model."""
    model_cfg = cfg.models_cfg[model_name]
    steps = [("preprocessor", StandardPreprocessor(drop_cols=list(cfg.data.drop_cols)))]

    if model_cfg.get("needs_scaling", False):
        steps.append(("scaler", StandardScaler()))

    estimator = build_model(model_name, params, cfg)
    steps.append(("model", estimator))
    return Pipeline(steps)


def _prepare_catboost_data(
    X: pd.DataFrame, cfg: DictConfig, preprocessor: CatBoostPreprocessor | None = None
) -> tuple[pd.DataFrame, CatBoostPreprocessor, list[int]]:
    """Fit (or apply) the CatBoost preprocessor and return (X_clean, prep, cat_indices)."""
    if preprocessor is None:
        preprocessor = CatBoostPreprocessor(
            drop_cols=list(cfg.data.catboost_drop_cols),
            cat_features=list(cfg.data.catboost_cat_features),
        )
        X_clean = preprocessor.fit_transform(X)
    else:
        X_clean = preprocessor.transform(X)

    cat_indices = preprocessor.get_cat_feature_indices(X_clean if isinstance(X_clean, pd.DataFrame) else X)
    return X_clean, preprocessor, cat_indices


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-validation helper (works for both Pipeline and CatBoost dict)
# ══════════════════════════════════════════════════════════════════════════════

SCORING = {
    "accuracy":  "accuracy",
    "recall":    "recall",
    "precision": "precision",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
    "neg_log_loss": "neg_log_loss",
}


def _cv_score_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: DictConfig,
) -> dict[str, list[float]]:
    cv = StratifiedKFold(n_splits=cfg.pipeline.cv_folds, shuffle=True,
                         random_state=cfg.pipeline.random_state)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING,
                             n_jobs=cfg.pipeline.n_jobs, error_score="raise")
    return {
        "accuracy":  results["test_accuracy"].tolist(),
        "recall":    results["test_recall"].tolist(),
        "precision": results["test_precision"].tolist(),
        "f1":        results["test_f1"].tolist(),
        "roc_auc":   results["test_roc_auc"].tolist(),
        "log_loss":  (-results["test_neg_log_loss"]).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MLflow helpers
# ══════════════════════════════════════════════════════════════════════════════

def _log_system_tags(run):
    info = get_system_info()
    for k, v in info.items():
        mlflow.set_tag(k, str(v))


def _get_project_root(cfg: DictConfig) -> Path:
    """Derive absolute project root from the known-absolute raw_train path."""
    try:
        model_dir = cfg.data.get("model_output_dir") or cfg.data["model_output_dir"]
        if model_dir:
            # model_output_dir is set → go up 2 levels (models/titanic → project root)
            return Path(model_dir).parent.parent
    except Exception:
        pass
    # Fallback: data/raw/train.csv → up 3 levels
    return Path(str(cfg.data.raw_train)).parent.parent.parent


def _save_and_log_model(
    pipeline_or_obj: Any,
    model_name: str,
    cfg: DictConfig,
    stage: str = "stage1",
    metrics: dict | None = None,
):
    """
    Save model as .pkl locally and log to MLflow with proper naming.

    Naming convention:
        stage1__logistic_regression.pkl
        stage2__xgboost_tuned.pkl
        ensemble__voting_classifier.pkl
    """
    project_root = _get_project_root(cfg)
    out_dir = project_root / "models" / "titanic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Consistent file name ───────────────────────────────────────
    clean_name = model_name.replace(" ", "_").lower()
    file_name = f"{stage}__{clean_name}.pkl"
    path = out_dir / file_name

    with open(path, "wb") as f:
        pickle.dump(pipeline_or_obj, f)
    log.info(f"  Saved → {path}")

    # ── Log pkl as raw artifact (always) ──────────────────────────
    mlflow.log_artifact(str(path), artifact_path="models")

    # ── Log as proper MLflow model (shows in Models column) ───────
    # Determine the underlying estimator type for correct flavor
    try:
        obj = pipeline_or_obj
        # Unwrap dict artifacts (CatBoost, TabNet)
        if isinstance(obj, dict):
            inner = obj.get("model", obj)
        else:
            inner = obj

        model_artifact_path = f"model_{clean_name}"

        # Check model type and use appropriate MLflow flavor
        type_name = type(inner).__name__.lower()

        if "catboost" in type_name:
            mlflow.catboost.log_model(inner, artifact_path=model_artifact_path)

        elif "tabnet" in type_name or "tabmodel" in type_name:
            # TabNet → log as pyfunc with pkl
            class _TabNetWrapper(mlflow.pyfunc.PythonModel):
                def load_context(self, ctx):
                    import pickle
                    with open(ctx.artifacts["pkl"], "rb") as f:
                        self.artifact = pickle.load(f)
                def predict(self, ctx, data):
                    import numpy as np
                    m = self.artifact.get("model") if isinstance(self.artifact, dict) else self.artifact
                    p = self.artifact.get("preprocessor") if isinstance(self.artifact, dict) else None
                    X = p.transform(data).astype(np.float32) if p else data.values.astype(np.float32)
                    return m.predict(X)
            mlflow.pyfunc.log_model(
                artifact_path=model_artifact_path,
                python_model=_TabNetWrapper(),
                artifacts={"pkl": str(path)},
            )

        elif hasattr(inner, "predict"):
            # Standard sklearn-compatible model (Pipeline, RF, XGB, LGBM, etc.)
            actual = inner if not isinstance(pipeline_or_obj, dict) else pipeline_or_obj
            # For dict artifacts wrap in pyfunc
            if isinstance(pipeline_or_obj, dict):
                class _DictWrapper(mlflow.pyfunc.PythonModel):
                    def load_context(self, ctx):
                        import pickle
                        with open(ctx.artifacts["pkl"], "rb") as f:
                            self.artifact = pickle.load(f)
                    def predict(self, ctx, data):
                        m = self.artifact.get("model") if isinstance(self.artifact, dict) else self.artifact
                        p = self.artifact.get("preprocessor") if isinstance(self.artifact, dict) else None
                        X = p.transform(data) if p else data
                        return m.predict(X)
                mlflow.pyfunc.log_model(
                    artifact_path=model_artifact_path,
                    python_model=_DictWrapper(),
                    artifacts={"pkl": str(path)},
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=pipeline_or_obj,
                    artifact_path=model_artifact_path,
                    registered_model_name=None,  # register separately after best selection
                )
        log.info(f"  MLflow model logged as '{model_artifact_path}' ✓")
    except Exception as e:
        log.warning(f"  MLflow model logging failed ({e}) — pkl artifact still saved")

    # ── Metadata tags ─────────────────────────────────────────────
    mlflow.set_tag("artifact_name", file_name)
    mlflow.set_tag("artifact_stage", stage)
    mlflow.set_tag("artifact_model", clean_name)
    if metrics:
        mlflow.log_metrics({f"saved_{k}": v for k, v in metrics.items()})

    return path


def _log_cv_metrics(cv_scores: dict, prefix: str = "cv_"):
    summary = summarise_cv_scores(cv_scores)
    mlflow.log_metrics({f"{prefix}{k}": v for k, v in summary.items()})


def _log_test_metrics(y_test, y_pred, y_proba, report: str, report_dir: str = "reports/titanic"):
    metrics = compute_metrics(y_test, y_pred, y_proba, prefix="test_")
    mlflow.log_metrics(metrics)
    # Log classification report as text artifact
    report_path = Path(report_dir) / "classification_report_tmp.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    mlflow.log_artifact(str(report_path), artifact_path="reports")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Base model training
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: DictConfig,
    experiment_id: str,
) -> dict[str, float]:
    """Train all models with fixed base_params. Returns {model_name: test_roc_auc}."""
    log.info("═" * 60)
    log.info("STAGE 1 — Base model training")
    log.info("═" * 60)

    results: dict[str, float] = {}

    for model_name in cfg.models.stage1:
        model_cfg = cfg.models_cfg[model_name]
        base_params = OmegaConf.to_container(model_cfg.base_params, resolve=True)

        log.info(f"[Stage 1] Training {model_cfg.display_name}…")
        timer = RuntimeTimer().start()

        run_name = f"stage1__{model_name}"
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            _log_system_tags(run)
            mlflow.set_tag("stage", "1_base")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("display_name", model_cfg.display_name)
            mlflow.log_params({f"base_{k}": v for k, v in base_params.items()})

            # ── CatBoost special path ────────────────────────────
            if model_cfg.get("is_catboost", False):
                cb_prep = CatBoostPreprocessor(
                    drop_cols=list(cfg.data.catboost_drop_cols),
                    cat_features=list(cfg.data.catboost_cat_features),
                )
                X_tr_cb = cb_prep.fit_transform(X_train)
                X_te_cb = cb_prep.transform(X_test)
                cat_indices = cb_prep.get_cat_feature_indices(X_tr_cb)

                model = build_model(model_name, base_params, cfg)
                model.fit(
                    X_tr_cb, y_train,
                    cat_features=cat_indices,
                    eval_set=(X_te_cb, y_test),
                )
                y_pred = model.predict(X_te_cb)
                y_proba = model.predict_proba(X_te_cb)[:, 1]
                artifact = {"model": model, "preprocessor": cb_prep, "cat_indices": cat_indices}
                _save_and_log_model(artifact, model_name, cfg, stage="stage1")

            # ── TabNet special path ──────────────────────────────
            elif model_name == "tabnet":
                pipeline = _build_standard_pipeline(model_name, {}, cfg)
                prep_pipeline = Pipeline(pipeline.steps[:-1])
                prep_pipeline.fit(X_train, y_train)
                X_tr_np = prep_pipeline.transform(X_train).astype(np.float32)
                X_te_np = prep_pipeline.transform(X_test).astype(np.float32)

                # build_model handles optimizer_lr -> optimizer_params internally
                fit_params = get_tabnet_fit_params(base_params)
                model = build_model(model_name, base_params, cfg)
                model.fit(
                    X_tr_np, y_train.values,
                    eval_set=[(X_te_np, y_test.values)],
                    eval_metric=["auc"],
                    **fit_params,
                )
                y_pred = model.predict(X_te_np)
                y_proba = model.predict_proba(X_te_np)[:, 1]
                artifact = {"model": model, "preprocessor": prep_pipeline}
                _save_and_log_model(artifact, model_name, cfg, stage="stage1")

            # ── Standard sklearn Pipeline path ───────────────────
            else:
                pipeline = _build_standard_pipeline(model_name, base_params, cfg)
                cv_scores = _cv_score_pipeline(pipeline, X_train, y_train, cfg)
                _log_cv_metrics(cv_scores)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_proba = (pipeline.predict_proba(X_test)[:, 1]
                           if hasattr(pipeline, "predict_proba") else None)
                _save_and_log_model(pipeline, model_name, cfg, stage="stage1")

            # ── Common: test metrics + report ────────────────────
            report = get_classification_report(y_test, y_pred)
            project_root = Path(str(cfg.data.raw_train)).parent.parent.parent
            report_dir = str(project_root / "reports" / "titanic")
            _log_test_metrics(y_test, y_pred, y_proba, report, report_dir=report_dir)

            elapsed = timer.elapsed()
            mlflow.log_metric("training_time_sec", elapsed)

            test_auc = compute_metrics(y_test, y_pred, y_proba).get("roc_auc", 0.0)
            results[model_name] = test_auc

            log.info(
                f"  ✓ {model_cfg.display_name} | "
                f"AUC={test_auc:.4f} | {elapsed:.1f}s"
            )
            log.info(f"\n{report}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Optuna hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════

def _sample_params(trial: optuna.Trial, search_space: dict) -> dict[str, Any]:
    """Sample hyperparameters from a search_space config dict."""
    params: dict[str, Any] = {}
    for param_name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"])
        elif ptype == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"],
                log=spec.get("log", False))
        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, spec["choices"])
    return params


def _make_objective(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
    experiment_id: str,
    parent_run_id: str,
) -> Any:
    """Return an Optuna objective closure for the given model."""
    model_cfg = cfg.models_cfg[model_name]
    raw_ss = model_cfg.search_space
    search_space = OmegaConf.to_container(raw_ss, resolve=True) if hasattr(raw_ss, "_metadata") else dict(raw_ss)
    cv = StratifiedKFold(n_splits=cfg.pipeline.cv_folds, shuffle=True,
                         random_state=cfg.pipeline.random_state)
    objective_metric = cfg.optuna.metric  # e.g. "roc_auc"

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, search_space)
        trial_name = f"stage2__{model_name}__trial_{trial.number}"

        with mlflow.start_run(
            run_name=trial_name,
            experiment_id=experiment_id,
            nested=True,
        ):
            mlflow.set_tag("stage", "2_optuna_trial")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("parent_run_id", parent_run_id)
            mlflow.log_params(params)

            # ── CatBoost ──────────────────────────────────────────
            if model_cfg.get("is_catboost", False):
                cb_prep = CatBoostPreprocessor(
                    drop_cols=list(cfg.data.catboost_drop_cols),
                    cat_features=list(cfg.data.catboost_cat_features),
                )
                scores = []
                for tr_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                    prep = CatBoostPreprocessor(
                        drop_cols=list(cfg.data.catboost_drop_cols),
                        cat_features=list(cfg.data.catboost_cat_features),
                    )
                    X_tr_cb = prep.fit_transform(X_tr)
                    X_val_cb = prep.transform(X_val)
                    cat_idx = prep.get_cat_feature_indices(X_tr_cb)
                    model = build_model(model_name, params, cfg)
                    model.fit(X_tr_cb, y_tr, cat_features=cat_idx)
                    proba = model.predict_proba(X_val_cb)[:, 1]
                    scores.append(roc_auc_score_safe(y_val, proba))
                score = float(np.mean(scores))

            # ── TabNet ────────────────────────────────────────────
            elif model_name == "tabnet":
                fit_params = get_tabnet_fit_params(params)
                scores = []
                for tr_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                    pipeline = _build_standard_pipeline("tabnet", {}, cfg)
                    prep_pipeline = Pipeline(pipeline.steps[:-1])
                    prep_pipeline.fit(X_tr, y_tr)
                    X_tr_np = prep_pipeline.transform(X_tr).astype(np.float32)
                    X_val_np = prep_pipeline.transform(X_val).astype(np.float32)
                    model = build_model(model_name, params, cfg)
                    model.fit(X_tr_np, y_tr.values,
                              eval_set=[(X_val_np, y_val.values)],
                              eval_metric=["auc"],
                              **fit_params)
                    proba = model.predict_proba(X_val_np)[:, 1]
                    scores.append(roc_auc_score_safe(y_val, proba))
                score = float(np.mean(scores))

            # ── Standard sklearn ──────────────────────────────────
            else:
                pipeline = _build_standard_pipeline(model_name, params, cfg)
                cv_results = cross_validate(
                    pipeline, X_train, y_train,
                    cv=cv, scoring="roc_auc",
                    n_jobs=cfg.pipeline.n_jobs,
                )
                score = float(cv_results["test_score"].mean())

            mlflow.log_metric(f"cv_{objective_metric}_mean", score)
            return score

    return objective


def roc_auc_score_safe(y_true, y_proba) -> float:
    """Wraps roc_auc_score with import (avoids circular at module level)."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_proba)


def run_stage2(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: DictConfig,
    experiment_id: str,
) -> dict[str, dict]:
    """
    Run Optuna hyperparameter search for each model.
    Returns {model_name: {"best_params": ..., "best_score": ..., "pipeline": ...}}
    """
    log.info("═" * 60)
    log.info("STAGE 2 — Optuna hyperparameter tuning")
    log.info("═" * 60)

    best_results: dict[str, dict] = {}

    for model_name in cfg.models.stage2:
        model_cfg = cfg.models_cfg[model_name]
        log.info(f"[Stage 2] Tuning {model_cfg.display_name} "
                 f"({cfg.optuna.n_trials} trials)…")
        timer = RuntimeTimer().start()

        parent_run_name = f"stage2_parent__{model_name}"
        with mlflow.start_run(
            run_name=parent_run_name, experiment_id=experiment_id
        ) as parent_run:
            _log_system_tags(parent_run)
            mlflow.set_tag("stage", "2_optuna_parent")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("display_name", model_cfg.display_name)
            mlflow.log_param("n_trials", cfg.optuna.n_trials)
            mlflow.log_param("optuna_metric", cfg.optuna.metric)

            study = optuna.create_study(
                direction=cfg.optuna.direction,
                sampler=optuna.samplers.TPESampler(seed=cfg.pipeline.random_state),
            )
            objective = _make_objective(
                model_name, X_train, y_train, cfg,
                experiment_id, parent_run.info.run_id,
            )
            study.optimize(
                objective,
                n_trials=cfg.optuna.n_trials,
                timeout=cfg.optuna.get("timeout"),
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_score = study.best_value

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric(f"best_cv_{cfg.optuna.metric}", best_score)

            # ── Retrain on full train set with best params ────────
            if model_cfg.get("is_catboost", False):
                cb_prep = CatBoostPreprocessor(
                    drop_cols=list(cfg.data.catboost_drop_cols),
                    cat_features=list(cfg.data.catboost_cat_features),
                )
                X_tr_cb = cb_prep.fit_transform(X_train)
                X_te_cb = cb_prep.transform(X_test)
                cat_indices = cb_prep.get_cat_feature_indices(X_tr_cb)
                model = build_model(model_name, best_params, cfg)
                model.fit(X_tr_cb, y_train, cat_features=cat_indices,
                          eval_set=(X_te_cb, y_test))
                y_pred = model.predict(X_te_cb)
                y_proba = model.predict_proba(X_te_cb)[:, 1]
                artifact = {"model": model, "preprocessor": cb_prep,
                            "cat_indices": cat_indices}
                _save_and_log_model(artifact, f"{model_name}_tuned", cfg, stage="stage2")
                best_results[model_name] = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "artifact": artifact,
                    "type": "catboost",
                    "preprocessor": cb_prep,
                    "cat_indices": cat_indices,
                }

            elif model_name == "tabnet":
                fit_params = get_tabnet_fit_params(best_params)
                pipeline = _build_standard_pipeline("tabnet", {}, cfg)
                prep_pipeline = Pipeline(pipeline.steps[:-1])
                prep_pipeline.fit(X_train, y_train)
                X_tr_np = prep_pipeline.transform(X_train).astype(np.float32)
                X_te_np = prep_pipeline.transform(X_test).astype(np.float32)
                model = build_model("tabnet", best_params, cfg)
                model.fit(X_tr_np, y_train.values,
                          eval_set=[(X_te_np, y_test.values)],
                          eval_metric=["auc"], **fit_params)
                y_pred = model.predict(X_te_np)
                y_proba = model.predict_proba(X_te_np)[:, 1]
                artifact = {"model": model, "preprocessor": prep_pipeline}
                _save_and_log_model(artifact, f"{model_name}_tuned", cfg, stage="stage2")
                best_results[model_name] = {
                    "best_params": best_params, "best_score": best_score,
                    "artifact": artifact, "type": "tabnet",
                    "preprocessor": prep_pipeline,
                }

            else:
                pipeline = _build_standard_pipeline(model_name, best_params, cfg)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_proba = (pipeline.predict_proba(X_test)[:, 1]
                           if hasattr(pipeline, "predict_proba") else None)
                _save_and_log_model(pipeline, f"{model_name}_tuned", cfg, stage="stage2")
                best_results[model_name] = {
                    "best_params": best_params, "best_score": best_score,
                    "pipeline": pipeline, "type": "sklearn",
                }

            # ── Log test metrics ──────────────────────────────────
            report = get_classification_report(y_test, y_pred)
            project_root = Path(str(cfg.data.raw_train)).parent.parent.parent
            report_dir = str(project_root / "reports" / "titanic")
            _log_test_metrics(y_test, y_pred, y_proba, report, report_dir=report_dir)
            elapsed = timer.elapsed()
            mlflow.log_metric("tuning_time_sec", elapsed)

            test_auc = compute_metrics(y_test, y_pred, y_proba).get("roc_auc", 0.0)
            log.info(
                f"  ✓ {model_cfg.display_name} | Best CV AUC={best_score:.4f} | "
                f"Test AUC={test_auc:.4f} | {elapsed:.1f}s"
            )
            log.info(f"\n{report}")

    return best_results
