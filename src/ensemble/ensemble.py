"""
Ensemble module.

Builds VotingClassifier and StackingClassifier from the best tuned models.
Logs everything to MLflow.
"""
from __future__ import annotations

import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.evaluation.metrics import compute_metrics, get_classification_report
from src.preprocessing.preprocess import StandardPreprocessor
from src.training.trainer import (
    _log_system_tags,
    _log_test_metrics,
    _save_and_log_model,
)
from src.utils.logger import get_logger
from src.utils.system_metrics import RuntimeTimer

log = get_logger("ensemble")


def _get_sklearn_estimators(
    stage2_results: dict, member_names: list[str]
) -> list[tuple[str, object]]:
    """
    Extract UNFITTED clones of tuned sklearn pipelines for ensemble members.

    We clone (not reuse) so that Voting/StackingClassifier can refit them
    internally on the correct CV folds.  CatBoost and TabNet are excluded
    because they have incompatible fit interfaces with sklearn meta-estimators.
    """
    from sklearn.base import clone

    estimators = []
    for name in member_names:
        result = stage2_results.get(name)
        if result is None:
            log.warning(f"  Skipping {name} — no Stage 2 result found.")
            continue
        rtype = result.get("type", "sklearn")
        if rtype == "sklearn":
            # Use best_params from tuning to rebuild a fresh (unfitted) pipeline
            # so the ensemble can refit it on its own CV splits
            tuned_pipeline = result["pipeline"]
            estimators.append((name, clone(tuned_pipeline)))
            log.info(f"  ✓ Added tuned {name} to ensemble")
        elif rtype in ("catboost", "tabnet"):
            log.warning(
                f"  {name} ({rtype}) skipped — incompatible with sklearn "
                f"VotingClassifier/StackingClassifier fit interface."
            )
        else:
            log.warning(f"  Unknown type for {name}, skipping.")
    return estimators


def run_ensemble(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    stage2_results: dict,
    cfg: DictConfig,
    experiment_id: str,
) -> dict[str, float]:
    """Build and evaluate Voting + Stacking classifiers."""
    log.info("═" * 60)
    log.info("ENSEMBLE — Voting & Stacking")
    log.info("═" * 60)

    ensemble_results: dict[str, float] = {}

    # ── Voting Classifier ─────────────────────────────────────────────────────
    voting_members = list(cfg.models.ensemble.voting_members)
    voting_estimators = _get_sklearn_estimators(stage2_results, voting_members)

    if len(voting_estimators) >= 2:
        timer = RuntimeTimer().start()
        with mlflow.start_run(run_name="ensemble__voting", experiment_id=experiment_id) as run:
            _log_system_tags(run)
            mlflow.set_tag("stage", "ensemble")
            mlflow.set_tag("ensemble_type", "voting")
            mlflow.set_tag("members", str([e[0] for e in voting_estimators]))
            mlflow.log_param("voting", "soft")
            mlflow.log_param("n_members", len(voting_estimators))

            voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft",
                                          n_jobs=-1)

            # CV
            cv = StratifiedKFold(n_splits=cfg.pipeline.cv_folds, shuffle=True,
                                 random_state=cfg.pipeline.random_state)
            cv_results = cross_validate(voting_clf, X_train, y_train, cv=cv,
                                        scoring="roc_auc",
                                        n_jobs=cfg.pipeline.n_jobs)
            cv_auc_mean = float(cv_results["test_score"].mean())
            mlflow.log_metric("cv_roc_auc_mean", cv_auc_mean)

            voting_clf.fit(X_train, y_train)
            y_pred = voting_clf.predict(X_test)
            y_proba = voting_clf.predict_proba(X_test)[:, 1]

            report = get_classification_report(y_test, y_pred)
            report_dir = str(Path(cfg.data.get("model_output_dir", "models/titanic")).parent.parent / "reports" / "titanic")
            _log_test_metrics(y_test, y_pred, y_proba, report, report_dir=report_dir)

            elapsed = timer.elapsed()
            mlflow.log_metric("training_time_sec", elapsed)
            _save_and_log_model(voting_clf, "voting_classifier", cfg)

            test_auc = compute_metrics(y_test, y_pred, y_proba).get("roc_auc", 0.0)
            ensemble_results["voting"] = test_auc
            log.info(f"  ✓ VotingClassifier | AUC={test_auc:.4f} | {elapsed:.1f}s")
            log.info(f"\n{report}")
    else:
        log.warning("Not enough compatible members for VotingClassifier (need ≥ 2).")

    # ── Stacking Classifier ───────────────────────────────────────────────────
    stacking_members = list(cfg.models.ensemble.stacking_members)
    stacking_estimators = _get_sklearn_estimators(stage2_results, stacking_members)

    if len(stacking_estimators) >= 2:
        meta_name = cfg.models.ensemble.stacking_meta_learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=cfg.pipeline.random_state)

        timer = RuntimeTimer().start()
        with mlflow.start_run(run_name="ensemble__stacking", experiment_id=experiment_id) as run:
            _log_system_tags(run)
            mlflow.set_tag("stage", "ensemble")
            mlflow.set_tag("ensemble_type", "stacking")
            mlflow.set_tag("members", str([e[0] for e in stacking_estimators]))
            mlflow.set_tag("meta_learner", meta_name)
            mlflow.log_param("n_members", len(stacking_estimators))

            stacking_clf = StackingClassifier(
                estimators=stacking_estimators,
                final_estimator=meta_learner,
                cv=cfg.pipeline.cv_folds,
                n_jobs=-1,
                passthrough=False,
            )

            # CV
            cv = StratifiedKFold(n_splits=cfg.pipeline.cv_folds, shuffle=True,
                                 random_state=cfg.pipeline.random_state)
            cv_results = cross_validate(stacking_clf, X_train, y_train, cv=cv,
                                        scoring="roc_auc",
                                        n_jobs=cfg.pipeline.n_jobs)
            cv_auc_mean = float(cv_results["test_score"].mean())
            mlflow.log_metric("cv_roc_auc_mean", cv_auc_mean)

            stacking_clf.fit(X_train, y_train)
            y_pred = stacking_clf.predict(X_test)
            y_proba = stacking_clf.predict_proba(X_test)[:, 1]

            report = get_classification_report(y_test, y_pred)
            report_dir = str(Path(cfg.data.get("model_output_dir", "models/titanic")).parent.parent / "reports" / "titanic")
            _log_test_metrics(y_test, y_pred, y_proba, report, report_dir=report_dir)

            elapsed = timer.elapsed()
            mlflow.log_metric("training_time_sec", elapsed)
            _save_and_log_model(stacking_clf, "stacking_classifier", cfg)

            test_auc = compute_metrics(y_test, y_pred, y_proba).get("roc_auc", 0.0)
            ensemble_results["stacking"] = test_auc
            log.info(f"  ✓ StackingClassifier | AUC={test_auc:.4f} | {elapsed:.1f}s")
            log.info(f"\n{report}")
    else:
        log.warning("Not enough compatible members for StackingClassifier (need ≥ 2).")

    return ensemble_results
