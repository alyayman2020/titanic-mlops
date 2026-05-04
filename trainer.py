"""
trainer.py — main pipeline entry point.

Usage:
    python trainer.py                              # run everything (default config)
    python trainer.py pipeline.run_stage2=false   # Stage 1 only
    python trainer.py optuna.n_trials=50          # override Optuna trials
    python trainer.py models.stage1=[xgboost,lightgbm]  # run specific models only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

# Load .env before anything else
load_dotenv()

from src.ensemble.ensemble import run_ensemble
from src.training.registry import register_best_model
from src.training.trainer import load_data, run_stage1, run_stage2
from src.utils.logger import get_logger
from src.utils.system_metrics import get_system_info

console = Console()
log = get_logger("main")


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty results table
# ══════════════════════════════════════════════════════════════════════════════

def _print_results_table(
    stage1: dict[str, float],
    stage2: dict[str, dict],
    ensemble: dict[str, float],
):
    table = Table(title="🏆 Pipeline Results Summary", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Stage 1 AUC", justify="right")
    table.add_column("Stage 2 Best CV AUC", justify="right")
    table.add_column("Stage 2 Test AUC", justify="right")

    all_models = set(stage1.keys()) | set(stage2.keys())
    for m in sorted(all_models):
        s1 = f"{stage1.get(m, 0.0):.4f}" if m in stage1 else "—"
        s2_cv = f"{stage2[m]['best_score']:.4f}" if m in stage2 else "—"
        # test AUC was logged to MLflow; we show CV here as proxy
        s2_test = "→ MLflow" if m in stage2 else "—"
        table.add_row(m, s1, s2_cv, s2_test)

    console.print(table)

    if ensemble:
        etable = Table(title="🔗 Ensemble Results", show_lines=True)
        etable.add_column("Ensemble", style="magenta")
        etable.add_column("Test AUC", justify="right")
        for ename, auc in ensemble.items():
            etable.add_row(ename, f"{auc:.4f}")
        console.print(etable)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ── 0. Load model configs separately (avoid Hydra struct restriction) ──
    from omegaconf import OmegaConf
    # Use original working directory (Hydra changes cwd to outputs/)
    from hydra.utils import get_original_cwd
    orig_cwd = Path(get_original_cwd())
    models_cfg_path = orig_cwd / "conf" / "model" / "models.yaml"
    models_cfg = OmegaConf.load(models_cfg_path)
    # Store as a plain Python dict on cfg using OmegaConf structured config
    # bypass struct mode by converting to a new unstructured container
    cfg = OmegaConf.merge(
        OmegaConf.to_container(cfg, resolve=True),
        {"models_cfg": OmegaConf.to_container(models_cfg, resolve=True)},
    )
    cfg = OmegaConf.create(cfg)

    # ── 1. Set MLflow tracking ────────────────────────────────────
    tracking_uri_cfg = cfg.mlflow.tracking_uri
    dagshub_token    = os.getenv("DAGSHUB_TOKEN", "")
    dagshub_user     = os.getenv("DAGSHUB_USERNAME", "")
    dagshub_repo     = os.getenv("DAGSHUB_REPO", "titanic-mlops")

    if tracking_uri_cfg.startswith("https://dagshub.com"):
        # ── DagsHub remote tracking ───────────────────────────────
        try:
            import dagshub
            dagshub.auth.add_app_token(token=dagshub_token)
            dagshub.init(
                repo_owner=dagshub_user,
                repo_name=dagshub_repo,
                mlflow=True,
            )
            mlflow_uri = tracking_uri_cfg
            log.info(f"DagsHub MLflow tracking enabled ✓")
        except Exception as e:
            log.warning(f"DagsHub init failed ({e}) — falling back to local SQLite")
            db_abs = (orig_cwd / "mlflow.db").resolve()
            mlflow_uri = f"sqlite:///{db_abs}"
    elif tracking_uri_cfg.startswith("sqlite:///"):
        # ── Local SQLite ──────────────────────────────────────────
        db_filename = tracking_uri_cfg.replace("sqlite:///", "")
        db_abs = (orig_cwd / db_filename).resolve()
        mlflow_uri = f"sqlite:///{db_abs}"
    else:
        # ── Plain folder (mlruns) — file:/// for Windows ──────────
        mlflow_path = (orig_cwd / tracking_uri_cfg).resolve()
        mlflow_uri = mlflow_path.as_uri()

    mlflow.set_tracking_uri(mlflow_uri)
    log.info(f"MLflow URI: {mlflow_uri}")
    # Patch data paths to be absolute
    cfg.data.raw_train  = str(orig_cwd / cfg.data.raw_train)
    cfg.data.raw_test   = str(orig_cwd / cfg.data.raw_test)
    cfg.data.processed_dir = str(orig_cwd / cfg.data.processed_dir)
    cfg.data.interim_dir   = str(orig_cwd / cfg.data.interim_dir)
    cfg.data["model_output_dir"] = str(orig_cwd / "models" / "titanic")
    experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)
    experiment_id = experiment.experiment_id

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║        TITANIC MLOps PIPELINE — Lab 3               ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"MLflow experiment : {cfg.mlflow.experiment_name}")
    log.info(f"Tracking URI      : {cfg.mlflow.tracking_uri}")

    # ── 2. System info ────────────────────────────────────────────
    sysinfo = get_system_info()
    log.info("System info:")
    for k, v in sysinfo.items():
        log.info(f"  {k}: {v}")

    # ── 3. Load & split data ──────────────────────────────────────
    log.info("Loading data…")
    X, y = load_data(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.split.test_size,
        random_state=cfg.data.split.random_state,
        stratify=y if cfg.data.split.stratify else None,
    )
    log.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Save interim splits
    interim_dir = Path(cfg.data.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(Survived=y_train.values).to_csv(interim_dir / "train_split.csv", index=False)
    X_test.assign(Survived=y_test.values).to_csv(interim_dir / "test_split.csv", index=False)

    # ── 4. Stage 1 — Base models ──────────────────────────────────
    stage1_results: dict[str, float] = {}
    if cfg.pipeline.run_stage1:
        stage1_results = run_stage1(X_train, X_test, y_train, y_test, cfg, experiment_id)
    else:
        log.info("Stage 1 skipped (pipeline.run_stage1=false)")

    # ── 5. Stage 2 — Optuna tuning ────────────────────────────────
    stage2_results: dict[str, dict] = {}
    if cfg.pipeline.run_stage2:
        stage2_results = run_stage2(X_train, X_test, y_train, y_test, cfg, experiment_id)
    else:
        log.info("Stage 2 skipped (pipeline.run_stage2=false)")

    # ── 6. Ensemble ───────────────────────────────────────────────
    ensemble_results: dict[str, float] = {}
    if cfg.pipeline.run_ensemble and stage2_results:
        ensemble_results = run_ensemble(
            X_train, X_test, y_train, y_test,
            stage2_results, cfg, experiment_id,
        )
    else:
        log.info("Ensemble skipped.")

    # ── 7. Model registration ─────────────────────────────────────
    if cfg.mlflow.register_best_model and (stage2_results or stage1_results):
        register_best_model(
            experiment_id=experiment_id,
            model_registry_name=cfg.mlflow.model_name,
            metric="test_roc_auc",
        )

    # ── 8. Summary table ──────────────────────────────────────────
    _print_results_table(stage1_results, stage2_results, ensemble_results)
    log.info("Pipeline complete ✓")


if __name__ == "__main__":
    main()
