"""
Model factory.

build_model(name, params, cfg) → estimator (NOT wrapped in a Pipeline;
the caller decides whether to attach a preprocessor).
"""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


# ── GPU helper ────────────────────────────────────────────────────────────────

def _use_gpu(model_cfg: DictConfig, global_cfg: DictConfig) -> bool:
    return bool(
        model_cfg.get("uses_gpu", False)
        and global_cfg.get("gpu", {}).get("use_gpu", False)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    name: str,
    params: dict[str, Any],
    cfg: DictConfig,
) -> Any:
    """
    Instantiate a classifier by name with given params.

    Parameters
    ----------
    name   : model key matching conf/model/models.yaml
    params : hyperparameter dict (base_params or Optuna-sampled)
    cfg    : full Hydra config (for GPU flag, random_state, etc.)
    """
    model_cfg = cfg.models_cfg[name]  # injected by trainer
    gpu = _use_gpu(model_cfg, cfg)
    rs = cfg.pipeline.random_state

    # Strip keys we always inject explicitly to avoid "multiple values" errors
    # Strip keys we always inject explicitly — covers ALL models to avoid duplicate kwargs
    ALWAYS_INJECTED = {
        "random_state", "random_seed",          # injected as rs
        "verbose", "verbosity",                  # injected per model
        "n_jobs",                                # always -1
        "device", "task_type",                   # GPU flags
        "use_label_encoder", "eval_metric",      # XGBoost
        "probability",                           # SVM
        "feature_name",                          # LightGBM (Dataset-level, not constructor)
        "algorithm",                             # AdaBoost (removed in sklearn 1.6+)
        "early_stopping", "validation_fraction", # MLP (set in base_params only)
    }
    params = {k: v for k, v in params.items() if k not in ALWAYS_INJECTED}

    # ── Logistic Regression ───────────────────────────────────────
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=rs, **params)

    # ── Random Forest ─────────────────────────────────────────────
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=rs, n_jobs=-1, **params)

    # ── XGBoost ───────────────────────────────────────────────────
    elif name == "xgboost":
        from xgboost import XGBClassifier
        device = "cuda" if gpu else "cpu"
        # Remove keys that XGBoost doesn't accept as __init__ params
        p = {k: v for k, v in params.items()
             if k not in ("use_label_encoder", "eval_metric")}
        return XGBClassifier(
            device=device,
            eval_metric="logloss",
            random_state=rs,
            verbosity=0,
            **p,
        )

    # ── LightGBM ──────────────────────────────────────────────────
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        device_type = "gpu" if gpu else "cpu"
        return LGBMClassifier(
            device=device_type,
            random_state=rs,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

    # ── CatBoost ──────────────────────────────────────────────────
    elif name == "catboost":
        from catboost import CatBoostClassifier
        task_type = "GPU" if gpu else "CPU"
        p = {k: v for k, v in params.items() if k != "random_state"}
        return CatBoostClassifier(
            task_type=task_type,
            random_seed=rs,
            verbose=0,
            **p,
        )

    # ── AdaBoost ──────────────────────────────────────────────────
    elif name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=rs, **params)

    # ── KNN ───────────────────────────────────────────────────────
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        p = dict(params)
        # p param only valid for minkowski metric — remove it otherwise
        if p.get("metric", "minkowski") != "minkowski" and "p" in p:
            del p["p"]
        return KNeighborsClassifier(n_jobs=-1, **p)

    # ── SVM ───────────────────────────────────────────────────────
    elif name == "svm":
        from sklearn.svm import SVC
        p = {k: v for k, v in params.items() if k != "random_state"}
        return SVC(probability=True, random_state=rs, **p)

    # ── MLP ───────────────────────────────────────────────────────
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        p = dict(params)
        # hidden_layer_sizes may come as list from yaml
        if "hidden_layer_sizes" in p and isinstance(p["hidden_layer_sizes"], list):
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        # early_stopping and validation_fraction are in ALWAYS_INJECTED (to avoid
        # duplicates in tuning), so add them back here explicitly for MLP
        return MLPClassifier(
            random_state=rs,
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=500,
            **p,
        )

    # ── TabNet ────────────────────────────────────────────────────
    elif name == "tabnet":
        # Auto-install PyTorch + TabNet if not present (e.g. fresh env)
        try:
            import torch  # noqa: F401
        except ImportError:
            import subprocess
            import sys
            print("\n[TabNet] PyTorch not found — installing with CUDA 12.1 support…")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                ])
                print("[TabNet] PyTorch (CUDA 12.1) installed ✓")
            except subprocess.CalledProcessError:
                print("[TabNet] CUDA wheel failed — falling back to CPU-only PyTorch…")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "torch", "torchvision",
                ])
                print("[TabNet] PyTorch (CPU) installed ✓")
        try:
            import pytorch_tabnet  # noqa: F401
        except ImportError:
            import subprocess
            import sys
            print("[TabNet] pytorch-tabnet not found — installing…")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet", "pytorch-tabnet",
            ])
            print("[TabNet] pytorch-tabnet installed ✓")
        from pytorch_tabnet.tab_model import TabNetClassifier
        device_name = "cuda" if gpu else "cpu"
        # Constructor params only — optimizer_lr goes here as optimizer_params
        init_keys = {"n_d", "n_a", "n_steps", "gamma", "n_independent",
                     "n_shared", "momentum", "mask_type", "seed"}
        init_params = {k: v for k, v in params.items() if k in init_keys}
        lr = params.get("optimizer_lr", 0.02)
        return TabNetClassifier(
            device_name=device_name,
            verbose=0,
            optimizer_params={"lr": lr},
            **init_params,
        )

    else:
        raise ValueError(f"Unknown model name: '{name}'")


def get_tabnet_fit_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Extract TabNet fit() kwargs from the model params dict.

    TabNet fit() accepts: max_epochs, patience, batch_size, virtual_batch_size.
    optimizer_lr goes into the constructor as optimizer_params — NOT fit().
    """
    fit_keys = {"max_epochs", "patience", "batch_size", "virtual_batch_size"}
    return {k: v for k, v in params.items() if k in fit_keys}
