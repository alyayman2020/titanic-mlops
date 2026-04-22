"""trainer.py — Titanic MLOps Lab 0 — Main Entry Point.

Orchestrates the full training pipeline:
    1. Load params.yaml
    2. Load and split raw Titanic data
    3. Cross-validate RandomForest + AdaBoost pipelines
    4. Fit both pipelines on the training set
    5. Evaluate on held-out test set
    6. Save model artifacts to models/titanic/

Usage
-----
    python trainer.py
    python trainer.py --params params.yaml
"""

import argparse

import yaml

from src.logger import get_logger
from src.training.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Titanic MLOps - Lab 0 Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Path to the YAML parameters file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger(log_file="logs/training.log")

    logger.info(f"Loading params from: {args.params}")
    with open(args.params) as f:
        params = yaml.safe_load(f)

    run_training(params=params, logger=logger)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
