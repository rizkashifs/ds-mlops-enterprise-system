"""Training pipeline orchestrator.

Loads config, selects a tracking backend, trains the model, validates metrics.

Tracker backend is selected via MLOPS_TRACKER environment variable (or config):
  MLOPS_TRACKER=local    → LocalFileTracker (default, no dependencies)
  MLOPS_TRACKER=mlflow   → MLflowTracker (requires mlflow + MLOPS_TRACKING_URI)
  MLOPS_TRACKER=none     → NoOpTracker

Run with:
  python pipelines/training_pipeline/train.py

Or import and call run_training_pipeline() for integration with a scheduler.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.lifecycle import ModelStatus, transition
from src.pipelines.training import TrainingConfig, TrainingResult, train_model
from src.pipelines.validation import ValidationThresholds, validate_model


def load_config(path: str = "configs/training.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_tracker(cfg: dict):
    """Select and configure a tracker based on MLOPS_TRACKER env var or config."""
    backend = os.environ.get(
        "MLOPS_TRACKER",
        cfg.get("tracker", {}).get("backend", "local"),
    )
    if backend == "mlflow":
        from src.tracking.mlflow_tracker import MLflowTracker
        tracking_uri = os.environ.get(
            "MLOPS_TRACKING_URI",
            cfg["experiment"].get("tracking_uri", "mlruns"),
        )
        return MLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=cfg["experiment"]["name"],
        )
    if backend == "none":
        from src.tracking.noop_tracker import NoOpTracker
        return NoOpTracker()
    # Default: local file tracker
    from src.tracking.local_tracker import LocalFileTracker
    store = cfg.get("artifacts", {}).get("store", "artifacts/runs")
    return LocalFileTracker(base_dir=store)


def run_training_pipeline(
    df: pd.DataFrame,
    config_path: str = "configs/training.yaml",
) -> dict:
    """Full training pipeline: select tracker → train → validate metrics → return result.

    Returns:
        dict with keys: run_id, metrics, model_uri, model, validation_passed, status
    """
    cfg = load_config(config_path)

    min_rows = cfg["data"].get("expected_min_rows", 0)
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows; expected at least {min_rows}")

    tracker = _build_tracker(cfg)

    train_cfg = TrainingConfig(
        experiment_name=cfg["experiment"]["name"],
        model_params=cfg["model"]["params"],
        target_column=cfg["model"]["target_column"],
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_seed"],
    )
    result = train_model(df, train_cfg, tracker=tracker)
    print(f"Training complete: run_id={result.run_id or '(no tracker)'}")
    print(f"  Metrics: {result.metrics}")
    if result.model_uri:
        print(f"  Saved to: {result.model_uri}")

    val_cfg = cfg["validation"]["thresholds"]
    thresholds = ValidationThresholds(
        min_accuracy=val_cfg["min_accuracy"],
        min_f1=val_cfg["min_f1"],
        min_roc_auc=val_cfg["min_roc_auc"],
    )
    validation = validate_model(result.metrics, thresholds)
    print(f"\n{validation.summary()}")

    status = ModelStatus.EXPERIMENTAL
    if validation.passed:
        status = transition(status, ModelStatus.CANDIDATE)
        print("Model promoted to CANDIDATE. Ready for review.")
    else:
        print("Model remains EXPERIMENTAL. Fix failures before promoting.")

    return {
        "run_id": result.run_id,
        "metrics": result.metrics,
        "model_uri": result.model_uri,
        "model": result.model,
        "validation_passed": validation.passed,
        "status": status.value,
    }


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "num_products": rng.integers(1, 5, n),
        "support_calls_90d": rng.integers(0, 10, n),
    })
    df["target"] = ((df["support_calls_90d"] > 5) | (df["tenure_months"] < 6)).astype(int)

    result = run_training_pipeline(df)
    print(f"\nFinal result: status={result['status']}, passed={result['validation_passed']}")
