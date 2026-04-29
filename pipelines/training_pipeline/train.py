"""Training pipeline orchestrator.

This is the entry point for running a training job. It:
  1. Loads config from configs/training.yaml
  2. Loads and validates data against the data contract
  3. Trains the model and logs to MLflow
  4. Validates the trained model against thresholds
  5. Prints a summary

Run with:
  python pipelines/training_pipeline/train.py

Or import and call run_training_pipeline() for integration with a scheduler.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.contracts import DataContract
from src.core.lifecycle import ModelStatus, transition
from src.pipelines.training import TrainingConfig, train_model
from src.pipelines.validation import ValidationThresholds, validate_model


def load_config(path: str = "configs/training.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_training_pipeline(df: pd.DataFrame, config_path: str = "configs/training.yaml") -> dict:
    """Full training pipeline: validate → train → validate metrics → return result.

    Args:
        df: Input DataFrame conforming to the configured data contract.
        config_path: Path to the training config YAML.

    Returns:
        dict with keys: run_id, metrics, model_uri, validation_passed, status
    """
    cfg = load_config(config_path)

    # --- Step 1: Validate row count ---
    min_rows = cfg["data"].get("expected_min_rows", 0)
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows; expected at least {min_rows}")

    # --- Step 2: Train ---
    train_cfg = TrainingConfig(
        experiment_name=cfg["experiment"]["name"],
        model_params=cfg["model"]["params"],
        target_column=cfg["model"]["target_column"],
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_seed"],
    )
    result = train_model(df, train_cfg)
    print(f"Training complete: run_id={result.run_id}")
    print(f"  Metrics: {result.metrics}")

    # --- Step 3: Validate metrics ---
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
        print(f"\nModel promoted to CANDIDATE. Ready for review.")
    else:
        print(f"\nModel remains EXPERIMENTAL. Fix failures before promoting.")

    return {
        "run_id": result.run_id,
        "metrics": result.metrics,
        "model_uri": result.model_uri,
        "validation_passed": validation.passed,
        "status": status.value,
    }


if __name__ == "__main__":
    # Quick smoke test with synthetic data — replace with real data loading
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
    print(f"\nFinal result: {result}")
