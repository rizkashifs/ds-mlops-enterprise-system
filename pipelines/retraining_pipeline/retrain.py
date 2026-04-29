"""Retraining pipeline orchestrator.

Retraining follows the SAME process as the initial training pipeline.
There are no shortcuts. The retrained model must:
  - Pass the validation gate
  - Outperform (or match within noise) the currently deployed model
  - Be promoted through CANDIDATE → APPROVED before deployment

Run with:
  python pipelines/retraining_pipeline/retrain.py

Or import and call run_retraining_pipeline() for integration with a scheduler or alert handler.
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.training_pipeline.train import run_training_pipeline
from src.pipelines.validation import ValidationThresholds, validate_model


def compare_against_production(
    new_metrics: dict,
    production_metrics: dict,
    tolerance: float = 0.02,
) -> tuple[bool, str]:
    """Check whether the retrained model is at least as good as production.

    Returns (passes, reason_string).
    Allows tolerance-point degradation to account for statistical noise.
    """
    failures = []
    for metric, prod_value in production_metrics.items():
        new_value = new_metrics.get(metric, 0.0)
        if new_value < prod_value - tolerance:
            failures.append(
                f"{metric}: retrained={new_value:.4f} vs production={prod_value:.4f} "
                f"(delta={new_value - prod_value:+.4f}, tolerance={tolerance})"
            )
    if failures:
        return False, "Retrained model does not match production:\n" + "\n".join(failures)
    return True, "Retrained model matches or exceeds production performance"


def run_retraining_pipeline(
    df: pd.DataFrame,
    production_metrics: Optional[dict] = None,
    config_path: str = "configs/training.yaml",
) -> dict:
    """Run the full retraining pipeline and compare against production.

    Args:
        df: Fresh training data conforming to the data contract.
        production_metrics: Metrics of the currently deployed model for comparison.
                            If None, production comparison step is skipped with a warning.
        config_path: Path to the training config YAML.

    Returns:
        dict with training result + comparison result
    """
    print("=" * 60)
    print("Retraining Pipeline")
    print("=" * 60)

    # --- Step 1: Train + validate (same as initial training) ---
    result = run_training_pipeline(df, config_path)

    if not result["validation_passed"]:
        print("\nRetrained model failed validation gate. Staying in EXPERIMENTAL.")
        return {**result, "production_comparison_passed": False, "comparison_reason": "Failed validation gate"}

    # --- Step 2: Compare against production ---
    if production_metrics is None:
        print("\nWARNING: No production metrics provided. Skipping production comparison.")
        print("A human reviewer MUST manually compare before promoting this model.")
        return {**result, "production_comparison_passed": None, "comparison_reason": "Skipped — no baseline"}

    passed, reason = compare_against_production(result["metrics"], production_metrics)
    print(f"\nProduction comparison: {'PASSED' if passed else 'FAILED'}")
    print(f"  {reason}")

    if not passed:
        print("\nRetrained model did not improve on production. Do not promote.")

    return {**result, "production_comparison_passed": passed, "comparison_reason": reason}
