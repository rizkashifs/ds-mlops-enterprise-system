"""Inference pipeline orchestrator.

Loads the production model, scores a DataFrame, runs monitoring checks,
evaluates retraining triggers, and writes output.
Config-driven — all settings come from configs/inference.yaml.

Run with:
  python pipelines/inference_pipeline/score.py

Or import and call run_inference_pipeline() for integration with a scheduler.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers
from mlops_platform.model_registry.registry import get_production_uri
from src.services.scoring import score_batch


def load_config(path: str = "configs/inference.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_inference_pipeline(
    df: pd.DataFrame,
    model_uri: str = None,
    config_path: str = "configs/inference.yaml",
    days_since_last_retrain: int = None,
    baseline_mean_score: float = None,
    baseline_metrics: dict = None,
    current_metrics: dict = None,
) -> dict:
    """Load model, score DataFrame, run monitoring, evaluate retrain triggers.

    Args:
        df: Input features (must not contain the target column).
        model_uri: If provided, overrides the registry lookup from config.
        config_path: Path to the inference config YAML.
        days_since_last_retrain: Days since last retrain (for time-based trigger).
        baseline_mean_score: Mean score at training time (for score shift trigger).
        baseline_metrics: Validation metrics at training time (for perf drop trigger).
        current_metrics: Windowed metrics from recent actuals (for perf drop trigger).

    Returns:
        dict with keys: scores_df, monitoring_report, trigger_decision, num_records
    """
    cfg = load_config(config_path)

    # --- Step 1: Validate input ---
    exclude = cfg["data"].get("exclude_columns", [])
    df = df.drop(columns=[c for c in exclude if c in df.columns])

    if df.empty:
        raise ValueError("Input DataFrame is empty — scoring aborted")

    # --- Step 2: Load model ---
    uri = model_uri or get_production_uri(cfg["model"]["registry_name"])

    # --- Step 3: Score ---
    result = score_batch(df, uri)
    print(f"Scored {result.num_records:,} records at {result.scored_at}")

    # --- Step 4: Monitoring ---
    report = build_monitoring_report(
        model_name=cfg["model"]["registry_name"],
        scores=result.probabilities,
        psi_alert_threshold=cfg["monitoring"]["psi_alert_threshold"],
    )
    print(f"  mean_score={report.mean_score:.4f}  p90={report.p90:.4f}")

    if report.has_alerts():
        print("MONITORING ALERTS:")
        for alert in report.alerts:
            print(f"  {alert}")

    # --- Step 5: Evaluate retraining triggers ---
    trigger_cfg = TriggerConfig(
        psi_alert_threshold=cfg["monitoring"]["psi_alert_threshold"],
    )
    trigger = evaluate_triggers(
        monitoring_report=report,
        config=trigger_cfg,
        days_since_last_retrain=days_since_last_retrain,
        baseline_mean_score=baseline_mean_score,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
    )

    if trigger.should_retrain:
        urgency_label = f"[{trigger.urgency.upper()}]"
        print(f"\nRETRAIN {urgency_label} triggered by: {trigger.triggered_by}")
        for reason in trigger.reasons:
            print(f"  {reason}")
        # In production: route to alert system or retrain queue based on urgency
        # if trigger.urgency == "immediate":
        #     send_alert(...)
        # else:
        #     queue_retrain(...)

    return {
        "scores_df": result.to_dataframe(),
        "monitoring_report": report,
        "trigger_decision": trigger,
        "num_records": result.num_records,
    }

