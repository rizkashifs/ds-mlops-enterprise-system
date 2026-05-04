"""Inference pipeline orchestrator.

Loads a model (from a local path, MLflow URI, or in-memory object), scores a
DataFrame, runs monitoring checks, and evaluates retraining triggers.

Model source priority:
  1. model_uri argument (explicit override)
  2. configs/inference.yaml → model.uri (direct path or MLflow URI)
  3. configs/inference.yaml → model.registry_name (MLflow registry lookup)

Run with:
  python pipelines/inference_pipeline/score.py

Or import and call run_inference_pipeline() from a scheduler or trigger handler.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers
from src.services.scoring import score_batch


def load_config(path: str = "configs/inference.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_model_uri(cfg: dict, model_uri_override: str = None) -> str:
    """Resolve model URI from explicit arg, config direct URI, or MLflow registry."""
    if model_uri_override:
        return model_uri_override
    model_cfg = cfg.get("model", {})
    if model_cfg.get("uri"):
        return model_cfg["uri"]
    # Fall back to MLflow registry lookup
    registry_name = model_cfg.get("registry_name")
    if registry_name:
        try:
            from mlops_platform.model_registry.registry import get_production_uri
            return get_production_uri(registry_name)
        except ImportError:
            raise ValueError(
                "mlflow is not installed and no model.uri is set in inference.yaml. "
                "Either install mlflow or set model.uri to a local model path."
            )
    raise ValueError("No model URI configured. Set model.uri in configs/inference.yaml.")


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

    Returns:
        dict with keys: scores_df, monitoring_report, trigger_decision, num_records
    """
    cfg = load_config(config_path)

    exclude = cfg["data"].get("exclude_columns", [])
    df = df.drop(columns=[c for c in exclude if c in df.columns])

    if df.empty:
        raise ValueError("Input DataFrame is empty — scoring aborted")

    uri = _resolve_model_uri(cfg, model_uri)
    result = score_batch(df, uri)
    print(f"Scored {result.num_records:,} records at {result.scored_at}")

    report = build_monitoring_report(
        model_name=cfg["model"].get("registry_name", "model"),
        scores=result.probabilities,
        psi_alert_threshold=cfg["monitoring"]["psi_alert_threshold"],
    )
    print(f"  mean_score={report.mean_score:.4f}  p90={report.p90:.4f}")

    if report.has_alerts():
        print("MONITORING ALERTS:")
        for alert in report.alerts:
            print(f"  {alert}")

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
        print(f"\nRETRAIN [{trigger.urgency.upper()}] triggered by: {trigger.triggered_by}")
        for reason in trigger.reasons:
            print(f"  {reason}")

    return {
        "scores_df": result.to_dataframe(),
        "monitoring_report": report,
        "trigger_decision": trigger,
        "num_records": result.num_records,
    }
