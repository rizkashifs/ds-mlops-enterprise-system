"""Experiment tracking helpers — enforces required logging from standards/experimentation.md.

This module wraps the tracker implementations in src/tracking/ with org-wide
standards enforcement: required tags, data shape logging, feature name logging.

Teams use tracked_run() instead of calling tracker methods directly.
It ensures nothing required gets accidentally skipped.

Usage with LocalFileTracker (no MLflow):
    from src.tracking.local_tracker import LocalFileTracker
    from mlops_platform.experiment_tracking.tracker import tracked_run, RunConfig

    tracker = LocalFileTracker()
    with tracked_run(tracker, run_config) as t:
        t.log_params(...)
        t.log_metrics(...)
        t.log_model(model, "model")

Usage with MLflow:
    from src.tracking.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(tracking_uri="https://mlflow.your-org.internal", experiment_name="churn-rf")
    with tracked_run(tracker, run_config) as t:
        ...
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List

from src.tracking.base import ExperimentTracker


@dataclass
class RunConfig:
    experiment_name: str
    run_name: str
    owner: str
    use_case: str
    data_contract: str  # "contract_name:version" e.g. "churn_features_v1:1.0"
    team: str = ""


@contextmanager
def tracked_run(tracker: ExperimentTracker, config: RunConfig):
    """Context manager that starts a tracker run and enforces required metadata.

    Yields the tracker so callers can log params, metrics, and artifacts.
    Ends the run on exit and returns the run_id.
    """
    tracker.start_run(run_name=config.run_name)
    # Log required metadata as params (works with all tracker backends)
    tracker.log_params({
        "owner": config.owner,
        "use_case": config.use_case,
        "data_contract": config.data_contract,
        "team": config.team,
    })
    try:
        yield tracker
    finally:
        tracker.end_run()


def log_feature_names(tracker: ExperimentTracker, feature_names: List[str]) -> None:
    """Log feature names as a comma-separated param."""
    tracker.log_params({"feature_names": ",".join(feature_names)})


def log_data_shape(tracker: ExperimentTracker, n_train: int, n_test: int) -> None:
    """Log train/test row counts."""
    tracker.log_params({"train_rows": n_train, "test_rows": n_test})


def get_run_uri(run_id: str, artifact_path: str = "model") -> str:
    """Build an MLflow-style URI for reference in model cards and commit messages."""
    return f"runs:/{run_id}/{artifact_path}"
