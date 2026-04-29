"""Standard experiment tracker wrapper.

Enforces the required logging list from standards/experimentation.md.
Teams use this instead of calling mlflow directly — it ensures nothing
required gets accidentally skipped.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlflow


@dataclass
class RunConfig:
    experiment_name: str
    run_name: str
    owner: str
    use_case: str
    data_contract: str  # "contract_name:version", e.g. "churn_features_v1:1.0"
    team: str = ""


@contextmanager
def tracked_run(config: RunConfig):
    """Context manager that starts an MLflow run and enforces required tags.

    Usage:
        with tracked_run(run_config) as run:
            mlflow.log_params(...)
            mlflow.log_metrics(...)
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id
    """
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.set_tags({
            "owner": config.owner,
            "use_case": config.use_case,
            "data_contract": config.data_contract,
            "team": config.team,
            "lifecycle_stage": "experimental",
        })
        yield run


def log_feature_names(feature_names: List[str]) -> None:
    mlflow.set_tag("feature_names", ",".join(feature_names))


def log_data_shape(n_train: int, n_test: int) -> None:
    mlflow.log_params({"train_rows": n_train, "test_rows": n_test})


def get_run_uri(run_id: str, artifact_path: str = "model") -> str:
    return f"runs:/{run_id}/{artifact_path}"
