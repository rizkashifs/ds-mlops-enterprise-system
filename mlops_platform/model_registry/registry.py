"""Model registry wrapper.

Enforces lifecycle rules when promoting models in the MLflow model registry.
Teams use this instead of calling MlflowClient directly.
"""
from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class RegistrationResult:
    model_name: str
    version: str
    stage: str
    run_id: str


def register_model(
    run_id: str,
    model_name: str,
    description: str = "",
) -> RegistrationResult:
    """Register a trained model in the MLflow model registry at Staging stage."""
    client = MlflowClient()

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass  # Already exists

    version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        description=description,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Staging",
    )

    return RegistrationResult(
        model_name=model_name,
        version=version.version,
        stage="Staging",
        run_id=run_id,
    )


def promote_to_production(model_name: str, version: str) -> None:
    """Promote a model version from Staging to Production.

    Automatically archives the previous Production version.
    Only call this after the pre-deployment checklist is complete.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )


def archive_model(model_name: str, version: str, reason: str = "") -> None:
    """Archive a model version (e.g., after rollback or retirement)."""
    client = MlflowClient()
    if reason:
        client.update_model_version(name=model_name, version=version, description=reason)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived",
    )


def get_production_uri(model_name: str) -> str:
    """Return the MLflow URI for the current Production version of a model."""
    return f"models:/{model_name}/Production"
