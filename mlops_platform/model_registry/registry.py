"""Model registry — two implementations, same interface.

FileRegistry   — JSON-backed, no external dependencies. Good for teams
                 without MLflow or with a custom tracking solution.

MLflow functions — wraps the MLflow model registry. Requires mlflow.
                   Use when your team is on MLflow for experiment tracking.

Both enforce the same lifecycle rules. Teams choose one based on their
infrastructure. See standards/git-and-release.md §6 for model versioning rules.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RegistrationResult:
    model_name: str
    version: str
    stage: str
    run_id: str


# ---------------------------------------------------------------------------
# File registry — no dependencies beyond stdlib + pathlib
# ---------------------------------------------------------------------------

class FileRegistry:
    """Simple JSON-backed model registry for teams not using MLflow.

    Stores model metadata in {base_dir}/registry.json.
    Stages follow the same vocabulary as MLflow: staging, production, archived.

    Usage:
        registry = FileRegistry()
        reg = registry.register_model(run_id, "churn-rf", model_uri="/path/to/model.joblib")
        registry.promote_to_production("churn-rf", reg.version)
        uri = registry.get_production_uri("churn-rf")  # → "/path/to/model.joblib"
    """

    def __init__(self, base_dir: str = "artifacts/registry"):
        self._path = Path(base_dir) / "registry.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

    def _save(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2))

    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_uri: str,
        description: str = "",
    ) -> RegistrationResult:
        data = self._load()
        versions = data.get(model_name, [])
        version = str(len(versions) + 1)
        versions.append({
            "version": version,
            "run_id": run_id,
            "model_uri": model_uri,
            "stage": "staging",
            "description": description,
        })
        data[model_name] = versions
        self._save(data)
        return RegistrationResult(model_name=model_name, version=version, stage="staging", run_id=run_id)

    def promote_to_production(self, model_name: str, version: str) -> None:
        data = self._load()
        for entry in data.get(model_name, []):
            if entry["stage"] == "production":
                entry["stage"] = "archived"
            if entry["version"] == version:
                entry["stage"] = "production"
        self._save(data)

    def archive_model(self, model_name: str, version: str, reason: str = "") -> None:
        data = self._load()
        for entry in data.get(model_name, []):
            if entry["version"] == version:
                entry["stage"] = "archived"
                if reason:
                    entry["description"] = reason
        self._save(data)

    def get_production_uri(self, model_name: str) -> Optional[str]:
        """Return the model_uri for the current production version, or None."""
        data = self._load()
        for entry in reversed(data.get(model_name, [])):
            if entry["stage"] == "production":
                return entry["model_uri"]
        return None

    def list_versions(self, model_name: str) -> list:
        return self._load().get(model_name, [])


# ---------------------------------------------------------------------------
# MLflow registry functions — require: pip install mlflow
# ---------------------------------------------------------------------------

def register_model(
    run_id: str,
    model_name: str,
    description: str = "",
) -> RegistrationResult:
    """Register a trained model in the MLflow model registry at Staging stage."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError(
            "mlflow is not installed. Use FileRegistry instead:\n"
            "  registry = FileRegistry()\n"
            "  registry.register_model(run_id, model_name, model_uri)"
        )
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass
    version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        description=description,
    )
    client.transition_model_version_stage(
        name=model_name, version=version.version, stage="Staging",
    )
    return RegistrationResult(
        model_name=model_name, version=version.version, stage="Staging", run_id=run_id,
    )


def promote_to_production(model_name: str, version: str) -> None:
    """Promote a model version from Staging to Production in MLflow."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError("mlflow is not installed. Use FileRegistry.promote_to_production().")
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True,
    )


def archive_model(model_name: str, version: str, reason: str = "") -> None:
    """Archive a model version in MLflow."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError("mlflow is not installed. Use FileRegistry.archive_model().")
    client = MlflowClient()
    if reason:
        client.update_model_version(name=model_name, version=version, description=reason)
    client.transition_model_version_stage(name=model_name, version=version, stage="Archived")


def get_production_uri(model_name: str) -> str:
    """Return the MLflow URI for the current Production version of a model."""
    return f"models:/{model_name}/Production"
