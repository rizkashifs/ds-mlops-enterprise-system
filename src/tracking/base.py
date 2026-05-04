"""ExperimentTracker protocol — the interface every tracking backend must implement.

Three implementations are provided:
  - LocalFileTracker  — saves to disk, no external dependencies (default)
  - MLflowTracker     — logs to MLflow (requires mlflow to be installed)
  - NoOpTracker       — does nothing; useful for unit tests and scripts

To use a different backend (W&B, Neptune, custom), implement this protocol.
"""
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    def start_run(self, run_name: str = "") -> None: ...
    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: Dict[str, float]) -> None: ...
    def log_model(self, model: Any, artifact_name: str) -> str: ...
    def end_run(self) -> str: ...
