"""No-op tracker — records nothing, returns empty strings.

Use in unit tests or when you only need the model object (result.model)
and don't need artifacts persisted to disk or a tracking server.
"""
from typing import Any, Dict


class NoOpTracker:
    def start_run(self, run_name: str = "") -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    def log_model(self, model: Any, artifact_name: str) -> str:
        return ""

    def end_run(self) -> str:
        return ""
