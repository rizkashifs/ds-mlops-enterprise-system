"""Local file tracker — saves runs to disk with no external dependencies.

Artifacts are stored at:
  {base_dir}/{run_id}/model.joblib
  {base_dir}/{run_id}/meta.json    ← params + metrics

Use this for local development, CI, or any environment where MLflow is
not available. Swap to MLflowTracker for full experiment comparison UI.
"""
import json
import uuid
from pathlib import Path
from typing import Any, Dict


class LocalFileTracker:
    def __init__(self, base_dir: str = "artifacts/runs"):
        self._base_dir = Path(base_dir)
        self._run_id: str = ""
        self._run_dir: Path = Path(".")
        self._meta: dict = {}

    def start_run(self, run_name: str = "") -> None:
        self._run_id = uuid.uuid4().hex[:8]
        self._run_dir = self._base_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {"run_id": self._run_id, "run_name": run_name, "params": {}, "metrics": {}}

    def log_params(self, params: Dict[str, Any]) -> None:
        self._meta["params"].update(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self._meta["metrics"].update(metrics)

    def log_model(self, model: Any, artifact_name: str) -> str:
        import joblib
        model_path = self._run_dir / f"{artifact_name}.joblib"
        joblib.dump(model, model_path)
        (self._run_dir / "meta.json").write_text(json.dumps(self._meta, indent=2))
        return str(model_path)

    def end_run(self) -> str:
        return self._run_id
