"""MLflow tracker — logs runs to an MLflow tracking server.

Requires: pip install mlflow

Set the tracking server via environment variable before running:
  export MLOPS_TRACKING_URI=https://mlflow.your-org.internal

Or pass tracking_uri directly to the constructor.

The model_uri returned by log_model() is an MLflow runs:/ URI that can be
loaded by score_batch() or registered in the MLflow model registry.
"""
from typing import Any, Dict, Optional


class MLflowTracker:
    def __init__(self, tracking_uri: str = "mlruns", experiment_name: str = "default"):
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Run: pip install mlflow\n"
                "Or use LocalFileTracker for tracking without MLflow."
            )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: Optional[Any] = None

    def start_run(self, run_name: str = "") -> None:
        import mlflow
        self._run = mlflow.start_run(run_name=run_name or None)

    def log_params(self, params: Dict[str, Any]) -> None:
        import mlflow
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        import mlflow
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_name: str) -> str:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_name)
        run_id = self._run.info.run_id if self._run else ""
        return f"runs:/{run_id}/{artifact_name}"

    def end_run(self) -> str:
        import mlflow
        run_id = self._run.info.run_id if self._run else ""
        mlflow.end_run()
        return run_id
