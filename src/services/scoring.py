"""Batch scoring: load a model and score a DataFrame.

Accepts the model as:
  - A fitted model object (returned by train_model as result.model)
  - A local file path to a .joblib file (saved by LocalFileTracker)
  - An MLflow URI (runs:/ or models:/) — requires mlflow to be installed

No MLflow import at module level — only imported lazily if an MLflow URI is passed.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class ScoringResult:
    scored_at: str
    model_uri: str
    num_records: int
    predictions: pd.Series
    probabilities: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "prediction": self.predictions,
            "probability": self.probabilities,
        })


def _load_model(path_or_uri: str):
    """Load a model from a local path or MLflow URI."""
    uri = str(path_or_uri)
    if uri.startswith(("runs:/", "models:/")):
        try:
            import mlflow.pyfunc
            return mlflow.pyfunc.load_model(uri)
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Install it to load from MLflow URIs: pip install mlflow\n"
                "Or use a local file path (LocalFileTracker saves .joblib files)."
            )
    import joblib
    return joblib.load(uri)


def score_batch(df: pd.DataFrame, model_or_path: Union[str, Path, object]) -> ScoringResult:
    """Score all rows in df using the provided model, path, or MLflow URI.

    df must NOT include the target column — only feature columns.
    """
    if isinstance(model_or_path, (str, Path)):
        model = _load_model(str(model_or_path))
        label = str(model_or_path)
    else:
        model = model_or_path
        label = "in-memory"

    predictions = pd.Series(model.predict(df), index=df.index, name="prediction")
    probabilities = pd.Series(
        model.predict_proba(df)[:, 1], index=df.index, name="probability"
    )

    return ScoringResult(
        scored_at=datetime.now(timezone.utc).isoformat(),
        model_uri=label,
        num_records=len(df),
        predictions=predictions,
        probabilities=probabilities,
    )
