"""Batch scoring service: load a registered model and score a DataFrame.

In production, this module is called by a scheduler (e.g., nightly churn run).
The model_uri comes from the model registry after promotion through the approval gate.
Results are returned as a ScoringResult — callers decide how to persist or route them.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow.sklearn
import pandas as pd


@dataclass
class ScoringResult:
    scored_at: str
    model_uri: str
    num_records: int
    predictions: pd.Series
    probabilities: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        """Combine predictions and probabilities into a single output DataFrame."""
        return pd.DataFrame({
            "prediction": self.predictions,
            "probability": self.probabilities,
        })


def score_batch(df: pd.DataFrame, model_uri: str) -> ScoringResult:
    """Load a model by URI and score all rows in df.

    df must NOT include the target column — only feature columns.
    model_uri is the MLflow URI returned by train_model or registered in the model registry.
    """
    model = mlflow.sklearn.load_model(model_uri)
    predictions = pd.Series(model.predict(df), index=df.index, name="prediction")
    probabilities = pd.Series(
        model.predict_proba(df)[:, 1], index=df.index, name="probability"
    )

    return ScoringResult(
        scored_at=datetime.now(timezone.utc).isoformat(),
        model_uri=model_uri,
        num_records=len(df),
        predictions=predictions,
        probabilities=probabilities,
    )
