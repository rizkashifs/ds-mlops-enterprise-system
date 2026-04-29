"""Training pipeline: fits a model, logs everything to MLflow, returns a result.

Every training run must produce a TrainingResult. That result is the input to
the validation stage. Nothing advances to review without a run_id.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainingConfig:
    experiment_name: str
    model_params: Dict[str, Any]
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class TrainingResult:
    run_id: str
    metrics: Dict[str, float]
    model_uri: str
    feature_names: List[str]
    params: Dict[str, Any] = field(default_factory=dict)


def train_model(df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    """Fit a RandomForest, log params + metrics + artifact to MLflow.

    Returns a TrainingResult with the run_id and model URI needed for
    validation and promotion.
    """
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            **config.model_params, random_state=config.random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }

        mlflow.log_params(config.model_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return TrainingResult(
            run_id=run.info.run_id,
            metrics=metrics,
            model_uri=f"runs:/{run.info.run_id}/model",
            feature_names=list(X.columns),
            params=config.model_params,
        )
