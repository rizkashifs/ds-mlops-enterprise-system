"""Training module: fits any sklearn-compatible model, tracks via any backend.

Algorithm choice:
  Pass any sklearn-compatible estimator via config.estimator.
  If None, defaults to RandomForestClassifier(**config.model_params).

Tracking choice:
  Pass any ExperimentTracker implementation.
  Defaults to NoOpTracker — model is always accessible via result.model
  even when no tracking backend is configured.

Examples:
  # Default: no tracking, use result.model directly for scoring
  result = train_model(df, config)

  # Local file tracking (no external dependencies)
  from src.tracking.local_tracker import LocalFileTracker
  result = train_model(df, config, tracker=LocalFileTracker())

  # MLflow tracking
  from src.tracking.mlflow_tracker import MLflowTracker
  result = train_model(df, config, tracker=MLflowTracker(tracking_uri="..."))

  # Custom estimator (XGBoost, LightGBM, LogisticRegression, etc.)
  from xgboost import XGBClassifier
  config = TrainingConfig(experiment_name="...", estimator=XGBClassifier(n_estimators=200))
  result = train_model(df, config, tracker=LocalFileTracker())
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.tracking.base import ExperimentTracker
from src.tracking.noop_tracker import NoOpTracker


@dataclass
class TrainingConfig:
    experiment_name: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    estimator: Any = None  # if None → RandomForestClassifier(**model_params)


@dataclass
class TrainingResult:
    run_id: str
    metrics: Dict[str, float]
    model_uri: str
    feature_names: List[str]
    model: Any = None  # always set — use directly when no tracker saves to disk
    params: Dict[str, Any] = field(default_factory=dict)


def train_model(
    df: pd.DataFrame,
    config: TrainingConfig,
    tracker: Optional[ExperimentTracker] = None,
) -> TrainingResult:
    """Fit a model, log everything via tracker, return a TrainingResult.

    The returned result.model is always the fitted estimator — useful when
    you want to score immediately without loading from disk or a registry.
    result.model_uri is the saved artifact path (or empty if using NoOpTracker).
    """
    if tracker is None:
        tracker = NoOpTracker()

    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    tracker.start_run(run_name=config.experiment_name)

    model = (
        config.estimator
        if config.estimator is not None
        else RandomForestClassifier(**config.model_params, random_state=config.random_state)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    tracker.log_params(config.model_params)
    tracker.log_metrics(metrics)
    model_uri = tracker.log_model(model, "model")
    run_id = tracker.end_run()

    return TrainingResult(
        run_id=run_id,
        metrics=metrics,
        model_uri=model_uri,
        feature_names=list(X.columns),
        model=model,
        params=config.model_params,
    )
