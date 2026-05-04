"""End-to-end MLOps lifecycle demo — Customer Churn Prediction.

Demonstrates the full lifecycle with LocalFileTracker (no MLflow required).
To switch to MLflow: replace LocalFileTracker with MLflowTracker — everything else stays the same.
To use a custom estimator: set config.estimator to any sklearn-compatible model.

Run:
  python examples/churn_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.core.contracts import ColumnSpec, ColumnType, DataContract, ModelCard
from src.core.lifecycle import ModelStatus, transition
from src.pipelines.training import TrainingConfig, train_model
from src.pipelines.validation import ValidationThresholds, validate_model
from src.services.scoring import score_batch
from src.tracking.local_tracker import LocalFileTracker

SEED = 42

CHURN_CONTRACT = DataContract(
    name="churn_features_v1",
    version="1.0",
    owner="data-engineering",
    description="Customer-level features for monthly churn prediction",
    columns=[
        ColumnSpec(name="tenure_months", dtype=ColumnType.NUMERIC, description="Months as a customer"),
        ColumnSpec(name="monthly_charges", dtype=ColumnType.NUMERIC, description="Average monthly spend"),
        ColumnSpec(name="num_products", dtype=ColumnType.NUMERIC, description="Number of active products"),
        ColumnSpec(name="support_calls_90d", dtype=ColumnType.NUMERIC, description="Support contacts in 90 days"),
        ColumnSpec(name="target", dtype=ColumnType.NUMERIC, description="1 = churned within 30 days"),
    ],
)


def make_synthetic_churn_data(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    num_products = rng.integers(1, 5, n)
    support_calls = rng.integers(0, 10, n)
    churn = ((support_calls > 5) | (tenure < 6) | (monthly_charges > 100)).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "num_products": num_products,
        "support_calls_90d": support_calls,
        "target": churn,
    })


def main() -> None:
    print("=" * 60)
    print("MLOps Lifecycle Demo — Customer Churn Prediction")
    print("=" * 60)

    # --- 1. Data contract validation ---
    df = make_synthetic_churn_data()
    violations = CHURN_CONTRACT.validate_dataframe(df)
    if violations:
        print("[FAIL] Contract violations:\n" + "\n".join(f"  - {v}" for v in violations))
        sys.exit(1)
    print(f"\n[1/6] Data Contract: OK")
    print(f"      {len(df):,} rows | churn rate {df['target'].mean():.1%}")

    # --- 2. Training (LocalFileTracker — no MLflow needed) ---
    # Swap LocalFileTracker() for MLflowTracker(...) to use MLflow instead.
    # Set config.estimator = XGBClassifier(...) to use a different algorithm.
    tracker = LocalFileTracker()
    config = TrainingConfig(
        experiment_name="churn-prediction-demo",
        model_params={"n_estimators": 100, "max_depth": 5, "class_weight": "balanced"},
        target_column="target",
    )
    result = train_model(df, config, tracker=tracker)
    print(f"\n[2/6] Training: complete")
    print(f"      run_id   : {result.run_id}")
    print(f"      metrics  : {result.metrics}")
    print(f"      saved to : {result.model_uri or '(in-memory only)'}")

    # --- 3. Validation gate ---
    thresholds = ValidationThresholds(min_accuracy=0.70, min_f1=0.55, min_roc_auc=0.70)
    validation = validate_model(result.metrics, thresholds)
    print(f"\n[3/6] Validation Gate:")
    print("     ", validation.summary().replace("\n", "\n      "))
    if not validation.passed:
        print("\nModel did not pass the promotion gate.")
        sys.exit(1)

    # --- 4. Lifecycle transitions ---
    status = ModelStatus.EXPERIMENTAL
    status = transition(status, ModelStatus.CANDIDATE)
    status = transition(status, ModelStatus.APPROVED)
    status = transition(status, ModelStatus.DEPLOYED)
    print(f"\n[4/6] Lifecycle: EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED")
    print(f"      current status: {status.value}")

    # --- 5. Batch scoring ---
    # Score using the saved file path (model_uri) — no need for MLflow or a registry.
    # Can also pass result.model directly to skip disk I/O entirely.
    scoring_df = df.drop(columns=["target"])
    score_result = score_batch(scoring_df, result.model_uri or result.model)
    print(f"\n[5/6] Batch Scoring:")
    print(f"      records scored   : {score_result.num_records:,}")
    print(f"      avg churn prob   : {score_result.probabilities.mean():.2%}")
    print(f"      high-risk (>70%) : {(score_result.probabilities > 0.7).sum():,}")

    # --- 6. Model card ---
    card = ModelCard(
        model_name="churn-rf-v1",
        version="1.0",
        owner="ds-team",
        created_date="2026-05-04",
        description="Random forest classifier predicting 30-day customer churn.",
        intended_use="Monthly batch scoring of active customers for retention campaigns.",
        out_of_scope_use="Real-time scoring, new customer segments, regulatory decisions.",
        training_data="churn_features_v1 v1.0 (2,000 synthetic records)",
        evaluation_metrics=result.metrics,
        known_limitations="Trained on synthetic data. Fairness audit required before production.",
        ethical_considerations="No protected attributes in features. Fairness audit required.",
        approval_status="approved",
    )
    print(f"\n[6/6] Model Card: {card.model_name} v{card.version} — {card.approval_status}")

    print("\n" + "=" * 60)
    print("Demo complete. All lifecycle stages passed.")
    print("Tracker: LocalFileTracker (swap for MLflowTracker to use MLflow)")
    print("=" * 60)


if __name__ == "__main__":
    main()
