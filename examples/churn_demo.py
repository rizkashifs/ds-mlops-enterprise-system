"""End-to-end MLOps lifecycle demo using a synthetic churn dataset.

Run this to see all five stages in action:
  python examples/churn_demo.py

What this covers:
  1. Define a data contract for input features
  2. Validate data against the contract
  3. Train a model and log to MLflow
  4. Run validation checks against promotion thresholds
  5. Transition the model through lifecycle stages
  6. Score a batch using the trained model
  7. Produce a model card for governance review

This is a reference implementation. Real teams replace the synthetic data
with their own DataContract definitions and feature pipelines.
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

SEED = 42


# ---------------------------------------------------------------------------
# 1. Define the data contract
# ---------------------------------------------------------------------------

CHURN_CONTRACT = DataContract(
    name="churn_features_v1",
    version="1.0",
    owner="data-engineering",
    description="Customer-level features for monthly churn prediction",
    columns=[
        ColumnSpec(name="tenure_months", dtype=ColumnType.NUMERIC, description="Months as a customer"),
        ColumnSpec(name="monthly_charges", dtype=ColumnType.NUMERIC, description="Average monthly spend"),
        ColumnSpec(name="num_products", dtype=ColumnType.NUMERIC, description="Number of active products"),
        ColumnSpec(name="support_calls_90d", dtype=ColumnType.NUMERIC, description="Support contacts in last 90 days"),
        ColumnSpec(name="target", dtype=ColumnType.NUMERIC, description="1 = churned within 30 days"),
    ],
)


def make_synthetic_churn_data(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic data that conforms to CHURN_CONTRACT."""
    rng = np.random.default_rng(SEED)
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    num_products = rng.integers(1, 5, n)
    support_calls = rng.integers(0, 10, n)
    # Simple rule-based label: high support calls, short tenure, or high charges → churn
    churn = (
        (support_calls > 5) | (tenure < 6) | (monthly_charges > 100)
    ).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "num_products": num_products,
        "support_calls_90d": support_calls,
        "target": churn,
    })


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("MLOps Lifecycle Demo — Customer Churn Prediction")
    print("=" * 60)

    # --- Stage 1: Data contract validation ---
    df = make_synthetic_churn_data()
    violations = CHURN_CONTRACT.validate_dataframe(df)
    if violations:
        print(f"[FAIL] Contract violations:\n" + "\n".join(f"  - {v}" for v in violations))
        sys.exit(1)
    churn_rate = df["target"].mean()
    print(f"\n[1/6] Data Contract: OK")
    print(f"      {len(df):,} rows  |  {df.shape[1]} columns  |  churn rate {churn_rate:.1%}")

    # --- Stage 2: Training ---
    config = TrainingConfig(
        experiment_name="churn-prediction-demo",
        model_params={"n_estimators": 100, "max_depth": 5},
        target_column="target",
    )
    result = train_model(df, config)
    print(f"\n[2/6] Training: complete")
    print(f"      run_id   : {result.run_id}")
    print(f"      metrics  : {result.metrics}")
    print(f"      model_uri: {result.model_uri}")

    # --- Stage 3: Validation gate ---
    thresholds = ValidationThresholds(min_accuracy=0.70, min_f1=0.55, min_roc_auc=0.70)
    validation = validate_model(result.metrics, thresholds)
    print(f"\n[3/6] Validation Gate:")
    print("     ", validation.summary().replace("\n", "\n      "))
    if not validation.passed:
        print("\nModel did not pass the promotion gate. Staying in CANDIDATE.")
        sys.exit(1)

    # --- Stage 4: Lifecycle transitions ---
    status = ModelStatus.EXPERIMENTAL
    status = transition(status, ModelStatus.CANDIDATE)
    status = transition(status, ModelStatus.APPROVED)
    status = transition(status, ModelStatus.DEPLOYED)
    print(f"\n[4/6] Lifecycle: promoted through EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED")
    print(f"      current status: {status.value}")

    # --- Stage 5: Batch scoring ---
    scoring_df = df.drop(columns=["target"])
    score_result = score_batch(scoring_df, result.model_uri)
    avg_probability = score_result.probabilities.mean()
    high_risk_count = (score_result.probabilities > 0.7).sum()
    print(f"\n[5/6] Batch Scoring:")
    print(f"      records scored    : {score_result.num_records:,}")
    print(f"      avg churn prob    : {avg_probability:.2%}")
    print(f"      high-risk (>70%)  : {high_risk_count:,}")
    print(f"      scored_at         : {score_result.scored_at}")

    # --- Stage 6: Model card ---
    card = ModelCard(
        model_name="churn-rf-v1",
        version="1.0",
        owner="ds-team",
        created_date="2026-04-29",
        description="Random forest classifier predicting 30-day customer churn.",
        intended_use="Monthly batch scoring of active customers for retention campaigns.",
        out_of_scope_use="Real-time scoring, new customer segments, regulatory decisions.",
        training_data="churn_features_v1 v1.0 (2,000 synthetic records, SEED=42)",
        evaluation_metrics=result.metrics,
        known_limitations=(
            "Trained on synthetic data. Label is deterministic, not probabilistic. "
            "Not validated on real customer distribution."
        ),
        ethical_considerations=(
            "No protected attributes (age, gender, ethnicity) included in features. "
            "Fairness audit against protected groups required before production use."
        ),
        approval_status="approved",
    )
    print(f"\n[6/6] Model Card: {card.model_name} v{card.version}")
    print(f"      owner    : {card.owner}")
    print(f"      status   : {card.approval_status}")
    print(f"      metrics  : {card.evaluation_metrics}")

    print("\n" + "=" * 60)
    print("Demo complete. All lifecycle stages passed.")
    print("See docs/mlops_standards.md for the full playbook.")
    print("=" * 60)


if __name__ == "__main__":
    main()
