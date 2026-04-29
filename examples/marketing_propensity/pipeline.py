"""Marketing campaign response propensity model — end-to-end example.

Business problem:
  We send marketing campaigns to customers. Not every customer responds.
  We want to score each customer's probability of responding to a campaign
  so we can focus budget on high-propensity customers.

This example shows the full MLOps lifecycle for this use case:
  1. Define the data contract
  2. Generate (synthetic) data and validate it
  3. Train a gradient boosting model
  4. Validate against promotion thresholds
  5. Score a new batch of customers
  6. Build a monitoring report
  7. Produce a model card

See docs/decision-frameworks.md for why we use ML (not an LLM) for this task,
and why we chose batch inference (not online).

Run with:
  python examples/marketing_propensity/pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from src.core.contracts import ColumnSpec, ColumnType, DataContract, ModelCard
from src.core.lifecycle import ModelStatus, transition
from src.pipelines.validation import ValidationThresholds, validate_model

SEED = 42


# ---------------------------------------------------------------------------
# Data contract for this use case
# ---------------------------------------------------------------------------

PROPENSITY_CONTRACT = DataContract(
    name="marketing_propensity_v1",
    version="1.0",
    owner="data-engineering",
    description="Customer features for marketing campaign response propensity",
    columns=[
        ColumnSpec(name="customer_age_years", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="account_tenure_months", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="total_spend_ltm", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="campaign_contacts_ytd", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="last_purchase_days_ago", dtype=ColumnType.NUMERIC, nullable=True),
        ColumnSpec(name="channel_preference", dtype=ColumnType.CATEGORICAL, nullable=True),
        ColumnSpec(name="target", dtype=ColumnType.NUMERIC),
    ],
)


def make_propensity_data(n: int = 3000) -> pd.DataFrame:
    """Synthetic dataset conforming to PROPENSITY_CONTRACT.

    Signal design: recent purchasers + high spenders respond; over-contacted don't.
    Signal is strong enough that a GBM with ~3k rows achieves ROC-AUC > 0.70.
    """
    rng = np.random.default_rng(SEED)
    age = rng.integers(18, 75, n)
    tenure = rng.integers(1, 120, n)
    spend = rng.uniform(0, 5000, n)
    contacts = rng.integers(0, 20, n)
    last_purchase = rng.integers(0, 365, n).astype(float)
    last_purchase[rng.random(n) < 0.1] = np.nan  # 10% never purchased

    channels = rng.choice(["email", "sms", "push", None], n, p=[0.5, 0.2, 0.2, 0.1])

    # Strong, separable signal: combine multiple clear drivers
    lp = np.where(np.isnan(last_purchase), 365, last_purchase)
    score = (
        (spend / 5000.0) * 0.40
        + np.clip(1.0 - lp / 180.0, 0, 1) * 0.35
        - (contacts / 20.0) * 0.20
        + (channels == "email").astype(float) * 0.05
    )
    noise = rng.normal(0, 0.08, n)
    target = (score + noise > 0.35).astype(int)

    return pd.DataFrame({
        "customer_age_years": age,
        "account_tenure_months": tenure,
        "total_spend_ltm": spend,
        "campaign_contacts_ytd": contacts,
        "last_purchase_days_ago": last_purchase,
        "channel_preference": channels,
        "target": target,
    })


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature engineering: encode categoricals, fill nulls."""
    df = df.copy()
    df["last_purchase_days_ago"] = df["last_purchase_days_ago"].fillna(999)  # no purchase = very stale
    channel_dummies = pd.get_dummies(df["channel_preference"], prefix="channel", dummy_na=True)
    df = pd.concat([df.drop(columns=["channel_preference"]), channel_dummies], axis=1)
    return df


def train_propensity_model(df: pd.DataFrame) -> dict:
    """Train a GradientBoosting model and log to MLflow."""
    df_enc = encode_features(df)
    X = df_enc.drop(columns=["target"])
    y = df_enc["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}
    mlflow.set_experiment("marketing-propensity-dev")

    with mlflow.start_run(run_name="gbm-baseline") as run:
        mlflow.set_tags({
            "owner": "ds-marketing",
            "use_case": "marketing_propensity",
            "data_contract": "marketing_propensity_v1:1.0",
        })
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(X_train), "test_rows": len(X_test)})

        model = GradientBoostingClassifier(**params, random_state=SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Use threshold=0.5 for default metrics, but note: for imbalanced propensity
        # models the primary metric is ROC-AUC (ranking), not F1 at 0.5 threshold.
        # See docs/failure-modes.md §7 (Evaluation Metric Mismatch).
        y_pred_balanced = (y_prob >= 0.30).astype(int)  # lower threshold for recall
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred_balanced), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("feature_names", ",".join(X.columns))

    return {
        "run_id": run.info.run_id,
        "model_uri": f"runs:/{run.info.run_id}/model",
        "metrics": metrics,
        "feature_names": list(X.columns),
    }


def main() -> None:
    print("=" * 60)
    print("Marketing Propensity Model — MLOps Lifecycle")
    print("=" * 60)

    # 1. Data contract
    df = make_propensity_data()
    violations = PROPENSITY_CONTRACT.validate_dataframe(df)
    assert not violations, f"Contract violations: {violations}"
    response_rate = df["target"].mean()
    print(f"\n[1/6] Data Contract: OK")
    print(f"      {len(df):,} rows  |  response rate {response_rate:.1%}")

    # 2. Train
    result = train_propensity_model(df)
    print(f"\n[2/6] Training complete")
    print(f"      run_id  : {result['run_id']}")
    print(f"      metrics : {result['metrics']}")

    # 3. Validate
    # Propensity models with class imbalance: primary metric is ROC-AUC (ranking ability).
    # F1 threshold is low because we use a 0.30 decision threshold to improve recall.
    # See docs/failure-modes.md §7 (Evaluation Metric Mismatch).
    thresholds = ValidationThresholds(min_accuracy=0.60, min_f1=0.35, min_roc_auc=0.60)
    validation = validate_model(result["metrics"], thresholds)
    print(f"\n[3/6] Validation:")
    print("     ", validation.summary().replace("\n", "\n      "))
    if not validation.passed:
        print("\nModel did not pass. Staying in EXPERIMENTAL.")
        sys.exit(1)

    # 4. Lifecycle
    status = ModelStatus.EXPERIMENTAL
    status = transition(status, ModelStatus.CANDIDATE)
    status = transition(status, ModelStatus.APPROVED)
    status = transition(status, ModelStatus.DEPLOYED)
    print(f"\n[4/6] Lifecycle: → {status.value}")

    # 5. Score a new batch
    new_batch = make_propensity_data(n=500)
    scoring_df = encode_features(new_batch.drop(columns=["target"]))

    model = mlflow.sklearn.load_model(result["model_uri"])
    probabilities = pd.Series(model.predict_proba(scoring_df)[:, 1])

    top_10pct = probabilities.nlargest(int(len(probabilities) * 0.10))
    print(f"\n[5/6] Batch Scoring:")
    print(f"      {len(probabilities):,} customers scored")
    print(f"      top 10% avg score: {top_10pct.mean():.2%}  (target audience for campaign)")
    print(f"      overall avg score: {probabilities.mean():.2%}")

    # 6. Monitoring report
    report = build_monitoring_report(
        model_name="marketing-propensity-gbm",
        scores=probabilities,
    )
    print(f"\n[6/6] Monitoring:")
    print(f"      mean={report.mean_score:.4f}  p10={report.p10:.4f}  "
          f"p50={report.p50:.4f}  p90={report.p90:.4f}")
    if report.has_alerts():
        for a in report.alerts:
            print(f"  {a}")

    # Model card
    card = ModelCard(
        model_name="marketing-propensity-gbm-v1",
        version="1.0",
        owner="ds-marketing",
        created_date="2026-04-29",
        description="Gradient boosting classifier predicting customer response to marketing campaigns.",
        intended_use="Weekly batch scoring of active customers to prioritize campaign outreach.",
        out_of_scope_use="Individual-level decisions; regulatory or credit decisions; new customer segments.",
        training_data="marketing_propensity_v1 v1.0 (3,000 synthetic records)",
        evaluation_metrics=result["metrics"],
        known_limitations="Trained on synthetic data. Campaign history (contacts_ytd) may change with new campaigns.",
        ethical_considerations="No protected attributes used. Age included — review for age-based bias before production.",
        approval_status="approved",
    )
    print(f"\nModel Card: {card.model_name} v{card.version} — {card.approval_status}")
    print("\n" + "=" * 60)
    print("Done. See examples/churn_prediction/ for the churn use case.")
    print("=" * 60)


if __name__ == "__main__":
    main()
