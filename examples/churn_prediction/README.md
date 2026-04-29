# Example: Customer Churn Prediction

## Business Problem

Customers who are about to cancel their subscription are expensive to lose. The retention team can intervene with offers or outreach — but only if they know who's at risk.

**Goal:** Score every active customer's probability of churning in the next 30 days. Flag high-risk customers for the retention team.

## MLOps Decisions

| Decision | Choice | Reason |
|---|---|---|
| ML vs LLM | ML | Structured account data, binary label, existing training data |
| Algorithm | Random Forest | Good baseline; interpretable feature importances |
| Inference pattern | Batch (monthly) | Retention team works on monthly cycles |
| Primary metric | ROC-AUC + F1 | Both ranking quality and precision/recall matter |
| Retraining trigger | Monthly + PSI > 0.20 on `support_calls_90d` | Support call pattern is most volatile feature |

## Features

| Feature | Description | Null handling |
|---|---|---|
| `tenure_months` | Months since account creation | Required |
| `monthly_charges` | Average monthly spend | Required |
| `num_products` | Count of active products | Required |
| `support_calls_90d` | Support contacts in last 90 days | Required |

## Running the Example

```bash
# From the repo root:
python examples/churn_demo.py
```

Or use the dedicated pipeline:

```bash
python pipelines/training_pipeline/train.py
```

## Adapting to a Real Dataset

1. Replace `make_churn_data()` in `examples/churn_demo.py` with your data source
2. Update `CHURN_CONTRACT` to match your real feature names
3. Update `configs/pipeline_contracts.yaml` with your real schema
4. Set thresholds in `configs/training.yaml` based on business requirements
5. Add feature engineering for any new features (nulls, categoricals, scaling)

## Feature Importance Reference

From the synthetic training run, key drivers of churn (highest importance):

1. `support_calls_90d` — high contact frequency is the strongest signal
2. `tenure_months` — new customers churn more
3. `monthly_charges` — higher charges → more churn risk

In a real deployment, use SHAP values for individual-level explanations.

## Common Failure Modes for This Use Case

- **Data leakage**: don't include any feature that's computed from the churn event itself (e.g., "last_activity_date" for churned customers)
- **Label timing**: make sure `target=1` means "churned in the next 30 days FROM the prediction date", not "ever churned"
- **Class imbalance**: churn rates are typically 5–15%; set `class_weight='balanced'` and report F1, not accuracy
- **Support call lag**: if support data has a 48h delay, don't score on data that's fresher than 48h
