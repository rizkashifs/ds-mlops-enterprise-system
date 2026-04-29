# Example: Marketing Campaign Response Propensity

## Business Problem

We run marketing campaigns (email, SMS, push). Not every customer responds. Contacting low-propensity customers wastes budget and hurts deliverability scores.

**Goal:** Score every active customer's probability of responding to a campaign. Use the scores to prioritize who to contact.

## MLOps Decisions

| Decision | Choice | Reason |
|---|---|---|
| ML vs LLM | ML | Structured tabular data, defined label, large training set |
| Algorithm | Gradient Boosting | Better calibration than Random Forest for imbalanced response rates |
| Inference pattern | Batch (weekly) | Campaign lists are prepared weekly; no real-time need |
| Primary metric | ROC-AUC | We care about ranking customers, not classifying them |
| Retraining trigger | Monthly + PSI > 0.20 | Campaign patterns shift seasonally |

See `docs/decision-frameworks.md` for the reasoning behind each choice.

## Features

| Feature | Description | Null handling |
|---|---|---|
| `customer_age_years` | Age at scoring date | Required, no nulls |
| `account_tenure_months` | Months since account creation | Required |
| `total_spend_ltm` | Total spend in last 12 months | Required |
| `campaign_contacts_ytd` | Marketing contacts year-to-date | Required |
| `last_purchase_days_ago` | Days since last purchase | Fill with 999 (never purchased) |
| `channel_preference` | Preferred contact channel | One-hot encode; null → own category |

## Running the Example

```bash
# From the repo root:
python examples/marketing_propensity/pipeline.py
```

This will:
1. Generate synthetic data and validate against the data contract
2. Train a Gradient Boosting model and log to MLflow
3. Validate metrics against thresholds
4. Score a new batch of customers
5. Produce a monitoring report and model card

## Expected Output

```
[1/6] Data Contract: OK — 3,000 rows, response rate 18.4%
[2/6] Training complete — accuracy=0.82, f1=0.42, roc_auc=0.73
[3/6] Validation: PASSED
[4/6] Lifecycle: → deployed
[5/6] Batch Scoring — 500 customers, top 10% avg score 45.2%
[6/6] Monitoring — mean=0.18, p90=0.38
```

## Adapting to a Real Dataset

1. Replace `make_propensity_data()` with your actual data loading function
2. Update `PROPENSITY_CONTRACT` in `pipeline.py` to match your real column names
3. Update `configs/pipeline_contracts.yaml` to add your contract version
4. Update `encode_features()` for your actual feature engineering logic
5. Adjust validation thresholds in `configs/training.yaml` for your business requirements

## Known Limitations (from Model Card)

- Trained on synthetic data — real campaign patterns will differ
- Age is a feature — review for age-based bias before production use
- `campaign_contacts_ytd` will shift as campaign frequency changes — monitor PSI on this feature

## Failure Modes to Watch

- **Training-serving skew**: `encode_features()` must be called identically in training and scoring
- **Contact fatigue drift**: if campaign frequency changes significantly, retrain
- **Seasonal response shifts**: Q4 (holiday) response rates differ from Q1 — consider time-based retraining before high-volume periods
