# Monitoring Standards

> A deployed model without monitoring is a liability.
> Monitoring is not optional — it is a deployment requirement.

---

## What Can Go Wrong in Production

Models degrade in three ways:

1. **Sudden failure**: pipeline crashes, data source goes offline, schema breaks. Visible in hours.
2. **Distribution shift**: input data changes gradually. Visible in days to weeks via PSI.
3. **Concept drift**: the relationship between features and labels changes. Visible in weeks to months via performance metrics.

Monitoring catches all three — at different timescales, with different signals.

---

## The Four Monitoring Layers

### Layer 1: Pipeline Health (operational)

Check that the scoring job actually runs and produces sensible output.

| Signal | Metric | Alert threshold | Frequency |
|---|---|---|---|
| Job completion | Success / failure | Any failure | Every run |
| Output row count | Rows written | <80% or >120% of expected baseline | Every run |
| Latency | Run duration (seconds) | >2× rolling average | Every run |
| Error rate (online) | Errors / total requests | >0.1% | Real-time |

This layer is the easiest to implement and catches the most acute failures.

### Layer 2: Input Data Drift (feature monitoring)

Check that the input features look like what the model was trained on.

**Primary metric: Population Stability Index (PSI)**

PSI measures how much a distribution has shifted between a baseline (training data or a reference window) and the current period.

| PSI value | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | Continue normal operations |
| 0.10 – 0.20 | Moderate shift | Investigate; consider scheduling a retrain |
| > 0.20 | Significant shift | Alert; likely trigger retraining |

**How to compute PSI:**

```python
import numpy as np

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
```

**Which features to monitor:**

- Top 10 features by importance (from the training run's feature importance log)
- Any feature with known business volatility (e.g., `monthly_charges` if pricing changed)
- Always monitor the target variable distribution if actuals are available

### Layer 3: Output (Prediction) Distribution

Check that the model's output scores look like what was expected.

| Signal | Metric | Alert threshold | Frequency |
|---|---|---|---|
| Mean score | Average prediction probability | >10% relative shift from 30-day baseline | Daily |
| Score histogram | Distribution shape | Visual review | Weekly |
| High-risk count | Count of scores > 0.7 | >20% shift from baseline | Daily |
| Prediction 0/1 ratio (binary) | Class distribution | >10% shift | Daily |

A sudden shift in mean score without a business explanation is a signal — it could mean feature drift, a data pipeline issue, or (rarely) sudden real-world change.

### Layer 4: Model Performance (label monitoring)

When ground truth labels are available (delayed), compare predictions to actuals.

| Signal | Metric | Alert threshold | Frequency |
|---|---|---|---|
| Accuracy on actuals | Windowed accuracy | >5pp drop from validation baseline | Weekly |
| F1 on actuals | Windowed F1 | >5pp drop | Weekly |
| Prediction-label gap | Predicted rate vs. actual rate | >5pp sustained for 7 days | Weekly |

**Labels are often delayed.** A churn model predicts 30-day outcomes — you won't have actuals for 30 days. Design your monitoring cadence to match the label delay.

---

## Alert Levels and Routing

| Level | Condition | Recipients | Expected response time |
|---|---|---|---|
| **Info** | Metrics within bounds; periodic summary | Model owner (digest) | No action required |
| **Warning** | PSI 0.10–0.20, score mean shift 5–10% | Model owner + DS team | Investigate within 2 business days |
| **Critical** | PSI > 0.20, score mean shift >10%, pipeline failure, performance drop >5pp | Model owner + platform team + on-call | Acknowledge within 1 hour; resolve or rollback within 24 hours |

---

## Setting Up Monitoring (Checklist)

Before a model can be promoted to DEPLOYED:

- [ ] **Baseline computed**: training data distribution saved as `artifacts/psi_baseline.parquet`
- [ ] **PSI monitoring**: daily job comparing current features to baseline, logging PSI by feature
- [ ] **Prediction distribution**: daily job logging mean, p10, p50, p90, p99 of scores
- [ ] **Pipeline health**: job completion and output row count in alerting system
- [ ] **Alerts configured**: warning and critical thresholds set for each signal
- [ ] **Alert routing**: owners and on-call contacts defined in alerting system
- [ ] **Dashboard created**: stakeholders can view score trends without asking the DS team
- [ ] **Runbook linked**: critical alerts link to `docs/runbook.md` section

---

## Monitoring Dashboard (Template)

Every deployed model should have a dashboard with at least:

**Section 1: Pipeline Health**
- Last run timestamp and status
- Output row count (trend over 30 days)
- Run duration (trend)

**Section 2: Input Distribution**
- PSI by top-10 feature (trend)
- Feature mean and std over time (key features)

**Section 3: Prediction Distribution**
- Mean score over time
- Score histogram (weekly snapshot)
- High-risk count (scores > threshold) over time

**Section 4: Performance (if labels available)**
- Predicted vs. actual rate over time
- Rolling accuracy / F1

---

## Handling Monitoring Alerts

When a warning or critical alert fires, use `docs/runbook.md` §1.

Quick decision guide:

```
Alert fires:

Is it a pipeline failure?
  YES → Fix the data pipeline first; don't assume the model is the problem

Is the PSI > 0.20 on a key feature?
  YES → Investigate source. Is it real drift or a data bug?
        Real drift → schedule retraining
        Data bug   → fix the pipeline

Is model performance dropping (labels available)?
  YES → Is the drift in features too?
        Both drifting → retrain
        Only performance → investigate label definition or use case change

Is it a sudden drop (1 day) or gradual (weeks)?
  Sudden → likely deployment or data pipeline issue → rollback / investigate
  Gradual → likely concept drift → schedule retraining
```

---

## Monitoring is Not Just for Data Scientists

Platform Engineering owns the infrastructure that emits metrics and fires alerts.
Data Science owns the interpretation and response.
Both must agree on thresholds and routing before a model goes live.

Monitoring should be treated like application monitoring — not an afterthought, but a first-class part of the deployment.
