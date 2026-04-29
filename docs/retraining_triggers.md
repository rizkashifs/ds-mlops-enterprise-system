# Retraining Triggers

> When should a model be retrained? Who decides? What happens next?
> This document answers all three questions with working code and clear rules.

---

## Contents

1. [Why triggers matter](#1-why-triggers-matter)
2. [The four trigger types](#2-the-four-trigger-types)
3. [Trigger priority and urgency levels](#3-trigger-priority-and-urgency-levels)
4. [Configuration](#4-configuration)
5. [Using the trigger evaluator in code](#5-using-the-trigger-evaluator-in-code)
6. [Integrating triggers with the retraining pipeline](#6-integrating-triggers-with-the-retraining-pipeline)
7. [Human review is always required](#7-human-review-is-always-required)
8. [Anti-patterns](#8-anti-patterns)
9. [Trigger evaluation checklist](#9-trigger-evaluation-checklist)

---

## 1. Why triggers matter

Without a trigger system, teams face two failure modes:

**Retraining too rarely:** The model drifts silently for weeks. By the time someone notices, the business has been acting on bad scores for a month.

**Retraining too often:** Every minor fluctuation causes a retrain cycle, which burns engineering time, introduces risk (every retrain is a potential regression), and trains teams to ignore the signals.

A trigger system makes the retraining decision **explicit, auditable, and consistent** — the same signals that would prompt an alert to a data scientist also drive the decision to queue a retrain. Nothing falls through the cracks because someone was on holiday.

---

## 2. The four trigger types

### Type 1: Performance Degradation (highest priority)

**Signal:** Actual model metrics (accuracy, F1, ROC-AUC) computed against ground truth labels drop below a threshold relative to the validation baseline.

**Why it's highest priority:** It's direct evidence that the model is producing wrong outputs. All other trigger types are leading indicators — this is the consequence.

**Limitation:** Requires ground truth labels, which are often delayed. A churn model predicting 30-day outcomes won't have labels for 30 days. This trigger only fires when you have actuals.

**Default threshold:** 5pp absolute drop from validation baseline.

```python
current_metrics  = {"accuracy": 0.72, "f1": 0.61}
baseline_metrics = {"accuracy": 0.82, "f1": 0.73}
# accuracy dropped 10pp → above 5pp threshold → IMMEDIATE retrain
```

### Type 2: Feature Drift — PSI Alert (high priority)

**Signal:** Population Stability Index (PSI) on one or more input features crosses the alert threshold (default 0.20).

**Why it fires a retrain:** When input distributions shift significantly, the model is operating outside its training distribution. Predictions become unreliable even before you see it in performance metrics. PSI is a leading indicator — it fires before performance degrades.

**PSI reference:**

| PSI | Meaning | Action |
|---|---|---|
| < 0.10 | No significant shift | No action |
| 0.10 – 0.20 | Moderate shift | Schedule investigation; consider retraining |
| > 0.20 | Significant shift | Trigger immediate retraining |

**Which features to monitor:**
- Top features by importance from the training run (logged as `feature_names` in MLflow)
- Features with known business volatility (prices, seasonally-affected counts)
- Features that historically drift fastest

### Type 3: Score Distribution Shift (high priority)

**Signal:** The mean prediction score shifts more than N% relative to the training baseline.

**Why it matters:** A sudden shift in output scores — without a corresponding business event — often indicates data pipeline issues or input drift before PSI-level evidence is visible. It's also the fastest signal to compute (no per-feature PSI needed).

**Default threshold:** 10% relative shift in mean score.

```python
baseline_mean_score = 0.20  # mean at training time
current_mean_score  = 0.35  # today's mean
relative_shift      = |0.35 - 0.20| / 0.20 = 75%
# 75% > 10% threshold → IMMEDIATE
```

**When this fires falsely:** During genuinely high-risk periods (e.g., an economic shock actually does increase churn). Always investigate before retraining — the model may be correct.

### Type 4: Time-Based (safety net)

**Signal:** Too many days have elapsed since the last retrain, regardless of observed drift.

**Why include it:** The other three triggers require active monitoring infrastructure. A time-based trigger is a safety net for when monitoring coverage is incomplete, labels are unavailable, or drift accumulates slowly below detection thresholds.

**Default:** 90 days. Set lower for fast-changing environments (fraud: 14 days, recommendations: 7 days), higher for stable ones (risk models: 180 days).

**This trigger always results in `schedule` urgency**, not `immediate` — it indicates a routine refresh, not an emergency.

---

## 3. Trigger priority and urgency levels

Triggers are evaluated in priority order. Higher-priority triggers can upgrade urgency but can't downgrade it.

```
Priority order:
  1. Performance degradation   → urgency: IMMEDIATE
  2. Feature drift (PSI > 0.20) → urgency: IMMEDIATE
  3. Score distribution shift   → urgency: IMMEDIATE
  4. Feature drift (0.10–0.20)  → urgency: SCHEDULE
  5. Time-based                 → urgency: SCHEDULE
```

**IMMEDIATE:** Investigate within 24 hours. Likely requires an emergency retrain or rollback. Do not wait for the next scheduled review.

**SCHEDULE:** Queue a retrain for the next regular release cycle. No emergency action required, but the retrain should happen within 1–2 weeks.

**NONE:** No retrain needed. Continue monitoring.

---

## 4. Configuration

Set trigger thresholds per use case. Fast-changing environments need lower thresholds; stable environments can tolerate higher ones.

```yaml
# configs/training.yaml
retraining:
  triggers:
    psi_warn_threshold: 0.10
    psi_alert_threshold: 0.20
    score_shift_threshold: 0.10      # relative shift in mean score
    performance_drop_threshold: 0.05 # absolute pp drop
    max_days_since_retrain: 90
    min_records_for_drift_check: 500
```

**Thresholds by use case type:**

| Use case | PSI alert | Score shift | Performance drop | Max days |
|---|---|---|---|---|
| Fraud detection | 0.15 | 0.08 | 0.03 | 14 |
| Churn prediction | 0.20 | 0.10 | 0.05 | 60 |
| Marketing propensity | 0.20 | 0.12 | 0.05 | 30 |
| Credit risk / LTV | 0.25 | 0.15 | 0.03 | 90 |
| Recommendations | 0.15 | 0.10 | 0.05 | 7 |

---

## 5. Using the trigger evaluator in code

```python
from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers

# After a scoring run, build the monitoring report
report = build_monitoring_report(
    model_name="churn-rf",
    scores=score_result.probabilities,
    features_df=scoring_df,
    baseline_features=training_features_df,  # saved at training time
)

# Configure thresholds for this use case
config = TriggerConfig(
    psi_alert_threshold=0.20,
    score_shift_threshold=0.10,
    max_days_since_retrain=60,
)

# Evaluate all triggers
decision = evaluate_triggers(
    monitoring_report=report,
    config=config,
    days_since_last_retrain=45,
    baseline_mean_score=0.18,           # mean score at training time
    current_metrics={"accuracy": 0.74, "f1": 0.62},   # from windowed actuals
    baseline_metrics={"accuracy": 0.82, "f1": 0.75},  # from validation run
)

print(decision.summary())
# RETRAIN [IMMEDIATE] — triggered by: feature_drift
#   TRIGGER  feature drift: PSI=0.238 on 'monthly_charges' (threshold=0.20)
#   WARN     moderate feature drift: PSI=0.14 on 'tenure_months'

if decision.should_retrain:
    if decision.urgency == "immediate":
        # Page on-call; initiate emergency retrain or rollback review
        send_alert(f"[CRITICAL] Retrain needed: {decision.triggered_by}")
    else:
        # Queue for next release cycle
        queue_retrain_job(model_name="churn-rf", reason=decision.reasons)
```

### What to save at training time

To make trigger evaluation possible at scoring time, save these artifacts when the model is trained:

```python
import mlflow
import pandas as pd

# Save baseline feature distributions (needed for PSI)
training_features.to_parquet("artifacts/psi_baseline.parquet")
mlflow.log_artifact("artifacts/psi_baseline.parquet")

# Log baseline mean score (on the test set)
baseline_mean_score = y_prob.mean()
mlflow.log_metric("baseline_mean_score", baseline_mean_score)

# Log validation metrics (needed for performance drop check)
mlflow.log_metrics({"accuracy": 0.82, "f1": 0.75, "roc_auc": 0.88})
```

At scoring time, load these from the MLflow run record:

```python
import mlflow

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
baseline_mean_score = run.data.metrics["baseline_mean_score"]
baseline_metrics = {
    "accuracy": run.data.metrics["accuracy"],
    "f1": run.data.metrics["f1"],
}
baseline_features = pd.read_parquet(
    mlflow.artifacts.download_artifacts(f"runs:/{run_id}/psi_baseline.parquet")
)
```

---

## 6. Integrating triggers with the retraining pipeline

The trigger evaluator connects monitoring → retraining pipeline. Here's the full loop:

```
Scoring job (daily)
  → build_monitoring_report()       # compute PSI, score stats
  → evaluate_triggers()             # check all four trigger types
  → if IMMEDIATE: page on-call + queue emergency retrain
  → if SCHEDULE: add to retrain queue
  → log TriggerDecision to run record

Retrain job (triggered or scheduled)
  → run_retraining_pipeline()       # train + validate + compare vs. production
  → if passes: promote to CANDIDATE for review
  → human reviews and approves
  → promote to DEPLOYED
  → update days_since_last_retrain = 0
```

**Integration example (in `pipelines/inference_pipeline/score.py`):**

```python
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers

# After scoring and building monitoring report...
config = TriggerConfig(**cfg["retraining"]["triggers"])
decision = evaluate_triggers(
    monitoring_report=report,
    config=config,
    days_since_last_retrain=get_days_since_last_retrain(model_name),
    baseline_mean_score=get_baseline_metric(run_id, "baseline_mean_score"),
)

if decision.should_retrain:
    log_trigger_event(model_name, decision)
    if decision.urgency == "immediate":
        trigger_alert(model_name, decision.summary())
    else:
        schedule_retrain(model_name, decision.reasons)
```

---

## 7. Human review is always required

**A trigger fires a retrain. It does not auto-deploy a retrained model.**

Every retrained model must pass:
1. The same validation gate as the original model (all metrics above threshold)
2. A comparison against the currently deployed model (retrained must be at least as good)
3. Human sign-off before promotion to DEPLOYED

Automated retraining (training + validation without human review) may be considered only for:
- Low-stakes use cases (no regulatory or adverse consequences)
- Well-established pipelines with stable data and long operational history
- Explicit governance sign-off on the automation

Even then: auto-train and auto-validate. Never auto-promote to DEPLOYED without a human checkpoint.

---

## 8. Anti-patterns

**Retraining on every drift signal without investigation**
PSI > 0.20 means investigate, not automatically retrain. The drift may be caused by a data pipeline bug (fix the pipeline, not the model), a real-world event that makes the drift expected, or noise in a small sample.

**Using a single threshold for all use cases**
A fraud model and a quarterly LTV model have completely different tolerance for drift. Copy-paste thresholds without adjusting for your use case.

**Not saving baseline artifacts**
If you don't save `psi_baseline.parquet` and `baseline_mean_score` at training time, you can't compute PSI or score shift at scoring time. The trigger evaluator requires these — set them up when the model is first trained, not later.

**Treating urgency levels as suggestions**
IMMEDIATE means page someone and decide within 24 hours. Treating it as "we'll look at it next week" defeats the purpose of the trigger system.

**Skipping the comparison step in retrain**
A retrain that doesn't compare against the production model can promote a worse model. The comparison step in `run_retraining_pipeline()` is not optional.

---

## 9. Trigger evaluation checklist

Before enabling trigger-based retraining for a model:

- [ ] `psi_baseline.parquet` saved and logged as MLflow artifact at training time
- [ ] `baseline_mean_score` logged as MLflow metric at training time
- [ ] Validation metrics (`accuracy`, `f1`, `roc_auc`) logged at training time
- [ ] Trigger thresholds set in `configs/training.yaml` (not using defaults blindly)
- [ ] Alert routing configured: who gets paged on IMMEDIATE vs. SCHEDULE
- [ ] `days_since_last_retrain` tracked (e.g., in a model registry tag or external store)
- [ ] Retrain pipeline tested end-to-end in staging (including the comparison step)
- [ ] Human approval step defined and documented before any retrain goes to DEPLOYED
