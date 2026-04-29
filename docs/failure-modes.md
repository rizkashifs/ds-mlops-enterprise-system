# ML Failure Modes

> Most ML failures in production are not about model architecture.
> They are about the gaps between training and the real world.
> This document catalogs the most common failure modes and how to prevent each one.

---

## Contents

1. [Data Leakage](#1-data-leakage)
2. [Training-Serving Skew](#2-training-serving-skew)
3. [Silent Drift](#3-silent-drift)
4. [Broken Retraining Loops](#4-broken-retraining-loops)
5. [Label Leakage](#5-label-leakage)
6. [Class Imbalance Blindness](#6-class-imbalance-blindness)
7. [Evaluation Metric Mismatch](#7-evaluation-metric-mismatch)
8. [Stale Features](#8-stale-features)
9. [Silent Pipeline Failures](#9-silent-pipeline-failures)
10. [Over-Reliance on Holdout Accuracy](#10-over-reliance-on-holdout-accuracy)

---

## 1. Data Leakage

**What it is:** Information from the future (or from the target itself) leaks into the training features. The model learns a shortcut that won't exist in production.

**How it happens:**
- A feature is computed using data from after the label date
- The label is included as a feature (or highly correlated proxy)
- Data is aggregated across train/test splits before splitting
- A join brings in information that wouldn't exist at prediction time

**Example:**
A churn model includes `last_activity_date` which is populated when an account closes. Customers who churn have `last_activity_date = churn_date`. The model learns "if last_activity_date is recent, the customer churned" — but at prediction time, active customers also have a recent activity date.

**How to detect:**
- Test AUC suspiciously high (>0.98 for a business problem)
- Model performance collapses the moment you deploy
- Feature importance concentrates on 1–2 features that shouldn't be that powerful
- Performance gap between training set and holdout is large

**How to prevent:**
- Define a clear **prediction point in time** and validate that every feature could have been known at that point
- Use a time-based train/test split, not random split, for time-series data
- Review feature definitions with domain experts before training
- Build a checklist question into the model review: "for each feature, could we have known this value at prediction time?"

---

## 2. Training-Serving Skew

**What it is:** The features computed during training are computed differently from the features computed during inference. The model was trained on a different distribution than it operates on.

**This is the most common production ML failure.**

**How it happens:**
- Training pipeline uses Python/Pandas; inference pipeline uses SQL or a different library
- Training uses `fillna(mean_at_training_time)` but serving computes the mean on the fly
- Training clips outliers; serving does not
- Training uses a label-encoded categorical; serving uses one-hot
- Different teams own training vs. serving pipelines

**Example:**
Training encodes `channel = "email"` as `1`, `"sms"` as `2`. A new channel `"push"` is added after training. The serving pipeline assigns it `3`. The model was never trained on `3` — it might map it to a `NaN` or encode it as a known category.

**How to detect:**
- Model performance in production is significantly worse than on the holdout set
- Score distribution in production doesn't match the validation distribution
- Specific cohorts of predictions behave unexpectedly

**How to prevent:**
- **Use the same feature computation code** in training and serving. If training uses Python, serving should call the same Python function, not a SQL translation.
- Use a feature store with point-in-time correct lookups
- Log feature distributions at both training time and serving time; compare them
- Add a test that runs the feature pipeline on a known input and checks the output is identical to what training produced

---

## 3. Silent Drift

**What it is:** The world changes, making the model's training distribution stale, but no alert fires. Predictions get worse gradually, and nobody notices until a business metric suffers.

**How it happens:**
- No monitoring is in place for prediction distribution or feature drift
- Monitoring exists but alert thresholds are too wide
- Monitoring fires but goes to a team that doesn't know what to do with it
- The model is batch-scoring — errors accumulate for a full cycle before being noticed

**Example:**
A propensity model is trained in Q1. By Q3, a new product is launched that changes the customer base significantly. The model's predictions degrade over 3 months, but since nobody is tracking PSI or prediction distribution, the issue is only noticed when the marketing team reports low campaign conversion.

**Signs of silent drift:**
- Prediction mean shifts over time (e.g., average churn score was 0.18 in January, now 0.35 in August without a business reason)
- PSI > 0.10 on key features with no corresponding retrain
- Feature importances at serving time differ from training

**How to prevent:**
- Monitor prediction distribution daily (mean, p10, p50, p90, p99)
- Calculate PSI on input features daily against a rolling 30-day baseline
- Set alert thresholds at PSI > 0.10 (warning) and PSI > 0.20 (page)
- Require monitoring dashboards before any model can be DEPLOYED
- Review actuals vs. predictions on a windowed basis as soon as labels are available

---

## 4. Broken Retraining Loops

**What it is:** The retraining pipeline exists on paper but fails silently or produces a worse model, which then gets deployed without proper validation.

**How it happens:**
- Automated retraining skips the validation gate because "it's the same pipeline"
- The retraining pipeline uses stale data (the data fetch is broken but doesn't raise an error)
- The retrained model is auto-promoted without comparing against the current production model
- A retrained model has a data leakage bug introduced during pipeline updates

**Example:**
Automated retraining runs every month. One month, the feature store returns data from 2 years ago due to a date filter bug. The retrained model has a 0.95 AUC (due to leakage from a stale join) and is automatically promoted. In production, it performs terribly.

**How to prevent:**
- Automated retraining must still pass the same validation gate as manual training
- Every retrained model must be compared against the current production model — a retrain is not an auto-promotion
- Log the data date range in every training run; alert if the max date is more than 48h stale
- A human must review and approve a retrained model before it's promoted to DEPLOYED, even in automated workflows
- Test retraining pipelines in staging with known data before enabling automation in production

---

## 5. Label Leakage

**What it is:** The label definition references something that wouldn't be known at prediction time, or the label is computed from the same data the model will predict on.

**How it happens:**
- Labeling happens after the fact and inadvertently includes post-event features
- Events that cause the outcome also appear in the features
- The label date and feature snapshot date are misaligned

**Example:**
A model predicts whether a customer will file a complaint. One feature is `support_ticket_open = True/False`. Customers who file complaints often have open support tickets. But "filing a complaint" and "having an open ticket" are near-simultaneous events — the model learns this correlation, not the causal signal.

**How to prevent:**
- Write out the prediction scenario explicitly: "At time T, we observe features F and predict whether event E will occur by time T + horizon"
- Every feature must be observable at time T; every label must only reference events after T
- Review the label definition with domain experts, not just data engineers

---

## 6. Class Imbalance Blindness

**What it is:** The model is evaluated on accuracy, which looks great (95%) because it predicts the majority class almost always. In reality, it has poor recall on the minority class (the interesting one).

**How it happens:**
- Default evaluation metric is accuracy
- No class weighting in training
- The team only reviews aggregate metrics, not per-class metrics

**Example:**
A fraud detection model has 99.5% accuracy and 0.9% fraud recall. It correctly ignores almost all fraud cases because predicting "not fraud" is always right. This is useless — you want the minority class detected.

**How to prevent:**
- Always report precision, recall, and F1 per class, not just aggregate accuracy
- For imbalanced datasets, use ROC-AUC or PR-AUC as primary metrics
- Set `class_weight='balanced'` in sklearn models as a default
- Set explicit F1 or recall thresholds for the minority class in the validation gate
- Review the confusion matrix, not just the headline metric

---

## 7. Evaluation Metric Mismatch

**What it is:** The model is optimized and validated on a metric that doesn't reflect business value. The model scores well on the metric but delivers poor business outcomes.

**How it happens:**
- Data scientists default to accuracy or AUC without connecting them to business impact
- Business stakeholders are given a number they can't interpret ("We have 0.87 AUC!")
- The model ranks customers correctly overall but the top decile (the ones who get action) is noisy

**Example:**
A propensity model has 0.82 AUC. Great! But the business only contacts the top 5% of scores. When you look at precision in the top decile, it's 0.30 — meaning 70% of the people being contacted won't convert. The model is decent overall but poor at exactly the task it's used for.

**How to prevent:**
- Before training, define: "The business will take action on the top X% of scores. What precision do we need in that decile to make the campaign profitable?"
- Add a **precision at top K%** metric to your validation thresholds
- Present metrics in business language: "Of the 10,000 customers we'll contact, we expect 2,300 to respond (23% precision)"
- Use lift curves and cumulative response curves to communicate model value

---

## 8. Stale Features

**What it is:** Features are computed correctly at training time, but at serving time they are computed from data that's hours, days, or weeks old. The model receives a stale view of reality.

**How it happens:**
- Feature computation pipelines have slow SLAs not aligned with scoring SLAs
- A feature store has a stale partition due to a failed upstream job
- Features are cached but the cache isn't invalidated properly

**Example:**
A model uses `support_calls_last_7d`. The feature is refreshed daily. A customer calls support 3 times in one day, but the scoring job runs before the daily refresh. The feature shows `0` instead of `3` — the model misses a high-risk signal.

**How to prevent:**
- Define the **feature freshness SLA** for each feature and check it at scoring time
- Log feature timestamps alongside predictions
- Alert when a feature partition hasn't been refreshed within its SLA
- For time-sensitive features, push updates on event rather than schedule

---

## 9. Silent Pipeline Failures

**What it is:** A pipeline runs to completion but produces empty, incorrect, or partial output with no error logged. Downstream consumers receive bad data silently.

**How it happens:**
- An empty DataFrame is a valid result for a filter, but an empty output from a scoring job is not
- A join drops rows silently (inner join instead of left join)
- A configuration change causes a pipeline to score zero records

**Example:**
A scoring pipeline filters customers by `is_active = True`. A configuration bug changes this to `is_active = "true"` (string vs. boolean). The filter returns 0 rows. The pipeline completes successfully, writes an empty output file, and no alert fires. The next day, no scores appear in the marketing tool.

**How to prevent:**
- Add output assertions: check that output row count is within expected bounds before finishing
- Log output row count as a metric on every run
- Alert on zero-row outputs — they are almost always a bug
- Add an integration test that runs the pipeline on a small known dataset and checks the output count

---

## 10. Over-Reliance on Holdout Accuracy

**What it is:** The team treats strong holdout metrics as a guarantee of production performance. It isn't.

**The holdout only tells you how the model performs on historical data from the same distribution as training.**

It does not tell you:
- How the model performs on future data (distribution shift)
- How the model performs on edge cases not well-represented in the holdout
- How the model performs when features are computed by the serving pipeline (training-serving skew)
- Whether the model's output will actually improve the business outcome it's designed for

**How to prevent:**
- Treat holdout accuracy as a **necessary but not sufficient** condition for deployment
- Always run a period of shadow mode (new model runs in parallel, results not acted on) before full deployment
- Define a production success metric (conversion rate, incident reduction, revenue) and measure it after deployment
- Run an A/B test if the use case allows for it

---

## Quick Reference: Failure Mode Checklist

Before promoting a model to APPROVED, verify:

- [ ] No feature uses data that would only exist after the prediction date
- [ ] Feature computation code is identical between training and serving
- [ ] Model evaluated on per-class metrics, not just aggregate accuracy
- [ ] Holdout uses time-based split (not random) for time-series data
- [ ] Label definition reviewed: correct time alignment, no leakage
- [ ] Output row count assertions in place
- [ ] Monitoring plan includes prediction distribution and feature drift
- [ ] Retraining plan includes validation gate (no auto-promotion without comparison)
