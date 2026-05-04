# Architecture Overview

> This document describes the reference architecture this repository implements. It defines system boundaries, component responsibilities, data flows, and the rationale behind key structural decisions.

---

## Contents

1. [System Overview](#1-system-overview)
2. [Component Map](#2-component-map)
3. [Data Flow](#3-data-flow)
4. [Training Architecture](#4-training-architecture)
5. [Inference Architecture](#5-inference-architecture)
6. [Monitoring Architecture](#6-monitoring-architecture)
7. [Retraining Architecture](#7-retraining-architecture)
8. [MLflow Integration Points](#8-mlflow-integration-points)
9. [Environment Architecture](#9-environment-architecture)
10. [What This Architecture Does Not Include](#10-what-this-architecture-does-not-include)

---

## 1. System Overview

This repository defines a **reference MLOps architecture** for enterprise data science teams. It is not a framework or a library — it is a template that any team can adopt for any supervised ML use case.

The architecture is built around one principle: **every model must be reproducible, auditable, and replaceable.** This means:

- Every training run produces an immutable artifact with a unique ID
- Every prediction can be traced back to a specific model version and input data snapshot
- Every model can be rolled back to a previous version within minutes
- Every change to model behaviour goes through code review

### Architecture style

The architecture follows a **pipeline-oriented design** with clear separation between:

1. **Library code** (`src/`) — reusable, tested, no side effects
2. **Orchestrators** (`pipelines/`) — entry points that call library code
3. **Platform tools** (`mlops_platform/`) — monitoring and trigger evaluation
4. **Configuration** (`configs/`) — all parameters and thresholds, no secrets

This separation means business logic is testable in isolation, and pipelines are thin enough to read in minutes.

---

## 2. Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│   (object store, data warehouse, feature pipeline output)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Contract Layer                         │
│   src/core/contracts.py — validates schema before processing    │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────┐       ┌─────────────────────────┐
│  Training Pipeline  │       │   Inference Pipeline     │
│  pipelines/         │       │   pipelines/             │
│  training_pipeline/ │       │   inference_pipeline/    │
└─────────┬───────────┘       └───────────┬─────────────┘
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────┐
│  src/pipelines/     │       │  src/services/           │
│  training.py        │       │  scoring.py              │
│  validation.py      │       └───────────┬─────────────┘
└─────────┬───────────┘                   │
          │                               ▼
          ▼                   ┌─────────────────────────┐
┌─────────────────────┐       │  mlops_platform/         │
│  MLflow             │       │  monitoring_hooks/       │
│  - Experiments      │◄──────│  hooks.py, triggers.py   │
│  - Model Registry   │       └───────────┬─────────────┘
│  - Artifact Store   │                   │
└─────────────────────┘                   ▼
          │                   ┌─────────────────────────┐
          ▼                   │  Retraining Pipeline     │
┌─────────────────────┐       │  pipelines/              │
│  src/core/          │       │  retraining_pipeline/    │
│  lifecycle.py       │       └─────────────────────────┘
│  contracts.py       │
└─────────────────────┘
```

---

## 3. Data Flow

### Training flow

```
Raw data (object store)
    │
    ▼
DataContract.validate_dataframe()     ← blocks if schema invalid
    │
    ▼
Feature engineering
    │
    ▼
train_model(df, config)
    ├── Fit model
    ├── Save imputer / encoder artifacts → MLflow
    ├── Log metrics → MLflow
    └── Register model version → MLflow Model Registry (EXPERIMENTAL)
```

### Inference flow

```
Scoring batch (object store)
    │
    ▼
DataContract.validate_dataframe()     ← validates contract at ingestion
    │
    ▼
Load model artifact from MLflow
Load imputer / encoder from MLflow    ← same artifacts as training
    │
    ▼
Feature engineering                   ← identical computation to training
    │
    ▼
score_batch(df, model_uri)
    │
    ▼
build_monitoring_report()             ← score stats, PSI per feature
    │
    ▼
evaluate_triggers()                   ← check if retrain needed
    │
    ▼
Write scores to output store
```

### Retraining flow

```
Trigger signal (performance drop / drift / schedule)
    │
    ▼
run_retraining_pipeline()
    ├── Load fresh training data
    ├── Run training pipeline (same as training flow above)
    ├── Run validation gate
    └── compare_against_production()
             ├── New model metrics vs production model
             └── If new model wins → transition to CANDIDATE
                 If not → retain current production model
```

---

## 4. Training Architecture

### Key module: `src/pipelines/training.py`

The training module is the single entry point for all model training. It:
- Accepts a DataFrame and a `TrainingConfig`
- Splits data, trains a RandomForest classifier
- Logs all params, metrics (accuracy, F1, ROC-AUC), and the model artifact to MLflow
- Returns a `TrainingResult` with run_id, metrics, model_uri, feature names

### Design decisions

**Why a single `train_model()` function?**
All use cases share the same training contract. Experiment-specific logic (feature engineering, preprocessing) happens before this call. This makes validation, logging, and reproducibility uniform across use cases.

**Why RandomForest as the default?**
Interpretable, robust to missing values (with imputation), performs well on tabular data without tuning, and supports `class_weight="balanced"`. Teams can replace the estimator by passing custom `model_params` — the framework is not opinionated about algorithm choice.

**Why MLflow for every run?**
Every training run must be reproducible. MLflow's run_id is the key linking a code version (git commit) to a model artifact. Without this link, debugging production failures is guesswork.

### Config-driven training

All training parameters come from `configs/training.yaml`:

```yaml
model:
  n_estimators: 100
  max_depth: 10
  class_weight: "balanced"

validation:
  min_accuracy: 0.70
  min_f1: 0.60
  min_roc_auc: 0.70
```

Changing a hyperparameter requires a code review. This is intentional — hyperparameter changes are model changes.

---

## 5. Inference Architecture

### Key module: `src/services/scoring.py`

The scoring module loads a model by URI and scores a DataFrame. It:
- Loads the model artifact from MLflow using the provided URI
- Returns a `ScoringResult` with entity IDs, probabilities, and binary predictions
- Does not compute features — the caller must provide a feature-ready DataFrame

### Why inference is separated from training

The inference pipeline loads artifacts saved during training (imputer, encoder, model). It does not recompute any statistics from the scoring batch. This is the fundamental design choice that prevents training-serving skew.

```
Training time:  fit imputer on training set → save to MLflow
Scoring time:   load imputer from MLflow → transform scoring batch
                                          ↑ never refit
```

### Stateful transforms as artifacts

Any transformation that computes statistics from data (imputation means, encoding categories, scaling parameters) must be saved as an MLflow artifact at training time and loaded at scoring time:

```python
# Training
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)
mlflow.sklearn.log_model(imputer, "imputer")

# Scoring (inference pipeline)
imputer_uri = f"runs:/{run_id}/imputer"
imputer = mlflow.sklearn.load_model(imputer_uri)
X_scored = imputer.transform(X_scoring)
```

---

## 6. Monitoring Architecture

### Key modules: `mlops_platform/monitoring_hooks/`

Monitoring runs at the end of every inference pipeline execution. It produces a structured report and evaluates whether retraining is needed.

### What is monitored

| Signal | Implementation | Alert threshold |
|---|---|---|
| Score distribution mean | `build_monitoring_report()` | Shift > 0.10 from baseline |
| Score distribution std | `build_monitoring_report()` | Used as context |
| PSI per feature | `compute_psi()` | < 0.10 stable, 0.10–0.20 warn, > 0.20 alert |
| Low confidence predictions | `build_monitoring_report()` | > 20% of batch in (0.4, 0.6) range |

### PSI computation

```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
```

Computed per feature using 10 equal-frequency buckets. The expected distribution is the training set baseline, saved as an artifact at training time.

### Trigger evaluation

After building the monitoring report, `evaluate_triggers()` checks five signals in priority order:

```
1. Performance degradation (if ground truth available)   → immediate
2. PSI alert (> 0.20)                                    → immediate
3. PSI warning (> 0.10)                                  → schedule
4. Score distribution shift                              → schedule
5. Time-based schedule exceeded                          → schedule
```

Urgency levels: `immediate` (retrain now), `schedule` (queue for next window), `none`.

---

## 7. Retraining Architecture

### Key module: `pipelines/retraining_pipeline/retrain.py`

Retraining is triggered by a `TriggerDecision` from the monitoring layer. The retraining pipeline:

1. Loads fresh training data
2. Runs the standard training pipeline
3. Runs the validation gate
4. Compares the new model against the current production model using `compare_against_production()`
5. If the new model is better (within tolerance), transitions it to CANDIDATE
6. A human then reviews and approves promotion to APPROVED/DEPLOYED

### Why human approval is required for production promotion

Automated retraining to CANDIDATE is acceptable. Automated deployment to production is not. This is a deliberate constraint:

- Model behaviour changes can have downstream business impact
- Drift may indicate a data pipeline problem, not just a model problem
- Compliance requirements in many domains require human sign-off

### Champion-challenger comparison

```python
def compare_against_production(new_metrics, prod_metrics, tolerance=0.02):
    """New model must beat production on F1 by at least tolerance."""
```

The tolerance parameter prevents unnecessary model churn — a new model must clearly outperform, not just marginally.

---

## 8. MLflow Integration Points

MLflow is the central store for all run data and model artifacts. Integration points:

| Component | MLflow call | What is stored |
|---|---|---|
| `train_model()` | `mlflow.start_run()` | Params, metrics, model artifact |
| `train_model()` | `mlflow.log_artifact()` | PSI baseline, imputer, encoder |
| `train_model()` | `mlflow.register_model()` | Model version in registry |
| `score_batch()` | `mlflow.pyfunc.load_model()` | Loads model by URI |
| Inference pipeline | `mlflow.sklearn.load_model()` | Loads imputer/encoder by URI |
| Lifecycle | `mlflow.tracking.MlflowClient()` | Transitions registry stage |

### Run ID as the audit key

Every training run produces a unique `run_id`. This ID must be:
- Referenced in the commit message body when committing training results
- Included in the model card
- Used to retrieve any artifact from that run

```bash
# In git commit message body:
MLflow run: 174f8900b34e4fada1d7067625648da0
```

---

## 9. Environment Architecture

The system runs identically in all three environments. The only difference is configuration via environment variables.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Development    │    │     Staging       │    │   Production     │
│                  │    │                  │    │                  │
│ MLOPS_ENV=dev    │    │ MLOPS_ENV=staging │    │ MLOPS_ENV=prod   │
│                  │    │                  │    │                  │
│ MLflow: local    │───►│ MLflow: staging   │───►│ MLflow: prod     │
│ Data: synthetic  │    │ Data: masked prod │    │ Data: full prod  │
│ Model: EXPRMNTL  │    │ Model: CANDIDATE  │    │ Model: DEPLOYED  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                    Same code, same config structure
                    Different environment variables only
```

### Code is environment-agnostic

Pipeline code contains no `if env == "prod":` branches. All environment-specific behaviour is controlled by environment variables. This means the staging run is a genuine test of the production code path.

---

## 10. What This Architecture Does Not Include

This reference architecture is intentionally minimal. The following components are not implemented here and would be added by a platform team for production scale:

| Component | Description | Common choices |
|---|---|---|
| **Feature Store** | Centralised, versioned feature serving for both training and inference | Feast, Tecton, Hopsworks |
| **Workflow Orchestrator** | Scheduled pipeline execution with DAG dependencies | Airflow, Prefect, Dagster |
| **Serving Layer** | Real-time model serving via REST API | FastAPI + MLflow, Seldon, BentoML |
| **Data Versioning** | Immutable, versioned snapshots of training datasets | DVC, Delta Lake, LakeFS |
| **A/B Testing Framework** | Traffic splitting and statistical significance testing | Custom, or platform-provided |
| **Alerting Integration** | Monitoring report → PagerDuty / Slack alerts | Via webhook from trigger evaluation |
| **Model Explainability Dashboard** | SHAP values visualised per prediction | SHAP + custom dashboard |
| **Data Quality Platform** | Automated data quality checks beyond contract validation | Great Expectations, dbt tests |

The architecture is designed so these components can be added incrementally. Each integration point is explicit and documented.
