# ML Lifecycle: Data to Production

> This document traces the complete journey of an ML use case from initial problem statement to retired model.
> Each layer has an owner, required inputs, and required outputs.

---

## The Seven Layers

```
┌─────────────────────────────────────────────────────┐
│  1. DATA           Raw sources, contracts, quality  │
├─────────────────────────────────────────────────────┤
│  2. FEATURES       Engineering, store, versioning   │
├─────────────────────────────────────────────────────┤
│  3. TRAINING       Experiment tracking, artifacts   │
├─────────────────────────────────────────────────────┤
│  4. EVALUATION     Validation gates, model cards    │
├─────────────────────────────────────────────────────┤
│  5. DEPLOYMENT     Packaging, serving, rollout      │
├─────────────────────────────────────────────────────┤
│  6. MONITORING     Drift, performance, ops health   │
├─────────────────────────────────────────────────────┤
│  7. RETRAINING     Triggers, cadence, governance    │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Data

**Owner:** Data Engineering
**Input:** Raw data sources (databases, APIs, event streams)
**Output:** Contract-validated, versioned datasets ready for feature engineering

### What happens here

- **Data contracts** define what datasets must look like: columns, types, nullability, ownership
- **Data validation** runs at ingestion to catch schema violations, nulls, and out-of-range values early
- **Data versioning** ensures every model can point back to the exact dataset it was trained on
- **Data lineage** tracks where each row came from

### Required artifacts

- `configs/pipeline_contracts.yaml` — schema definitions for all datasets
- Contract validation passing with zero violations before any downstream pipeline runs

### Layer health checklist

- [ ] All datasets used by models have a defined contract
- [ ] Contracts are version-controlled (changes require review)
- [ ] Validation runs on every data load, not just once at setup
- [ ] Row count and freshness SLAs are monitored
- [ ] Null rates and value distributions are tracked over time

---

## Layer 2: Features

**Owner:** Data Engineering / Data Science (shared)
**Input:** Validated raw datasets
**Output:** A feature set (DataFrame or feature store entries) ready for training

### What happens here

- Features are computed from raw data according to a defined specification
- Feature engineering logic is version-controlled and documented
- For shared features, a **feature store** ensures the same computation is used in training and serving
- Feature importance is tracked across model versions

### Critical rule: training-serving parity

The biggest failure mode in ML systems is computing features differently in training vs. serving. The feature engineering code that runs during training must be the same code (not a re-implementation) that runs at scoring time. See `docs/failure-modes.md` §2.

### Feature documentation template

For each feature used in a model, document:

| Field | Example |
|---|---|
| Name | `support_calls_90d` |
| Definition | Count of inbound support contacts in the last 90 calendar days |
| Source table | `events.support_contacts` |
| Computation | `COUNT(contact_id) WHERE contact_date >= CURRENT_DATE - 90` |
| Refresh cadence | Daily at 02:00 UTC |
| Freshness SLA | Data must be <26h old at scoring time |
| Handling nulls | Fill with 0 (no contacts = no calls) |
| Expected range | [0, 50] |

---

## Layer 3: Training

**Owner:** Data Science
**Input:** Feature-engineered dataset conforming to a data contract
**Output:** MLflow run record (params, metrics, artifact), TrainingResult

### What happens here

- Training pipeline fits a model on the prepared features
- All parameters, metrics, and artifacts are logged to MLflow automatically
- Random seeds are fixed and logged for reproducibility
- The output is a `TrainingResult` containing the `run_id` and `model_uri`

### Required MLflow logs per run

| What | How |
|---|---|
| Hyperparameters | `mlflow.log_params(config.model_params)` |
| Evaluation metrics | `mlflow.log_metrics({"accuracy": ..., "f1": ..., "roc_auc": ...})` |
| Model artifact | `mlflow.sklearn.log_model(model, "model")` |
| Feature names | `mlflow.set_tag("features", str(feature_names))` |
| Data contract | `mlflow.set_tag("data_contract", "name:version")` |
| Owner | `mlflow.set_tag("owner", "team@company.com")` |

### What makes a training run reproducible

1. All inputs (data, params, random seed) are logged
2. All library versions are recorded (requirements.txt in artifact)
3. The artifact can be loaded months later with `mlflow.sklearn.load_model(run_id)`

---

## Layer 4: Evaluation

**Owner:** Data Science (training), Model Reviewer (approval)
**Input:** TrainingResult from Layer 3
**Output:** ValidationResult, signed ModelCard, lifecycle status = APPROVED

### What happens here

- **Validation gate**: compares metrics against thresholds defined in `configs/training.yaml`
- **Baseline comparison**: new model vs. current production model AND a simple baseline
- **Fairness check**: performance across protected attribute groups
- **Model card**: documents what the model does, who it's for, and what its limits are
- **Peer review**: another data scientist reviews the training approach and evaluation

### Validation failure is expected behavior

A model failing the validation gate is not a failure of the process — it's the process working correctly. The team should:

1. Understand why the thresholds weren't met
2. Decide whether to adjust features, hyperparameters, or (rarely) thresholds
3. Document the decision in the model card

### The model card is the approval artifact

The model card (see `docs/model_card_template.md`) is the primary document used by risk/compliance to sign off on a model. It must be complete before any model can be APPROVED.

---

## Layer 5: Deployment

**Owner:** ML Engineering / Platform Engineering
**Input:** APPROVED model with signed model card
**Output:** Running scoring service (batch job or API endpoint)

### What happens here

- Model is registered in the MLflow model registry with stage = `Production`
- Deployment pattern is chosen (batch, sync, async — see `docs/decision-frameworks.md`)
- Scoring pipeline is deployed and tested in staging
- Rollback plan is documented before deployment

### Deployment is not the finish line

Most failures in ML systems happen after deployment — not before it. Deploying is the beginning of the operational phase, not the end of the delivery phase.

---

## Layer 6: Monitoring

**Owner:** ML Engineering / Platform Engineering (alerts), Data Science (response)
**Input:** Running predictions, ground truth labels (when available)
**Output:** Drift reports, performance dashboards, retraining signals

### What to monitor (in priority order)

1. **Prediction distribution** — mean score, histogram; alert on >10% shift
2. **Feature drift (PSI)** — compare daily input distributions to training baseline
3. **Pipeline health** — output row count, latency, error rate
4. **Model performance** — accuracy/F1 on windowed actuals when labels are available
5. **Label drift** — actual outcome rate vs. predicted rate over time

### Monitoring is not optional

A model without monitoring is a liability. Monitoring must be configured before DEPLOYED status is granted. See `standards/monitoring.md` for threshold guidance and `docs/runbook.md` for response playbooks.

---

## Layer 7: Retraining

**Owner:** Data Science (trigger evaluation + training), Model Reviewer (approval)
**Input:** Monitoring signals or scheduled trigger
**Output:** New model promoted through the same Layers 3–5 as the initial model

### Retraining is not a shortcut

Retraining follows the exact same process as the initial training:
- Data validation (Layer 1)
- Feature computation (Layer 2)
- Training + artifact logging (Layer 3)
- Validation gate + model card update (Layer 4)
- Staged deployment (Layer 5)

**There are no fast-path promotions.** Urgency does not bypass governance.

### When to retrain

See `docs/decision-frameworks.md` §3 for the retraining strategy decision tree and trigger thresholds.

---

## Lifecycle Status State Machine

```
EXPERIMENTAL ──► CANDIDATE ──► APPROVED ──► DEPLOYED ──► RETIRED
     ▲                │              │             │
     └────────────────┘              └─────────────┘
          (rollback)                    (rollback to CANDIDATE)
```

See `src/core/lifecycle.py` for the enforced transition rules.

---

## Layer Ownership Summary

| Layer | Primary Owner | Secondary |
|---|---|---|
| Data | Data Engineering | — |
| Features | Data Engineering | Data Science |
| Training | Data Science | — |
| Evaluation | Data Science | Risk/Compliance |
| Deployment | ML Engineering | Platform |
| Monitoring | Platform Engineering | Data Science |
| Retraining | Data Science | Model Reviewer |
