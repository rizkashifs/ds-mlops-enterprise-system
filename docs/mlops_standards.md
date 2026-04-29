# MLOps Standards Reference

> **Audience:** Data Scientists, ML Engineers, Platform Engineers, Risk & Compliance, Product Owners
> **Scope:** All ML use cases moving from experimentation toward production
> **Last updated:** 2026-04-29

This document defines the operating standards for delivering machine learning systems at scale. It covers the full lifecycle from initial experiment to retired model, and is designed to be adopted by any team, project, or business unit.

---

## Contents

1. [Principles](#1-principles)
2. [Model Lifecycle Stages](#2-model-lifecycle-stages)
3. [Data Contracts](#3-data-contracts)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Model Validation Gates](#5-model-validation-gates)
6. [Model Cards](#6-model-cards)
7. [Deployment Standards](#7-deployment-standards)
8. [Monitoring Standards](#8-monitoring-standards)
9. [Retraining Policy](#9-retraining-policy)
10. [Governance & Approval](#10-governance--approval)
11. [Roles & Responsibilities](#11-roles--responsibilities)
12. [Naming & Versioning Conventions](#12-naming--versioning-conventions)
13. [Checklist: Experiment to Production](#13-checklist-experiment-to-production)

---

## 1. Principles

These five principles drive every standard in this document.

### 1.1 Documentation before deployment
Every model moving to production requires a data contract, experiment record, validation report, and model card. No exceptions. Documentation is not bureaucracy — it is the artifact that lets the next engineer understand what was built, why, and what its limits are.

### 1.2 Explicitness over assumption
Pipelines must surface failures loudly. Silent failures (missing metrics, empty outputs, unchecked schema drifts) cause production incidents. Every contract violation, threshold breach, or missing artifact should halt the pipeline and log the reason.

### 1.3 Modular ownership
Each lifecycle stage (data, training, validation, deployment, monitoring) has a clear owner and a well-defined input/output contract. Owners of one stage do not reach into another stage's implementation. This boundary reduces coupling and enables independent scaling.

### 1.4 Reproducibility by default
Every experiment must be reproducible from its logged artifacts. Random seeds, data versions, package versions, and environment configurations are tracked automatically. A team member who wasn't present during training must be able to reproduce the result from the MLflow run record alone.

### 1.5 Simplicity over cleverness
The standard pattern is always preferred over an exotic one. A well-documented RandomForest that meets the threshold beats a complex ensemble that nobody can explain. Complexity must be justified by a measurable outcome, not by novelty.

---

## 2. Model Lifecycle Stages

Every model moves through five stages. Promotion between stages requires explicit approval. Rollbacks are supported at every stage except RETIRED.

```
EXPERIMENTAL ──► CANDIDATE ──► APPROVED ──► DEPLOYED ──► RETIRED
     ▲                │              │             │
     └────────────────┘              └─────────────┘
          (rollback)                    (rollback)
```

### Stage Definitions

| Stage | Description | Who can promote | Exit criteria |
|---|---|---|---|
| **EXPERIMENTAL** | Active development. Not used in production. | Data Scientist | Training pipeline runs successfully |
| **CANDIDATE** | Training complete. Undergoing validation and review. | Data Scientist | All pipeline artifacts logged |
| **APPROVED** | Passed all validation gates and governance review. | Model Reviewer / Risk | Validation report + model card signed off |
| **DEPLOYED** | Serving predictions in production. | ML Engineer / Platform | Deployment checklist complete |
| **RETIRED** | No longer active. Archived. Terminal state. | Model Owner / Governance | Deprecation notice sent, replacement identified |

### Entry Requirements by Stage

**EXPERIMENTAL → CANDIDATE**
- [ ] Training pipeline ran to completion without errors
- [ ] All required metrics logged (accuracy, F1, ROC-AUC)
- [ ] Data contract validated — zero violations
- [ ] Artifacts stored in the MLflow run record
- [ ] Feature names logged

**CANDIDATE → APPROVED**
- [ ] All validation thresholds met (see §5)
- [ ] Model card complete and reviewed (see §6)
- [ ] No undisclosed data leakage
- [ ] Baseline comparison performed (new model vs. current production)
- [ ] Fairness check completed for protected attribute groups

**APPROVED → DEPLOYED**
- [ ] Deployment checklist signed off (see §7)
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured (see §8)
- [ ] Stakeholder sign-off obtained

**DEPLOYED → RETIRED**
- [ ] Deprecation notice sent to all consumers at least 2 weeks in advance
- [ ] Replacement model identified (or use case discontinued)
- [ ] Final audit log saved

---

## 3. Data Contracts

A **data contract** is a versioned schema declaration that defines what a dataset must look like before any pipeline may consume it.

### Why data contracts matter

Without contracts, schema changes in upstream data silently corrupt models. A contract is the earliest possible failure — catching problems at ingestion rather than at scoring time (or worse, in production output).

### Contract structure

```yaml
# configs/pipeline_contracts.yaml — example
churn_features_v1:
  version: "1.0"
  owner: data-engineering
  description: Customer features for churn prediction
  columns:
    - name: tenure_months
      dtype: numeric
      nullable: false
    - name: monthly_charges
      dtype: numeric
      nullable: false
    - name: num_products
      dtype: numeric
      nullable: false
    - name: support_calls_90d
      dtype: numeric
      nullable: false
```

In code:

```python
from src.core.contracts import DataContract, ColumnSpec, ColumnType

contract = DataContract(
    name="churn_features_v1",
    version="1.0",
    owner="data-engineering",
    columns=[
        ColumnSpec(name="tenure_months", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="monthly_charges", dtype=ColumnType.NUMERIC),
    ],
)

violations = contract.validate_dataframe(df)
if violations:
    raise ValueError(f"Contract violations: {violations}")
```

### Contract versioning rules

- **Minor version** (1.0 → 1.1): additive changes only (new nullable columns).
- **Major version** (1.x → 2.0): any breaking change (column removal, type change, nullability tightening).
- A model trained on v1.x **may not** be used with v2.x data without revalidation.
- All contracts are checked into `configs/pipeline_contracts.yaml`.

### What to validate

| Check | Required |
|---|---|
| All expected columns present | Yes |
| No nulls in non-nullable columns | Yes |
| Column types match specification | Yes |
| Row count within expected range | Recommended |
| No duplicate primary keys | Use-case dependent |
| Value distribution within historical bounds | Recommended (data drift) |

---

## 4. Experiment Tracking

Every training run is an experiment. Every experiment is logged to MLflow. This is non-negotiable — experiments that are not tracked cannot be reviewed, reproduced, or promoted.

### What to log in every run

| Artifact | MLflow call | Required |
|---|---|---|
| Hyperparameters | `mlflow.log_params(...)` | Yes |
| Evaluation metrics | `mlflow.log_metrics(...)` | Yes |
| Trained model artifact | `mlflow.sklearn.log_model(...)` | Yes |
| Feature names | `mlflow.log_param("features", ...)` | Yes |
| Data contract name + version | `mlflow.set_tag(...)` | Yes |
| Training dataset hash or version | `mlflow.set_tag(...)` | Recommended |
| Python + library versions | `mlflow.log_artifact(requirements.txt)` | Recommended |

### Experiment naming convention

```
{use-case}-{model-type}-{environment}

Examples:
  churn-prediction-rf-dev
  fraud-detection-xgb-staging
  ltv-regression-linear-prod
```

### Run tagging

All runs must include:

```python
mlflow.set_tags({
    "data_contract": "churn_features_v1:1.0",
    "owner": "jane.smith@company.com",
    "use_case": "customer_churn",
    "lifecycle_stage": "experimental",
})
```

### What not to do

- Do not log experiments manually from notebooks without using `mlflow.start_run()`.
- Do not commit model artifacts to git. Models belong in the artifact store (S3, GCS, Azure Blob).
- Do not reuse run IDs. Each training run produces a new run_id.

---

## 5. Model Validation Gates

The validation gate is the quality checkpoint between CANDIDATE and APPROVED. A model that fails this gate cannot advance in the lifecycle.

### Standard metric thresholds

These are **minimum defaults**. Each use case should define its own thresholds in `configs/config.yaml` based on business requirements.

| Metric | Default minimum | Notes |
|---|---|---|
| Accuracy | 0.70 | Binary classification baseline |
| F1 Score | 0.60 | Balances precision/recall |
| ROC-AUC | 0.70 | Discrimination ability |

### Setting thresholds in code

```python
from src.pipelines.validation import ValidationThresholds, validate_model

thresholds = ValidationThresholds(
    min_accuracy=0.75,
    min_f1=0.65,
    min_roc_auc=0.80,
)

result = validate_model(training_result.metrics, thresholds)

if not result.passed:
    print(result.summary())
    # FAILED
    #   FAIL  f1 = 0.5812  (threshold: 0.6500)
```

### Baseline comparison

Before promoting a model, compare it to the current production model on the same holdout set:

| Comparison | Required |
|---|---|
| New model vs. production model on same test set | Yes |
| New model vs. simple baseline (majority class, mean) | Yes |
| New model vs. previous champion on out-of-time sample | Recommended |

A model that outperforms production on the standard test set but underperforms on recent data (out-of-time) is a signal of distribution shift — investigate before promoting.

### Fairness validation

Before APPROVED status, run a fairness check if the use case may affect people differently based on protected attributes:

- Report performance (accuracy, F1) separately for each protected group.
- Flag groups where performance differs by more than 5pp from the overall metric.
- Document findings in the model card.

---

## 6. Model Cards

A model card is a 1–2 page document that answers the key governance questions about a model before it goes into production.

**Model cards are required for APPROVED and DEPLOYED status.** They cannot be partially filled in — all fields must be complete.

### Required fields

| Field | Description |
|---|---|
| `model_name` | Unique identifier for the model |
| `version` | Semantic version (e.g., 2.1.0) |
| `owner` | Team or individual accountable for the model |
| `created_date` | ISO date of initial training |
| `description` | Plain-language description of what the model does |
| `intended_use` | Specific use cases the model is designed for |
| `out_of_scope_use` | Use cases the model must NOT be used for |
| `training_data` | Description of the dataset: source, date range, size |
| `evaluation_metrics` | All tracked metrics with values |
| `known_limitations` | What the model is bad at; edge cases; distributional gaps |
| `ethical_considerations` | Potential for harm, bias analysis, fairness findings |
| `approval_status` | Current lifecycle status |

See `docs/model_card_template.md` for a fill-in template.

---

## 7. Deployment Standards

### Deployment patterns

| Pattern | Description | When to use |
|---|---|---|
| **Batch scoring** | Scheduled job scores a table and writes results | Daily/weekly predictions (churn, LTV, risk) |
| **Online inference (sync)** | REST API serves predictions in real time | <500ms latency required; low volume |
| **Online inference (async)** | Message queue + consumer pattern | High volume; latency tolerance >1s |
| **Shadow mode** | New model runs in parallel but results not used | Validating a new model in prod traffic before cutover |

### Pre-deployment checklist

- [ ] Model registered in MLflow model registry with `Production` tag
- [ ] Model card signed off by risk/governance
- [ ] Deployment tested in staging environment with prod-like data
- [ ] Rollback procedure documented (which model replaces this one on failure)
- [ ] Monitoring dashboards created and alert thresholds set
- [ ] Consumer teams notified of deployment date and any schema changes
- [ ] Load test completed (online inference only)

### Environment configuration

All environment-specific settings (endpoints, credentials, feature store URIs) live in environment variables — never in the codebase.

```yaml
# configs/config.yaml — structure only, no secrets
storage:
  artifact_store: ${ARTIFACT_STORE_URI}
  feature_store: ${FEATURE_STORE_URI}
scoring:
  batch_output_path: ${SCORING_OUTPUT_PATH}
```

---

## 8. Monitoring Standards

A deployed model that is not monitored is a liability. Monitoring catches three types of failure: performance degradation, data drift, and operational errors.

### What to monitor

| Signal | Metric | Threshold | Frequency |
|---|---|---|---|
| **Prediction distribution** | Mean score, score histogram | >10% shift from baseline | Daily |
| **Feature drift** | PSI (Population Stability Index) or KS test | PSI > 0.2 = alert | Daily |
| **Label drift** | Actual vs. predicted rate (when labels available) | >5pp gap sustained 7 days | Weekly |
| **Model performance** | Accuracy, F1 on windowed actuals | >5pp drop from validation | Weekly |
| **Data volume** | Record count | >20% drop or spike | Daily |
| **Pipeline latency** | p50, p95 inference time | p95 > 2× baseline | Real-time (online) |
| **Error rate** | Failed predictions / total | >0.1% | Real-time (online) |

### PSI interpretation

| PSI value | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | Continue as normal |
| 0.10 – 0.20 | Moderate shift | Investigate; consider retraining |
| > 0.20 | Significant shift | Alert; likely trigger retraining |

### Alert routing

- **Warning alerts** → model owner (email/Slack)
- **Critical alerts** → model owner + platform team (PagerDuty or equivalent)
- **Data pipeline failures** → data engineering team

---

## 9. Retraining Policy

### When to retrain

Retraining should be triggered by any of these conditions:

| Trigger | Condition | Priority |
|---|---|---|
| Performance degradation | Model performance drops >5pp below validation baseline | High |
| Data drift | PSI > 0.20 on any key feature | High |
| Label drift | Actual vs. predicted gap sustained for 7 days | High |
| Scheduled retraining | Configured cadence (e.g., quarterly) | Medium |
| Upstream schema change | Data contract version bump | Medium |
| Business change | New product, segment, or regulation | Low / manual |

### Retraining process

1. Pull fresh training data conforming to the current data contract version.
2. Run the standard training pipeline with the same `TrainingConfig`.
3. Evaluate against both current production thresholds AND the current production model.
4. The retrained model must outperform (or match within noise) the deployed model.
5. Follow the standard CANDIDATE → APPROVED → DEPLOYED promotion flow.
6. Do not skip validation gates for "urgent" retraining — urgency is not a reason to bypass governance.

### Automated vs. manual retraining

- **Manual retraining** is the default. A human reviews the trigger, approves the retrain, and signs off the promotion.
- **Automated retraining** (triggered by monitoring alerts without human review) requires explicit approval from the governance team and a documented rollback SLA.

---

## 10. Governance & Approval

### Review roles

| Role | Responsibility |
|---|---|
| **Model Owner** | Data Scientist accountable for correctness, documentation, and lifecycle transitions |
| **Model Reviewer** | Peer reviewer from the team who validates training and validation approach |
| **Risk/Compliance** | Reviews model card, fairness analysis, and ethical considerations |
| **Platform Engineering** | Reviews deployment architecture, monitoring setup, and operational readiness |

### Approval gates summary

| Gate | Required approvals | Blocking criteria |
|---|---|---|
| CANDIDATE → APPROVED | Model Reviewer, Risk/Compliance | Validation failure, incomplete model card, fairness issue |
| APPROVED → DEPLOYED | Platform Engineering | Missing monitoring, no rollback plan |
| DEPLOYED → RETIRED | Model Owner, consumers notified | Replacement not identified |

### Audit requirements

The following artifacts must be retained for **3 years** post-retirement (or per applicable regulation):

- MLflow run record (params, metrics, artifacts)
- Model card (all versions)
- Validation report
- Approval sign-off records
- Monitoring summaries (monthly aggregate)

---

## 11. Roles & Responsibilities

| Role | Primary responsibilities |
|---|---|
| **Data Scientist** | Data exploration, feature engineering, training pipeline, validation, model card |
| **ML Engineer** | Pipeline automation, deployment packaging, scoring services, CI/CD |
| **Data Engineer** | Data contracts, feature store, upstream data quality |
| **Platform Engineer** | Infrastructure, model registry, artifact store, monitoring infrastructure |
| **Risk / Compliance** | Fairness review, model card sign-off, audit readiness |
| **Product Owner** | Use case definition, success metric definition, stakeholder communication |

### RACI for lifecycle transitions

| Decision | Responsible | Accountable | Consulted | Informed |
|---|---|---|---|---|
| Start experiment | Data Scientist | DS Team Lead | — | Product Owner |
| Promote to CANDIDATE | Data Scientist | DS Team Lead | — | — |
| Promote to APPROVED | Model Reviewer | Risk/Compliance | DS Team Lead | Platform |
| Promote to DEPLOYED | Platform Engineer | DS Team Lead | Risk | Product Owner |
| Retire model | Model Owner | DS Team Lead | Platform | All consumers |

---

## 12. Naming & Versioning Conventions

### Model naming

```
{use-case}-{algorithm}-v{major}

Examples:
  churn-rf-v1
  fraud-xgb-v2
  ltv-linear-v1
```

### Experiment naming

```
{use-case}-{environment}

Examples:
  churn-prediction-dev
  fraud-detection-staging
```

### Data contract naming

```
{domain}_{entity}_{version}

Examples:
  churn_features_v1
  fraud_transactions_v2
  customer_profile_v3
```

### Semantic versioning for models

| Change type | Version bump | Example |
|---|---|---|
| New production deployment | Major | 1.0 → 2.0 |
| Retrained on new data, same architecture | Minor | 1.0 → 1.1 |
| Hyperparameter tuning only | Patch | 1.0 → 1.0.1 |

### Git branch naming

```
{team}/{use-case}/{short-description}

Examples:
  ds-team/churn/add-tenure-feature
  ml-eng/churn/batch-scoring-service
  platform/infra/mlflow-upgrade
```

---

## 13. Checklist: Experiment to Production

Use this checklist when moving a model from initial training to production.

### Phase 1: Experimentation (EXPERIMENTAL)
- [ ] Data contract defined and version-controlled in `configs/pipeline_contracts.yaml`
- [ ] Training data validated against contract — zero violations
- [ ] Training pipeline runs end-to-end without errors
- [ ] MLflow experiment created with correct naming convention
- [ ] All required metrics logged (accuracy, F1, ROC-AUC)
- [ ] Feature names logged in the run record
- [ ] Random seed set and logged
- [ ] Model artifact stored in MLflow run

### Phase 2: Review prep (CANDIDATE)
- [ ] Validation thresholds defined for this use case in `configs/config.yaml`
- [ ] All validation checks pass (no FAIL items in `ValidationResult`)
- [ ] Warnings reviewed and explained or mitigated
- [ ] Baseline comparison completed (vs. production model and simple baseline)
- [ ] Model card first draft complete
- [ ] Fairness check completed
- [ ] Peer review scheduled

### Phase 3: Approval (APPROVED)
- [ ] Model card reviewed and signed off by risk/compliance
- [ ] Peer code review of training pipeline complete
- [ ] Validation report shared with stakeholders
- [ ] Deployment pattern chosen (batch / online / shadow)
- [ ] Deployment checklist prepared

### Phase 4: Deployment (DEPLOYED)
- [ ] Model registered in MLflow model registry as `Production`
- [ ] Deployment tested in staging with prod-like data
- [ ] Monitoring dashboards created
- [ ] Alert thresholds configured and tested
- [ ] Rollback plan documented
- [ ] Consumer teams notified
- [ ] Go-live sign-off from Platform Engineering

### Phase 5: Production operations
- [ ] First 30 days: daily monitoring review
- [ ] Monthly: performance review against business KPIs
- [ ] Quarterly: scheduled retraining evaluation
- [ ] Annually: full model review (risk, compliance, business relevance)

---

*For questions about this standard, contact the ML Platform team. For exceptions to any requirement, document the justification and obtain sign-off from Risk/Compliance.*
