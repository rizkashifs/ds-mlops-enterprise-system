# Onboarding Guide — New Teams and Use Cases

> How to adopt this system for a new ML use case. From first meeting to production in a repeatable, auditable way.

---

## Contents

1. [Who This Guide Is For](#1-who-this-guide-is-for)
2. [System Overview — 5-Minute Version](#2-system-overview--5-minute-version)
3. [Roles and Responsibilities](#3-roles-and-responsibilities)
4. [New Use Case Checklist](#4-new-use-case-checklist)
5. [Repository Structure — What Goes Where](#5-repository-structure--what-goes-where)
6. [Step-by-Step: Your First Model](#6-step-by-step-your-first-model)
7. [Tools Setup](#7-tools-setup)
8. [Common Mistakes to Avoid](#8-common-mistakes-to-avoid)
9. [Getting Help](#9-getting-help)

---

## 1. Who This Guide Is For

This guide is for:
- A **data scientist** joining an existing ML team and starting their first project
- An **ML engineer** setting up a new use case for a business problem
- A **team lead** onboarding a squad to the org's MLOps standards

Assumptions:
- You have Python 3.10+ installed
- You have read access to this repository
- You have credentials for the MLflow tracking server (ask your Platform Engineer)

---

## 2. System Overview — 5-Minute Version

This repository is the **operating standard** for building, deploying, and maintaining ML models at this org. It answers four questions:

| Question | Where to look |
|---|---|
| How do we build models? | `src/`, `pipelines/training_pipeline/`, `docs/decision-frameworks.md` |
| How do we deploy them? | `standards/deployment.md`, `pipelines/inference_pipeline/`, `src/core/lifecycle.py` |
| How do we ensure consistency across teams? | `standards/`, `src/core/contracts.py`, `configs/` |
| How do we avoid common failures? | `docs/failure-modes.md`, `docs/decision-frameworks.md`, `mlops_platform/` |

**The seven layers:**

```
Data → Features → Training → Evaluation → Deployment → Monitoring → Retraining
```

Each layer has documented standards, reusable modules, and templates. A new use case plugs into this framework rather than reinventing it.

**Model lifecycle:**

```
EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED → RETIRED
```

Every model version moves through these stages explicitly. See `src/core/lifecycle.py` and `docs/lifecycle.md`.

---

## 3. Roles and Responsibilities

| Role | Core responsibilities |
|---|---|
| **Data Scientist** | Feature design, model training, experiment tracking, evaluation, model cards |
| **ML Engineer** | Pipelines, serving infrastructure, CI/CD, staging deployment, performance monitoring |
| **Data Engineer** | Data contracts, feature pipelines, schema ownership, data quality |
| **Platform Engineer** | MLflow infrastructure, artifact stores, secrets, production deployments |
| **Risk / Compliance** | Model card approval, governance sign-off for regulated use cases |

You don't need all five roles to start. For a small team, one person can cover multiple roles — but the *responsibilities* still need to be owned.

---

## 4. New Use Case Checklist

Work through this checklist in order. Each step has a clear output — don't move to the next step until the current output exists.

### Phase 1 — Scoping

- [ ] **Define the business problem** — What decision does this model inform? What is the cost of a false positive vs false negative?
- [ ] **Select the right approach** — ML, rules, or LLM? Use `docs/decision-frameworks.md §1` and §7
- [ ] **Define the prediction target** — What exactly is being predicted? What is the label? When is it observed?
- [ ] **Identify label delay** — How long after the prediction event is the outcome known? See `docs/decision-frameworks.md §14`
- [ ] **Assess data availability** — Is labeled training data available? How much? See `docs/decision-frameworks.md §9`
- [ ] **Define success metrics** — What accuracy / F1 / business KPI makes this model worth deploying?
- [ ] **Identify consumers** — Which downstream team or system will use the scores?

**Output:** A one-page problem statement shared with the team.

### Phase 2 — Data and Contracts

- [ ] **Define the data contract** — Column names, types, nullability, ownership. See `configs/pipeline_contracts.yaml`
- [ ] **Register the contract** — Add it to `configs/pipeline_contracts.yaml` with owner and version
- [ ] **Validate contract on sample data** — Run `DataContract.validate_dataframe()` on a sample
- [ ] **Document label computation** — Where does the label come from? Is there risk of label leakage?
- [ ] **Identify feature sources** — What tables / APIs provide each feature?

**Output:** A versioned entry in `configs/pipeline_contracts.yaml`.

### Phase 3 — Experiment

- [ ] **Create the MLflow experiment** — `{use-case}-{algorithm}` naming, e.g., `churn-rf`
- [ ] **Create a feature branch** — `ds-team/{use-case}/initial-experiment`
- [ ] **Build training pipeline** — Use `src/pipelines/training.py` as the foundation
- [ ] **Log all params, metrics, artifacts** — Every run must be reproducible from its MLflow entry
- [ ] **Apply class imbalance strategy** — See `docs/decision-frameworks.md §10`
- [ ] **Validate on hold-out set** — Use a time-based split if time is a factor
- [ ] **Pass validation gate** — All metrics above thresholds in `ValidationThresholds`
- [ ] **Document in model card** — Fill in `ModelCard` fields before any promotion

**Output:** A passing MLflow run with a registered model version in EXPERIMENTAL stage.

### Phase 4 — Staging

- [ ] **Promote to CANDIDATE** — Use `lifecycle.transition()`
- [ ] **Create inference pipeline** — Use `pipelines/inference_pipeline/` as template
- [ ] **Test on staging data** — Full run on production-like volume
- [ ] **Verify no training-serving skew** — Feature computation must be identical to training
- [ ] **Set up monitoring hooks** — PSI baseline saved; `build_monitoring_report()` wired in
- [ ] **Complete pre-deployment checklist** — See `standards/deployment.md`
- [ ] **Get required reviewers** — Data contract: Data Engineer. Model: Data Scientist peer

**Output:** A CANDIDATE model with a passing staging run and a complete pre-deployment checklist.

### Phase 5 — Production

- [ ] **Get Platform Engineering sign-off**
- [ ] **Get model card approval** — Risk/Compliance for regulated use cases
- [ ] **Promote to APPROVED, then DEPLOYED**
- [ ] **Tag the git commit** — `git tag -a model/{name}-v{version}`
- [ ] **Notify consumer teams** — At least 48 hours notice for planned releases
- [ ] **Validate post-deployment** — Smoke test confirms scores are flowing
- [ ] **Set up alerting** — Monitoring alerts wired to on-call rotation

**Output:** A DEPLOYED model with live scoring and active monitoring.

---

## 5. Repository Structure — What Goes Where

```
ds-mlops-enterprise-system/
│
├── src/                        # Reusable library code (import this, don't copy it)
│   ├── core/
│   │   ├── contracts.py        # DataContract, ModelCard, ColumnSpec
│   │   └── lifecycle.py        # ModelStatus, can_transition, transition
│   ├── pipelines/
│   │   ├── training.py         # train_model() — the standard training interface
│   │   └── validation.py       # validate_model(), ValidationThresholds
│   └── services/
│       └── scoring.py          # score_batch() — load model + score DataFrame
│
├── pipelines/                  # Thin orchestrators — entry points for pipelines
│   ├── training_pipeline/      # train.py — runs training, logs to MLflow
│   ├── inference_pipeline/     # score.py — loads model, scores batch, monitors
│   └── retraining_pipeline/    # retrain.py — triggered retrain with comparison
│
├── mlops_platform/             # Platform-level tools (monitoring, triggers)
│   └── monitoring_hooks/
│       ├── hooks.py            # build_monitoring_report(), compute_psi()
│       └── triggers.py         # evaluate_triggers(), TriggerConfig
│
├── configs/                    # All parameters, thresholds, contracts (no secrets)
│   ├── training.yaml           # Model params, artifact settings
│   ├── inference.yaml          # Scoring config, retraining trigger thresholds
│   └── pipeline_contracts.yaml # Data contract definitions
│
├── docs/                       # Documentation — the "why" and "how"
│   ├── decision-frameworks.md  # 18 critical decisions every ML team faces
│   ├── failure-modes.md        # How production ML fails and how to prevent it
│   ├── lifecycle.md            # Model lifecycle stages and promotion rules
│   ├── retraining_triggers.md  # When and why to retrain
│   ├── architecture.md         # System architecture overview
│   ├── glossary.md             # Shared vocabulary (this file's companion)
│   └── onboarding.md           # This file
│
├── standards/                  # Org-wide standards — every team follows these
│   ├── coding.md               # Code quality, style, patterns
│   ├── experimentation.md      # Experiment tracking, reproducibility
│   ├── deployment.md           # Pre-deployment checklist, rollback
│   ├── monitoring.md           # What to monitor, alert thresholds
│   └── git-and-release.md      # Branching, commits, CI/CD, versioning
│
├── templates/                  # Copy-paste starting points for new use cases
│   └── batch_inference/
│
├── examples/                   # Working reference implementations
│   ├── churn_demo.py           # End-to-end churn model demo
│   └── marketing_propensity/   # Full propensity model example
│
└── tests/                      # Unit tests for src/ modules
    ├── test_contracts.py
    ├── test_lifecycle.py
    ├── test_validation.py
    └── test_retraining_triggers.py
```

### The golden rule

**`src/` is a library. `pipelines/` are scripts. Don't call pipeline scripts from library code.**

When you write a new use case:
1. Business logic (feature computation, custom metrics) → `src/`
2. Orchestration (run training, save artifact, log to MLflow) → `pipelines/{your-use-case}/`
3. Config (params, thresholds) → `configs/{your-use-case}.yaml`
4. Tests → `tests/test_{your-module}.py`

---

## 6. Step-by-Step: Your First Model

This walkthrough creates a minimal working model using the churn example as reference. Adapt to your use case.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Start MLflow locally

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Set the environment variable:

```bash
export MLOPS_TRACKING_URI=http://127.0.0.1:5000
```

### Step 3 — Run the existing example

Before building your own use case, run the reference example to verify your setup:

```bash
python examples/churn_demo.py
```

Expected output: training run logged to MLflow, validation passing, model registered.

### Step 4 — Define your data contract

Add an entry to `configs/pipeline_contracts.yaml`:

```yaml
contracts:
  - name: your_use_case_v1
    version: "1.0"
    owner: your-team
    columns:
      - name: customer_id
        dtype: numeric
        nullable: false
      - name: feature_a
        dtype: numeric
        nullable: true
      - name: label
        dtype: numeric
        nullable: false
```

### Step 5 — Create your training pipeline

Copy the template:

```bash
cp -r pipelines/training_pipeline/ pipelines/{your-use-case}_training/
```

In your training script, use the standard training interface:

```python
from src.pipelines.training import TrainingConfig, train_model

config = TrainingConfig(
    experiment_name="your-use-case-rf",
    model_params={"n_estimators": 100, "class_weight": "balanced"},
    target_column="label",
)
result = train_model(df, config)
print(result.metrics)
```

### Step 6 — Validate the model

Validation runs automatically inside `train_model()`. If thresholds are appropriate for your use case, override them:

```python
from src.pipelines.validation import ValidationThresholds, validate_model

thresholds = ValidationThresholds(min_accuracy=0.75, min_f1=0.65, min_roc_auc=0.75)
result = validate_model(metrics, thresholds)
if not result.passed:
    raise ValueError(result.summary())
```

### Step 7 — Register and promote

```python
from src.core.lifecycle import ModelStatus, transition

# In MLflow, register the model
# Then in your system:
current = ModelStatus.EXPERIMENTAL
promoted = transition(current, ModelStatus.CANDIDATE)
```

### Step 8 — Wire up monitoring

In your inference pipeline, call:

```python
from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers

report = build_monitoring_report(scores, df, psi_baseline)
trigger = evaluate_triggers(report, TriggerConfig())
if trigger.should_retrain:
    # queue retraining job
    pass
```

---

## 7. Tools Setup

### Required

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| MLflow | 2.x | Experiment tracking, model registry |
| scikit-learn | 1.4+ | Model training |
| pandas | 2.x | Data manipulation |
| pytest | 8.x | Tests |

### Recommended

| Tool | Purpose |
|---|---|
| `ruff` | Fast Python linter |
| `mypy` | Type checking |
| `pre-commit` | Run linting before every commit |

### Environment variables

```bash
# Development (local)
export MLOPS_TRACKING_URI=http://localhost:5000
export MLOPS_ARTIFACT_STORE=./artifacts
export MLOPS_ENV=dev

# Staging
export MLOPS_TRACKING_URI=https://mlflow.staging.internal
export MLOPS_ARTIFACT_STORE=s3://mlops-staging-artifacts
export MLOPS_ENV=staging
```

Never put these in code or config files checked into git. Use a `.env` file (gitignored) for local development.

### Pre-commit setup (recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs linting and test checks before every commit locally, catching issues before CI.

---

## 8. Common Mistakes to Avoid

These are the most frequent mistakes teams make when adopting this system. They're in `docs/failure-modes.md` in full — read that document before going to production.

### Training-serving skew

**The mistake:** Computing features differently in training and scoring.

**The fix:** Save your imputer, encoder, and any stateful transformers as MLflow artifacts at training time. Load them at scoring time. Never recompute from the scoring batch.

```python
# Wrong — recomputes mean on scoring batch
df["feature"] = df["feature"].fillna(df["feature"].mean())

# Right — loads the saved imputer from the training run
imputer = mlflow.sklearn.load_model(imputer_uri)
df["feature"] = imputer.transform(df[["feature"]])
```

### Leaking future data into training

**The mistake:** Including features that wouldn't be available at prediction time.

**The fix:** Audit every feature for temporal leakage. Ask: "At prediction time T, would I actually have this value?"

### Not logging reproducibly

**The mistake:** Running training in a notebook, forgetting to log params, losing the run.

**The fix:** Every training run goes through `train_model()`. Every run logs params, metrics, and artifacts. Use the `run_id` to reproduce any result.

### Skipping the data contract

**The mistake:** Starting modeling without defining the contract, discovering schema issues in staging.

**The fix:** Define the contract first (step 4 in the checklist). Validate it on the first data sample. This takes 30 minutes and saves days.

### Hardcoding thresholds

**The mistake:** `if score > 0.5:` in your scoring script.

**The fix:** All thresholds in `configs/inference.yaml`. Reference via config. Changes go through git review.

---

## 9. Getting Help

| Problem | Who to ask | Where |
|---|---|---|
| Data access, schema questions | Data Engineer | data-engineering Slack channel |
| Model quality, feature ideas | Data Scientist peer | ds-team Slack channel |
| Pipeline infrastructure, MLflow | ML Engineer | ml-engineering Slack channel |
| Production deployments, environment | Platform Engineer | platform Slack channel |
| Governance, compliance questions | Risk / Compliance | risk-review Slack channel |

**For bugs in this system:** Open a GitHub issue in this repo.

**For improvements to standards:** Open a PR with your proposed change and tag a reviewer from the relevant team.

The standards in this repo are living documents. If something doesn't make sense for your use case, raise it — the goal is consistency, not rigidity.
