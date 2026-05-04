# Architecture Decision Records (ADRs)

> This document records the significant architectural decisions made in this system, why they were made, and what alternatives were considered. New decisions should be added here as the system evolves.

---

## ADR Index

| ID | Title | Status | Date |
|---|---|---|---|
| [ADR-001](#adr-001-documentation-first-foundation) | Documentation-first foundation | Accepted | 2026-01-01 |
| [ADR-002](#adr-002-modular-lifecycle-ownership) | Modular lifecycle ownership | Accepted | 2026-01-01 |
| [ADR-003](#adr-003-mlflow-as-experiment-and-registry-backend) | MLflow as experiment and registry backend | Accepted | 2026-01-15 |
| [ADR-004](#adr-004-config-driven-design-for-all-parameters) | Config-driven design for all parameters | Accepted | 2026-01-15 |
| [ADR-005](#adr-005-stateful-transforms-as-mlflow-artifacts) | Stateful transforms as MLflow artifacts | Accepted | 2026-01-20 |
| [ADR-006](#adr-006-data-contracts-as-versioned-code) | Data contracts as versioned code | Accepted | 2026-01-20 |
| [ADR-007](#adr-007-separate-library-code-from-pipeline-orchestrators) | Separate library code from pipeline orchestrators | Accepted | 2026-02-01 |
| [ADR-008](#adr-008-randomforest-as-default-algorithm) | RandomForest as default algorithm | Accepted | 2026-02-01 |
| [ADR-009](#adr-009-four-retraining-trigger-types) | Four retraining trigger types | Accepted | 2026-02-15 |
| [ADR-010](#adr-010-trunk-based-development-for-ml-repos) | Trunk-based development for ML repos | Accepted | 2026-02-15 |
| [ADR-011](#adr-011-human-approval-required-for-production-promotion) | Human approval required for production promotion | Accepted | 2026-03-01 |
| [ADR-012](#adr-012-semantic-versioning-for-models) | Semantic versioning for models | Accepted | 2026-03-01 |
| [ADR-013](#adr-013-psi-as-primary-drift-metric) | PSI as primary drift metric | Accepted | 2026-03-15 |
| [ADR-014](#adr-014-mlops_platform-namespace-for-platform-code) | `mlops_platform/` namespace for platform code | Accepted | 2026-03-15 |

---

## ADR-001: Documentation-first foundation

**Status:** Accepted  
**Date:** 2026-01-01

### Context

When building an MLOps system for an organisation, the first question is: do you start by writing code or writing standards? Teams that start with code often find they've optimised for one use case and the code doesn't generalise. Teams that start with documentation often produce docs no one reads because there's no working reference.

### Decision

Start with system boundaries, contracts, and conventions before writing production-grade implementation code. Document the *why* before the *how*.

The first artefacts in this repo were:
1. The lifecycle model (EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED → RETIRED)
2. The data contract schema
3. The decision frameworks

Code was written to demonstrate and validate these concepts, not to replace them.

### Consequences

**Positive:**
- New team members understand the operating model before they write a line of code
- Standards documents are the source of truth; code is an expression of those standards
- Decisions are traceable to specific documents, not buried in code comments

**Negative:**
- More upfront writing before a working system exists
- Documentation can drift from code if not maintained

**Mitigation:** Standards documents are reviewed in PRs alongside code changes. A standards change without a corresponding code change (or vice versa) is a flag for reviewers.

---

## ADR-002: Modular lifecycle ownership

**Status:** Accepted  
**Date:** 2026-01-01

### Context

ML systems often grow into monolithic pipelines where training, validation, deployment, and monitoring are tangled together. This makes each component hard to change, test, or reason about independently.

### Decision

Training, validation, deployment, monitoring, and governance are represented as distinct modules with explicit interfaces:

- `src/pipelines/training.py` — training only
- `src/pipelines/validation.py` — validation only
- `src/services/scoring.py` — inference only
- `mlops_platform/monitoring_hooks/` — monitoring only
- `src/core/lifecycle.py` — state transitions only
- `src/core/contracts.py` — contracts and governance only

Each module has a single responsibility. Pipelines call modules; modules do not call pipelines.

### Alternatives considered

**Monolithic pipeline class:** A single `MLPipeline` class with methods for train, validate, deploy, monitor. Rejected because it creates coupling that makes testing and partial execution difficult.

**Microservices from day one:** A separate service for each lifecycle stage. Rejected because it introduces operational complexity before the team has validated the core workflow.

### Consequences

**Positive:**
- Each module is independently testable
- Teams can replace one module (e.g., swap the scoring module for a real-time serving layer) without touching others
- Clear ownership boundaries — ML Engineers own pipelines, Data Scientists own `src/pipelines/training.py`

**Negative:**
- More files and directories to navigate initially
- Requires discipline to keep orchestration logic in `pipelines/` and business logic in `src/`

---

## ADR-003: MLflow as experiment and registry backend

**Status:** Accepted  
**Date:** 2026-01-15

### Context

The system needs a way to track experiments, compare runs, store model artifacts, and manage model versions. This could be built custom or use an existing platform.

### Decision

Use MLflow for all experiment tracking, model registry, and artifact storage. The MLflow tracking URI and artifact store are configured via environment variables, making the backend swappable.

### Alternatives considered

**Custom database + object storage:** Maximum control but significant build and maintenance cost. No standard UI for experiment comparison.

**Weights & Biases (W&B):** Excellent UI and features, but cloud-only (data residency concerns) and paid at scale.

**Neptune.ai:** Similar tradeoffs to W&B.

**No tracking:** Not viable — reproducibility is a hard requirement.

### Why MLflow

- Open source with no data residency concerns
- Self-hostable on any infrastructure
- Mature Python SDK with first-class scikit-learn integration
- Built-in model registry with staging/production/archived stages
- Widely adopted — most data scientists have used it

### Consequences

MLflow is a dependency for the entire system. If the org later migrates to a different platform, the integration points are in `src/pipelines/training.py`, `src/services/scoring.py`, and `mlops_platform/monitoring_hooks/`. These are the only files that would need to change.

---

## ADR-004: Config-driven design for all parameters

**Status:** Accepted  
**Date:** 2026-01-15

### Context

Model parameters, validation thresholds, and scoring configuration need to be changeable without modifying source code, and changes need to be reviewable.

### Decision

All parameters and thresholds live in `configs/*.yaml`. Source code reads from config files. No hardcoded values for anything that might change.

```yaml
# configs/training.yaml
model:
  n_estimators: 100
  max_depth: 10
  class_weight: "balanced"
```

Config files are checked into git. Changing a threshold requires a PR — which means it goes through code review.

### What goes in config vs environment variables

- **Config files:** Structure and defaults. Values that define model behaviour. Changes need review.
- **Environment variables:** Environment-specific endpoints, credentials, bucket paths. Never in git.

### Consequences

**Positive:**
- Every parameter change is auditable in git history
- Config changes can be reviewed by a domain expert without reading code
- No environment-specific logic in code

**Negative:**
- Slightly more setup friction for one-off experiments
- Config files can become cluttered if not maintained

---

## ADR-005: Stateful transforms as MLflow artifacts

**Status:** Accepted  
**Date:** 2026-01-20

### Context

Feature transformations like imputation, scaling, and encoding compute statistics from the training set. If these statistics are recomputed at scoring time, the model receives different input distributions than it was trained on — training-serving skew.

### Decision

Any transformation that computes statistics from data must be:
1. Fitted on the training set only
2. Saved as an MLflow artifact
3. Loaded from MLflow at scoring time

```python
# Training — fit and save
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)
mlflow.sklearn.log_model(imputer, "imputer")

# Inference — load and apply
imputer = mlflow.sklearn.load_model(f"runs:/{run_id}/imputer")
X_scored = imputer.transform(X_batch)
```

This applies to: imputers, scalers, ordinal encoders, label encoders, any fitted sklearn transformer.

### Why this matters

Training-serving skew is the #1 cause of silent production failures. A model that passes offline validation can fail in production simply because features look different. By loading the exact same fitted transformers, this risk is eliminated.

### Consequences

- Every training run saves more artifacts (imputer, encoder, etc.) — small storage cost
- Inference pipeline depends on the training run_id to load the right artifacts
- The training run_id must be tracked and accessible to the inference pipeline

---

## ADR-006: Data contracts as versioned code

**Status:** Accepted  
**Date:** 2026-01-20

### Context

Without a formal schema definition, upstream data changes (new columns, renamed fields, changed types) silently break downstream pipelines. Schema documentation written in Confluence or wikis drifts from reality.

### Decision

Data contracts are defined as code in `src/core/contracts.py` (the `DataContract` and `ColumnSpec` classes) and registered in `configs/pipeline_contracts.yaml`. They are validated programmatically at pipeline ingestion using `DataContract.validate_dataframe()`.

Because contracts are in git, a schema change requires a PR — which means data engineers, data scientists, and downstream consumers all see the change before it lands.

### Alternatives considered

**Great Expectations:** More powerful, but adds a significant dependency and operational overhead for teams just starting with contracts.

**JSON Schema / Pydantic models only:** Pydantic was used for the data model, but a simple Python class was preferred over a full Pydantic dependency for this base layer.

**Schema registry (Confluent, Glue):** Appropriate for streaming/event-driven systems. Over-engineered for batch pipelines.

### Consequences

- Schema validation catches upstream data issues at pipeline ingestion, not inside the model
- Contract changes go through PR review, giving consumers visibility
- Contract versioning (`v1.0`, `v1.1`) makes breaking changes explicit

---

## ADR-007: Separate library code from pipeline orchestrators

**Status:** Accepted  
**Date:** 2026-02-01

### Context

Pipelines that contain both business logic and orchestration logic are hard to test and hard to reuse. Testing a pipeline that writes to MLflow requires mocking MLflow.

### Decision

The codebase has two distinct layers:

**Library (`src/`):** Pure functions and classes with no side effects at import time. Testable without infrastructure. Business logic lives here.

**Orchestrators (`pipelines/`):** Entry-point scripts that call library code, write to MLflow, read from databases, write to object storage. Not unit-testable without mocking — integration tested instead.

The rule: if it has a side effect (writes a file, calls an API, logs to MLflow), it belongs in `pipelines/`, not `src/`.

### Consequences

- `src/` modules have high test coverage with pure unit tests
- Pipeline scripts are thin enough to review quickly
- A new use case can reuse `src/pipelines/training.py` without modifying it

---

## ADR-008: RandomForest as default algorithm

**Status:** Accepted  
**Date:** 2026-02-01

### Context

The training module needs a default algorithm for the reference implementation. This choice affects how accessible the examples are to teams evaluating the system.

### Decision

Use `RandomForestClassifier` from scikit-learn as the default. The training interface is algorithm-agnostic — the estimator can be swapped via config.

### Why RandomForest

- Works well on tabular data without extensive tuning
- Supports `class_weight="balanced"` natively — important for imbalanced datasets
- Provides feature importance out of the box
- Familiar to most data scientists
- No GPU required — works in any environment
- Robust to outliers and missing values (after imputation)

### Not chosen

- **XGBoost / LightGBM:** Higher performance but more hyperparameter sensitivity; better for teams that have already solved data contracts and monitoring
- **Logistic Regression:** Simple but often underperforms on non-linear tabular data
- **Neural networks:** Overkill for most tabular business problems; requires GPU for training

### Consequences

The default examples (churn, propensity) use RandomForest. Teams adopting this system for XGBoost or LightGBM use cases replace the estimator in their use-case config — the framework supports this without code changes.

---

## ADR-009: Four retraining trigger types

**Status:** Accepted  
**Date:** 2026-02-15

### Context

Models degrade over time for different reasons. A single "retrain on a schedule" policy misses cases where the model has degraded between scheduled runs (drift) and wastes resources when nothing has changed (schedule-based retrain on a stable signal).

### Decision

Implement four complementary trigger types, evaluated in priority order:

1. **Performance degradation** — measured F1 drops below threshold (requires ground truth)
2. **Feature drift** — PSI exceeds alert threshold for any feature
3. **Score distribution shift** — mean prediction score has shifted significantly
4. **Time-based schedule** — a configured number of days has passed since last retrain

Each trigger has a configurable threshold in `configs/inference.yaml`. The system returns the highest-urgency trigger signal.

### Why four types

| Trigger | What it catches |
|---|---|
| Performance | Direct model quality degradation |
| Feature drift | Upstream data changes before they degrade the model |
| Score shift | Changes in the scoring population (a leading indicator) |
| Time-based | Silent drift that no single metric has caught yet |

Multiple signals provide redundancy. If ground truth is unavailable (label delay), drift and score shift signals still fire.

### Consequences

- Retraining is triggered only when there's evidence, not blindly on schedule alone
- Multiple signals may fire simultaneously; priority order ensures the highest-severity wins
- Teams must configure appropriate thresholds per use case (fraud models have tighter thresholds than propensity models)

---

## ADR-010: Trunk-based development for ML repos

**Status:** Accepted  
**Date:** 2026-02-15

### Context

ML teams frequently debate branching strategy. GitFlow (with long-lived `develop` branches) is common in software engineering but creates specific problems for ML.

### Decision

Use trunk-based development with short-lived feature branches:
- `master` is always deployable
- Feature branches live for at most 7 days
- Branch names follow `{team}/{use-case}/{description}`
- All merges via reviewed PR

### Why not GitFlow

- Long-lived `develop` branches accumulate merge conflicts between experiments
- It becomes hard to trace which code version produced which model artifact
- `develop` branch often diverges from `master`, making the comparison meaningless
- Short-lived branches force smaller, more reviewable changes

### Consequences

- Teams must merge more frequently — some initial friction
- Stale branches are deleted after 7 days — no more zombie branches
- Every commit to master is deployable, which raises the bar for what gets committed

---

## ADR-011: Human approval required for production promotion

**Status:** Accepted  
**Date:** 2026-03-01

### Context

Automated retraining pipelines can train and validate a new model without human involvement. The question is: should deployment to production also be automated?

### Decision

Automated promotion is permitted up to **CANDIDATE** (staging). Promotion from CANDIDATE to APPROVED and from APPROVED to DEPLOYED requires human sign-off.

### Why not fully automated

1. **Model behaviour change has business impact.** A retrained model may perform better on metrics but worse on specific segments that matter to the business.
2. **Drift may indicate a data problem.** If PSI is high because an upstream pipeline changed, the right response may be to fix the data, not retrain the model.
3. **Compliance.** Many industries require documented human approval for model changes in production.
4. **First-deployment risk.** Challenger models have passed validation but have never seen live production traffic. Shadow mode first, then A/B, then full promotion.

### When fully automated promotion is acceptable

For low-stakes, high-frequency use cases (e.g., ad bid optimisation where models retrain daily), the governance overhead may be disproportionate. Teams can propose an exception with documented justification.

### Consequences

- Deployment velocity is lower than fully automated systems
- Humans are accountable for production model behaviour
- Post-incident reviews are cleaner because there's a clear approval log

---

## ADR-012: Semantic versioning for models

**Status:** Accepted  
**Date:** 2026-03-01

### Context

Model versions need to communicate what kind of change was made. `model-v1`, `model-v2` tells you nothing about whether this is a retrain, a new architecture, or a threshold tweak.

### Decision

Apply semantic versioning to model versions with ML-specific meaning:

```
{major}.{minor}.{patch}
   │       │       │
   │       │       └── Hyperparameter tuning only
   │       └────────── Retrained on fresh data; same architecture
   └────────────────── New architecture, features, or breaking change
```

Model naming: `{use-case}-{algorithm}-v{major}` (e.g., `churn-rf-v1`).

### Why ML semantics differ from software semantics

In software, "patch" means a bug fix with no behaviour change. In ML, there's no such thing as a change with no behaviour change — even a hyperparameter tweak changes predictions. So:
- **Patch** = smallest possible change (tuning)
- **Minor** = same pipeline, fresh data (regular retrain)
- **Major** = fundamentally different model (new architecture, new features)

### Consequences

- Version bumps communicate model change severity to downstream consumers
- Major version bumps (new model architecture) require consumer team notification
- Git tags use the same convention: `git tag -a model/churn-rf-v1.1`

---

## ADR-013: PSI as primary drift metric

**Status:** Accepted  
**Date:** 2026-03-15

### Context

Drift detection requires a statistical metric to compare training and production feature distributions. Several options exist.

### Decision

Use Population Stability Index (PSI) as the primary feature drift metric, with thresholds:
- PSI < 0.10: stable
- PSI 0.10–0.20: warn
- PSI > 0.20: alert

### Why PSI

- Industry standard in financial services (credit scoring, fraud) — widely understood
- Interpretable: a PSI of 0.25 means "significant distribution shift"
- Works on both numeric and categorical features
- Thresholds are well-established and defensible to risk/compliance teams
- Easy to compute and explain

### Alternatives considered

**KL Divergence:** Information-theoretically grounded but lacks industry-standard thresholds and can be infinite if a bucket has zero counts.

**Kolmogorov-Smirnov test:** Valid for numeric distributions but doesn't handle categoricals and requires choosing a significance level.

**Jensen-Shannon Divergence:** Symmetric and bounded (0–1) but less industry-familiar.

**Wasserstein Distance:** Strong theoretical properties but computationally expensive for high-dimensional feature spaces.

### Consequences

PSI baseline distributions are saved as MLflow artifacts on every training run. The inference pipeline loads these baselines and computes PSI per feature on every scoring batch.

---

## ADR-014: `mlops_platform/` namespace for platform code

**Status:** Accepted  
**Date:** 2026-03-15

### Context

Platform-level tools (monitoring hooks, trigger evaluation) need a clear home in the repository. During development, these were initially placed in a `platform/` directory, which caused a Python import conflict with the standard library `platform` module.

### Decision

Place platform tools under `mlops_platform/` to avoid the namespace conflict and to make the purpose of the directory explicit.

### The error that drove this decision

```
ModuleNotFoundError: No module named 'platform.monitoring_hooks';
'platform' is not a package
```

`platform` is a Python standard library module. A directory named `platform/` shadows it, causing import failures.

### Lesson

When naming directories that will contain Python packages, check that the name doesn't conflict with the Python standard library or any installed package. Prefer organisation-specific namespaces (`mlops_platform/`, `ds_tools/`, `org_utils/`).

### Consequences

All imports use `from mlops_platform.monitoring_hooks.hooks import ...`. The directory name clearly communicates that this is platform-level infrastructure code, not use-case-specific code.
