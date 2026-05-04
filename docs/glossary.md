# MLOps Glossary

> Shared vocabulary for every role working with this system. Definitions are scoped to how terms are used in this org — not textbook definitions.

---

## A

**A/B Test**
A live experiment splitting production traffic between two models (or a model and a rule). Used to measure real business impact. Requires statistical significance before declaring a winner. See `docs/decision-frameworks.md §16`.

**Artifact**
Any file produced by a training run — model binary, imputer, encoder, feature importance plot, PSI baseline. Artifacts are stored in MLflow, not in git.

**Artifact Store**
The object storage backend where MLflow saves artifacts. In production: `s3://mlops-prod-artifacts`. In dev: `./artifacts`. Configured via `MLOPS_ARTIFACT_STORE` environment variable.

---

## B

**Batch Scoring**
Running model predictions on a dataset periodically (hourly, daily, weekly). The most common serving pattern. See `docs/decision-frameworks.md §2`.

**Baseline Distribution**
The feature value distribution captured from the training set. Used as the reference when computing PSI drift scores at serving time. Saved as an artifact per training run.

**Branch (git)**
A short-lived pointer to a set of changes. Feature branches in this org must follow `{team}/{use-case}/{description}` naming and be merged within 7 days. See `standards/git-and-release.md §1`.

---

## C

**CANDIDATE**
A model lifecycle stage. A model that has passed validation thresholds and is ready for pre-production testing. Equivalent to MLflow's "Staging" registry stage. Can be promoted to APPROVED or rolled back to EXPERIMENTAL.

**Champion Model**
The currently deployed model serving production traffic. The model to beat. All challenger models are compared against it before any promotion. See `docs/decision-frameworks.md §12`.

**Challenger Model**
A candidate model deployed alongside the champion (in shadow mode or A/B) to measure whether it outperforms the champion. If it wins, it becomes the new champion.

**Class Imbalance**
When one outcome class (e.g., churners) is far rarer than the other. Causes naive models to predict the majority class constantly. Mitigation strategies: class weights, oversampling (SMOTE), undersampling, threshold adjustment. See `docs/decision-frameworks.md §10`.

**Config-Driven Design**
All parameters, thresholds, and environment settings come from `configs/*.yaml` or environment variables — never hardcoded. This makes changes reviewable and auditable.

**Contract (Data Contract)**
A versioned specification of a dataset's schema: column names, data types, nullability, ownership. Validated at pipeline ingestion. Defined in `src/core/contracts.py`.

---

## D

**Data Drift**
When the distribution of input features in production diverges from what the model was trained on. Measured using PSI. A PSI above 0.20 triggers a retraining alert.

**Data Lineage**
The traceable path from raw source data to model prediction. Includes data transforms, joins, feature computations, and imputation steps. Required for debugging production failures.

**DEPLOYED**
A model lifecycle stage. The model is actively serving production scoring requests. Only one version per use case should be DEPLOYED at a time.

**Deployment**
The act of promoting a model into production serving. Requires Platform Engineering sign-off, a complete pre-deployment checklist, and a rollback procedure. See `standards/deployment.md`.

---

## E

**Experiment**
An MLflow concept grouping related training runs. One experiment per model use case (e.g., `churn-rf`, `propensity-gbm`). Experiments track all runs for comparison.

**EXPERIMENTAL**
The entry-point model lifecycle stage. Any newly trained model starts here. No production traffic. Can be promoted to CANDIDATE after passing validation thresholds.

**Explainability**
The ability to explain why a model made a specific prediction. Global explainability (feature importance) and local explainability (SHAP values per prediction). Required for high-risk use cases. See `docs/decision-frameworks.md §13`.

---

## F

**F1 Score**
The harmonic mean of precision and recall. The primary metric for imbalanced classification problems. Default minimum threshold: 0.60. See `src/pipelines/validation.py`.

**Feature**
A measurable property of an entity (customer, transaction) used as model input. Features must be defined in the data contract and computed identically in training and serving.

**Feature Engineering**
Transforming raw data into model-ready features. Includes encoding, scaling, imputation, and derived calculations. The most common source of training-serving skew.

**Feature Importance**
A ranking of features by their contribution to model predictions. Logged as an MLflow artifact on every training run. Used for model debugging and drift investigation.

**Feature Store**
A centralised repository for computed, versioned features. Ensures identical feature values between training and serving. Not yet implemented in this reference system — see `docs/architecture.md`.

---

## G

**Governance**
The policies, approvals, and audit trails required before a model reaches production. Includes model card approval, risk review, compliance sign-off. See `src/core/contracts.py:ModelCard`.

---

## H

**Hotfix**
An unplanned urgent change to fix a critical production issue. Bypasses the normal release cycle but not review. Requires a post-incident note within 48 hours. See `standards/git-and-release.md §8`.

**Human-in-the-Loop (HITL)**
A design pattern where a human reviews or overrides model decisions for high-stakes cases. Required when model confidence is low, stakes are high, or regulation demands it. See `docs/decision-frameworks.md §15`.

---

## I

**Imputer**
A transformation that fills in missing values. In this system: trained on the training set, saved as an MLflow artifact, loaded at serving time. Never recomputed on the scoring batch — that causes training-serving skew.

**Inference Pipeline**
The production pipeline that loads a model artifact and scores a new dataset. Lives in `pipelines/inference_pipeline/`. Must use the same feature computation as the training pipeline.

---

## L

**Label**
The outcome variable a model is trained to predict (e.g., `churned_30d`, `converted`). Labels are often only available after a delay. See `docs/decision-frameworks.md §14`.

**Label Delay**
The gap between when a prediction is made and when the true outcome is known. Affects how quickly model performance can be measured and how the train/test split must be constructed.

**Lifecycle Stage**
The current promotion status of a model version. Five stages: `EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED → RETIRED`. Transitions are governed by `src/core/lifecycle.py`.

---

## M

**MLflow**
The experiment tracking and model registry platform used by this system. Tracks runs, metrics, params, and artifacts. Model versions are registered and promoted through registry stages.

**Model Card**
A governance document for every production model. Captures intended use, out-of-scope use, training data, evaluation metrics, known limitations, and ethical considerations. Defined in `src/core/contracts.py:ModelCard`.

**Model Registry**
The MLflow component that manages named, versioned model artifacts. Models progress through None → Staging → Production → Archived registry stages. The canonical source of which model is deployed.

**Monitoring Report**
A structured report of scoring batch statistics: score distribution, PSI per feature, alert flags. Built by `mlops_platform/monitoring_hooks/hooks.py:build_monitoring_report()`.

---

## O

**Offline Evaluation**
Measuring model performance on a held-out historical dataset. Fast and cheap but may not reflect real-world behaviour. See `docs/decision-frameworks.md §16`.

**Online Evaluation**
Measuring model performance on live production traffic — A/B test or interleaved experiment. Expensive but reflects true business impact. See `docs/decision-frameworks.md §16`.

**Orchestrator**
A pipeline runner that executes steps in order with logging and error handling. In this system, pipelines are thin orchestrators that call into `src/` modules.

---

## P

**Pipeline**
A reproducible sequence of steps that produces a model, score batch, or monitoring report. Lives in `pipelines/`. Pipelines are thin orchestrators — business logic lives in `src/`.

**Population Stability Index (PSI)**
A metric measuring how much a feature's distribution has shifted between training and production. PSI < 0.10: stable. 0.10–0.20: warn. > 0.20: significant drift, investigate. See `mlops_platform/monitoring_hooks/hooks.py`.

**Precision**
Of all the cases a model predicted as positive, what fraction actually were positive? High precision = few false alarms.

**Pre-deployment Checklist**
A gated list of checks required before any model reaches production. Defined in `standards/deployment.md`. Includes smoke tests, rollback plan, consumer notification.

**Production**
The live environment serving real business decisions on real customer data. The highest-stakes environment. Promotion requires explicit sign-off. See `standards/git-and-release.md §4`.

**Pull Request (PR)**
The code review gate. Every change to `master` requires a PR with a description, test evidence, and at least one qualified reviewer. See `standards/git-and-release.md §3`.

---

## R

**Recall**
Of all the actual positives, what fraction did the model correctly identify? High recall = few missed positives.

**Retraining**
Training a new model version to replace or supplement the current production model. Triggered by performance degradation, data drift, score shift, or time schedule. See `docs/retraining_triggers.md`.

**Retraining Trigger**
A signal that initiates a retraining pipeline. Four types: performance degradation, feature drift (PSI), score distribution shift, time-based schedule. Evaluated by `mlops_platform/monitoring_hooks/triggers.py`.

**RETIRED**
The terminal model lifecycle stage. A model version no longer used. Retained for audit and reproducibility. Cannot be promoted to any other stage.

**ROC-AUC**
Area Under the Receiver Operating Characteristic Curve. Measures a classifier's ability to distinguish classes across all thresholds. Default minimum threshold: 0.70. Range: 0.5 (random) to 1.0 (perfect).

**Rollback**
Reverting to a previous model version when the current one fails. Requires a pre-documented rollback procedure before any deployment. See `standards/deployment.md`.

**Run**
A single execution of a training pipeline. Each run logs params, metrics, and artifacts to MLflow and gets a unique `run_id`. The `run_id` is the link between code and model artifact.

---

## S

**Scoring**
Applying a trained model to new data to produce predictions. Also called "inference." See `src/services/scoring.py`.

**Score Distribution Shift**
When the statistical distribution of prediction scores changes significantly compared to baseline. Can indicate data drift, model degradation, or upstream data issues. Monitored via mean score comparison.

**Shadow Mode**
Running a new model in production without using its predictions — only logging them for comparison. The safest way to validate a challenger model. See `docs/decision-frameworks.md §12`.

**SHAP (SHapley Additive exPlanations)**
A framework for explaining individual model predictions. Each feature gets a SHAP value showing its positive or negative contribution to the prediction. See `docs/decision-frameworks.md §13`.

**Smoke Test**
A minimal end-to-end test confirming a pipeline runs without crashing. Not a performance test — just a sanity check that the wiring is correct.

**Staging**
The pre-production environment. Uses production-like data (real but access-controlled). Where integration testing and full pipeline validation happen. See `standards/git-and-release.md §4`.

---

## T

**Threshold**
The decision boundary converting a model's probability score into a binary prediction. Default: 0.50. Should be tuned per business context using a cost matrix. See `docs/decision-frameworks.md §8`.

**Training Pipeline**
The pipeline that trains a model on historical data, evaluates it, and registers the result in MLflow. Lives in `pipelines/training_pipeline/`. See `src/pipelines/training.py`.

**Training-Serving Skew**
When the feature computation in training differs from the feature computation in scoring, causing the model to receive different input distributions than it was trained on. The #1 cause of silent production failures.

**Trunk-Based Development**
A git branching strategy where all development happens on short-lived branches that merge frequently to `master`. Prevents long-lived divergent branches. Used by ML teams in this org. See `standards/git-and-release.md §1`.

---

## U

**Unit Test**
A test that verifies a single function or module in isolation. Required for all `src/` modules. Lives in `tests/`. Must pass in CI before any PR is merged.

---

## V

**Validation Gate**
A CI check or pipeline step that rejects a trained model if its metrics fall below defined thresholds. Prevents under-performing models from reaching production. See `src/pipelines/validation.py`.

**Validation Thresholds**
Minimum acceptable metric values for a model to pass validation. Defaults: accuracy ≥ 0.70, F1 ≥ 0.60, ROC-AUC ≥ 0.70. Can be overridden per use case.

---

## W

**Warning Zone**
When a metric is above threshold but within 5 percentage points of it — a caution signal without a hard failure. Appears in validation output. See `src/pipelines/validation.py`.
