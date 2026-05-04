# DS MLOps Enterprise System

**An operating system for data science teams.**

This repository answers four questions every DS org needs to answer before it can scale:

1. **How do we build models?** → `standards/`, `src/`, `pipelines/`
2. **How do we deploy them?** → `standards/deployment.md`, `templates/`
3. **How do we ensure consistency across teams?** → `docs/`, `configs/`, `standards/`
4. **How do we avoid common failures?** → `docs/failure-modes.md`, `docs/decision-frameworks.md`

---

## Key Philosophy

Most machine learning systems fail not because of poor models, but because of inconsistent processes and lack of system design.

This repository focuses on standardization, reproducibility, and lifecycle management.

## Why This Matters

Most real-world AI systems fail not due to model performance, but due to:
- lack of standardization
- poor observability
- weak lifecycle management

This repository addresses these challenges through system design and architectural patterns.

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

See `docs/lifecycle.md` for the full layer-by-layer guide.

---

## Design Principles

These principles govern every decision in the repo. When in doubt, apply them.

| Principle | What it means |
|---|---|
| **Reproducibility is non-negotiable** | Data + code + config must produce the same model every time. Data versioning, locked configs, and seeded randomness are not optional. |
| **Standards define what; teams choose how** | Orchestration, serving, and tracking are defined as standards — not bundled as fixed tools. Teams pick AWS Step Functions or Airflow, MLflow or a local tracker, FastAPI or their own serving layer. |
| **Config over code** | Every decision point (thresholds, paths, algorithm params, tracker backend) has a config key. Nothing is hardcoded in pipelines. |
| **Fail fast at validation gates** | Problems caught before production are cheap. Problems caught during scoring are not. Every pipeline has explicit validation gates: data contracts, metric thresholds, fairness checks. |
| **Incremental adoption** | Teams adopt what they need. MLflow is optional. Orchestration is optional. DVC is optional. The minimum viable implementation is documented for each component. |
| **Algorithm-independent** | Any sklearn-compatible estimator works — RandomForest, XGBoost, LightGBM, custom. No lock-in. Pass your estimator to `TrainingConfig.estimator`. |
| **Pluggable tracking** | The `ExperimentTracker` protocol decouples training from tracking. Use `LocalFileTracker` (default), `MLflowTracker`, `NoOpTracker`, or implement your own. |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the churn prediction end-to-end demo
python examples/churn_demo.py

# Run the marketing propensity demo
python examples/marketing_propensity/pipeline.py

# Run tests
pytest tests/ -v
```

---

## Repository Structure

```
ds-mlops-enterprise-system/
│
├── README.md
│
├── docs/                          # System documentation
│   ├── lifecycle.md               # The seven layers explained
│   ├── decision-frameworks.md     # 18 critical ML decisions with frameworks
│   ├── failure-modes.md           # What goes wrong and how to prevent it
│   ├── retraining_triggers.md     # Trigger types, config, priority, integration
│   ├── architecture.md            # System architecture, data flow, component map
│   ├── decisions.md               # Architecture Decision Records (ADRs)
│   ├── glossary.md                # Shared vocabulary for all roles
│   ├── onboarding.md              # New team checklist — from scoping to production
│   ├── mlops_standards.md         # Full MLOps standards reference
│   ├── model_card_template.md     # Fill-in model card template
│   ├── data_contract_guide.md     # How to define and use data contracts
│   └── runbook.md                 # Day-2 operations playbook
│
├── standards/                     # Org-wide standards — every team follows these
│   ├── coding.md                  # Code style, naming, testing, config-driven design
│   ├── experimentation.md         # MLflow logging requirements, experiment hygiene
│   ├── deployment.md              # Deployment patterns, pre-deploy checklist, API contracts
│   ├── monitoring.md              # What to monitor, PSI, alert thresholds
│   └── git-and-release.md         # Branching, commits, CI/CD, environments, model versioning
│
├── src/                           # Reusable core modules
│   ├── core/
│   │   ├── contracts.py           # DataContract, ModelCard, ColumnSpec
│   │   └── lifecycle.py           # ModelStatus state machine (EXPERIMENTAL → RETIRED)
│   ├── pipelines/
│   │   ├── training.py            # train_model() — fits, logs to MLflow, returns TrainingResult
│   │   └── validation.py          # validate_model() — checks metrics against thresholds
│   └── services/
│       └── scoring.py             # score_batch() — loads model, scores DataFrame
│
├── pipelines/                     # Orchestrators (entry points for scheduled jobs)
│   ├── training_pipeline/train.py
│   ├── inference_pipeline/score.py
│   └── retraining_pipeline/retrain.py
│
├── mlops_platform/                # Shared platform tools
│   ├── experiment_tracking/tracker.py        # Enforces required MLflow logging
│   ├── model_registry/registry.py            # Model registration and promotion
│   └── monitoring_hooks/
│       ├── hooks.py                           # PSI, score distribution, monitoring reports
│       └── triggers.py                        # Retraining trigger evaluation (4 trigger types)
│
├── templates/                     # Copy-paste starting points for new use cases
│   ├── tabular_ml_pipeline/       # Standard ML pipeline (copy this first)
│   ├── genai_pipeline/            # LLM extraction/generation pipeline
│   ├── batch_inference/           # Scheduled scoring job
│   └── realtime_api/              # FastAPI online inference endpoint
│
├── examples/                      # Runnable end-to-end examples
│   ├── churn_prediction/          # Customer churn (Random Forest, batch)
│   │   └── README.md
│   ├── marketing_propensity/      # Campaign response (GBM, batch)
│   │   ├── pipeline.py
│   │   └── README.md
│   └── churn_demo.py              # Single-file end-to-end lifecycle demo
│
├── configs/                       # All configuration — no hardcoded values in code
│   ├── config.yaml                # Project-level settings
│   ├── training.yaml              # Training pipeline config
│   ├── inference.yaml             # Inference pipeline config
│   └── pipeline_contracts.yaml   # All dataset schemas (data contracts)
│
├── tests/
│   ├── test_contracts.py
│   ├── test_lifecycle.py
│   └── test_validation.py
│
└── requirements.txt
```

---

## Decision Frameworks

Before writing any code, read `docs/decision-frameworks.md`. These 22 frameworks answer the questions teams ask at the start of every ML project. Using them prevents the most common class of failure: building the wrong thing with the wrong tool.

| Question | Where |
|---|---|
| Should I use ML or an LLM? | §1 |
| Should I use batch or real-time inference? | §2 |
| When and how should I retrain? | §3 |
| Should I build, buy, or use a pre-built API? | §4 |
| Do I need a simple or complex model? | §5 |
| Do I need a feature store? | §6 |
| Which deployment strategy — shadow, canary, A/B, or blue-green? | §7 |
| Should I build one model or segment it? | §8 |
| Do I need explainability, and how much? | §9 |
| Should I label manually, programmatically, or use active learning? | §10 |
| Cloud, on-premise, or hybrid? | §11 |
| When should I retire a model? | §12 |
| Do I need a model at all — or will rules do? | §13 |
| How do I set the right classification threshold? | §14 |
| How much labeled data do I need? | §15 |
| How do I handle class imbalance? | §16 |
| When should I A/B test a new model (champion-challenger)? | §17 |
| How do I handle label delay? | §18 |
| Human-in-the-loop or full automation? | §19 |
| Offline evaluation or online experiment — which comes first? | §20 |
| Missing data: impute, drop, or model through? | §21 |
| Do I need a GPU for inference? | §22 |

All 22 frameworks are in `docs/decision-frameworks.md`.

---

## Standards Overview

| Standard | What it covers | File |
|---|---|---|
| Coding | Config-driven design, naming, testing, code review | `standards/coding.md` |
| Experimentation | Required MLflow logs, experiment naming, reproducibility | `standards/experimentation.md` |
| Deployment | Patterns, pre-deploy checklist, API versioning, rollback | `standards/deployment.md` |
| Monitoring | PSI, prediction distribution, alerts, dashboards | `standards/monitoring.md` |
| Git & Release | Branching strategy, commit format, CI/CD, environments, model versioning | `standards/git-and-release.md` |

---

## Model Lifecycle

Every model goes through five stages. Promotion between stages requires explicit sign-off.

```
EXPERIMENTAL → CANDIDATE → APPROVED → DEPLOYED → RETIRED
```

Each stage has defined entry criteria, required artifacts, and approval owners.
See `docs/lifecycle.md` and `docs/mlops_standards.md` §2.

In code:

```python
from src.core.lifecycle import ModelStatus, transition

status = ModelStatus.EXPERIMENTAL
status = transition(status, ModelStatus.CANDIDATE)  # after training
status = transition(status, ModelStatus.APPROVED)   # after review
status = transition(status, ModelStatus.DEPLOYED)   # after deployment checklist
```

---

## Data Contracts

Every dataset used by a model must have a defined contract.
Contracts are validated at ingestion — violations halt the pipeline.

```python
from src.core.contracts import DataContract, ColumnSpec, ColumnType

contract = DataContract(
    name="churn_features_v1",
    version="1.0",
    owner="data-engineering",
    columns=[
        ColumnSpec(name="tenure_months", dtype=ColumnType.NUMERIC),
        ColumnSpec(name="target", dtype=ColumnType.NUMERIC),
    ],
)

violations = contract.validate_dataframe(df)
if violations:
    raise ValueError(f"Contract violations: {violations}")
```

See `docs/data_contract_guide.md` for versioning rules and advanced validation.

---

## Failure Modes

Read `docs/failure-modes.md` before your first production deployment. These are the ten most common ways ML systems fail in production and how to prevent them.

| Failure mode | What goes wrong | Prevention |
|---|---|---|
| **Data leakage** | A feature carries future information into training; offline metrics look great, production results collapse | Time-based splits; audit features for target correlation before training |
| **Training-serving skew** | Features are computed differently in training vs. inference; predictions shift without any model change | Shared feature logic in `src/`; contract validation at serving time |
| **Silent drift** | The world changes but no alert fires; model degrades unnoticed until a business review | PSI monitoring after every scoring run; threshold alerts wired to on-call |
| **Broken retraining loops** | Retraining runs on stale or corrupted data; new model is worse but passes offline checks | Data freshness checks; hash validation before training; challenger comparison |
| **Label leakage** | Labels are computed after the event you're predicting, contaminating training | Define the prediction timestamp; validate that all features predate the label |
| **Class imbalance blindness** | Model optimises accuracy, ignores the minority class, looks 97% accurate on a 97/3 split | Use F1, precision-recall, ROC-AUC; set `class_weight: balanced`; check confusion matrix |
| **Evaluation metric mismatch** | Model is tuned on AUC but business cares about precision at a fixed recall | Define the business metric before training; set the threshold explicitly |
| **Stale features** | A feature pipeline breaks silently; model scores on stale or missing values | Pipeline freshness checks; null-rate monitoring on scoring inputs |
| **Silent pipeline failures** | A cron job exits non-zero; nobody notices; yesterday's scores are served for a week | Exit-code alerting on every scheduled job — non-negotiable |
| **Over-reliance on holdout accuracy** | Holdout set leaks over time (same split used across runs); reported accuracy is optimistic | Single holdout evaluation at the end; use time-based splits; track per-segment performance |

Full detail and remediation steps in `docs/failure-modes.md`.

---

## Adding a New Use Case

See `docs/onboarding.md` for the full checklist — from scoping to production.

Quick steps:
1. Work through the scoping questions in `docs/onboarding.md §4`
2. Define your data contract and add it to `configs/pipeline_contracts.yaml`
3. Copy `templates/tabular_ml_pipeline/` to `pipelines/{your-use-case}/`
4. Fill in feature engineering; set thresholds in `configs/training.yaml`
5. Fill out the `ModelCard` fields before promoting past EXPERIMENTAL
6. Follow `standards/deployment.md` pre-deployment checklist before going to production

---

## Design Decisions

| Decision | Tradeoff |
|---|---|
| Documentation-first | Alignment before code; requires discipline to keep docs current |
| Config-driven pipelines | Fast iteration on params; all settings reviewable in one place |
| Explicit lifecycle stages | Clear ownership + audit trail; introduces review gates |
| Modular layers (data / features / training / evaluation / deployment) | Independent ownership; introduces integration points to manage |
| MLflow for tracking | Standard, open, self-hostable; requires infrastructure to run at scale |

See `docs/decisions.md` for formal Architecture Decision Records.

---

## Part of AI Platform

This repository is part of a modular AI platform:

- [ds-mlops-enterprise-system](https://github.com/rizkashifs/ds-mlops-enterprise-system) → defines standards and best practices
- [mlops-control-plane](https://github.com/rizkashifs/mlops-control-plane) → manages model lifecycle and governance
- [enterprise-rag-agent-system](https://github.com/rizkashifs/enterprise-rag-agent-system) → GenAI application layer
- [hybrid-ds-genai-agentic-mlops-system](https://github.com/rizkashifs/hybrid-ds-genai-agentic-mlops-system) → ML + LLM + agentic workflows
- [ai-observability-and-drift-platform](https://github.com/rizkashifs/ai-observability-and-drift-platform) → monitoring and reliability
- [multi-model-routing-engine](https://github.com/rizkashifs/multi-model-routing-engine) → model selection and optimization

These repositories together represent an enterprise-grade AI system.
