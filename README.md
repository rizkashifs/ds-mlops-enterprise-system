# DS MLOps Enterprise System

**An operating system for data science teams.**

This repository answers four questions every DS org needs to answer before it can scale:

1. **How do we build models?** → `standards/`, `src/`, `pipelines/`
2. **How do we deploy them?** → `standards/deployment.md`, `templates/`
3. **How do we ensure consistency across teams?** → `docs/`, `configs/`, `standards/`
4. **How do we avoid common failures?** → `docs/failure-modes.md`, `docs/decision-frameworks.md`

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

## Start Here: Decision Frameworks

Before writing any code, read `docs/decision-frameworks.md`. It answers:

| Question | Where |
|---|---|
| Should I use ML or an LLM? | §1 |
| Should I use batch or real-time inference? | §2 |
| When and how should I retrain? | §3 |
| Should I build or buy? | §4 |
| Do I need a simple or complex model? | §5 |
| Should I use rules, ML, or LLMs? | §7 |
| How do I set the right decision threshold? | §8 |
| How much labeled data do I need? | §9 |
| How do I handle class imbalance? | §10 |
| Should I build one model or segment it? | §11 |
| How do I safely roll out a new model? | §12 |
| Do I need explainability? | §13 |
| How do I handle label delay? | §14 |
| Do I need human-in-the-loop? | §15 |
| How do I test before going live? | §16 |
| How do I handle missing data? | §17 |
| Do I need a GPU? | §18 |

All 18 frameworks are in `docs/decision-frameworks.md`.

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

## Top Failure Modes to Know

Read `docs/failure-modes.md` before your first production deployment. The top 3:

1. **Training-serving skew** — features computed differently in training vs. inference
2. **Data leakage** — a feature that wouldn't exist at prediction time
3. **Silent drift** — the world changes but no alert fires

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
