# DS MLOps Enterprise System

**An operating system for data science teams.**

This repository answers four questions every DS org needs to answer before it can scale:

1. **How do we build models?** → `standards/`, `src/`, `pipelines/`
2. **How do we deploy them?** → `standards/deployment.md`, `templates/`
3. **How do we ensure consistency across teams?** → `docs/`, `configs/`, `platform/`
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
│   ├── decision-frameworks.md     # ML vs LLM, Batch vs Real-time, Retraining strategy
│   ├── failure-modes.md           # What goes wrong and how to prevent it
│   ├── mlops_standards.md         # Full MLOps standards reference
│   ├── model_card_template.md     # Fill-in model card template
│   ├── data_contract_guide.md     # How to define and use data contracts
│   ├── runbook.md                 # Day-2 operations playbook
│   └── architecture.md            # Architecture notes
│
├── standards/                     # Team coding and process standards
│   ├── coding.md                  # Code style, naming, testing, config-driven design
│   ├── experimentation.md         # MLflow logging requirements, experiment hygiene
│   ├── deployment.md              # Deployment patterns, pre-deploy checklist, API contracts
│   └── monitoring.md              # What to monitor, PSI, alert thresholds
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
│   ├── experiment_tracking/tracker.py   # Enforces required MLflow logging
│   ├── model_registry/registry.py       # Model registration and promotion
│   └── monitoring_hooks/hooks.py        # PSI, score distribution, monitoring reports
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
| Should I use ML or an LLM? | `docs/decision-frameworks.md` §1 |
| Should I use batch or real-time inference? | `docs/decision-frameworks.md` §2 |
| When and how should I retrain? | `docs/decision-frameworks.md` §3 |
| Should I build or buy? | `docs/decision-frameworks.md` §4 |
| Do I need a simple or complex model? | `docs/decision-frameworks.md` §5 |

---

## Standards Overview

| Standard | What it covers | File |
|---|---|---|
| Coding | Config-driven design, naming, testing, code review | `standards/coding.md` |
| Experimentation | Required MLflow logs, experiment naming, reproducibility | `standards/experimentation.md` |
| Deployment | Patterns, pre-deploy checklist, API versioning, rollback | `standards/deployment.md` |
| Monitoring | PSI, prediction distribution, alerts, dashboards | `standards/monitoring.md` |

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

1. Copy `templates/tabular_ml_pipeline/` to `examples/{your-use-case}/`
2. Define your data contract and add it to `configs/pipeline_contracts.yaml`
3. Fill in your feature engineering in `encode_features()`
4. Set thresholds in `configs/training.yaml`
5. Fill out `docs/model_card_template.md`
6. Run the pipeline and verify: contract passes, training runs, validation passes
7. Follow `docs/mlops_standards.md` §13 checklist for production

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
