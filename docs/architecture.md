# Architecture

This document describes the components of the enterprise MLOps platform, how they connect, and the data and model flow through the system.

---

## System overview

The platform is organized into four layers:

| Layer | Purpose | Key modules |
|---|---|---|
| **Contracts & governance** | Shared language between teams; enforced at every gate | `src/core/contracts.py`, `src/core/lifecycle.py` |
| **Platform services** | Reusable infrastructure for tracking, registry, and monitoring | `mlops_platform/` |
| **Pipelines** | Orchestrated workflows for training, inference, and retraining | `pipelines/` |
| **Templates** | Copy-and-adapt starting points for new use cases | `templates/` |

---

## Component map

```
configs/                     ← per-use-case YAML (thresholds, schedules, contracts)
src/
  core/
    contracts.py             ← DataContract, ModelCard (Pydantic models)
    lifecycle.py             ← ModelStatus state machine (5 stages)
    lineage.py               ← LineageRecord attached to MLflow runs
    secrets.py               ← env-var loader with startup validation
  pipelines/
    training.py              ← train/evaluate loop, returns metrics dict
    validation.py            ← ValidationThresholds gate (blocks bad models)
  services/
    scoring.py               ← score_batch() used by all inference paths

mlops_platform/
  experiment_tracking/
    tracker.py               ← MLflow run wrapper (log params, metrics, artifacts)
  model_registry/
    registry.py              ← register, promote, archive (wraps MlflowClient)
  monitoring_hooks/
    hooks.py                 ← compute_psi(), build_monitoring_report()
    triggers.py              ← evaluate_triggers() → TriggerDecision

pipelines/
  training_pipeline/train.py      ← end-to-end: load → validate → train → register
  inference_pipeline/score.py     ← load production model → score → write → monitor
  retraining_pipeline/retrain.py  ← triggered retrain with drift context

templates/
  tabular_ml_pipeline/     ← copy-and-adapt: structured tabular use cases
  genai_pipeline/          ← copy-and-adapt: LLM extraction / generation
  realtime_api/app.py      ← FastAPI inference endpoint (+ Dockerfile)
  batch_inference/scorer.py← scheduled batch scorer (+ Dockerfile)

docs/                        ← decision frameworks, standards, runbooks
standards/                   ← coding, deployment, experimentation, monitoring
tests/                       ← unit tests; run with: pytest tests/ -v
```

---

## Model and data flow

```
                         ┌─────────────────┐
  Raw data               │  DataContract   │  validates schema,
  ──────────────────────>│  .validate_df() │  nulls, types
                         └────────┬────────┘
                                  │ clean data
                         ┌────────▼────────┐
                         │ Training        │  logs params + metrics
                         │ Pipeline        │  to MLflow
                         └────────┬────────┘
                                  │ run_id
                         ┌────────▼────────┐
                         │ Validation Gate │  compare metrics vs.
                         │ validate_model()│  ValidationThresholds
                         └────────┬────────┘
                          pass    │    fail → stays CANDIDATE
                         ┌────────▼────────┐
                         │ Model Registry  │  EXPERIMENTAL → CANDIDATE
                         │ (MLflow)        │  → APPROVED → DEPLOYED
                         └────────┬────────┘
                                  │ model URI
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
 ┌────────▼───────┐    ┌──────────▼──────────┐  ┌─────────▼────────┐
 │ Batch Inference│    │ Real-time API        │  │ Retraining       │
 │ scorer.py      │    │ FastAPI /v1/predict  │  │ Pipeline         │
 └────────┬───────┘    └──────────┬──────────┘  └─────────▲────────┘
          │                       │                        │
          └───────────┬───────────┘                        │
                      │ scores + features                  │
             ┌────────▼────────┐                           │
             │ Monitoring      │  PSI, score stats,        │
             │ build_report()  │  alert thresholds         │
             └────────┬────────┘                           │
                      │ MonitoringReport                   │
             ┌────────▼────────┐                           │
             │ evaluate_       │  TriggerDecision          │
             │ triggers()      │──────────────────────────►│
             └─────────────────┘  should_retrain=True
```

---

## Lifecycle state machine

Models progress through five stages. All transitions are enforced by `lifecycle.py`.

```
EXPERIMENTAL ──► CANDIDATE ──► APPROVED ──► DEPLOYED ──► RETIRED
      ▲               │              │           │
      │               │              │           │
      └───────────────┘              └───────────┘
      (rollback)                     (rollback to CANDIDATE)
```

| Stage | Meaning | Who can advance |
|---|---|---|
| `EXPERIMENTAL` | Active development; not production-ready | Data scientist |
| `CANDIDATE` | Training complete; awaiting validation | Automated pipeline |
| `APPROVED` | Passed all validation gates | Model reviewer / automated gate |
| `DEPLOYED` | Serving predictions in production | Release engineer |
| `RETIRED` | Permanently decommissioned | Model owner (terminal) |

---

## Key design decisions

- **DataContract at every ingestion boundary.** Pipelines fail fast on bad data rather than producing wrong predictions silently. See `docs/data_contract_guide.md`.
- **ModelCard required before review.** No model enters the APPROVED stage without a completed card. This is the governance checkpoint. See `docs/model_card_template.md`.
- **Monitoring is always synchronous with inference.** `build_monitoring_report()` runs immediately after every batch scoring job. Alerts do not require a separate scheduled check.
- **Retraining is event-driven, not purely scheduled.** `evaluate_triggers()` fires on drift signals. A time-based fallback ensures a maximum interval. See `docs/retraining_triggers.md` and `decision-frameworks.md §3`.
- **Secrets are never in code.** `src/core/secrets.py` enforces env-var sourcing. See `docs/secrets-management.md`.

---

## Integration points

| System | How to connect | Where configured |
|---|---|---|
| **MLflow tracking server** | Set `MLFLOW_TRACKING_URI` env var | `configs/config.yaml` |
| **MLflow model registry** | Same URI; registry is part of tracking server | Automatic |
| **Data storage** (S3, GCS, ADLS) | Override `load_scoring_data()` in batch template | `templates/batch_inference/scorer.py` |
| **Orchestrator** (Prefect, Airflow) | Wrap pipeline `run()` functions as tasks/DAGs | `pipelines/` |
| **Alerting** (PagerDuty, Slack) | Hook into `MonitoringReport.has_alerts()` | `docs/runbook.md` |
| **Secrets vault** (AWS SM, Vault) | Pre-populate env vars at container start | `docs/secrets-management.md` |
