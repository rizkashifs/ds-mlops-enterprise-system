# Deployment Standards

> Deployment is not the end of delivery. It's the beginning of operations.
> These standards ensure models enter production in a controlled, observable, and reversible way.

---

## Deployment Patterns

Choose the right pattern before writing any deployment code. See `docs/decision-frameworks.md` §2 for the batch vs. real-time decision tree.

### Pattern 1: Batch Scoring

**When to use:** Predictions needed on a schedule (daily, weekly, monthly). Latency tolerance > a few minutes. Output is a table or file consumed by another process.

**Architecture:**
```
Scheduler (cron / Prefect / Airflow)
  → Load model from registry
  → Load features from data warehouse
  → Score batch
  → Write predictions to output table
  → Log scoring run metrics
```

**Example use cases:** Churn scores, LTV segments, risk ratings, propensity tiers

### Pattern 2: Synchronous Online Inference

**When to use:** Predictions must be served in the same HTTP request cycle. Latency tolerance <500ms. Interactive use cases.

**Architecture:**
```
Client request
  → API gateway
  → Inference service (FastAPI / Flask)
      → Load features from feature store (real-time lookup)
      → Load model from in-memory cache
      → Predict
      → Return response
```

**Example use cases:** Fraud detection at transaction time, real-time recommendations, dynamic pricing

### Pattern 3: Pre-Compute + Cache

**When to use:** "Real-time" need is actually a low-latency lookup. Predictions don't depend on events that happen at request time.

**Architecture:**
```
Batch job (every N hours)
  → Score all entities
  → Write to fast key-value store (Redis / DynamoDB)

API call
  → Read(entity_id) from key-value store
  → Return cached score (<10ms)
```

This pattern gives the simplicity of batch with the latency of online. Use it whenever possible before building a synchronous inference service.

### Pattern 4: Shadow Mode

**When to use:** Validating a new model against production traffic before cutover. No risk: new model runs but its output is not acted on.

**Architecture:**
```
Production traffic
  → Current model (outputs used)
  → New model in shadow (outputs logged, not acted on)

Monitoring:
  → Compare distributions: new model vs. current model
  → After N days: review, then switch
```

---

## Pre-Deployment Checklist

No model may be promoted to DEPLOYED status without all of the following:

### Model readiness
- [ ] Model card complete and signed off by risk/compliance
- [ ] Model registered in MLflow model registry with stage = `Staging`
- [ ] All validation thresholds passed (ValidationResult.passed = True)
- [ ] Baseline comparison complete (new model >= current production)

### Infrastructure readiness
- [ ] Deployment tested in staging environment with production-like data volume
- [ ] Output row count verified: scoring job produces expected number of records
- [ ] Latency tested (online inference) or throughput tested (batch)
- [ ] Rollback procedure documented: which model/version to fall back to

### Observability readiness
- [ ] Monitoring dashboards created (prediction distribution, feature PSI)
- [ ] Alert thresholds configured and tested (warning + critical levels)
- [ ] Pipeline health metrics in place (output count, latency, error rate)
- [ ] On-call runbook updated with this model's specifics

### Stakeholder readiness
- [ ] Consumer teams notified of deployment date and any schema changes
- [ ] Platform Engineering sign-off obtained
- [ ] Go/no-go decision recorded in the model card

---

## Model Versioning

Use semantic versioning for models:

| Change | Version bump | Example |
|---|---|---|
| New production deployment (new algorithm, architecture, major feature set) | Major | 1.x → 2.0 |
| Retrained on fresh data, same architecture | Minor | 1.0 → 1.1 |
| Hyperparameter tuning only | Patch | 1.0 → 1.0.1 |

Register models in MLflow with the version number:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_registered_model("churn-rf")
client.create_model_version(
    name="churn-rf",
    source=f"runs:/{run_id}/model",
    run_id=run_id,
    description="v1.0 — initial production model",
)
```

---

## API Contracts for Online Inference

If deploying as an API, the request and response schema must be versioned and documented.

### Request schema (example)

```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    customer_id: str
    tenure_months: int
    monthly_charges: float
    num_products: int
    support_calls_90d: int
```

### Response schema (example)

```python
class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int           # 0 or 1
    churn_probability: float  # [0.0, 1.0]
    model_version: str        # "churn-rf:1.0"
    scored_at: str            # ISO datetime
```

### API versioning

Version your inference endpoints: `/v1/predict`, `/v2/predict`.
Deprecate old versions with a sunset notice before removing them.
Never silently change response schemas.

---

## Rollback Procedure

Document the rollback procedure **before** deploying, not after something goes wrong.

For each deployment, record:

1. **Rollback trigger**: what conditions would require rollback (performance drop, error spike, data issue)
2. **Previous model version**: the name and version of the model that was running before this deployment
3. **Rollback steps**: the exact commands to revert
4. **Rollback owner**: who has authority and access to execute the rollback

```python
# Example rollback steps (document these before deploying)
# 1. Archive the current model in the registry:
client.transition_model_version_stage("churn-rf", version="2", stage="Archived")
# 2. Restore the previous version to Production:
client.transition_model_version_stage("churn-rf", version="1", stage="Production")
# 3. Redeploy the scoring service pointing to the restored version
# 4. Notify stakeholders
```

---

## Environment Promotion

Models should move through environments in order:

```
local dev → staging → production
```

**Local dev:** Data scientists run experiments on their machines or dev infrastructure. No production data.

**Staging:** Runs against a full-size sample of production data (or prod data with access controls). Used for integration testing and load testing.

**Production:** Live scoring, serving real decisions.

Never promote directly from local to production. Always validate in staging first.

---

## Post-Deployment Verification

Within 24 hours of deploying a new model:

1. **Output count**: does the scoring job produce the expected number of records?
2. **Score distribution**: does the distribution match what was seen in staging? Use KS test or visual check.
3. **No errors**: zero pipeline failures or uncaught exceptions in logs
4. **Consumer confirmation**: at least one downstream consumer confirms they're receiving data correctly
5. **Monitoring active**: confirm dashboards are receiving data and alerts are configured
