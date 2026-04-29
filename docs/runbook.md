# MLOps Operations Runbook

> **Audience:** ML Engineers, Platform Engineers, on-call Data Scientists
> **When to use this:** When something goes wrong in production, or when performing routine maintenance

This runbook covers the most common operational scenarios. For the full lifecycle standards, see `docs/mlops_standards.md`.

---

## Contents

1. [Responding to a monitoring alert](#1-responding-to-a-monitoring-alert)
2. [Emergency model rollback](#2-emergency-model-rollback)
3. [Retraining a deployed model](#3-retraining-a-deployed-model)
4. [Diagnosing a scoring pipeline failure](#4-diagnosing-a-scoring-pipeline-failure)
5. [Promoting a model to production](#5-promoting-a-model-to-production)
6. [Retiring a model](#6-retiring-a-model)

---

## 1. Responding to a monitoring alert

### Symptoms
- Performance metric drops below threshold (accuracy, F1, ROC-AUC)
- Score distribution shifts significantly (PSI > 0.20)
- Pipeline error rate spike

### Steps

**Step 1: Acknowledge and assess severity**
- Is this a data pipeline failure (no data, bad data) or a model quality issue?
- Check the monitoring dashboard for the last 7-day trend. Is this a sudden drop or gradual drift?
- Check whether upstream data sources or contracts changed recently.

**Step 2: Rule out data issues first**
```python
# Check row count and contract compliance on recent batch
violations = contract.validate_dataframe(latest_batch_df)
print(violations)  # empty = data is clean
print(f"Row count: {len(latest_batch_df)}")
```

If there are contract violations, page the data engineering team. The model is not the problem.

**Step 3: Compare current score distribution to baseline**
Run a PSI calculation between today's score distribution and the 30-day rolling baseline. If PSI > 0.20 on input features, the input data has drifted — initiate the retraining process (see §3).

**Step 4: Determine whether to rollback or retrain**
| Condition | Action |
|---|---|
| Sudden failure, recent deployment | Rollback to previous model (see §2) |
| Gradual drift over weeks | Initiate retraining (see §3) |
| Data pipeline failure | Fix data, re-run scoring |
| Unknown cause | Rollback + investigate |

---

## 2. Emergency model rollback

### When to use
When a newly deployed model is producing incorrect or harmful predictions and you need to immediately restore the previous model.

### Steps

**Step 1: Identify the previous production model**
```bash
# Check MLflow model registry for the last 'Production' model before today's deployment
mlflow models list --name churn-rf
```

Note the previous model version number and its `run_id`.

**Step 2: Update the model registry tag**
In MLflow UI or via CLI:
- Archive the current `Production` version
- Promote the previous version back to `Production`

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
# Archive the bad model
client.transition_model_version_stage(
    name="churn-rf",
    version="CURRENT_VERSION",
    stage="Archived",
)
# Restore the previous model
client.transition_model_version_stage(
    name="churn-rf",
    version="PREVIOUS_VERSION",
    stage="Production",
)
```

**Step 3: Update the lifecycle status in your tracking system**
- Set the bad model back to `CANDIDATE` with a note explaining the rollback
- Set the restored model back to `DEPLOYED`

**Step 4: Re-run the scoring pipeline with the restored model URI**

**Step 5: Post-incident**
- Write up what went wrong
- Document in the model card what the failure mode was
- Add a test case or monitoring rule that would catch this earlier next time

---

## 3. Retraining a deployed model

### When to use
When monitoring signals indicate the current model's performance is degrading and a retrain will address the root cause.

### Prerequisites
- Confirm the trigger meets the retraining policy (see `docs/mlops_standards.md` §9)
- Confirm fresh training data is available and passes contract validation
- Notify stakeholders that a retrain is in progress

### Steps

**Step 1: Pull and validate fresh training data**
```python
from src.core.contracts import DataContract

violations = contract.validate_dataframe(new_training_df)
assert not violations, f"Contract violations: {violations}"
```

**Step 2: Run the training pipeline**
```python
from src.pipelines.training import TrainingConfig, train_model

config = TrainingConfig(
    experiment_name="churn-prediction-retrain-2026Q2",
    model_params={"n_estimators": 100, "max_depth": 5},
)
result = train_model(new_training_df, config)
print(result.metrics)
```

**Step 3: Validate the new model**
```python
from src.pipelines.validation import ValidationThresholds, validate_model

thresholds = ValidationThresholds(min_accuracy=0.75, min_f1=0.65, min_roc_auc=0.80)
validation = validate_model(result.metrics, thresholds)
print(validation.summary())
```

The new model must also outperform (or match within noise) the currently deployed model on the same holdout set.

**Step 4: Follow the standard promotion flow**
```python
from src.core.lifecycle import ModelStatus, transition

status = transition(ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE)
# ... complete review and approval steps ...
status = transition(status, ModelStatus.APPROVED)
status = transition(status, ModelStatus.DEPLOYED)
```

Do not skip the approval gate even under time pressure. Log the urgency and explain the timeline to reviewers.

**Step 5: Monitor the retrained model closely for 30 days**
Daily monitoring review for the first 30 days post-deployment.

---

## 4. Diagnosing a scoring pipeline failure

### Common failure modes

| Symptom | Likely cause | Check |
|---|---|---|
| `KeyError: 'feature_name'` | Column missing from input data | Validate against data contract |
| `ValueError: model not found` | Wrong model URI or model registry issue | Check MLflow registry |
| Empty output DataFrame | Input data was empty | Check source data pipeline |
| Predictions all = 0 or 1 | Model collapsed; check training run | Review MLflow run metrics |
| Pipeline hung / timeout | Data volume spike or infrastructure issue | Check row count and job logs |

### Diagnostic steps

**Step 1: Check the input data**
```python
print(f"Input shape: {df.shape}")
print(f"Null counts:\n{df.isnull().sum()}")
print(f"Column names: {list(df.columns)}")
```

**Step 2: Validate against contract**
```python
violations = contract.validate_dataframe(df)
if violations:
    print("Contract violations found:")
    for v in violations:
        print(f"  {v}")
```

**Step 3: Verify the model URI**
```python
import mlflow

# Verify model exists and is loadable
model = mlflow.sklearn.load_model(model_uri)
print(f"Model type: {type(model)}")
print(f"Expected features: {model.feature_names_in_}")
```

**Step 4: Run a test prediction on a small sample**
```python
sample = df.head(10)
from src.services.scoring import score_batch
result = score_batch(sample, model_uri)
print(result.to_dataframe())
```

---

## 5. Promoting a model to production

### Pre-promotion checklist (run through before any DEPLOYED transition)

- [ ] Validation passed with all required metrics above threshold
- [ ] Model card complete and signed off
- [ ] MLflow run ID recorded in the model card
- [ ] Model registered in MLflow model registry
- [ ] Staging test completed on prod-like data
- [ ] Monitoring dashboards created
- [ ] Rollback procedure documented (which version to fall back to)
- [ ] Consumer teams notified (at least 48 hours in advance for non-emergency)

### Promotion command

```python
from src.core.lifecycle import ModelStatus, transition

# Only call this after checklist is complete
status = transition(ModelStatus.APPROVED, ModelStatus.DEPLOYED)
```

### Post-deployment verification

After deploying, verify the first live scoring run:
1. Check output row count matches expected input count
2. Check that score distribution matches what was seen in staging
3. Check that no errors appear in the pipeline logs
4. Confirm monitoring dashboards are receiving data

---

## 6. Retiring a model

### When to retire
- A replacement model has been deployed and is performing well
- The use case has been discontinued
- The model has been deprecated by the business

### Steps

**Step 1: Notify consumers (minimum 2 weeks notice)**
Send a deprecation notice to all teams consuming this model's output. Include:
- Retirement date
- Replacement model name and version (or confirmation that the use case is discontinued)
- Any changes to output format

**Step 2: Verify no active consumers remain**
Check that no scheduled jobs or APIs are still calling this model after the retirement date.

**Step 3: Archive in MLflow**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="churn-rf",
    version="VERSION_TO_RETIRE",
    stage="Archived",
)
```

**Step 4: Update lifecycle status**
```python
status = transition(ModelStatus.DEPLOYED, ModelStatus.RETIRED)
```

**Step 5: Preserve audit artifacts**
Ensure the following are retained for 3 years:
- MLflow run record
- Model card (final version)
- Validation reports
- Approval sign-off records

The model artifact itself can be moved to cold storage but must not be deleted for 3 years.
