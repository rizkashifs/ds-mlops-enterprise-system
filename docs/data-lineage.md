# Data Lineage Tracking

> Lineage answers: "For any model prediction, what data created it?"
> Without it, you cannot reproduce a model, investigate a wrong prediction, or satisfy an audit.

---

## Contents

1. [What lineage tracks](#1-what-lineage-tracks)
2. [Implementation: LineageRecord](#2-implementation-lineagerecord)
3. [Integrating into a training pipeline](#3-integrating-into-a-training-pipeline)
4. [Integrating into a scoring pipeline](#4-integrating-into-a-scoring-pipeline)
5. [Querying lineage](#5-querying-lineage)
6. [What to track vs. what to skip](#6-what-to-track-vs-what-to-skip)
7. [Compliance use cases](#7-compliance-use-cases)

---

## 1. What lineage tracks

A lineage record answers three questions about a run:

| Question | Captured by |
|---|---|
| **What data was used?** | `DataSource` — name, version, location, snapshot timestamp, row count |
| **How was it transformed?** | `TransformationStep` — step name, description, parameters |
| **What did it produce?** | `output_artifact` — MLflow artifact URI or model registry path |

Lineage is attached to the MLflow run as a JSON artifact. This means it is co-located with model weights, metrics, and parameters — no separate database required to get started.

---

## 2. Implementation: LineageRecord

`src/core/lineage.py` provides three dataclasses and two functions:

```python
from src.core.lineage import (
    DataSource,
    LineageRecord,
    TransformationStep,
    create_lineage_record,
    log_lineage_to_mlflow,
)
```

### DataSource

```python
DataSource(
    name="customer_features",           # logical name, not a path
    version="v3",                       # data contract version
    location="s3://bucket/path/",       # physical location at snapshot time
    snapshot_timestamp="2026-04-30T00:00:00Z",
    row_count=420_000,                  # optional; useful for audits
    schema_hash="sha256:abc...",        # optional; detect schema drift
)
```

### TransformationStep

```python
TransformationStep(
    name="standard_scaler",
    description="StandardScaler fitted on training split only",
    parameters={"with_mean": True, "with_std": True},
)
```

### Creating and logging a record

```python
record = create_lineage_record(run_id=mlflow.active_run().info.run_id, model_name="churn-v2")
record.inputs.append(DataSource(...))
record.transformations.append(TransformationStep(...))
record.output_artifact = f"runs:/{record.run_id}/model"
log_lineage_to_mlflow(record)
```

This writes `lineage/lineage_<timestamp>.json` as an artifact on the active MLflow run.

---

## 3. Integrating into a training pipeline

Add lineage logging at the end of `pipelines/training_pipeline/train.py`, inside the active MLflow run:

```python
import mlflow
from src.core.lineage import DataSource, TransformationStep, create_lineage_record, log_lineage_to_mlflow

with mlflow.start_run() as run:
    # ... existing train / validate / register logic ...

    record = create_lineage_record(run_id=run.info.run_id, model_name=model_name)

    record.inputs.append(DataSource(
        name="training_dataset",
        version=data_contract.version,
        location=training_data_path,
        snapshot_timestamp=snapshot_ts,
        row_count=len(train_df),
    ))

    for step_name, step_params in preprocessing_steps:
        record.transformations.append(TransformationStep(
            name=step_name,
            description=f"Applied during feature engineering",
            parameters=step_params,
        ))

    record.output_artifact = f"runs:/{run.info.run_id}/model"
    log_lineage_to_mlflow(record)
```

---

## 4. Integrating into a scoring pipeline

For batch inference, log a lineage record that captures the scoring dataset and the model version used:

```python
record = create_lineage_record(run_id=scoring_run_id, model_name=model_name)
record.inputs.append(DataSource(
    name="scoring_population",
    version="live",
    location=scoring_data_path,
    snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
    row_count=len(scoring_df),
))
record.output_artifact = scores_output_path
log_lineage_to_mlflow(record)
```

For real-time inference, log lineage at the request level only when required by compliance (e.g., credit decisions). Per-request lineage at high volume is expensive — batch it or sample.

---

## 5. Querying lineage

### Via MLflow UI

Navigate to the run → Artifacts → `lineage/` → open the JSON file.

### Via MLflow client

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
artifacts = client.list_artifacts(run_id, path="lineage")
for artifact in artifacts:
    content = client.download_artifacts(run_id, artifact.path)
    # parse as JSON
```

### Tracing a model version back to its data

1. Find the run ID for a model version in the MLflow registry.
2. Download the `lineage/` artifact from that run.
3. The `inputs[].location` and `inputs[].snapshot_timestamp` fields identify the exact data snapshot.

---

## 6. What to track vs. what to skip

**Always track:**
- Training dataset name, version, location, and snapshot timestamp
- Number of rows in the training split
- Any feature engineering step whose parameters are fitted on data (scalers, encoders, imputers)
- The output model artifact URI

**Track when it matters:**
- Validation and test split metadata (row counts, class balance)
- External data sources (third-party enrichment, reference tables)
- Model ensemble lineage (if output model depends on other models)

**Skip:**
- Hyperparameter search artifacts (log these as MLflow params, not lineage)
- Intermediate temp files
- Transformations with no fitted state (e.g., a column drop)

---

## 7. Compliance use cases

### Model audit

An auditor asks: "What data trained the fraud model deployed on 2026-03-15?"

1. Look up the production model version in the registry.
2. Find the MLflow run ID for that version.
3. Download the lineage artifact: `inputs[0].location` and `snapshot_timestamp`.
4. Retrieve the data snapshot from that location.
5. Rerun training to verify reproducibility.

### Data deletion request (GDPR Right to Erasure)

A customer requests deletion. You need to know whether their data influenced any deployed model.

1. For each deployed model, retrieve training lineage.
2. Check `inputs[].location` — does the snapshot include the customer's data?
3. If yes: log it, assess retraining obligation, and document the decision.
4. Lineage records themselves may need to be anonymized if they contain individual identifiers — keep only aggregate `row_count`, not row-level IDs.

### Training-serving skew investigation

If a model performs differently in production than in evaluation:

1. Retrieve training lineage: `inputs[].snapshot_timestamp`.
2. Compare feature distributions at that timestamp against current scoring input distributions.
3. PSI > 0.20 on a feature with a transformation step confirms skew origin.
