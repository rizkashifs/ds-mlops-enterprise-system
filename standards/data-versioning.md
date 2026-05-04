# Data Versioning Standards

> This document defines what data versioning must solve and how to implement it.
> DVC is the reference implementation. Delta Lake, Iceberg, and S3 versioning
> are covered as alternatives — the principles apply to all.

---

## Contents

1. [Why Data Versioning Matters](#1-why-data-versioning-matters)
2. [What Every Team Must Track](#2-what-every-team-must-track)
3. [DVC — Reference Implementation](#3-dvc--reference-implementation)
4. [Alternative Approaches](#4-alternative-approaches)
5. [Minimum Viable Implementation](#5-minimum-viable-implementation)
6. [Anti-Patterns](#6-anti-patterns)

---

## 1. Why Data Versioning Matters

A model is only as reproducible as its training data. Without data versioning:

- You cannot reproduce a training run from 6 months ago when a production issue is reported
- You cannot tell which data version produced which model version
- You cannot roll back to the previous data version when an upstream pipeline corrupts data
- You cannot audit training data for compliance ("what data was this model trained on?")

The goal is: given a model artifact and its MLflow run_id, you should be able to identify the exact training dataset that produced it.

---

## 2. What Every Team Must Track

Regardless of which tool you use, every training run must record:

| Attribute | How to track |
|---|---|
| Dataset name and version | Log as MLflow param or in meta.json |
| Row count at training time | Log as MLflow param |
| Source location (S3 path, table name, query) | Log as MLflow param |
| Date range of the data | Log as MLflow param |
| Data contract version | Log as MLflow param (already enforced by DataContract) |
| Hash or fingerprint of training file | Log as MLflow param (DVC handles this automatically) |

Minimum logging in every training run:
```python
tracker.log_params({
    "data.source": "s3://bucket/churn/2026-04-01/",
    "data.row_count": len(df),
    "data.date_range": "2025-01-01 to 2026-03-31",
    "data.contract": "churn_features_v1:1.0",
    "data.version": "v20260401",  # or DVC hash
})
```

---

## 3. DVC — Reference Implementation

DVC (Data Version Control) tracks large data files outside of git while keeping lightweight pointers (`.dvc` files) inside git. This means data versions are tied to code versions without storing large files in git.

### Setup

```bash
pip install dvc
# With remote storage (choose your backend):
pip install "dvc[s3]"      # AWS S3
pip install "dvc[gs]"      # Google Cloud Storage
pip install "dvc[azure]"   # Azure Blob Storage

dvc init
```

### Configure a remote

```bash
# S3 example (set credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
dvc remote add -d myremote s3://my-bucket/dvc-store

# Local remote (for dev — not recommended for team use)
dvc remote add -d localremote /tmp/dvc-store
```

### Track a dataset

```bash
# Add a dataset to DVC tracking
dvc add data/churn_training.parquet

# This creates data/churn_training.parquet.dvc — commit this to git
git add data/churn_training.parquet.dvc data/.gitignore
git commit -m "data: add churn training dataset v1.0"

# Push data to remote storage
dvc push
```

The `.dvc` file looks like:
```yaml
outs:
- md5: d8e8fca2dc0f896fd7cb4cb0031ba249
  size: 1048576
  path: churn_training.parquet
```

The `md5` hash is the data fingerprint. Log it to your training tracker alongside the model artifact.

### Reproduce a training run

```bash
# Check out the code version that produced the model
git checkout model/churn-rf-v1.1

# Pull the exact data that was used at that commit
dvc pull

# Run training — guaranteed to use the same data
python pipelines/training_pipeline/train.py
```

### Define a reproducible pipeline

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/churn_raw.parquet
      - src/data/preprocess.py
    outs:
      - data/processed/churn_features.parquet

  train:
    cmd: python pipelines/training_pipeline/train.py
    deps:
      - data/processed/churn_features.parquet
      - configs/training.yaml
      - src/pipelines/training.py
    metrics:
      - metrics/metrics.json
```

Run the full pipeline:
```bash
dvc repro        # runs only stages where inputs have changed
dvc repro --force  # forces all stages to rerun
```

### Log the DVC revision in MLflow / local tracker

```python
import subprocess
dvc_rev = subprocess.check_output(["dvc", "params", "diff", "--all"]).decode().strip()
tracker.log_params({"data.dvc_revision": dvc_rev})
```

Or more simply:
```python
import hashlib, pandas as pd
df = pd.read_parquet("data/churn_features.parquet")
data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
tracker.log_params({"data.hash": data_hash})
```

---

## 4. Alternative Approaches

### Delta Lake / Apache Iceberg (data warehouse teams)

If your training data lives in a data warehouse or lakehouse, use its built-in versioning:

```python
# Delta Lake — log the table version used for training
from delta.tables import DeltaTable
dt = DeltaTable.forPath(spark, "s3://bucket/churn_features/")
version = dt.history(1).select("version").collect()[0][0]
tracker.log_params({"data.delta_version": version, "data.table": "churn_features"})

# Read a specific version for reproducibility
df = spark.read.format("delta").option("versionAsOf", version).load("s3://bucket/churn_features/")
```

```sql
-- Iceberg — log the snapshot ID
SELECT snapshot_id FROM my_catalog.churn_features.snapshots ORDER BY committed_at DESC LIMIT 1;
-- Read a specific snapshot
SELECT * FROM churn_features FOR SYSTEM_VERSION AS OF {snapshot_id};
```

### S3 object versioning

Enable S3 versioning on the training data bucket:
```bash
aws s3api put-bucket-versioning --bucket my-training-data \
  --versioning-configuration Status=Enabled
```

Log the version ID at training time:
```python
import boto3
s3 = boto3.client("s3")
head = s3.head_object(Bucket="my-training-data", Key="churn/features.parquet")
version_id = head["VersionId"]
tracker.log_params({"data.s3_version_id": version_id})
```

### Manual versioning (minimum viable)

If none of the above tools fit, use a naming convention:
- Store training datasets as `data/churn_features_v20260401.parquet`
- Log the filename as a training param
- Never overwrite versioned files

```python
DATA_VERSION = "v20260401"
df = pd.read_parquet(f"data/churn_features_{DATA_VERSION}.parquet")
tracker.log_params({"data.version": DATA_VERSION, "data.rows": len(df)})
```

---

## 5. Minimum Viable Implementation

If you're not ready for DVC or a lakehouse, implement this minimum before going to production:

```
1. Never overwrite training datasets — always write versioned files.
   churn_features_v20260401.parquet, not churn_features.parquet (overwritten).

2. Log the dataset name, version, and row count in every training run.
   tracker.log_params({"data.file": "churn_features_v20260401.parquet", "data.rows": 45231})

3. Keep training datasets for at least 12 months.
   Use lifecycle policies in S3/GCS/Azure Blob to enforce retention.

4. Add a data hash check before training.
   If the hash doesn't match the expected value, fail fast before spending
   hours training on corrupted data.
```

This takes 2 hours to implement and prevents the most common data reproducibility failures.

---

## 6. Anti-Patterns

**Overwriting training data files.** `churn_training.parquet` should never be overwritten. Once a version is written, it's immutable. Use dated or versioned filenames.

**Not logging data provenance.** A training run with no record of which data was used is not reproducible. Log data source, version, and row count as a non-negotiable minimum.

**Storing raw data in git.** Git is for code, not data. Files over 10MB don't belong in git. Use DVC, object storage, or a data warehouse.

**Training on live production data without a snapshot.** If the query runs against a live table, results change every time. Always create a point-in-time snapshot before training and log its location.

**Different data in dev vs staging.** Using synthetic data in dev and real data in staging is expected. But the data _contract_ must be identical — synthetic data must conform to the same schema. Divergence between dev and staging data contracts is a frequent source of staging failures.
