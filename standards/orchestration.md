# Orchestration Standards

> This document defines what a pipeline orchestrator must provide and how to integrate
> the pipelines in this repo with the orchestration platform your org uses.
> No orchestration framework is bundled here — the choice depends on your infrastructure.

---

## Contents

1. [What Orchestration Must Provide](#1-what-orchestration-must-provide)
2. [Platform Comparison](#2-platform-comparison)
3. [How to Wrap the Existing Pipelines](#3-how-to-wrap-the-existing-pipelines)
4. [Scheduling Patterns](#4-scheduling-patterns)
5. [What Not to Put in the Orchestrator](#5-what-not-to-put-in-the-orchestrator)

---

## 1. What Orchestration Must Provide

Every orchestration platform a team adopts must satisfy these requirements regardless of tool choice:

### Required capabilities

| Capability | Why |
|---|---|
| **Scheduling** | Pipelines run on a defined cadence without human trigger |
| **Retry logic** | Transient failures (network, data unavailability) are retried with backoff |
| **DAG dependencies** | Pipeline B doesn't start until Pipeline A succeeds |
| **Failure alerting** | On-call is notified when a pipeline fails; not discovered at scoring time |
| **Run history** | Every pipeline execution is logged with start time, end time, status |
| **Parameter passing** | Config and secrets are injected at runtime, not hardcoded |
| **Manual trigger** | A pipeline can be triggered on-demand for reruns, hotfixes |
| **Environment isolation** | Dev, staging, and prod pipelines run in separate contexts |

### Minimum viable (for teams getting started)

If you don't have full orchestration yet, implement at minimum:
1. A cron job that runs `python pipelines/training_pipeline/train.py`
2. An alert (email, Slack) if the cron job fails
3. A log file that records each run's start time, end time, and exit code

Add proper orchestration incrementally. Running a cron job with alerting is far better than no automation.

---

## 2. Platform Comparison

| Platform | Best for | Scheduling | DAGs | Managed | Notes |
|---|---|---|---|---|---|
| **AWS Step Functions** | AWS-native teams, serverless workloads | EventBridge (cron) | Yes (state machines) | Yes | No infra to manage; pay-per-execution; integrates with Glue, Lambda, SageMaker |
| **Azure Durable Functions** | Azure-native teams | Timer triggers | Yes (function chaining/fan-out) | Yes | Serverless; integrates with Azure ML, Data Factory |
| **GCP Cloud Workflows** | GCP-native teams | Cloud Scheduler | Yes | Yes | Integrates with Vertex AI, BigQuery |
| **Apache Airflow** | On-premise or Kubernetes; teams wanting full control | Native cron | Yes (Python DAGs) | No (unless managed) | Most widely adopted; highest operational overhead |
| **Prefect** | Python-first teams; hybrid cloud | Native scheduler | Yes (flows) | Yes (Prefect Cloud) | Lighter than Airflow; good local dev story |
| **Dagster** | Teams wanting asset-based orchestration | Native scheduler | Yes (jobs/assets) | Yes (Dagster+) | Strong data asset lineage model |
| **Cron + shell scripts** | Very small teams or PoC | OS cron | No | No | Zero overhead; insufficient at scale |

### Decision guidance

```
Is your infrastructure primarily on one cloud?
  YES → Use that cloud's native orchestrator (Step Functions / Durable Functions / Cloud Workflows)
  NO  → Are you Kubernetes-native?
          YES → Airflow on Kubernetes or Prefect with Kubernetes executor
          NO  → Prefect Cloud or Dagster+ (managed, minimal ops overhead)

Is this a PoC or early-stage team (< 3 pipelines)?
  YES → Cron + alerting is fine. Add proper orchestration when you have 5+ pipelines.
```

---

## 3. How to Wrap the Existing Pipelines

The pipelines in this repo (`pipelines/training_pipeline/train.py`, `pipelines/inference_pipeline/score.py`, `pipelines/retraining_pipeline/retrain.py`) are plain Python scripts. Each exposes a `run_*()` function that can be called from any orchestrator.

### AWS Step Functions

```python
# Lambda function wrapping the training pipeline
import json
import sys
sys.path.insert(0, "/var/task")

import pandas as pd
from pipelines.training_pipeline.train import run_training_pipeline

def handler(event, context):
    # Load data from S3, Glue, or another source
    df = load_data_from_s3(event["s3_path"])
    result = run_training_pipeline(df, config_path=event.get("config_path", "configs/training.yaml"))
    return {
        "statusCode": 200,
        "body": json.dumps({
            "run_id": result["run_id"],
            "validation_passed": result["validation_passed"],
            "status": result["status"],
        })
    }
```

State machine definition (simplified):
```json
{
  "StartAt": "TrainModel",
  "States": {
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:function:train-model",
      "Next": "CheckValidation",
      "Retry": [{"ErrorEquals": ["States.ALL"], "MaxAttempts": 2}]
    },
    "CheckValidation": {
      "Type": "Choice",
      "Choices": [
        {"Variable": "$.validation_passed", "BooleanEquals": true, "Next": "AwaitApproval"}
      ],
      "Default": "NotifyFailure"
    },
    "AwaitApproval": {"Type": "Task", "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken", "End": true},
    "NotifyFailure": {"Type": "Task", "Resource": "arn:aws:lambda:...:function:send-alert", "End": true}
  }
}
```

### Prefect

```python
from prefect import flow, task
import pandas as pd
from pipelines.training_pipeline.train import run_training_pipeline
from pipelines.inference_pipeline.score import run_inference_pipeline

@task(retries=2, retry_delay_seconds=60)
def train(df: pd.DataFrame) -> dict:
    return run_training_pipeline(df)

@task
def score(df: pd.DataFrame, model_uri: str) -> dict:
    return run_inference_pipeline(df, model_uri=model_uri)

@flow(name="churn-weekly-retrain")
def weekly_retrain_flow():
    df = load_training_data()
    result = train(df)
    if result["validation_passed"]:
        score_result = score(load_scoring_data(), result["model_uri"])
```

Deploy:
```bash
prefect deployment build pipelines/orchestration/prefect_flow.py:weekly_retrain_flow \
  --name "churn-weekly-retrain" \
  --cron "0 2 * * 1"   # every Monday at 02:00
prefect deployment apply weekly_retrain_flow-deployment.yaml
```

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_task(**context):
    import pandas as pd
    from pipelines.training_pipeline.train import run_training_pipeline
    df = load_data()
    result = run_training_pipeline(df)
    context["ti"].xcom_push(key="model_uri", value=result["model_uri"])
    return result["validation_passed"]

with DAG("churn_weekly_retrain", schedule_interval="0 2 * * 1", start_date=datetime(2026, 1, 1)) as dag:
    train = PythonOperator(task_id="train_model", python_callable=train_task)
    # Add score, monitor tasks similarly
```

### Azure Durable Functions

```python
import azure.functions as func
import azure.durable_functions as df

def orchestrator(context: df.DurableOrchestrationContext):
    result = yield context.call_activity("TrainModel", {"config_path": "configs/training.yaml"})
    if result["validation_passed"]:
        yield context.call_activity("ScoreModel", {"model_uri": result["model_uri"]})
    else:
        yield context.call_activity("SendAlert", {"message": "Training validation failed"})

main = df.Orchestrator.create(orchestrator)
```

---

## 4. Scheduling Patterns

### Standard ML pipeline schedule

| Pipeline | Trigger | Typical cadence |
|---|---|---|
| Inference / scoring | Schedule | Daily or weekly (depending on decision cadence) |
| Monitoring evaluation | After inference | Immediately after each scoring run |
| Retraining | Trigger-based or schedule | On trigger signal, or monthly as safety net |
| Data contract validation | Before inference | Before each scoring run |

### Event-driven retraining

The preferred pattern — retrain only when there's a signal, not blindly on schedule:

```
Inference pipeline → monitoring report → evaluate_triggers()
  → trigger.urgency == "immediate" → fire retrain job now
  → trigger.urgency == "schedule"  → add to retrain queue
  → trigger.urgency == "none"      → no action
```

Wire this by connecting the trigger output from `run_inference_pipeline()` to your orchestrator's event or queue system.

---

## 5. What Not to Put in the Orchestrator

The orchestrator's job is scheduling, retrying, and routing. **Business logic belongs in `src/`.**

| Put here (orchestrator) | Put here (`src/`) |
|---|---|
| Retry configuration | Feature engineering |
| Scheduling cadence | Validation thresholds |
| Alerting on failure | Model training logic |
| Environment variable injection | Contract validation |
| DAG dependencies | Monitoring calculations |
| Manual trigger configuration | Trigger evaluation logic |

If you find yourself putting feature computation or threshold checks in a DAG task definition, move it to `src/pipelines/` and call it from the orchestrator task.
