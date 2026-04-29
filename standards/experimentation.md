# Experimentation Standards

> Every training run is an experiment. Every experiment must be tracked.
> An experiment that isn't tracked cannot be reproduced, reviewed, or promoted.

---

## Why tracking matters

Without experiment tracking:
- You can't reproduce a result that was promising 3 weeks ago
- You can't compare two approaches fairly
- You can't audit what data or parameters produced the production model
- You can't explain to a reviewer why you chose this model over alternatives

With tracking:
- Every run is a permanent record: who ran it, when, with what data and params
- The production model has a full provenance chain back to its training run
- Experiments are comparable and sortable
- Rollbacks are possible because you can load any past model artifact

---

## The non-negotiable logging list

Every training run must log the following before the run closes. Runs missing any of these items are not eligible for promotion.

### Parameters

```python
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 5,
    "test_size": 0.20,
    "random_seed": 42,
})
```

### Metrics

```python
mlflow.log_metrics({
    "accuracy": 0.82,
    "f1": 0.75,
    "roc_auc": 0.88,
    "precision": 0.79,
    "recall": 0.71,
})
```

Always log test-set metrics. Optionally log train-set metrics alongside for comparing overfitting.

### Model artifact

```python
mlflow.sklearn.log_model(model, "model")
# or for other frameworks:
mlflow.pytorch.log_model(model, "model")
mlflow.xgboost.log_model(model, "model")
```

### Tags (metadata)

```python
mlflow.set_tags({
    "owner": "jane.smith@company.com",
    "use_case": "customer_churn",
    "data_contract": "churn_features_v1:1.0",
    "lifecycle_stage": "experimental",
    "team": "ds-retention",
})
```

### Optional but recommended

```python
# Feature names (important for debugging training-serving skew)
mlflow.set_tag("feature_names", str(list(X.columns)))

# Data shape
mlflow.log_params({"train_rows": len(X_train), "test_rows": len(X_test)})

# Requirements snapshot
mlflow.log_artifact("requirements.txt")

# Feature importances
pd.Series(model.feature_importances_, index=X.columns).to_csv("feat_importance.csv")
mlflow.log_artifact("feat_importance.csv")
```

---

## Experiment naming convention

```
{use-case}-{environment}

Examples:
  churn-prediction-dev
  churn-prediction-staging
  fraud-detection-dev
  marketing-propensity-dev
```

Use a consistent name for all runs within the same use case. MLflow groups runs by experiment name. Using "churn-v2-FINAL-real" as an experiment name creates a mess and is not permitted.

---

## Structuring an experiment in code

Always use a context manager to ensure the run closes cleanly:

```python
import mlflow

mlflow.set_experiment("churn-prediction-dev")

with mlflow.start_run(run_name="rf-baseline-100trees") as run:
    # Log params BEFORE training (so they appear even if training crashes)
    mlflow.log_params(config.model_params)

    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
```

---

## Comparing runs fairly

When comparing two experiments:
- Use the **same test set** for both
- Use the **same random seed** for train/test splits
- Use the **same data contract version**
- Log both runs to the **same MLflow experiment**

A comparison is not valid if the two runs used different data or different splits.

---

## Experiment hygiene

### Do

- Name runs descriptively: `rf-balanced-nofeature-z` not `run_1_final`
- Tag all runs with the owner and use case
- Archive runs you've decided to abandon (don't delete — they're part of the audit trail)
- Add a note to the run description if there's context that isn't captured in params

### Don't

- Don't run experiments from notebooks without using `mlflow.start_run()`
- Don't commit model artifacts to git — models belong in the artifact store
- Don't manually create run records — always use the tracking API
- Don't reuse run IDs — they are immutable records

---

## From experiment to model candidate

When you have a run worth promoting, document:

1. **Which run is the candidate**: record the `run_id` in the model card
2. **Why this run**: explain what made this run better than others (different features, tuned params, better data)
3. **What you compared it against**: at minimum, the current production model and a simple baseline
4. **What the features are**: list the feature names logged in the run

Only runs that have all required metrics and artifacts can advance to CANDIDATE status.

---

## Reproducibility checklist

Before tagging a run as a candidate for promotion:

- [ ] `random_state` / `seed` is fixed and logged
- [ ] Data contract name and version are tagged on the run
- [ ] All hyperparameters are logged (nothing hardcoded without logging)
- [ ] Model artifact is stored in MLflow (not just locally)
- [ ] Feature names are logged
- [ ] `requirements.txt` or equivalent is attached as an artifact
- [ ] Running `train_model(same_data, same_config)` produces the same metrics (±noise)
