# Data Contract Guide

> A data contract is the agreement between data producers and data consumers about what a dataset looks like, who owns it, and what guarantees come with it.

---

## Why we use data contracts

Without a contract, a schema change in an upstream table can silently corrupt a downstream model. The model keeps running, producing wrong scores, until someone notices something off in business metrics weeks later.

A data contract makes that failure **loud and immediate** — it fires at ingestion, not at scoring time. The earlier a failure surfaces, the cheaper it is to fix.

Contracts also serve as living documentation. When a data scientist asks "what does `tenure_months` actually measure?", the contract is the answer.

---

## Contract fields

Every contract defines:

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Unique identifier for this dataset (e.g., `churn_features_v1`) |
| `version` | Yes | Semantic version of this contract (e.g., `1.0`, `2.1`) |
| `owner` | Yes | Team accountable for producing this data (e.g., `data-engineering`) |
| `description` | No | Plain-language description of what this dataset represents |
| `columns` | Yes | List of column specs (see below) |

Each **column spec** defines:

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Column name as it appears in the DataFrame |
| `dtype` | Yes | One of: `numeric`, `categorical`, `boolean`, `datetime` |
| `nullable` | No | Whether null values are permitted (default: `false`) |
| `description` | No | What this column measures |

---

## Defining a contract in code

```python
from src.core.contracts import DataContract, ColumnSpec, ColumnType

contract = DataContract(
    name="churn_features_v1",
    version="1.0",
    owner="data-engineering",
    description="Customer-level features for monthly churn prediction",
    columns=[
        ColumnSpec(
            name="tenure_months",
            dtype=ColumnType.NUMERIC,
            nullable=False,
            description="Number of months since account creation",
        ),
        ColumnSpec(
            name="monthly_charges",
            dtype=ColumnType.NUMERIC,
            nullable=False,
            description="Average monthly spend over last 3 months",
        ),
        ColumnSpec(
            name="segment",
            dtype=ColumnType.CATEGORICAL,
            nullable=True,
            description="Customer segment label; null for unclassified accounts",
        ),
    ],
)
```

---

## Validating a DataFrame against a contract

```python
violations = contract.validate_dataframe(df)

if violations:
    for v in violations:
        print(f"VIOLATION: {v}")
    raise ValueError("Data does not conform to contract. Pipeline halted.")
```

What `validate_dataframe` checks:

1. **All required columns are present.** Missing columns are listed by name.
2. **Non-nullable columns have no null values.** Each violating column is reported individually.

The validator returns a list of strings. An empty list means the data is compliant. The pipeline should halt on any non-empty list.

---

## Storing contracts

Contracts are version-controlled in `configs/pipeline_contracts.yaml`. This makes them:

- Reviewable in pull requests (schema changes require code review)
- Discoverable (one place to look up any dataset's schema)
- Auditable (git history shows when schemas changed)

```yaml
# configs/pipeline_contracts.yaml
contracts:
  churn_features_v1:
    version: "1.0"
    owner: data-engineering
    description: Customer features for churn prediction
    columns:
      - name: tenure_months
        dtype: numeric
        nullable: false
        description: Months since account creation
      - name: monthly_charges
        dtype: numeric
        nullable: false
```

---

## Versioning rules

### When to bump the minor version (1.0 → 1.1)

Minor version bumps are **backwards-compatible**. Models trained on v1.0 can still use v1.1 data.

- Adding a new **nullable** column
- Improving a column description
- Relaxing nullability (non-nullable → nullable)

### When to bump the major version (1.x → 2.0)

Major version bumps are **breaking changes**. Any model trained on v1.x must be re-evaluated before being used with v2.x data.

- Removing a column
- Renaming a column
- Changing a column's data type
- Tightening nullability (nullable → non-nullable)
- Changing the semantic meaning of a column

### What to do when a breaking change happens

1. Create a new contract version (e.g., `churn_features_v2`)
2. Update the training pipeline to reference the new version
3. Retrain the model on data conforming to the new contract
4. Go through the standard validation and approval process
5. Do not silently migrate a DEPLOYED model to a new contract version

---

## Adding data quality checks beyond schema

Schema validation is the baseline. For production datasets, add these additional checks:

### Row count check

```python
EXPECTED_ROW_RANGE = (10_000, 5_000_000)

if not (EXPECTED_ROW_RANGE[0] <= len(df) <= EXPECTED_ROW_RANGE[1]):
    raise ValueError(f"Unexpected row count: {len(df)}")
```

### Value range check (for numeric columns)

```python
if not df["tenure_months"].between(0, 120).all():
    raise ValueError("tenure_months contains values outside expected range [0, 120]")
```

### Categorical value check

```python
KNOWN_SEGMENTS = {"consumer", "business", "enterprise"}
unknown = set(df["segment"].dropna().unique()) - KNOWN_SEGMENTS
if unknown:
    raise ValueError(f"Unknown segment values: {unknown}")
```

### Distribution drift check (PSI)

Use PSI (Population Stability Index) to compare today's data distribution to the training baseline:

| PSI | Interpretation |
|---|---|
| < 0.10 | No significant shift |
| 0.10 – 0.20 | Moderate — investigate |
| > 0.20 | Significant shift — halt pipeline or trigger review |

---

## Common mistakes

**Mistake: Defining a contract but not enforcing it**
The contract is only useful if it's called before the pipeline runs. Add `contract.validate_dataframe(df)` as the first step of every ingestion function.

**Mistake: Using the same contract across major versions**
When upstream data changes structure, update the contract version rather than quietly changing the contract definition in place. The old version should remain in the config file for traceability.

**Mistake: Not including the contract version in the MLflow run record**
Log the contract name and version as a tag on every training run: `mlflow.set_tag("data_contract", "churn_features_v1:1.0")`. This makes every model's provenance traceable.

**Mistake: Treating nullable columns as optional features**
A column being nullable means it can have nulls — it doesn't mean the model can ignore it. Decide explicitly how nulls are handled in feature engineering and document that decision.
