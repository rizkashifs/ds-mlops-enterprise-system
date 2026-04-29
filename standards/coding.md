# Coding Standards for ML Systems

> These standards apply to all code that runs in production or feeds production models.
> The goal: code that any engineer on the team can read, run, debug, and extend.

---

## Core Principles

**Config-driven over hardcoded.** Parameters, thresholds, paths, and environment settings belong in `configs/`, not in the code. Code changes require review; config changes are lower risk and faster to iterate.

**Modular, not monolithic.** Each pipeline stage (ingest, feature, train, validate, score) lives in its own module with clear inputs and outputs. Stages should be independently runnable and testable.

**Explicit over implicit.** Functions return typed dataclasses or Pydantic models, not tuples or raw dicts. Failures raise exceptions with messages; they don't return `None` silently.

**Reproducible by default.** Random seeds are always fixed and logged. Data versions are logged. The same input should always produce the same output.

---

## Project Structure

```
project/
├── configs/            # All configuration — no hardcoded values in code
│   ├── training.yaml
│   └── inference.yaml
├── pipelines/          # Orchestrators — thin; call src/ modules
│   ├── training_pipeline/
│   ├── inference_pipeline/
│   └── retraining_pipeline/
├── src/ or platform/   # Reusable modules and platform tools
│   ├── core/           # Contracts, lifecycle, shared types
│   ├── pipelines/      # Training and validation logic
│   └── services/       # Scoring and inference services
├── tests/              # One test file per source module
├── docs/               # Architecture, decisions, runbooks
└── examples/           # Runnable end-to-end examples
```

---

## Naming Conventions

### Files and modules

```python
# Good: lowercase, underscores, descriptive
training_pipeline.py
feature_engineering.py
model_validation.py

# Bad: vague, camelCase, or generic
utils.py          # What utilities?
Helper.py         # Helper for what?
ml_stuff.py       # Never acceptable
```

### Functions

```python
# Good: verb + noun, clear intent
def train_model(df, config) -> TrainingResult: ...
def validate_dataframe(df) -> List[str]: ...
def score_batch(df, model_uri) -> ScoringResult: ...

# Bad: vague or noun-only
def run(data): ...
def model(x): ...
def process_data(df): ...
```

### Variables

```python
# Good: descriptive, matches domain language
training_result = train_model(df, config)
validation_result = validate_model(metrics, thresholds)
churn_probability = score_result.probabilities

# Bad: single-letter or abbreviated
r = train_model(df, cfg)
v = validate_model(m, t)
p = sr.p
```

### Constants

```python
# Good: UPPER_SNAKE_CASE, at module level, with a comment if not obvious
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TEST_SIZE = 0.20
PSI_ALERT_THRESHOLD = 0.20
```

---

## Function Design

### Keep functions small and single-purpose

```python
# Good: each function does one thing
def load_training_data(contract: DataContract) -> pd.DataFrame:
    df = _read_parquet(contract)
    violations = contract.validate_dataframe(df)
    if violations:
        raise ValueError(f"Contract violations: {violations}")
    return df

def train_model(df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    ...

# Bad: one function that does everything
def run_training(path, params, test_size, seed, experiment_name, ...):
    df = pd.read_parquet(path)
    # validate, train, log, evaluate — all mixed together
    ...
```

### Use typed dataclasses for inputs and outputs

```python
# Good: explicit, discoverable, testable
@dataclass
class TrainingConfig:
    experiment_name: str
    model_params: Dict[str, Any]
    target_column: str = "target"
    test_size: float = 0.20

@dataclass
class TrainingResult:
    run_id: str
    metrics: Dict[str, float]
    model_uri: str
    feature_names: List[str]

def train_model(df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    ...

# Bad: positional arguments, no return type, unclear contract
def train(df, experiment, params, target, size):
    ...
    return run_id, metrics, uri
```

### Surface failures explicitly

```python
# Good: fail loudly with a useful message
violations = contract.validate_dataframe(df)
if violations:
    raise ValueError(f"Data contract violations: {violations}")

# Bad: silently continue
violations = contract.validate_dataframe(df)
if violations:
    print("Warning: some violations found")  # Then continues anyway
```

---

## Config-Driven Design

Load configuration from YAML at the top of a pipeline, not scattered through code.

```python
# Good: config loaded once, passed around
import yaml

with open("configs/training.yaml") as f:
    raw = yaml.safe_load(f)

config = TrainingConfig(
    experiment_name=raw["experiment"]["name"],
    model_params=raw["model"]["params"],
    target_column=raw["model"]["target_column"],
)
result = train_model(df, config)

# Bad: magic strings and numbers scattered through code
model = RandomForestClassifier(n_estimators=100, max_depth=5)  # Why these values?
mlflow.set_experiment("churn-prediction-v3-final-FINAL")        # Not from config
```

### Environment variables for secrets and infra

Anything environment-specific (endpoint URLs, credentials, bucket names) comes from environment variables — never from YAML files in the repo.

```python
import os

TRACKING_URI = os.environ.get("MLOPS_TRACKING_URI", "mlruns")  # local default
ARTIFACT_STORE = os.environ["MLOPS_ARTIFACT_STORE"]  # required; crash if missing
```

---

## Testing Standards

### What to test

| Test type | What it covers | Examples |
|---|---|---|
| Unit | Individual functions with simple inputs | `test_validate_dataframe`, `test_lifecycle_transitions` |
| Integration | Two modules working together | `test_train_then_validate` |
| Contract | Data schema compliance | `test_churn_features_conforms_to_contract` |
| Smoke | Pipeline runs end-to-end without error | `test_full_training_pipeline` |

### Test file naming

```
tests/
  test_contracts.py      # matches src/core/contracts.py
  test_lifecycle.py      # matches src/core/lifecycle.py
  test_validation.py     # matches src/pipelines/validation.py
  test_training.py       # matches src/pipelines/training.py
```

### Test conventions

```python
# Good: test names describe the scenario, not just the function
def test_contract_fails_on_missing_column():
def test_lifecycle_raises_on_invalid_transition():
def test_validation_passes_when_all_metrics_meet_threshold():

# Bad: vague names
def test_contract():
def test_lifecycle():
def test_1():
```

### Run tests before committing

```bash
pytest tests/ -v
```

A failing test must be fixed before a PR can be merged.

---

## Code Review Checklist

Before approving any PR that touches training or scoring code:

- [ ] No hardcoded parameters, thresholds, or paths
- [ ] All functions have type annotations on parameters and return value
- [ ] Input validation is present where data crosses a module boundary
- [ ] New functions have at least one unit test
- [ ] No commented-out code blocks
- [ ] No secrets, credentials, or sensitive data in the diff
- [ ] Config changes include a comment explaining the reasoning
