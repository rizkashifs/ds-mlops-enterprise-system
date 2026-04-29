"""Template: Tabular ML pipeline.

Copy this template to start a new tabular ML use case.
Replace every TODO comment with your implementation.
Delete this docstring once you've customized the file.

Steps to adapt:
  1. Define YOUR_CONTRACT with your actual columns
  2. Replace load_data() with your actual data source
  3. Adjust model params and thresholds in configs/training.yaml
  4. Add feature engineering in encode_features() if needed
  5. Run this file to verify the pipeline works end-to-end
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.contracts import ColumnSpec, ColumnType, DataContract
from src.core.lifecycle import ModelStatus, transition
from src.pipelines.training import TrainingConfig, train_model
from src.pipelines.validation import ValidationThresholds, validate_model

# ---------------------------------------------------------------------------
# TODO: Define your data contract
# ---------------------------------------------------------------------------

YOUR_CONTRACT = DataContract(
    name="your_dataset_v1",                  # TODO: update
    version="1.0",
    owner="your-team",                        # TODO: update
    description="TODO: describe the dataset",
    columns=[
        ColumnSpec(name="feature_1", dtype=ColumnType.NUMERIC),    # TODO: replace
        ColumnSpec(name="feature_2", dtype=ColumnType.CATEGORICAL), # TODO: replace
        ColumnSpec(name="target", dtype=ColumnType.NUMERIC),
    ],
)


# ---------------------------------------------------------------------------
# TODO: Load your data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    # TODO: replace with your actual data loading logic
    # Examples:
    #   return pd.read_parquet("s3://bucket/path/to/data.parquet")
    #   return pd.read_sql("SELECT * FROM features WHERE ...", conn)
    raise NotImplementedError("Replace load_data() with your data source")


# ---------------------------------------------------------------------------
# TODO: Add feature engineering (optional)
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: handle nulls, encode categoricals, create derived features
    # If no feature engineering is needed, return df as-is
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config_path: str = "configs/training.yaml") -> None:
    # 1. Load + validate
    df = load_data()
    violations = YOUR_CONTRACT.validate_dataframe(df)
    if violations:
        raise ValueError(f"Contract violations: {violations}")
    print(f"Data: {len(df):,} rows, {df.shape[1]} columns")

    # 2. Feature engineering
    df = encode_features(df)

    # 3. Train
    config = TrainingConfig(
        experiment_name="your-use-case-dev",  # TODO: update
        model_params={"n_estimators": 100, "max_depth": 5},
        target_column="target",
    )
    result = train_model(df, config)
    print(f"Training: run_id={result.run_id}, metrics={result.metrics}")

    # 4. Validate
    thresholds = ValidationThresholds(
        min_accuracy=0.70,  # TODO: set based on business requirements
        min_f1=0.60,
        min_roc_auc=0.70,
    )
    validation = validate_model(result.metrics, thresholds)
    print(validation.summary())

    # 5. Lifecycle
    if validation.passed:
        status = transition(ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE)
        print(f"Promoted to: {status.value}")
    else:
        print("Validation failed. Fix issues before promoting.")


if __name__ == "__main__":
    run()
