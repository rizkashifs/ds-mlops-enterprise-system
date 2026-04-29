"""Template: Batch inference / scoring job.

Copy this for any use case that needs to score a table on a schedule.
Replace all TODO items with your implementation.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops_platform.model_registry.registry import get_production_uri
from mlops_platform.monitoring_hooks.hooks import build_monitoring_report
from src.services.scoring import score_batch


def load_scoring_data() -> pd.DataFrame:
    # TODO: load the population to score (e.g., all active customers)
    # Must NOT include the target column
    raise NotImplementedError("Replace with your data loading logic")


def write_scores(scores_df: pd.DataFrame) -> None:
    # TODO: write output to your destination
    # Examples:
    #   scores_df.to_parquet("s3://bucket/scores/date=2026-04-29/scores.parquet")
    #   scores_df.to_sql("model_scores", conn, if_exists="append")
    raise NotImplementedError("Replace with your output destination")


def run(model_name: str) -> None:
    # 1. Load data
    df = load_scoring_data()
    print(f"Loaded {len(df):,} records for scoring")

    # 2. Score
    model_uri = get_production_uri(model_name)
    result = score_batch(df, model_uri)

    # 3. Write output
    write_scores(result.to_dataframe())
    print(f"Scores written: {result.num_records:,} records")

    # 4. Monitoring
    report = build_monitoring_report(model_name=model_name, scores=result.probabilities)
    print(f"Monitoring: mean={report.mean_score:.4f}, alerts={len(report.alerts)}")
    for alert in report.alerts:
        print(f"  ALERT: {alert}")


if __name__ == "__main__":
    run(model_name="your-model-name")  # TODO: replace
