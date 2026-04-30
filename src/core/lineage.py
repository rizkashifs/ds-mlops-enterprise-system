"""Data lineage tracking.

Records what data fed into a run, what transformations were applied, and
what artifact was produced. Records are attached to MLflow runs as JSON
artifacts so lineage survives alongside model weights.

Usage:
    record = create_lineage_record(run_id="abc123", model_name="churn-v2")
    record.inputs.append(DataSource(
        name="customer_features",
        version="v3",
        location="s3://bucket/features/2026-04-30/",
        snapshot_timestamp="2026-04-30T00:00:00Z",
        row_count=420_000,
    ))
    record.transformations.append(TransformationStep(
        name="standard_scaler",
        description="StandardScaler fitted on training split",
        parameters={"with_mean": True, "with_std": True},
    ))
    record.output_artifact = "runs:/abc123/model"
    log_lineage_to_mlflow(record)
"""
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class DataSource:
    name: str
    version: str
    location: str
    snapshot_timestamp: str
    row_count: Optional[int] = None
    schema_hash: Optional[str] = None


@dataclass
class TransformationStep:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageRecord:
    run_id: str
    model_name: str
    recorded_at: str
    inputs: List[DataSource] = field(default_factory=list)
    transformations: List[TransformationStep] = field(default_factory=list)
    output_artifact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def create_lineage_record(run_id: str, model_name: str) -> LineageRecord:
    return LineageRecord(
        run_id=run_id,
        model_name=model_name,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )


def log_lineage_to_mlflow(record: LineageRecord) -> None:
    """Attach the lineage record as a JSON artifact on the active MLflow run."""
    import mlflow

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="lineage_"
    ) as f:
        f.write(record.to_json())
        tmp_path = f.name
    try:
        mlflow.log_artifact(tmp_path, artifact_path="lineage")
    finally:
        os.unlink(tmp_path)
