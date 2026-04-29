"""Data contracts and model cards — the shared language of this MLOps system.

A DataContract defines what a dataset must look like before any pipeline touches it.
A ModelCard documents what a model does, who owns it, and what its limits are.
Both are required at every lifecycle gate.
"""
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class ColumnSpec(BaseModel):
    name: str
    dtype: ColumnType
    nullable: bool = False
    description: str = ""


class DataContract(BaseModel):
    """Schema + ownership declaration for a dataset.

    Teams define one contract per dataset version. Pipelines validate against
    it at ingestion time. Violations block pipeline execution.
    """
    name: str
    version: str
    owner: str
    columns: List[ColumnSpec]
    description: str = ""

    def validate_dataframe(self, df) -> List[str]:
        """Return violation messages. Empty list means the data is contract-compliant."""
        violations = []
        expected_cols = {c.name for c in self.columns}
        actual_cols = set(df.columns)
        for missing in expected_cols - actual_cols:
            violations.append(f"Missing required column: {missing}")
        for col in self.columns:
            if col.name in df.columns and not col.nullable and df[col.name].isnull().any():
                violations.append(f"Null values in non-nullable column: {col.name}")
        return violations


class ModelCard(BaseModel):
    """Standard metadata required for any model entering the review process.

    Fills in the 'who, what, why, and limits' before a model gets an approval
    number. Governance teams read this; keep it honest and plain-language.
    """
    model_name: str
    version: str
    owner: str
    created_date: str
    description: str
    intended_use: str
    out_of_scope_use: str
    training_data: str
    evaluation_metrics: Dict[str, float]
    known_limitations: str
    ethical_considerations: str
    approval_status: str = "pending"
