import pandas as pd
import pytest

from src.core.contracts import ColumnSpec, ColumnType, DataContract, ModelCard


def make_contract():
    return DataContract(
        name="test_contract",
        version="1.0",
        owner="test-team",
        columns=[
            ColumnSpec(name="age", dtype=ColumnType.NUMERIC),
            ColumnSpec(name="category", dtype=ColumnType.CATEGORICAL, nullable=True),
        ],
    )


def test_contract_valid_dataframe():
    contract = make_contract()
    df = pd.DataFrame({"age": [25, 30], "category": ["A", None]})
    assert contract.validate_dataframe(df) == []


def test_contract_missing_column():
    contract = make_contract()
    df = pd.DataFrame({"age": [25, 30]})  # missing "category"
    violations = contract.validate_dataframe(df)
    assert any("category" in v for v in violations)


def test_contract_null_in_non_nullable():
    contract = make_contract()
    df = pd.DataFrame({"age": [25, None], "category": ["A", "B"]})
    violations = contract.validate_dataframe(df)
    assert any("age" in v for v in violations)


def test_model_card_defaults():
    card = ModelCard(
        model_name="test-model",
        version="1.0",
        owner="ds-team",
        created_date="2026-01-01",
        description="test",
        intended_use="testing",
        out_of_scope_use="none",
        training_data="synthetic",
        evaluation_metrics={"accuracy": 0.85},
        known_limitations="none",
        ethical_considerations="none",
    )
    assert card.approval_status == "pending"
