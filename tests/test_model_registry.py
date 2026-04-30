from unittest.mock import MagicMock, patch

from mlops_platform.model_registry.registry import (
    archive_model,
    get_production_uri,
    promote_to_production,
    register_model,
)


def test_get_production_uri():
    assert get_production_uri("fraud-model") == "models:/fraud-model/Production"
    assert get_production_uri("churn-v2") == "models:/churn-v2/Production"


@patch("mlops_platform.model_registry.registry.MlflowClient")
def test_register_model_returns_staging_result(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_version = MagicMock()
    mock_version.version = "3"
    mock_client.create_model_version.return_value = mock_version

    result = register_model(run_id="abc123", model_name="fraud-model", description="test run")

    assert result.model_name == "fraud-model"
    assert result.version == "3"
    assert result.stage == "Staging"
    assert result.run_id == "abc123"
    mock_client.transition_model_version_stage.assert_called_once_with(
        name="fraud-model", version="3", stage="Staging"
    )


@patch("mlops_platform.model_registry.registry.MlflowClient")
def test_register_model_tolerates_existing_registered_model(mock_client_cls):
    import mlflow.exceptions

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.create_registered_model.side_effect = mlflow.exceptions.RestException(
        {"error_code": "RESOURCE_ALREADY_EXISTS", "message": "already exists"}
    )
    mock_version = MagicMock()
    mock_version.version = "2"
    mock_client.create_model_version.return_value = mock_version

    result = register_model(run_id="xyz", model_name="existing-model")
    assert result.version == "2"


@patch("mlops_platform.model_registry.registry.MlflowClient")
def test_promote_to_production_archives_existing(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    promote_to_production("fraud-model", "3")

    mock_client.transition_model_version_stage.assert_called_once_with(
        name="fraud-model",
        version="3",
        stage="Production",
        archive_existing_versions=True,
    )


@patch("mlops_platform.model_registry.registry.MlflowClient")
def test_archive_model_with_reason(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    archive_model("fraud-model", "2", reason="retired after v3 rollout")

    mock_client.update_model_version.assert_called_once_with(
        name="fraud-model", version="2", description="retired after v3 rollout"
    )
    mock_client.transition_model_version_stage.assert_called_once_with(
        name="fraud-model", version="2", stage="Archived"
    )


@patch("mlops_platform.model_registry.registry.MlflowClient")
def test_archive_model_without_reason_skips_description_update(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    archive_model("fraud-model", "2")

    mock_client.update_model_version.assert_not_called()
    mock_client.transition_model_version_stage.assert_called_once()
