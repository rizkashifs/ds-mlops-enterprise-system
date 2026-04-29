import pytest

from src.pipelines.validation import ValidationThresholds, validate_model


GOOD_METRICS = {"accuracy": 0.82, "f1": 0.75, "roc_auc": 0.88}
DEFAULT_THRESHOLDS = ValidationThresholds()


def test_passing_metrics():
    result = validate_model(GOOD_METRICS, DEFAULT_THRESHOLDS)
    assert result.passed
    assert result.failures == []


def test_failing_accuracy():
    metrics = {**GOOD_METRICS, "accuracy": 0.60}
    result = validate_model(metrics, DEFAULT_THRESHOLDS)
    assert not result.passed
    assert any("accuracy" in f for f in result.failures)


def test_failing_f1():
    metrics = {**GOOD_METRICS, "f1": 0.50}
    result = validate_model(metrics, DEFAULT_THRESHOLDS)
    assert not result.passed
    assert any("f1" in f for f in result.failures)


def test_missing_metric_is_hard_failure():
    metrics = {"accuracy": 0.85, "f1": 0.75}  # missing roc_auc
    result = validate_model(metrics, DEFAULT_THRESHOLDS)
    assert not result.passed
    assert any("roc_auc" in f for f in result.failures)


def test_near_threshold_warning():
    metrics = {**GOOD_METRICS, "accuracy": 0.71}  # just above 0.70, within 5pp
    result = validate_model(metrics, DEFAULT_THRESHOLDS)
    assert result.passed
    assert any("accuracy" in w for w in result.warnings)


def test_custom_thresholds():
    strict = ValidationThresholds(min_accuracy=0.90, min_f1=0.85, min_roc_auc=0.90)
    result = validate_model(GOOD_METRICS, strict)
    assert not result.passed


def test_summary_string_on_failure():
    metrics = {**GOOD_METRICS, "accuracy": 0.50}
    result = validate_model(metrics, DEFAULT_THRESHOLDS)
    summary = result.summary()
    assert "FAILED" in summary
    assert "accuracy" in summary
