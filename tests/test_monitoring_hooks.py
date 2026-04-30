import numpy as np
import pandas as pd
import pytest

from mlops_platform.monitoring_hooks.hooks import build_monitoring_report, compute_psi


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(42)
    data = rng.random(1000)
    assert compute_psi(data, data.copy()) < 0.01


def test_psi_fully_separated_distributions_is_large():
    rng = np.random.default_rng(42)
    expected = rng.uniform(0.0, 0.3, 1000)
    actual = rng.uniform(0.7, 1.0, 1000)
    assert compute_psi(expected, actual) > 0.20


def test_monitoring_report_basic_stats():
    scores = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
    report = build_monitoring_report("test-model", scores)

    assert report.num_records == 5
    assert report.mean_score == pytest.approx(scores.mean())
    assert report.p50 == pytest.approx(np.percentile(scores, 50))
    assert report.model_name == "test-model"
    assert not report.has_alerts()


def test_monitoring_report_empty_scores_raises_alert():
    scores = pd.Series([], dtype=float)
    report = build_monitoring_report("test-model", scores)
    assert report.has_alerts()
    assert any("empty" in a.lower() for a in report.alerts)


def test_monitoring_report_psi_alert_fires_above_threshold():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"age": rng.uniform(0.0, 0.3, 1000)})
    current = pd.DataFrame({"age": rng.uniform(0.7, 1.0, 1000)})
    scores = pd.Series(rng.random(1000))

    report = build_monitoring_report(
        "test-model",
        scores,
        features_df=current,
        baseline_features=baseline,
    )

    assert "age" in report.psi_by_feature
    assert report.psi_by_feature["age"] > 0.20
    assert report.has_alerts()
    assert any("age" in a for a in report.alerts)


def test_monitoring_report_no_psi_without_baseline():
    rng = np.random.default_rng(1)
    scores = pd.Series(rng.random(200))
    report = build_monitoring_report("test-model", scores)
    assert report.psi_by_feature == {}


def test_monitoring_report_skips_columns_not_in_baseline():
    rng = np.random.default_rng(2)
    baseline = pd.DataFrame({"age": rng.random(500)})
    current = pd.DataFrame({"age": rng.random(500), "income": rng.random(500)})
    scores = pd.Series(rng.random(500))

    report = build_monitoring_report(
        "test-model", scores, features_df=current, baseline_features=baseline
    )
    assert "age" in report.psi_by_feature
    assert "income" not in report.psi_by_feature
