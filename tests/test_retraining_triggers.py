import pytest

from mlops_platform.monitoring_hooks.hooks import MonitoringReport
from mlops_platform.monitoring_hooks.triggers import TriggerConfig, TriggerDecision, evaluate_triggers

SCORED_AT = "2026-04-29T00:00:00+00:00"


def make_report(
    psi_by_feature=None,
    mean_score=0.20,
    num_records=1000,
) -> MonitoringReport:
    return MonitoringReport(
        model_name="test-model",
        scored_at=SCORED_AT,
        num_records=num_records,
        mean_score=mean_score,
        p10=0.05,
        p50=0.18,
        p90=0.45,
        p99=0.80,
        psi_by_feature=psi_by_feature or {},
    )


DEFAULT_CONFIG = TriggerConfig()


# ---------------------------------------------------------------------------
# No retrain expected
# ---------------------------------------------------------------------------

def test_no_trigger_when_all_stable():
    report = make_report(psi_by_feature={"tenure_months": 0.05})
    decision = evaluate_triggers(report, DEFAULT_CONFIG, days_since_last_retrain=30, baseline_mean_score=0.20)
    assert not decision.should_retrain
    assert decision.urgency == "none"


def test_no_trigger_insufficient_records():
    report = make_report(psi_by_feature={"tenure": 0.25}, num_records=100)
    config = TriggerConfig(min_records_for_drift_check=500)
    decision = evaluate_triggers(report, config)
    assert not decision.should_retrain
    assert any("drift checks skipped" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# Feature drift triggers
# ---------------------------------------------------------------------------

def test_psi_above_alert_triggers_immediate():
    report = make_report(psi_by_feature={"monthly_charges": 0.25})
    decision = evaluate_triggers(report, DEFAULT_CONFIG)
    assert decision.should_retrain
    assert decision.urgency == "immediate"
    assert decision.triggered_by == "feature_drift"
    assert any("monthly_charges" in r for r in decision.reasons)


def test_psi_in_warn_range_triggers_schedule():
    report = make_report(psi_by_feature={"monthly_charges": 0.15})
    decision = evaluate_triggers(report, DEFAULT_CONFIG)
    assert decision.should_retrain
    assert decision.urgency == "schedule"
    assert decision.triggered_by == "feature_drift_moderate"
    assert any("monthly_charges" in w for w in decision.warnings)


def test_psi_below_warn_no_trigger():
    report = make_report(psi_by_feature={"monthly_charges": 0.05})
    decision = evaluate_triggers(report, DEFAULT_CONFIG)
    assert not decision.should_retrain


# ---------------------------------------------------------------------------
# Score distribution shift
# ---------------------------------------------------------------------------

def test_score_shift_triggers_immediate():
    report = make_report(mean_score=0.35)  # 75% relative shift from baseline 0.20
    decision = evaluate_triggers(report, DEFAULT_CONFIG, baseline_mean_score=0.20)
    assert decision.should_retrain
    assert decision.urgency == "immediate"
    assert decision.triggered_by == "score_distribution_shift"


def test_small_score_shift_no_trigger():
    report = make_report(mean_score=0.21)  # 5% shift — below 10% threshold
    decision = evaluate_triggers(report, DEFAULT_CONFIG, baseline_mean_score=0.20)
    assert not decision.should_retrain


# ---------------------------------------------------------------------------
# Performance degradation
# ---------------------------------------------------------------------------

def test_performance_drop_triggers_immediate():
    report = make_report()
    current = {"accuracy": 0.70, "f1": 0.60, "roc_auc": 0.75}
    baseline = {"accuracy": 0.82, "f1": 0.72, "roc_auc": 0.87}
    decision = evaluate_triggers(report, DEFAULT_CONFIG, current_metrics=current, baseline_metrics=baseline)
    assert decision.should_retrain
    assert decision.urgency == "immediate"
    assert decision.triggered_by == "performance_degradation"


def test_small_performance_drop_no_trigger():
    report = make_report()
    current = {"accuracy": 0.80, "f1": 0.70}
    baseline = {"accuracy": 0.82, "f1": 0.72}
    decision = evaluate_triggers(report, DEFAULT_CONFIG, current_metrics=current, baseline_metrics=baseline)
    assert not decision.should_retrain


# ---------------------------------------------------------------------------
# Time-based trigger
# ---------------------------------------------------------------------------

def test_time_based_triggers_schedule():
    report = make_report()
    decision = evaluate_triggers(report, DEFAULT_CONFIG, days_since_last_retrain=95)
    assert decision.should_retrain
    assert decision.urgency == "schedule"
    assert decision.triggered_by == "time_based"


def test_within_time_limit_no_trigger():
    report = make_report()
    decision = evaluate_triggers(report, DEFAULT_CONFIG, days_since_last_retrain=45)
    assert not decision.should_retrain


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_performance_drop_overrides_time_based():
    """Performance degradation is priority 1; should win over time-based."""
    report = make_report()
    current = {"accuracy": 0.70, "f1": 0.60}
    baseline = {"accuracy": 0.82, "f1": 0.72}
    decision = evaluate_triggers(
        report, DEFAULT_CONFIG,
        days_since_last_retrain=95,
        current_metrics=current,
        baseline_metrics=baseline,
    )
    assert decision.triggered_by == "performance_degradation"
    assert decision.urgency == "immediate"


def test_summary_string_on_trigger():
    report = make_report(psi_by_feature={"feature_x": 0.30})
    decision = evaluate_triggers(report, DEFAULT_CONFIG)
    summary = decision.summary()
    assert "RETRAIN" in summary
    assert "feature_x" in summary


def test_summary_string_no_trigger():
    report = make_report()
    decision = evaluate_triggers(report, DEFAULT_CONFIG)
    assert "No retrain needed" in decision.summary()
