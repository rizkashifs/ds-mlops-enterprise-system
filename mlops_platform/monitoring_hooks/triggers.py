"""Retraining trigger evaluation.

This module answers one question after every monitoring run:
"Given what we observed, should we retrain the model?"

There are four trigger types, evaluated in priority order:
  1. Performance degradation  — model quality dropped (requires labels)
  2. Feature drift (PSI)      — input distribution shifted
  3. Score distribution shift — output scores shifted
  4. Time-based              — safety net; max days since last retrain

Usage:
    from mlops_platform.monitoring_hooks.triggers import TriggerConfig, evaluate_triggers

    config = TriggerConfig(psi_alert_threshold=0.20, max_days_since_retrain=90)
    decision = evaluate_triggers(
        monitoring_report=report,
        config=config,
        days_since_last_retrain=45,
    )

    if decision.should_retrain:
        print(f"[{decision.urgency.upper()}] Retrain triggered by: {decision.triggered_by}")
        for r in decision.reasons:
            print(f"  - {r}")
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlops_platform.monitoring_hooks.hooks import MonitoringReport


@dataclass
class TriggerConfig:
    """Thresholds for each retraining trigger type.

    Set these per use case in configs/training.yaml under a `retraining` key.
    Defaults are conservative starting points — adjust based on how quickly
    your use case's data distribution changes.
    """
    # Feature drift (PSI)
    psi_warn_threshold: float = 0.10   # schedule investigation
    psi_alert_threshold: float = 0.20  # trigger retrain

    # Score distribution shift — max relative change in mean score before triggering
    score_shift_threshold: float = 0.10  # 10% relative shift from baseline

    # Performance degradation (absolute pp drop from validation baseline)
    performance_drop_threshold: float = 0.05  # 5pp drop in accuracy or F1

    # Time-based safety net — retrain at minimum every N days regardless of drift
    max_days_since_retrain: int = 90

    # Minimum records needed before drift checks are trusted
    min_records_for_drift_check: int = 500


@dataclass
class TriggerDecision:
    """Result of evaluating all retraining triggers against a monitoring report."""
    should_retrain: bool
    urgency: str                          # "immediate" | "schedule" | "none"
    triggered_by: Optional[str]          # primary trigger category
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.should_retrain:
            lines = ["No retrain needed."]
        else:
            lines = [f"RETRAIN [{self.urgency.upper()}] — triggered by: {self.triggered_by}"]
        for r in self.reasons:
            lines.append(f"  TRIGGER  {r}")
        for w in self.warnings:
            lines.append(f"  WARN     {w}")
        return "\n".join(lines)


def evaluate_triggers(
    monitoring_report: MonitoringReport,
    config: TriggerConfig,
    days_since_last_retrain: Optional[int] = None,
    baseline_mean_score: Optional[float] = None,
    current_metrics: Optional[Dict[str, float]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> TriggerDecision:
    """Evaluate all retraining triggers against a MonitoringReport.

    Triggers are checked in priority order. The highest-priority trigger that fires
    sets the urgency level. All fired triggers are included in the reasons list.

    Priority:
      1. Performance degradation (immediate) — requires current_metrics + baseline_metrics
      2. Feature drift PSI > alert threshold (immediate)
      3. Score distribution shift (immediate)
      4. Feature drift PSI > warn threshold (schedule)
      5. Time-based max days (schedule)

    Args:
        monitoring_report: Output of build_monitoring_report() from this scoring run.
        config: Trigger thresholds (TriggerConfig).
        days_since_last_retrain: How many days ago the last retrain completed. None = unknown.
        baseline_mean_score: Mean score from training/reference window. None = skip check.
        current_metrics: Current windowed model metrics (accuracy, f1, roc_auc).
        baseline_metrics: Metrics from the original validation run to compare against.

    Returns:
        TriggerDecision with should_retrain, urgency, triggered_by, and reasons.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    urgency = "none"
    triggered_by: Optional[str] = None

    enough_records = monitoring_report.num_records >= config.min_records_for_drift_check

    # --- Priority 1: Performance degradation (requires ground truth labels) ---
    if current_metrics and baseline_metrics:
        for metric in ("accuracy", "f1", "roc_auc"):
            current = current_metrics.get(metric)
            baseline = baseline_metrics.get(metric)
            if current is not None and baseline is not None:
                drop = baseline - current
                if drop >= config.performance_drop_threshold:
                    reasons.append(
                        f"performance: {metric} dropped {drop:.3f}pp "
                        f"(current={current:.4f}, baseline={baseline:.4f})"
                    )
                    if urgency != "immediate":
                        urgency = "immediate"
                        triggered_by = "performance_degradation"

    # --- Priority 2: Feature drift — PSI above alert threshold ---
    if enough_records and monitoring_report.psi_by_feature:
        for feature, psi in monitoring_report.psi_by_feature.items():
            if psi >= config.psi_alert_threshold:
                reasons.append(
                    f"feature drift: PSI={psi:.3f} on '{feature}' "
                    f"(threshold={config.psi_alert_threshold})"
                )
                if urgency != "immediate":
                    urgency = "immediate"
                    triggered_by = triggered_by or "feature_drift"

    # --- Priority 3: Score distribution shift ---
    if baseline_mean_score is not None and baseline_mean_score > 0:
        relative_shift = abs(monitoring_report.mean_score - baseline_mean_score) / baseline_mean_score
        if relative_shift >= config.score_shift_threshold:
            reasons.append(
                f"score shift: mean score moved {relative_shift:.1%} "
                f"(current={monitoring_report.mean_score:.4f}, "
                f"baseline={baseline_mean_score:.4f})"
            )
            if urgency != "immediate":
                urgency = "immediate"
                triggered_by = triggered_by or "score_distribution_shift"

    # --- Priority 4: Feature drift — PSI above warn threshold (schedule) ---
    if enough_records and monitoring_report.psi_by_feature:
        for feature, psi in monitoring_report.psi_by_feature.items():
            if config.psi_warn_threshold <= psi < config.psi_alert_threshold:
                warnings.append(
                    f"moderate feature drift: PSI={psi:.3f} on '{feature}' "
                    f"(investigate; alert at {config.psi_alert_threshold})"
                )
                if urgency == "none":
                    urgency = "schedule"
                    triggered_by = triggered_by or "feature_drift_moderate"

    # --- Priority 5: Time-based safety net ---
    if days_since_last_retrain is not None:
        if days_since_last_retrain >= config.max_days_since_retrain:
            warnings.append(
                f"time-based: {days_since_last_retrain} days since last retrain "
                f"(max={config.max_days_since_retrain})"
            )
            if urgency == "none":
                urgency = "schedule"
                triggered_by = triggered_by or "time_based"

    # --- Not enough records to trust drift checks ---
    if not enough_records and monitoring_report.num_records > 0:
        warnings.append(
            f"drift checks skipped: only {monitoring_report.num_records} records "
            f"(min={config.min_records_for_drift_check} required)"
        )

    should_retrain = urgency in ("immediate", "schedule")

    return TriggerDecision(
        should_retrain=should_retrain,
        urgency=urgency,
        triggered_by=triggered_by,
        reasons=reasons,
        warnings=warnings,
    )
