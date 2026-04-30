"""Standard monitoring hooks.

Drop-in functions for computing PSI, logging scoring run metrics,
and emitting the standard monitoring payload after each batch score.
These functions enforce the monitoring standards from standards/monitoring.md.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MonitoringReport:
    model_name: str
    scored_at: str
    num_records: int
    mean_score: float
    p10: float
    p50: float
    p90: float
    p99: float
    psi_by_feature: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    def has_alerts(self) -> bool:
        return len(self.alerts) > 0


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """Population Stability Index between two 1-D arrays.

    PSI < 0.10: no significant shift
    PSI 0.10-0.20: moderate shift, investigate
    PSI > 0.20: significant shift, alert
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def build_monitoring_report(
    model_name: str,
    scores: pd.Series,
    features_df: Optional[pd.DataFrame] = None,
    baseline_features: Optional[pd.DataFrame] = None,
    psi_alert_threshold: float = 0.20,
) -> MonitoringReport:
    """Compute the standard monitoring payload after a batch scoring run.

    Args:
        model_name: Name of the model that produced the scores.
        scores: Probability scores from the scoring run.
        features_df: Input features used for scoring (optional, for PSI).
        baseline_features: Training-time feature distributions (optional, for PSI).
        psi_alert_threshold: PSI level above which an alert is raised.

    Returns:
        MonitoringReport with score stats, PSI, and any raised alerts.
    """
    alerts: List[str] = []

    if scores.empty or len(scores) == 0:
        alerts.append("CRITICAL: scoring output is empty — pipeline may have failed")
        report = MonitoringReport(
            model_name=model_name,
            scored_at=datetime.now(timezone.utc).isoformat(),
            num_records=0,
            mean_score=0.0,
            p10=0.0,
            p50=0.0,
            p90=0.0,
            p99=0.0,
            alerts=alerts,
        )
        return report

    report = MonitoringReport(
        model_name=model_name,
        scored_at=datetime.now(timezone.utc).isoformat(),
        num_records=len(scores),
        mean_score=float(scores.mean()),
        p10=float(np.percentile(scores, 10)),
        p50=float(np.percentile(scores, 50)),
        p90=float(np.percentile(scores, 90)),
        p99=float(np.percentile(scores, 99)),
    )

    if features_df is not None and baseline_features is not None:
        shared_cols = [c for c in features_df.columns if c in baseline_features.columns]
        for col in shared_cols:
            psi = compute_psi(
                baseline_features[col].dropna().values,
                features_df[col].dropna().values,
            )
            report.psi_by_feature[col] = round(psi, 4)
            if psi > psi_alert_threshold:
                alerts.append(f"WARNING: PSI={psi:.3f} on feature '{col}' (threshold {psi_alert_threshold})")

    report.alerts = alerts
    return report
