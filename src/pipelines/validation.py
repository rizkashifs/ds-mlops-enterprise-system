"""Validation pipeline: checks whether a trained model meets promotion thresholds.

This is the gate between CANDIDATE and APPROVED. A model that fails validation
stays in CANDIDATE and cannot be deployed. All failures are surfaced explicitly
so data scientists can act on them.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ValidationThresholds:
    """Minimum acceptable values for a model to pass the promotion gate.

    Teams set these thresholds per use case in configs/config.yaml.
    Defaults are conservative starting points, not universal targets.
    """
    min_accuracy: float = 0.70
    min_f1: float = 0.60
    min_roc_auc: float = 0.70


@dataclass
class ValidationResult:
    passed: bool
    failures: List[str]
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation: {status}"]
        for f in self.failures:
            lines.append(f"  FAIL  {f}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        return "\n".join(lines)


def validate_model(
    metrics: Dict[str, float],
    thresholds: ValidationThresholds,
) -> ValidationResult:
    """Compare model metrics against thresholds. Returns a ValidationResult.

    Checks accuracy, F1, and ROC-AUC. Any missing metric is a hard failure —
    pipelines must log all three for a result to be promotable.
    """
    failures: List[str] = []
    warnings: List[str] = []

    required_checks = [
        ("accuracy", thresholds.min_accuracy),
        ("f1", thresholds.min_f1),
        ("roc_auc", thresholds.min_roc_auc),
    ]

    for metric_name, threshold in required_checks:
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"Required metric not logged: {metric_name}")
        elif value < threshold:
            failures.append(
                f"{metric_name} = {value:.4f}  (threshold: {threshold:.4f})"
            )
        elif value < threshold + 0.05:
            warnings.append(
                f"{metric_name} = {value:.4f} is within 5pp of threshold {threshold:.4f}"
            )

    return ValidationResult(
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        metrics=metrics,
    )
