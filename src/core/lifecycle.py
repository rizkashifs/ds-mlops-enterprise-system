"""Model lifecycle: the five stages every model passes through.

EXPERIMENTAL  — active development, not for production use
CANDIDATE     — training complete, ready for validation review
APPROVED      — passed review gates, cleared for deployment
DEPLOYED      — serving predictions in production
RETIRED       — no longer active; archived for audit

Transitions are one-directional except for rollbacks (APPROVED → EXPERIMENTAL,
DEPLOYED → CANDIDATE). Retiring a model is permanent.
"""
from enum import Enum
from typing import Set


class ModelStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    RETIRED = "retired"


# Explicit allowlist keeps accidental promotions from compiling silently.
_ALLOWED_TRANSITIONS: dict[ModelStatus, Set[ModelStatus]] = {
    ModelStatus.EXPERIMENTAL: {ModelStatus.CANDIDATE},
    ModelStatus.CANDIDATE: {ModelStatus.APPROVED, ModelStatus.EXPERIMENTAL},
    ModelStatus.APPROVED: {ModelStatus.DEPLOYED, ModelStatus.EXPERIMENTAL},
    ModelStatus.DEPLOYED: {ModelStatus.RETIRED, ModelStatus.CANDIDATE},
    ModelStatus.RETIRED: set(),
}


def can_transition(from_status: ModelStatus, to_status: ModelStatus) -> bool:
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, set())


def transition(from_status: ModelStatus, to_status: ModelStatus) -> ModelStatus:
    """Advance a model's status. Raises ValueError for invalid moves."""
    if not can_transition(from_status, to_status):
        allowed = {s.value for s in _ALLOWED_TRANSITIONS.get(from_status, set())}
        raise ValueError(
            f"Cannot go from '{from_status}' to '{to_status}'. "
            f"Allowed next states: {allowed or 'none (terminal)'}"
        )
    return to_status
