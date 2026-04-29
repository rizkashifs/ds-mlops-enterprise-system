"""Runtime services: batch scoring and online inference."""
from .scoring import ScoringResult, score_batch

__all__ = ["ScoringResult", "score_batch"]
