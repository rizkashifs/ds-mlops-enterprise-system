import pytest

from src.core.lifecycle import ModelStatus, can_transition, transition


def test_valid_forward_promotion():
    assert can_transition(ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE)
    assert can_transition(ModelStatus.CANDIDATE, ModelStatus.APPROVED)
    assert can_transition(ModelStatus.APPROVED, ModelStatus.DEPLOYED)
    assert can_transition(ModelStatus.DEPLOYED, ModelStatus.RETIRED)


def test_valid_rollback():
    assert can_transition(ModelStatus.CANDIDATE, ModelStatus.EXPERIMENTAL)
    assert can_transition(ModelStatus.APPROVED, ModelStatus.EXPERIMENTAL)
    assert can_transition(ModelStatus.DEPLOYED, ModelStatus.CANDIDATE)


def test_invalid_skip_stage():
    assert not can_transition(ModelStatus.EXPERIMENTAL, ModelStatus.APPROVED)
    assert not can_transition(ModelStatus.EXPERIMENTAL, ModelStatus.DEPLOYED)


def test_retired_is_terminal():
    for status in ModelStatus:
        assert not can_transition(ModelStatus.RETIRED, status)


def test_transition_returns_new_status():
    result = transition(ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE)
    assert result == ModelStatus.CANDIDATE


def test_transition_raises_on_invalid():
    with pytest.raises(ValueError, match="Cannot go from"):
        transition(ModelStatus.EXPERIMENTAL, ModelStatus.DEPLOYED)
