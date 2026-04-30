import os
import pytest

from src.core.secrets import get, require, require_all


def test_require_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "hunter2")
    assert require("MY_SECRET") == "hunter2"


def test_require_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    with pytest.raises(EnvironmentError, match="MISSING_SECRET"):
        require("MISSING_SECRET")


def test_require_raises_when_empty_string(monkeypatch):
    monkeypatch.setenv("EMPTY_SECRET", "")
    with pytest.raises(EnvironmentError, match="EMPTY_SECRET"):
        require("EMPTY_SECRET")


def test_get_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("OPT_VAR", "value")
    assert get("OPT_VAR") == "value"


def test_get_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert get("MISSING_VAR", default="fallback") == "fallback"
    assert get("MISSING_VAR") is None


def test_require_all_returns_dict_when_all_set(monkeypatch):
    monkeypatch.setenv("A", "1")
    monkeypatch.setenv("B", "2")
    result = require_all("A", "B")
    assert result == {"A": "1", "B": "2"}


def test_require_all_raises_listing_all_missing(monkeypatch):
    monkeypatch.delenv("X", raising=False)
    monkeypatch.delenv("Y", raising=False)
    with pytest.raises(EnvironmentError, match="X") as exc:
        require_all("X", "Y")
    assert "Y" in str(exc.value)
