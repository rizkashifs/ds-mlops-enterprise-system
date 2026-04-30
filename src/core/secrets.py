"""Secrets and credentials loader.

Reads secrets from environment variables — never from files committed to the
repository. Raises clear errors when required secrets are missing so failures
are obvious at startup, not buried in runtime exceptions.

Usage:
    from src.core.secrets import require, get

    # Hard-fail at startup if missing:
    db_password = require("DB_PASSWORD")
    mlflow_token = require("MLFLOW_TRACKING_TOKEN")

    # Optional with fallback:
    log_level = get("LOG_LEVEL", default="INFO")

Local development: set variables in a .env file and load it before starting
(e.g. `export $(cat .env | xargs)` or use python-dotenv). Never commit .env.
"""
import os
from typing import Optional


def require(name: str) -> str:
    """Return the value of an environment variable. Raises if missing or empty."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required secret '{name}' is not set. "
            f"Set it as an environment variable before starting the application. "
            f"Do not hardcode it in source code."
        )
    return value


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the value of an environment variable, or default if not set."""
    return os.environ.get(name, default)


def require_all(*names: str) -> dict:
    """Return a dict of all named secrets. Raises if any are missing."""
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        raise EnvironmentError(
            f"Required secrets not set: {', '.join(missing)}. "
            f"Set them as environment variables before starting the application."
        )
    return {name: os.environ[name] for name in names}
