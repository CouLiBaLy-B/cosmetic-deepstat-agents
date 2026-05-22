"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect every workspace / audit path into a per-test temporary directory.

    We also clear the ``get_settings`` LRU cache so each test sees a fresh
    Settings object using the new env vars.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        monkeypatch.setenv("WORKSPACE_ROOT", str(root / "workspace"))
        monkeypatch.setenv("MEMORY_ROOT", str(root / "memories"))
        monkeypatch.setenv("AUDIT_LOG_PATH", str(root / "audit.jsonl"))
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{root / 'cdsa.db'}")
        monkeypatch.setenv("APP_ENV", "dev")
        monkeypatch.setenv("LLM_PROVIDER", "mock")

        # Reset the cached singleton.
        from app.core import settings as settings_mod
        settings_mod.get_settings.cache_clear()

        # Reset in-memory repositories.
        from app.storage import db as db_mod
        db_mod.studies().clear()
        db_mod.claims().clear()
        db_mod.approvals().clear()

        yield root
