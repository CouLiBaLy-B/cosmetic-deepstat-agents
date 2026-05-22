"""Tests for the append-only audit trail."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.audit import hash_file, write_audit_event
from app.core.paths import StudyWorkspace
from app.core.settings import get_settings


def test_write_audit_event_writes_global_and_per_study(tmp_path: Path) -> None:
    StudyWorkspace("STUDY_AUDIT_001").ensure()
    evt = write_audit_event(
        actor="test",
        action="unit.test",
        study_id="STUDY_AUDIT_001",
        metadata={"k": "v"},
    )
    settings = get_settings()
    global_log = settings.audit_log_path
    per_study_log = (
        settings.workspace_root_abs / "STUDY_AUDIT_001" / "audit" / "audit_trail.jsonl"
    )

    assert global_log.exists()
    assert per_study_log.exists()

    with global_log.open() as fh:
        lines = [json.loads(line) for line in fh]
    assert any(line["event_id"] == evt["event_id"] for line in lines)

    with per_study_log.open() as fh:
        per_study_lines = [json.loads(line) for line in fh]
    assert per_study_lines[-1]["action"] == "unit.test"


def test_hash_file(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    h = hash_file(f)
    assert h == "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
