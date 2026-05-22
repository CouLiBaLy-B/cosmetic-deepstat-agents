"""Unit tests for app.core.settings and app.core.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.paths import (
    SUBDIRS,
    InvalidStudyIdError,
    StudyWorkspace,
    WorkspacePathError,
    validate_study_id,
)
from app.core.settings import get_settings


def test_settings_defaults_mock_provider() -> None:
    s = get_settings()
    assert s.llm_provider == "mock"
    assert s.workspace_root_abs.exists()


def test_validate_study_id_accepts_valid_ids() -> None:
    assert validate_study_id("STUDY_001") == "STUDY_001"
    assert validate_study_id("study-abc-123") == "study-abc-123"


@pytest.mark.parametrize("bad", ["", "ab", "x" * 65, "study/../etc", "with space", "héllo"])
def test_validate_study_id_rejects_bad(bad: str) -> None:
    with pytest.raises(InvalidStudyIdError):
        validate_study_id(bad)


def test_study_workspace_ensure_creates_all_subdirs() -> None:
    ws = StudyWorkspace("STUDY_TEST_001").ensure()
    for sub in SUBDIRS:
        assert (ws.root / sub).is_dir(), f"missing {sub}"


def test_safe_join_blocks_path_traversal() -> None:
    ws = StudyWorkspace("STUDY_TEST_002").ensure()
    # Legitimate file path inside the workspace is fine.
    p = ws.safe_join("raw", "subjects.csv")
    assert isinstance(p, Path)
    assert ws.root in p.parents

    # Any attempt to escape must be refused.
    with pytest.raises(WorkspacePathError):
        ws.safe_join("..", "evil.txt")
    with pytest.raises(WorkspacePathError):
        ws.safe_join("raw", "..", "..", "secrets")
