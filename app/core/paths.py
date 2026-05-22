"""Per-study workspace path helpers.

Every other module that needs to read/write a file inside a study workspace
MUST go through ``StudyWorkspace``. This guarantees:

- the directory layout is consistent (`raw/`, `clean/`, `scripts/`, …),
- no module accidentally writes outside `WORKSPACE_ROOT` (path traversal
  protection),
- the `raw/` directory is read-only after first write (enforced at the
  service layer in ``app.storage.object_store``).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.settings import get_settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")


class InvalidStudyIdError(ValueError):
    """Raised when a study_id does not match the safe identifier pattern."""


class WorkspacePathError(ValueError):
    """Raised when an attempt is made to escape the workspace root."""


SUBDIRS: tuple[str, ...] = (
    "raw",
    "clean",
    "scripts",
    "results",
    "figures",
    "reports",
    "audit",
    "approvals",
)


def validate_study_id(study_id: str) -> str:
    """Validate the study_id is safe to use as a directory name."""
    if not _SAFE_ID.match(study_id):
        raise InvalidStudyIdError(
            f"Invalid study_id {study_id!r}. Must match {_SAFE_ID.pattern}."
        )
    return study_id


class StudyWorkspace:
    """Path manager for a single study's workspace directory."""

    def __init__(self, study_id: str) -> None:
        self.study_id = validate_study_id(study_id)
        self.root: Path = (get_settings().workspace_root_abs / self.study_id).resolve()

    def ensure(self) -> StudyWorkspace:
        """Create the workspace directory tree if it does not exist."""
        for sub in SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def clean(self) -> Path:
        return self.root / "clean"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    @property
    def approvals(self) -> Path:
        return self.root / "approvals"

    def safe_join(self, *parts: str) -> Path:
        """Join a relative path under the study root, refusing any traversal."""
        candidate = (self.root.joinpath(*parts)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspacePathError(
                f"Path {candidate} escapes the study workspace {self.root}."
            ) from exc
        return candidate
