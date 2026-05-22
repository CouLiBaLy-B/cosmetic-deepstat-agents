"""Append-only audit trail (JSONL).

Every tool, every approval, every dataset write goes through this module.
The format is intentionally simple (newline-delimited JSON) so it can be
shipped to a WORM object store or rotated to S3 without code changes.
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from app.core.settings import get_settings

_LOCK = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def hash_bytes(data: bytes, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(data)
    return f"{algo}:{h.hexdigest()}"


def hash_file(path: str | Path, algo: str = "sha256") -> str:
    """Stream-hash a file. Returns ``"<algo>:<hex>"``."""
    p = Path(path)
    h = hashlib.new(algo)
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def write_audit_event(
    *,
    actor: str,
    action: str,
    study_id: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Append a structured event to the audit log and return it.

    The audit log is, by default, a single global JSONL file. When ``study_id``
    is given, the event is **also** appended to the per-study audit file at
    ``workspace/{study_id}/audit/audit_trail.jsonl`` so each study is
    self-contained.
    """
    settings = get_settings()
    event: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "timestamp": _utcnow_iso(),
        "actor": actor,
        "action": action,
        "study_id": study_id,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "metadata": metadata or {},
        "pid": os.getpid(),
    }
    if not settings.enable_audit_trail:
        return event

    targets = []
    targets.append(audit_path or settings.audit_log_path)
    if study_id:
        targets.append(
            settings.workspace_root_abs / study_id / "audit" / "audit_trail.jsonl"
        )

    line = orjson.dumps(event) + b"\n"
    with _LOCK:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("ab") as fh:
                fh.write(line)
    return event
