"""Audit-trail event schema."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class AuditEvent(BaseModel):
    """A single immutable entry in the audit trail."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    action: str
    study_id: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    pid: int | None = None
