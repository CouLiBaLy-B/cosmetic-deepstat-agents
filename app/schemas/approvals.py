"""Schemas for human-in-the-loop approval requests."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecision(BaseModel):
    """Payload accepted by ``POST /api/approvals/{id}``."""

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalStatus
    reviewer: str = Field(..., min_length=1, max_length=120)
    comment: str | None = None
    edited_payload: dict[str, object] | None = None


class ApprovalRequest(BaseModel):
    """An item awaiting a human review/approval/edit/rejection."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(..., min_length=1)
    study_id: str
    object_type: str = Field(
        ...,
        description='What is being approved (e.g. "sap", "primary_endpoint", "claim_wording").',
    )
    object_id: str = Field(..., description="Identifier of the object under review.")
    reason: str
    payload: dict[str, object] = Field(
        default_factory=dict, description="Snapshot of the object being approved."
    )
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: str | None = None
    decision_at: datetime | None = None
    comment: str | None = None
    edited_payload: dict[str, object] | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
