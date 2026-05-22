"""Human-in-the-loop approval endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from app.core.audit import write_audit_event
from app.schemas.approvals import ApprovalDecision, ApprovalRequest, ApprovalStatus
from app.storage import db

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@router.get("", response_model=list[ApprovalRequest])
def list_approvals(study_id: str | None = None, status: ApprovalStatus | None = None) -> list[ApprovalRequest]:
    items = db.approvals().list()
    if study_id:
        items = [a for a in items if a.study_id == study_id]
    if status:
        items = [a for a in items if a.status == status]
    return items


@router.get("/{approval_id}", response_model=ApprovalRequest)
def get_approval(approval_id: str) -> ApprovalRequest:
    item = db.approvals().get(approval_id)
    if item is None:
        raise HTTPException(404, f"Approval {approval_id!r} not found.")
    return item


@router.post("/{approval_id}", response_model=ApprovalRequest)
def decide(approval_id: str, payload: ApprovalDecision) -> ApprovalRequest:
    item = db.approvals().get(approval_id)
    if item is None:
        raise HTTPException(404, f"Approval {approval_id!r} not found.")
    if item.status != ApprovalStatus.PENDING:
        raise HTTPException(409, f"Approval {approval_id!r} is already {item.status.value}.")

    item.status = payload.decision
    item.reviewer = payload.reviewer
    item.comment = payload.comment
    item.edited_payload = payload.edited_payload
    item.decision_at = datetime.now(UTC)
    db.approvals().upsert(item.approval_id, item)

    write_audit_event(
        actor=f"reviewer:{payload.reviewer}",
        action=f"approval.{payload.decision.value}",
        study_id=item.study_id,
        metadata={
            "approval_id": item.approval_id,
            "object_type": item.object_type,
            "object_id": item.object_id,
            "comment": payload.comment,
        },
    )
    return item
