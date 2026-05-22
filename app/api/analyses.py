"""Endpoints to launch and inspect the agentic pipeline.

The launch is **synchronous** in the MVP (the pipeline finishes in a few
seconds on the demo dataset). For production, swap to a background task /
RQ worker — the pipeline function is already idempotent and re-entrant.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.agents.master_agent import build_master_agent
from app.core.audit import write_audit_event
from app.storage import db

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@router.post("/{study_id}", status_code=status.HTTP_202_ACCEPTED)
def launch_analysis(study_id: str) -> dict[str, object]:
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    write_audit_event(actor="api", action="analysis.launch", study_id=study_id)

    agent = build_master_agent()
    try:
        summary = agent.invoke({"study_id": study_id})
    except PermissionError as exc:
        # SAP not yet approved — this is an expected, recoverable state.
        return {
            "study_id": study_id,
            "status": "paused",
            "reason": str(exc),
            "pending_approvals": [
                a.approval_id
                for a in db.approvals().list()
                if a.study_id == study_id and a.status.value == "pending"
            ],
        }
    except Exception as exc:  # surface as 422 so the caller can retry
        raise HTTPException(422, f"Pipeline failed: {exc}") from exc

    return {"study_id": study_id, "status": "ok", "summary": summary}


@router.get("/{study_id}/status")
def analysis_status(study_id: str) -> dict[str, object]:
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    pending = [a for a in db.approvals().list() if a.study_id == study_id and a.status.value == "pending"]
    return {
        "study_id": study_id,
        "status": study.status.value,
        "pending_approvals": [a.approval_id for a in pending],
    }
