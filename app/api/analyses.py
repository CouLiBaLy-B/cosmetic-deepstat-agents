"""Endpoints to launch and inspect the agentic pipeline.

In the MVP the launch is **synchronous** but stubbed: it returns an
"accepted" envelope. The actual pipeline orchestration is implemented in
``app/services/pipeline.py`` (next phase) and will be invoked from here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.audit import write_audit_event
from app.storage import db

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@router.post("/{study_id}", status_code=status.HTTP_202_ACCEPTED)
def launch_analysis(study_id: str) -> dict[str, object]:
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    write_audit_event(actor="api", action="analysis.launch", study_id=study_id)
    return {
        "study_id": study_id,
        "accepted": True,
        "message": (
            "Pipeline launch accepted. Wire app/services/pipeline.py to actually run the "
            "DeepAgents orchestration. See docs/architecture.md §4 for the expected flow."
        ),
    }


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
