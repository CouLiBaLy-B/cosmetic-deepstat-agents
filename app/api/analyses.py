"""Endpoints to launch, resume and inspect the agentic pipeline.

The launch is **synchronous** in the MVP (the pipeline finishes in a few
seconds on the demo dataset). For production, swap to a background task /
RQ worker — the pipeline function is already idempotent and re-entrant.

C3 audit fix: added ``POST /api/analyses/{study_id}/resume`` for HITL
continuation in deepagents mode via ``Command(resume=...)``.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.master_agent import build_master_agent
from app.core.audit import write_audit_event
from app.core.paths import StudyWorkspace
from app.storage import db

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


# ---------------------------------------------------------------------------
# POST /api/analyses/{study_id}  — launch or re-launch the pipeline
# ---------------------------------------------------------------------------


@router.post("/{study_id}", status_code=status.HTTP_202_ACCEPTED)
def launch_analysis(study_id: str) -> dict[str, object]:
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    write_audit_event(actor="api", action="analysis.launch", study_id=study_id)

    agent = build_master_agent()
    try:
        summary = agent.invoke({"study_id": study_id}, thread_id=study_id)
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


# ---------------------------------------------------------------------------
# POST /api/analyses/{study_id}/resume  — resume after HITL pause (C3 fix)
# ---------------------------------------------------------------------------


class ResumeRequest(BaseModel):
    """Payload for resuming a paused pipeline after HITL interrupt."""

    decisions: list[dict[str, object]] = Field(
        default_factory=lambda: [{"type": "approve"}],  # type: ignore[arg-type]
        description=(
            "List of decisions, one per interrupted tool call, in order. "
            "Each dict: {\"type\": \"approve\"} | {\"type\": \"reject\"} | "
            "{\"type\": \"edit\", \"args\": {...}}."
        ),
    )


@router.post("/{study_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume_analysis(study_id: str, body: ResumeRequest | None = None) -> dict[str, object]:
    """Resume the pipeline after a HITL interrupt.

    In ``mock`` mode this is equivalent to re-launching the pipeline
    (approvals are persisted via ``POST /api/approvals/{id}``).

    In ``deepagents`` mode this calls ``Command(resume={"decisions": ...})``
    on the graph with the same ``thread_id``.
    """
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    decisions = (body.decisions if body else None) or [{"type": "approve"}]

    write_audit_event(
        actor="api",
        action="analysis.resume",
        study_id=study_id,
        metadata={"n_decisions": len(decisions)},
    )

    agent = build_master_agent()
    try:
        result = agent.resume(study_id, decisions=decisions)
    except Exception as exc:
        raise HTTPException(422, f"Resume failed: {exc}") from exc

    return {"study_id": study_id, "status": "ok", "result": result}


# ---------------------------------------------------------------------------
# GET /api/analyses/{study_id}/status  — detailed pipeline status
# ---------------------------------------------------------------------------


@router.get("/{study_id}/status")
def analysis_status(study_id: str) -> dict[str, object]:
    """Detailed pipeline status including step progress and pending gates."""
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    ws = StudyWorkspace(study_id)
    pending = [
        {
            "approval_id": a.approval_id,
            "object_type": a.object_type,
            "reason": a.reason,
        }
        for a in db.approvals().list()
        if a.study_id == study_id and a.status.value == "pending"
    ]
    approved = [
        {
            "approval_id": a.approval_id,
            "object_type": a.object_type,
            "reviewer": a.reviewer,
        }
        for a in db.approvals().list()
        if a.study_id == study_id and a.status.value == "approved"
    ]

    # Check which artefacts exist
    artefacts: dict[str, bool] = {
        "claim_evidence_map": (ws.results / "claim_evidence_map.json").exists(),
        "qc_report": (ws.results / "qc_report.json").exists(),
        "sap_draft": (ws.results / "sap_draft.json").exists(),
        "statistical_results": (ws.results / "statistical_results.json").exists(),
        "claim_decisions": (ws.results / "claim_decisions.json").exists(),
        "safety_report": (ws.results / "safety_report.json").exists(),
        "reports": (ws.reports / "statistical_analysis_report.md").exists(),
        "qa_audit": (ws.audit / "qa_audit_report.json").exists(),
    }

    # QA pass/fail
    qa_passed: bool | None = None
    qa_path = ws.audit / "qa_audit_report.json"
    if qa_path.exists():
        qa_data = json.loads(qa_path.read_text())
        qa_passed = qa_data.get("passed")

    return {
        "study_id": study_id,
        "status": study.status.value,
        "artefacts": artefacts,
        "pending_approvals": pending,
        "approved": approved,
        "qa_passed": qa_passed,
    }
