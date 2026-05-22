"""Endpoints to launch and inspect the agentic pipeline.

The launch is **synchronous** in the MVP (the pipeline finishes in a few
seconds on the demo dataset). For production, swap to a background task /
RQ worker — the pipeline function is already idempotent and re-entrant.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status

from app.agents.master_agent import build_master_agent
from app.core.audit import write_audit_event
from app.core.paths import StudyWorkspace
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
    """Detailed pipeline status including step progress and pending gates."""
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    ws = StudyWorkspace(study_id)
    pending = [
        {"approval_id": a.approval_id, "object_type": a.object_type, "reason": a.reason}
        for a in db.approvals().list()
        if a.study_id == study_id and a.status.value == "pending"
    ]
    approved = [
        {"approval_id": a.approval_id, "object_type": a.object_type, "reviewer": a.reviewer}
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
