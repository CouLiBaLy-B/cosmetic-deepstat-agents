"""Endpoints to download generated reports."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.paths import StudyWorkspace, WorkspacePathError
from app.storage import db

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{study_id}/{report_name}")
def download_report(study_id: str, report_name: str) -> FileResponse:
    if db.studies().get(study_id) is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    ws = StudyWorkspace(study_id)
    try:
        path = ws.safe_join("reports", report_name)
    except WorkspacePathError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, f"Report {report_name!r} not found for study {study_id!r}.")
    return FileResponse(path, filename=report_name)


@router.get("/{study_id}")
def list_reports(study_id: str) -> list[str]:
    if db.studies().get(study_id) is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    ws = StudyWorkspace(study_id)
    if not ws.reports.exists():
        return []
    return sorted(p.name for p in ws.reports.iterdir() if p.is_file())
