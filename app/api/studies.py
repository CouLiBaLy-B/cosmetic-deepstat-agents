"""Study CRUD + raw-data upload endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.audit import hash_file, write_audit_event
from app.core.paths import StudyWorkspace
from app.schemas.claims import Claim
from app.schemas.study import Study, StudyCreate, StudyStatus
from app.storage import db

router = APIRouter(prefix="/api/studies", tags=["studies"])


@router.post("", response_model=Study, status_code=status.HTTP_201_CREATED)
def create_study(payload: StudyCreate) -> Study:
    """Create a study and provision its workspace directory tree."""
    repo = db.studies()
    if repo.get(payload.study_id) is not None:
        raise HTTPException(409, f"Study {payload.study_id!r} already exists.")

    ws = StudyWorkspace(payload.study_id).ensure()
    study = Study(**payload.model_dump())
    repo.upsert(study.study_id, study)

    write_audit_event(
        actor="api",
        action="study.create",
        study_id=study.study_id,
        metadata={"workspace": str(ws.root)},
    )
    return study


@router.get("/{study_id}", response_model=Study)
def get_study(study_id: str) -> Study:
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    return study


@router.get("", response_model=list[Study])
def list_studies() -> list[Study]:
    return db.studies().list()


@router.post("/{study_id}/data", status_code=status.HTTP_201_CREATED)
def upload_data(study_id: str, file: UploadFile) -> dict[str, object]:
    """Upload a raw dataset file. Raw files are never modified afterwards."""
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    if not file.filename:
        raise HTTPException(400, "Missing file name.")

    ws = StudyWorkspace(study_id).ensure()
    target: Path = ws.safe_join("raw", file.filename)

    if target.exists():
        raise HTTPException(
            409,
            f"Raw file {file.filename!r} already exists. Raw data is immutable; "
            "delete it manually with a documented audit reason if required.",
        )

    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    file_hash = hash_file(target)
    if file.filename not in study.data_paths:
        study.data_paths.append(file.filename)
        study.status = StudyStatus.DATA_UPLOADED
        db.studies().upsert(study.study_id, study)

    write_audit_event(
        actor="api",
        action="study.upload_data",
        study_id=study.study_id,
        output_hash=file_hash,
        metadata={"path": f"raw/{file.filename}", "size": target.stat().st_size},
    )
    return {"path": f"raw/{file.filename}", "sha256": file_hash, "size": target.stat().st_size}


@router.post("/{study_id}/claims", response_model=list[Claim], status_code=status.HTTP_201_CREATED)
def attach_claims(study_id: str, claims_payload: list[Claim]) -> list[Claim]:
    """Attach a list of marketing claims to the study."""
    study = db.studies().get(study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")

    attached: list[Claim] = []
    for claim in claims_payload:
        claim.study_id = study_id
        claim.product_id = claim.product_id or study.product_id
        db.claims().upsert(claim.claim_id, claim)
        attached.append(claim)

    write_audit_event(
        actor="api",
        action="study.attach_claims",
        study_id=study_id,
        metadata={"claim_ids": [c.claim_id for c in attached]},
    )
    return attached


@router.get("/{study_id}/claims", response_model=list[Claim])
def list_claims(study_id: str) -> list[Claim]:
    if db.studies().get(study_id) is None:
        raise HTTPException(404, f"Study {study_id!r} not found.")
    return [c for c in db.claims().list() if c.study_id == study_id]
