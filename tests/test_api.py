"""End-to-end tests on the FastAPI app using TestClient."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_create_study_and_upload_and_attach_claims(client: TestClient) -> None:
    payload = {
        "study_id": "STUDY_E2E_001",
        "product_id": "CREAM_001",
        "title": "Hydration study",
        "design_type": "before_after_longitudinal",
        "population": "women 40-60 dry skin",
        "visits": ["D0", "D7", "D14", "D28"],
        "endpoints": [
            {
                "name": "corneometer_hydration",
                "data_type": "continuous",
                "unit": "a.u.",
                "timepoints": ["D0", "D7", "D14", "D28"],
                "primary_or_secondary": "primary",
                "practical_threshold": 5.0,
                "multiplicity_family": "hydration",
            }
        ],
        "jurisdiction": "EU",
    }
    r = client.post("/api/studies", json=payload)
    assert r.status_code == 201, r.text
    assert r.json()["study_id"] == "STUDY_E2E_001"

    # Duplicate is rejected.
    r = client.post("/api/studies", json=payload)
    assert r.status_code == 409

    # Get it back.
    r = client.get("/api/studies/STUDY_E2E_001")
    assert r.status_code == 200
    assert r.json()["status"] == "draft"

    # Upload a raw CSV file.
    files = {"file": ("subjects.csv", io.BytesIO(b"subject_id,visit,value\n1,D0,42\n"), "text/csv")}
    r = client.post("/api/studies/STUDY_E2E_001/data", files=files)
    assert r.status_code == 201, r.text
    assert r.json()["path"] == "raw/subjects.csv"
    assert r.json()["sha256"].startswith("sha256:")

    # Re-upload same file is refused (raw is immutable).
    r = client.post("/api/studies/STUDY_E2E_001/data", files=files)
    assert r.status_code == 409

    # Attach claims.
    claims = [
        {"claim_id": "C001", "text": "Hydrate pendant 24h", "claim_type": "instrumental"},
        {"claim_id": "C002", "text": "Bien toléré", "claim_type": "safety"},
    ]
    r = client.post("/api/studies/STUDY_E2E_001/claims", json=claims)
    assert r.status_code == 201, r.text
    assert {c["claim_id"] for c in r.json()} == {"C001", "C002"}

    r = client.get("/api/studies/STUDY_E2E_001/claims")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_analysis_launch_returns_422_when_no_data(client: TestClient) -> None:
    """Without uploaded data the pipeline raises ValueError → API returns 422."""
    payload = {
        "study_id": "STUDY_E2E_002",
        "product_id": "CREAM_002",
        "title": "tiny study",
        "design_type": "before_after",
        "population": "subjects",
    }
    client.post("/api/studies", json=payload)

    r = client.post("/api/analyses/STUDY_E2E_002")
    assert r.status_code == 422, r.text

    r = client.get("/api/analyses/STUDY_E2E_002/status")
    assert r.status_code == 200
    body = r.json()
    assert body["study_id"] == "STUDY_E2E_002"
    assert body["pending_approvals"] == []


def test_reports_listing_when_empty(client: TestClient) -> None:
    payload = {
        "study_id": "STUDY_E2E_003",
        "product_id": "P3",
        "title": "tiny",
        "design_type": "before_after",
        "population": "subjects",
    }
    client.post("/api/studies", json=payload)
    r = client.get("/api/reports/STUDY_E2E_003")
    assert r.status_code == 200
    assert r.json() == []


def test_404_paths(client: TestClient) -> None:
    assert client.get("/api/studies/UNKNOWN").status_code == 404
    assert client.get("/api/analyses/UNKNOWN/status").status_code == 404
    assert client.get("/api/reports/UNKNOWN").status_code == 404
