"""End-to-end integration test of the deterministic pipeline.

Runs the demo study from the API:
- create study
- upload synthetic CSV
- attach 3 claims
- launch pipeline → should pause at SAP approval
- approve SAP → relaunch → should produce results + claim decisions
                 and pause at claim wording
- approve claim wording → relaunch → should produce reports + QA audit
- list reports
- QA audit passes
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

EX = Path(__file__).parent.parent / "examples" / "sample_study"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def demo_csv(tmp_path: Path) -> Path:
    """Re-use the bundled demo dataset if present, else regenerate."""
    p = EX / "data" / "measurements_long.csv"
    if p.exists():
        return p
    # Fallback: generate inline (kept very small to keep test fast).
    import numpy as np

    rng = np.random.default_rng(1729)
    rows = []
    for s in range(1, 26):
        sid = f"S{s:03d}"
        base = float(rng.normal(35.0, 6.0))
        for v, drift in [("D0", 0.0), ("D7", 3.0), ("D14", 5.0), ("D28", 7.0)]:
            rows.append({
                "subject_id": sid, "visit": v, "endpoint": "corneometer_hydration",
                "value": round(base + drift + float(rng.normal(0, 1.5)), 3),
            })
        bw = float(rng.normal(0.22, 0.05))
        rows.append({"subject_id": sid, "visit": "D0", "endpoint": "wrinkle_depth", "value": round(bw, 4)})
        rows.append({"subject_id": sid, "visit": "D28", "endpoint": "wrinkle_depth",
                     "value": round(bw + float(rng.normal(-0.07, 0.025)), 4)})
    out = tmp_path / "demo.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _create_demo_study(client: TestClient) -> str:
    meta = json.loads((EX / "study_metadata.json").read_text())
    r = client.post("/api/studies", json=meta)
    assert r.status_code == 201, r.text
    return meta["study_id"]


def test_full_pipeline_with_auto_approval(client: TestClient, demo_csv: Path) -> None:
    study_id = _create_demo_study(client)

    # Upload synthetic dataset
    with demo_csv.open("rb") as fh:
        r = client.post(
            f"/api/studies/{study_id}/data",
            files={"file": (demo_csv.name, fh, "text/csv")},
        )
    assert r.status_code == 201, r.text

    # Attach claims
    claims = json.loads((EX / "claims.json").read_text())
    r = client.post(f"/api/studies/{study_id}/claims", json=claims)
    assert r.status_code == 201, r.text

    # 1st launch → should pause at SAP approval
    r = client.post(f"/api/analyses/{study_id}")
    body = r.json()
    assert r.status_code == 202, r.text
    assert body["status"] == "ok"
    assert body["summary"]["sap_locked"] is False
    assert len(body["summary"]["pending_approvals"]) >= 1

    # Approve every pending request and re-launch up to 5 times
    for _ in range(5):
        pending_resp = client.get(f"/api/approvals?study_id={study_id}&status=pending")
        pending = pending_resp.json()
        if not pending:
            break
        for a in pending:
            r = client.post(
                f"/api/approvals/{a['approval_id']}",
                json={"decision": "approved", "reviewer": "test-auto"},
            )
            assert r.status_code == 200, r.text
        r = client.post(f"/api/analyses/{study_id}")
        assert r.status_code == 202, r.text

    # Final summary — pipeline should have produced reports
    r = client.get(f"/api/reports/{study_id}")
    assert r.status_code == 200
    reports = r.json()
    assert "statistical_analysis_report.md" in reports
    assert "executive_summary.md" in reports

    # Download one report to make sure FileResponse works
    r = client.get(f"/api/reports/{study_id}/executive_summary.md")
    assert r.status_code == 200
    text = r.text
    assert study_id in text


def test_pipeline_refuses_to_run_analyses_without_sap_approval(
    client: TestClient, demo_csv: Path
) -> None:
    """The first launch must pause and the analyses must NOT have been written."""
    study_id = _create_demo_study(client)
    with demo_csv.open("rb") as fh:
        client.post(f"/api/studies/{study_id}/data", files={"file": (demo_csv.name, fh, "text/csv")})
    claims = json.loads((EX / "claims.json").read_text())
    client.post(f"/api/studies/{study_id}/claims", json=claims)

    r = client.post(f"/api/analyses/{study_id}")
    body = r.json()
    assert body["summary"]["sap_locked"] is False

    # statistical_results.json must NOT exist
    from app.core.paths import StudyWorkspace
    ws = StudyWorkspace(study_id)
    assert not (ws.results / "statistical_results.json").exists()


def test_qa_audit_passes_after_full_run(client: TestClient, demo_csv: Path) -> None:
    study_id = _create_demo_study(client)
    with demo_csv.open("rb") as fh:
        client.post(f"/api/studies/{study_id}/data", files={"file": (demo_csv.name, fh, "text/csv")})
    client.post(f"/api/studies/{study_id}/claims", json=json.loads((EX / "claims.json").read_text()))
    client.post(f"/api/analyses/{study_id}")

    for _ in range(5):
        pending = client.get(f"/api/approvals?study_id={study_id}&status=pending").json()
        if not pending:
            break
        for a in pending:
            client.post(
                f"/api/approvals/{a['approval_id']}",
                json={"decision": "approved", "reviewer": "test-auto"},
            )
        client.post(f"/api/analyses/{study_id}")

    from app.core.paths import StudyWorkspace
    ws = StudyWorkspace(study_id)
    audit = json.loads((ws.audit / "qa_audit_report.json").read_text())
    assert audit["passed"] is True, audit["issues"]
