"""Phase 6+7 integration tests.

Tests cover:
- Full pipeline with auto-model selection (MMRM for longitudinal)
- Safety analysis step (with and without safety claims)
- HITL gates: SAP, claim_wording, safety_conclusion, final_report
- QA audit with hash-integrity and no-raw-IDs checks
- Idempotence: re-running the pipeline does not duplicate approvals
- Pipeline status endpoint returns detailed artefact/approval info
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
    p = EX / "data" / "measurements_long.csv"
    if p.exists():
        return p
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


def _create_demo_study(client: TestClient, suffix: str = "") -> str:
    meta = json.loads((EX / "study_metadata.json").read_text())
    meta["study_id"] = meta["study_id"] + suffix
    r = client.post("/api/studies", json=meta)
    if r.status_code == 409:
        pass
    else:
        assert r.status_code == 201, r.text
    return meta["study_id"]


def _setup_study(client: TestClient, demo_csv: Path, suffix: str = "") -> str:
    study_id = _create_demo_study(client, suffix)
    with demo_csv.open("rb") as fh:
        r = client.post(
            f"/api/studies/{study_id}/data",
            files={"file": (demo_csv.name, fh, "text/csv")},
        )
    assert r.status_code in (201, 409), r.text
    claims = json.loads((EX / "claims.json").read_text())
    r = client.post(f"/api/studies/{study_id}/claims", json=claims)
    assert r.status_code == 201, r.text
    return study_id


def _approve_all(client: TestClient, study_id: str) -> int:
    """Approve all pending approvals; return count approved."""
    count = 0
    pending = client.get(f"/api/approvals?study_id={study_id}&status=pending").json()
    for a in pending:
        r = client.post(
            f"/api/approvals/{a['approval_id']}",
            json={"decision": "approved", "reviewer": "test-auto"},
        )
        assert r.status_code == 200, r.text
        count += 1
    return count


class TestFullPipelineWithModelSelection:
    """Verify that the pipeline selects the right model per endpoint."""

    def test_pipeline_uses_mmrm_for_longitudinal(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_01")

        # First launch → pauses at SAP
        r = client.post(f"/api/analyses/{study_id}")
        body = r.json()
        assert r.status_code == 202
        assert body["summary"]["sap_locked"] is False

        # Approve SAP and relaunch
        _approve_all(client, study_id)
        r = client.post(f"/api/analyses/{study_id}")
        body = r.json()
        assert body["summary"]["sap_locked"] is True

        # Check statistical_results.json
        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)
        sres = json.loads((ws.results / "statistical_results.json").read_text())

        # corneometer_hydration has 4 timepoints → should use MMRM
        hyd_res = next(r for r in sres if r.get("endpoint") == "corneometer_hydration")
        assert "MMRM" in hyd_res["model"], f"Expected MMRM, got {hyd_res['model']}"

        # wrinkle_depth has 2 timepoints → should use paired test
        wr_res = next(r for r in sres if r.get("endpoint") == "wrinkle_depth")
        assert wr_res["model"] in {"paired_t", "wilcoxon_signed_rank"}


class TestSafetyAnalysis:
    """Verify the safety analysis step."""

    def test_safety_step_creates_report(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_02")

        # Run full pipeline with auto-approve
        for _ in range(5):
            r = client.post(f"/api/analyses/{study_id}")
            assert r.status_code == 202
            n = _approve_all(client, study_id)
            if n == 0:
                break

        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)
        assert (ws.results / "safety_report.json").exists()
        safety = json.loads((ws.results / "safety_report.json").read_text())
        assert "summary" in safety
        # The demo has a safety claim (C003) → should trigger safety review
        assert len(safety.get("safety_claims", [])) >= 1


class TestHITLGates:
    """Verify HITL gates block correctly."""

    def test_sap_gate_blocks_analysis(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_03")
        r = client.post(f"/api/analyses/{study_id}")
        body = r.json()
        assert body["summary"]["sap_locked"] is False
        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)
        assert not (ws.results / "statistical_results.json").exists()

    def test_all_four_hitl_gates_fire(self, client: TestClient, demo_csv: Path) -> None:
        """SAP + claim_wording + safety_conclusion + final_report gates."""
        study_id = _setup_study(client, demo_csv, "_P67_04")

        # Collect all approval types created during the pipeline
        approval_types_seen: set[str] = set()

        for _ in range(6):
            r = client.post(f"/api/analyses/{study_id}")
            assert r.status_code == 202
            pending = client.get(f"/api/approvals?study_id={study_id}&status=pending").json()
            for a in pending:
                approval_types_seen.add(a["object_type"])
                client.post(
                    f"/api/approvals/{a['approval_id']}",
                    json={"decision": "approved", "reviewer": "test-auto"},
                )
            if not pending:
                break

        # All four gates should have been triggered
        assert "sap" in approval_types_seen
        assert "claim_wording" in approval_types_seen
        assert "final_report" in approval_types_seen
        # safety_conclusion only fires if there's a safety claim
        assert "safety_conclusion" in approval_types_seen


class TestQAAudit:
    """Verify QA audit checks."""

    def test_qa_audit_passes_with_all_artefacts(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_05")

        for _ in range(6):
            r = client.post(f"/api/analyses/{study_id}")
            assert r.status_code == 202
            n = _approve_all(client, study_id)
            if n == 0:
                break

        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)
        qa = json.loads((ws.audit / "qa_audit_report.json").read_text())
        assert qa["passed"] is True, f"QA failed: {qa['issues']}"
        assert qa["n_checks"] >= 10  # at least 10 checks performed
        # Hash integrity check should have passed
        assert qa["checks"].get("hash_integrity:analysis_dataset") is True

    def test_qa_detects_no_raw_ids_in_reports(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_06")
        for _ in range(6):
            client.post(f"/api/analyses/{study_id}")
            if _approve_all(client, study_id) == 0:
                break

        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)
        qa = json.loads((ws.audit / "qa_audit_report.json").read_text())
        # No raw IDs should appear in reports
        for k, v in qa["checks"].items():
            if k.startswith("no_raw_ids:"):
                assert v is True, f"Raw IDs found: {k}"


class TestIdempotence:
    """Verify that re-running the pipeline does not create duplicate approvals."""

    def test_no_duplicate_approvals(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_07")

        # First run
        client.post(f"/api/analyses/{study_id}")

        # Count SAP approvals
        all_approvals = client.get(f"/api/approvals?study_id={study_id}").json()
        sap_count_1 = sum(1 for a in all_approvals if a["object_type"] == "sap")

        # Re-run (should NOT create a duplicate SAP approval)
        client.post(f"/api/analyses/{study_id}")
        all_approvals = client.get(f"/api/approvals?study_id={study_id}").json()
        sap_count_2 = sum(1 for a in all_approvals if a["object_type"] == "sap")

        assert sap_count_2 == sap_count_1


class TestStatusEndpoint:
    """Verify the detailed status endpoint."""

    def test_status_shows_artefacts_and_approvals(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_08")
        client.post(f"/api/analyses/{study_id}")

        r = client.get(f"/api/analyses/{study_id}/status")
        assert r.status_code == 200
        body = r.json()

        assert "artefacts" in body
        assert body["artefacts"]["claim_evidence_map"] is True
        assert body["artefacts"]["qc_report"] is True
        assert body["artefacts"]["sap_draft"] is True
        # Before SAP approval, no statistical results
        assert body["artefacts"]["statistical_results"] is False
        assert len(body["pending_approvals"]) >= 1


class TestReportContent:
    """Verify generated reports have the expected sections."""

    def test_reports_are_complete(self, client: TestClient, demo_csv: Path) -> None:
        study_id = _setup_study(client, demo_csv, "_P67_09")
        for _ in range(6):
            client.post(f"/api/analyses/{study_id}")
            if _approve_all(client, study_id) == 0:
                break

        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study_id)

        # Statistical Analysis Report
        sar = (ws.reports / "statistical_analysis_report.md").read_text()
        assert "## Endpoints analysed" in sar
        assert "## Claim substantiation" in sar
        assert "## Limitations" in sar

        # Claim Substantiation Report
        csr = (ws.reports / "claim_substantiation_report.md").read_text()
        assert "## Claim inventory" in csr
        assert "## Evidence matrix" in csr
        assert "## Summary decision table" in csr

        # Safety Report
        sr = (ws.reports / "safety_report.md").read_text()
        assert "## Summary" in sr

        # Executive Summary
        es = (ws.reports / "executive_summary.md").read_text()
        assert "Claims confirmed:" in es

        # Reports listing via API
        r = client.get(f"/api/reports/{study_id}")
        reports = r.json()
        assert "statistical_analysis_report.md" in reports
        assert "claim_substantiation_report.md" in reports
        assert "safety_report.md" in reports
        assert "executive_summary.md" in reports
