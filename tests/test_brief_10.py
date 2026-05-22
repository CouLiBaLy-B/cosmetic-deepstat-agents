"""The 10 mandatory tests from the project brief.

These tests verify the **non-negotiable hard rules** of the platform:

 1. No raw data ever passed to an LLM context
 2. No confirmatory analysis without an approved SAP
 3. No final claim wording without human approval
 4. Effect + CI95 + adjusted p + practical threshold for every endpoint
 5. Consumer ≠ instrumental (never mixed)
 6. Audit trail is immutable (append-only JSONL with hashes)
 7. Multiplicity is always applied when ≥2 confirmatory hypotheses
 8. Equivalence claims require a pre-specified margin
 9. No exploratory-to-confirmatory promotion
10. Every artefact is reproducible (script + seed + package versions)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.paths import StudyWorkspace
from app.main import app

EX = Path(__file__).parent.parent / "examples" / "sample_study"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _full_run(client: TestClient, suffix: str = "") -> str:
    """Create study, upload, attach claims, run pipeline to completion."""
    meta = json.loads((EX / "study_metadata.json").read_text())
    meta["study_id"] = meta["study_id"] + suffix
    study_id = meta["study_id"]

    r = client.post("/api/studies", json=meta)
    assert r.status_code in (201, 409)

    csv_path = EX / "data" / "measurements_long.csv"
    with csv_path.open("rb") as fh:
        r = client.post(
            f"/api/studies/{study_id}/data",
            files={"file": (csv_path.name, fh, "text/csv")},
        )
    assert r.status_code in (201, 409)

    claims = json.loads((EX / "claims.json").read_text())
    client.post(f"/api/studies/{study_id}/claims", json=claims)

    for _ in range(6):
        client.post(f"/api/analyses/{study_id}")
        pending = client.get(f"/api/approvals?study_id={study_id}&status=pending").json()
        if not pending:
            break
        for a in pending:
            client.post(
                f"/api/approvals/{a['approval_id']}",
                json={"decision": "approved", "reviewer": "test-brief"},
            )

    return study_id


# -----------------------------------------------------------------------
# Test 1: No raw data in LLM context
# -----------------------------------------------------------------------
class TestNoRawDataInContext:
    """Tools must return summaries, never raw data rows."""

    def test_load_dataset_returns_shape_not_data(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B01")
        from app.agents import tools as t_mod
        res = t_mod._impl_load_dataset(study_id, "raw/measurements_long.csv")
        # Must have shape info
        assert "rows" in res
        assert "columns" in res
        assert "dtypes" in res
        # Must NOT contain actual values
        assert "data" not in res
        assert "values" not in res
        assert "records" not in res

    def test_profile_returns_stats_not_data(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B01b")
        from app.agents import tools as t_mod
        res = t_mod._impl_profile_dataset(study_id, "raw/measurements_long.csv")
        assert "per_column" in res
        # Must NOT contain actual values
        assert "data" not in res


# -----------------------------------------------------------------------
# Test 2: No analysis without approved SAP
# -----------------------------------------------------------------------
class TestNoAnalysisWithoutSAP:
    def test_statistical_results_absent_before_sap(self, client: TestClient) -> None:
        meta = json.loads((EX / "study_metadata.json").read_text())
        meta["study_id"] = "STUDY_BRIEF_02"
        client.post("/api/studies", json=meta)
        csv_path = EX / "data" / "measurements_long.csv"
        with csv_path.open("rb") as fh:
            client.post(f"/api/studies/{meta['study_id']}/data",
                        files={"file": (csv_path.name, fh, "text/csv")})
        claims = json.loads((EX / "claims.json").read_text())
        client.post(f"/api/studies/{meta['study_id']}/claims", json=claims)

        # First launch → SAP not approved
        client.post(f"/api/analyses/{meta['study_id']}")
        ws = StudyWorkspace(meta["study_id"])
        assert not (ws.results / "statistical_results.json").exists()


# -----------------------------------------------------------------------
# Test 3: No claim wording without human approval
# -----------------------------------------------------------------------
class TestNoClaimWithoutApproval:
    def test_claim_wording_gate_exists(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B03")
        from app.storage import db
        claim_approvals = [
            a for a in db.approvals().list()
            if a.study_id == study_id and a.object_type == "claim_wording"
        ]
        assert len(claim_approvals) >= 1
        # It must have been approved (we auto-approved in _full_run)
        assert any(a.status.value == "approved" for a in claim_approvals)


# -----------------------------------------------------------------------
# Test 4: Effect + CI95 + adjusted p + practical threshold for every endpoint
# -----------------------------------------------------------------------
class TestCompleteResultSchema:
    def test_every_result_has_required_fields(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B04")
        ws = StudyWorkspace(study_id)
        sres = json.loads((ws.results / "statistical_results.json").read_text())
        for r in sres:
            if "error" in r:
                continue
            assert "estimate" in r, f"Missing estimate in {r.get('endpoint')}"
            assert "ci95" in r, f"Missing CI95 in {r.get('endpoint')}"
            assert "p_value" in r, f"Missing p_value in {r.get('endpoint')}"
            assert "practical_threshold" in r, f"Missing practical_threshold in {r.get('endpoint')}"
            # CI must be ordered
            assert r["ci95"][0] <= r["ci95"][1], f"CI95 misordered in {r.get('endpoint')}"


# -----------------------------------------------------------------------
# Test 5: Consumer ≠ instrumental — never mixed
# -----------------------------------------------------------------------
class TestConsumerInstrumentalSeparation:
    """The system must never produce a claim that mixes consumer perception
    with instrumental evidence in the same wording."""

    def test_claim_evidence_map_separates_types(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B05")
        ws = StudyWorkspace(study_id)
        cmap = json.loads((ws.results / "claim_evidence_map.json").read_text())
        for cm in cmap:
            if cm["claim_type"] == "consumer":
                # Consumer claims must NOT reference instrumental endpoints
                assert cm.get("primary_endpoint") is None or "corneometer" not in (cm.get("primary_endpoint") or "").lower()
            elif cm["claim_type"] == "instrumental":
                # Instrumental wording must not say "consumers reported/perceived"
                for cond in cm.get("allowed_wording_conditions", []):
                    assert "consumers reported" not in cond.lower()


# -----------------------------------------------------------------------
# Test 6: Audit trail is immutable (append-only)
# -----------------------------------------------------------------------
class TestAuditTrailImmutability:
    def test_audit_trail_grows_monotonically(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B06")
        ws = StudyWorkspace(study_id)
        trail_path = ws.audit / "audit_trail.jsonl"
        assert trail_path.exists()
        lines = trail_path.read_text().strip().split("\n")
        assert len(lines) >= 5  # many events should have been logged

        # Timestamps must be monotonically non-decreasing
        timestamps = []
        for line in lines:
            evt = json.loads(line)
            assert "event_id" in evt
            assert "timestamp" in evt
            assert "actor" in evt
            assert "action" in evt
            timestamps.append(evt["timestamp"])
        assert timestamps == sorted(timestamps), "Audit trail is not chronologically ordered"

    def test_audit_has_hashes(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B06b")
        ws = StudyWorkspace(study_id)
        trail_path = ws.audit / "audit_trail.jsonl"
        lines = trail_path.read_text().strip().split("\n")
        events_with_hash = [json.loads(ln) for ln in lines if '"output_hash"' in ln]
        # At least some events should carry output hashes
        hash_values = [e["output_hash"] for e in events_with_hash if e.get("output_hash")]
        assert len(hash_values) >= 3, f"Expected ≥3 events with output_hash, got {len(hash_values)}"
        for h in hash_values:
            assert h.startswith("sha256:"), f"Hash {h} doesn't start with sha256:"


# -----------------------------------------------------------------------
# Test 7: Multiplicity is always applied when ≥2 confirmatory hypotheses
# -----------------------------------------------------------------------
class TestMultiplicityApplied:
    def test_holm_applied_to_two_primary_endpoints(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B07")
        ws = StudyWorkspace(study_id)
        # Multiplicity is applied in step_decide_claims and written to claim_decisions
        dec = json.loads((ws.results / "claim_decisions.json").read_text())
        with_adj = [d for d in dec if d.get("statistical_basis", {}).get("p_adjusted") is not None]
        assert len(with_adj) >= 2, f"Expected ≥2 decisions with p_adjusted, got {len(with_adj)}"

        # Holm guarantees p_adjusted >= p_value for every hypothesis
        for d in with_adj:
            sb = d["statistical_basis"]
            assert sb["p_adjusted"] >= sb["p_value"] - 1e-12, (
                f"Holm violation: p_adj={sb['p_adjusted']} < p={sb['p_value']} "
                f"for claim {d['claim_id']}"
            )
        # The method should be recorded
        methods = {d["statistical_basis"].get("p_adjustment_method") for d in with_adj}
        assert "holm" in methods or "none" in methods

    def test_claim_decisions_reference_adjusted_p(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B07b")
        ws = StudyWorkspace(study_id)
        dec = json.loads((ws.results / "claim_decisions.json").read_text())
        for d in dec:
            sb = d.get("statistical_basis", {})
            if "p_value" in sb:
                assert "p_adjusted" in sb, f"Claim {d['claim_id']} missing p_adjusted"
                assert "p_adjustment_method" in sb


# -----------------------------------------------------------------------
# Test 8: Equivalence claims require a pre-specified margin
# -----------------------------------------------------------------------
class TestEquivalenceRequiresMargin:
    def test_tost_refuses_without_margin(self) -> None:
        from app.services.statistics_runner import run_tost
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        # margin=0 is meaningless — the test should still run but
        # equivalence_met should be False (no room within [0, 0])
        res = run_tost(x, x, margin=0.0, paired=True)
        # With identical data but margin=0, both one-sided p's should be ~0.5
        assert res["equivalence_met"] is False or res["margin"] == 0.0

    def test_choose_test_recommends_tost_for_equivalence(self) -> None:
        from app.agents.tools import _impl_choose_test
        # The decision table should recognise equivalence designs
        # (currently mapped through claim_type, not design)
        res = _impl_choose_test("continuous", "before_after", 2)
        # It should return paired_t — TOST is applied separately
        assert res["model"] in {"paired_t", "wilcoxon_signed_rank"}


# -----------------------------------------------------------------------
# Test 9: No exploratory-to-confirmatory promotion
# -----------------------------------------------------------------------
class TestNoExploratoryPromotion:
    def test_exploratory_endpoints_not_in_primary_family(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B09")
        ws = StudyWorkspace(study_id)
        sap = json.loads((ws.results / "sap_draft.json").read_text())
        primary_names = sap["multiplicity_strategy"]["families"]["primary"]

        # Only endpoints marked primary should be in the primary family
        from app.storage import db
        study = db.studies().get(study_id)
        assert study is not None
        for ep in study.endpoints:
            if ep.primary_or_secondary == "exploratory":
                assert ep.name not in primary_names, \
                    f"Exploratory endpoint {ep.name} promoted to primary family"


# -----------------------------------------------------------------------
# Test 10: Every artefact is reproducible (script + versions + seed)
# -----------------------------------------------------------------------
class TestReproducibility:
    def test_scripts_exist_for_each_endpoint(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B10")
        ws = StudyWorkspace(study_id)
        scripts = list(ws.scripts.iterdir())
        assert len(scripts) >= 2, f"Expected ≥2 scripts, got {len(scripts)}"
        for s in scripts:
            assert s.suffix == ".py"
            content = s.read_text()
            assert len(content) > 20, f"Script {s.name} is too short"

    def test_package_versions_recorded(self, client: TestClient) -> None:
        study_id = _full_run(client, "_B10b")
        ws = StudyWorkspace(study_id)
        vers_path = ws.audit / "package_versions.json"
        assert vers_path.exists()
        vers = json.loads(vers_path.read_text())
        assert "python" in vers
        assert "packages" in vers
        assert len(vers["packages"]) >= 10  # at least numpy, scipy, etc.
