"""Contract tests: every tool's JSON output conforms to its documented schema.

Each test calls the _impl_* function on synthetic data and verifies the
returned dict has exactly the documented keys, correct types, and valid
value ranges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents import tools as t_mod
from app.core.paths import StudyWorkspace

SID = "STUDY_CONTRACT_001"


@pytest.fixture(autouse=True)
def _setup_data(tmp_path: Path) -> None:
    """Create a comprehensive dataset for all contract tests."""
    ws = StudyWorkspace(SID).ensure()
    rng = np.random.default_rng(777)
    rows = []
    for s in range(1, 21):
        base = float(rng.normal(35, 6))
        for i, v in enumerate(["D0", "D7", "D14", "D28"]):
            rows.append({
                "subject_id": f"S{s:03d}", "visit": v,
                "endpoint": "hyd", "value": round(base + 5 * (i / 3) + float(rng.normal(0, 1.5)), 3),
            })
        rows.append({"subject_id": f"S{s:03d}", "visit": "D0", "endpoint": "tol", "value": int(rng.random() < 0.3)})
        rows.append({"subject_id": f"S{s:03d}", "visit": "D28", "endpoint": "tol", "value": int(rng.random() < 0.8)})

    pd.DataFrame(rows).to_csv(ws.raw / "data.csv", index=False)
    pd.DataFrame(rows).to_parquet(ws.clean / "analysis_dataset.parquet", index=False)

    # Consumer data
    crows = []
    for s in range(1, 31):
        crows.append({"subject_id": f"C{s:03d}", "question": "smooth", "value": int(rng.choice([1, 2, 3, 4, 5]))})
    pd.DataFrame(crows).to_csv(ws.raw / "consumer.csv", index=False)


# ---- 1. load_dataset ----
class TestLoadDatasetContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_load_dataset(SID, "raw/data.csv")
        assert isinstance(r["study_id"], str)
        assert isinstance(r["path"], str)
        assert isinstance(r["rows"], int) and r["rows"] > 0
        assert isinstance(r["cols"], int) and r["cols"] > 0
        assert isinstance(r["columns"], list)
        assert all(isinstance(c, str) for c in r["columns"])
        assert isinstance(r["dtypes"], dict)
        assert isinstance(r["sha256"], str) and r["sha256"].startswith("sha256:")


# ---- 2. profile_dataset ----
class TestProfileDatasetContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_profile_dataset(SID, "raw/data.csv")
        assert isinstance(r["rows"], int)
        assert isinstance(r["per_column"], dict)
        assert "output_path" in r
        assert "output_sha256" in r
        for _col, stats in r["per_column"].items():
            assert "dtype" in stats
            assert "missing" in stats
            assert isinstance(stats["missing"], int)


# ---- 3. validate_paired_data ----
class TestValidatePairedContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_validate_paired_data(SID, "raw/data.csv", expected_visits=["D0", "D28"])
        assert isinstance(r["valid"], bool)
        assert isinstance(r["n_subjects"], int)
        assert isinstance(r["visits_present"], list)
        assert isinstance(r["duplicate_pairs"], int)
        assert isinstance(r["missing_pairs"], list)
        assert "output_path" in r


# ---- 4. detect_missingness ----
class TestDetectMissingnessContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_detect_missingness(SID, "raw/data.csv")
        assert isinstance(r["rows"], int)
        assert isinstance(r["per_column"], dict)
        for _col, info in r["per_column"].items():
            assert "missing" in info
            assert "pct_missing" in info
            assert 0 <= info["pct_missing"] <= 100
        assert "output_path" in r


# ---- 5. detect_outliers ----
class TestDetectOutliersContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_detect_outliers(SID, "raw/data.csv", value_col="value")
        assert isinstance(r["n_flagged"], int) and r["n_flagged"] >= 0
        assert r["method"] in {"iqr", "zscore"}
        assert isinstance(r.get("flagged_sample", r.get("flagged_rows", [])), list)
        assert "csv_path" in r


# ---- 6. pseudonymize ----
class TestPseudonymizeContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_pseudonymize(SID, "raw/data.csv")
        assert isinstance(r["n_subjects"], int)
        assert isinstance(r["analysis_dataset_path"], str)
        assert isinstance(r["analysis_dataset_sha256"], str)
        assert r["analysis_dataset_sha256"].startswith("sha256:")


# ---- 7. hash_file ----
class TestHashFileContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_hash_file(SID, "raw/data.csv")
        assert isinstance(r["sha256"], str)
        assert r["sha256"].startswith("sha256:")
        assert len(r["sha256"]) == 7 + 64  # "sha256:" + 64 hex chars


# ---- 8. write_audit_event ----
class TestWriteAuditEventContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_write_audit_event(SID, "test", "test.action", {"key": "val"})
        assert isinstance(r["event_id"], str)
        assert r["actor"] == "test"
        assert r["action"] == "test.action"


# ---- 9. choose_statistical_test ----
class TestChooseTestContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_choose_test("continuous", "before_after", 2)
        assert isinstance(r["model"], str) and len(r["model"]) > 0
        assert isinstance(r["rationale"], str)
        assert isinstance(r["rationale"], str)


# ---- 10. apply_multiplicity ----
class TestApplyMultiplicityContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_apply_multiplicity([0.01, 0.04, 0.06], method="holm")
        assert isinstance(r["p_adjusted"], list)
        assert len(r["p_adjusted"]) == 3
        assert isinstance(r["reject"], list)
        assert len(r["reject"]) == 3
        assert all(isinstance(x, float) for x in r["p_adjusted"])
        assert all(isinstance(x, bool) for x in r["reject"])
        assert all(0 <= p <= 1 for p in r["p_adjusted"])


# ---- 11. run_paired_test ----
class TestRunPairedTestContract:
    def test_output_schema(self) -> None:
        t_mod._impl_pseudonymize(SID, "raw/data.csv")
        r = t_mod._impl_run_paired_test(
            SID, "clean/analysis_dataset.parquet",
            endpoint="hyd", baseline="D0", timepoint="D28",
            practical_threshold=3.0, direction="increase",
        )
        assert isinstance(r["endpoint"], str)
        assert isinstance(r["model"], str)
        assert r["model"] in {"paired_t", "wilcoxon_signed_rank"}
        assert isinstance(r["estimate"], float)
        assert isinstance(r["ci95"], (list, tuple)) and len(r["ci95"]) == 2
        assert r["ci95"][0] <= r["ci95"][1]
        assert isinstance(r["p_value"], float) and 0 <= r["p_value"] <= 1
        assert isinstance(r["n"], int) and r["n"] > 0
        assert isinstance(r["practical_threshold_met"], bool)
        assert isinstance(r["conclusion"], str)
        assert isinstance(r["artefacts"], dict)
        assert "script" in r["artefacts"]


# ---- 12. run_mmrm ----
class TestRunMMRMContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_run_mmrm(
            SID, "clean/analysis_dataset.parquet",
            endpoint="hyd", baseline="D0", primary_timepoint="D28",
            practical_threshold=0.5, direction="increase",
        )
        assert isinstance(r["endpoint"], str)
        assert r["model"].startswith("MMRM")
        assert isinstance(r["estimate"], float)
        assert isinstance(r["ci95"], (list, tuple)) and len(r["ci95"]) == 2
        assert isinstance(r["p_value"], float) and 0 <= r["p_value"] <= 1
        assert isinstance(r["n"], int) and r["n"] > 0
        assert isinstance(r["practical_threshold_met"], bool)
        assert isinstance(r["artefacts"], dict)


# ---- 13. run_glmm_logit ----
class TestRunGLMMLogitContract:
    def test_output_schema(self) -> None:
        # Need longitudinal binary data
        ws = StudyWorkspace(SID).ensure()
        rng = np.random.default_rng(88)
        rows = []
        for s in range(1, 21):
            for v in ["D0", "D7", "D14", "D28"]:
                p = 0.3 if v == "D0" else 0.6
                rows.append({"subject_id": f"S{s:03d}", "visit": v, "endpoint": "tol",
                              "value": int(rng.random() < p)})
        pd.DataFrame(rows).to_csv(ws.raw / "binary_long.csv", index=False)

        r = t_mod._impl_run_glmm_logit(SID, "raw/binary_long.csv", endpoint="tol",
                                     baseline="D0", primary_timepoint="D28")
        assert r["model"] == "glmm_logit_gee"
        assert isinstance(r["estimate"], float) and r["estimate"] > 0  # OR
        assert isinstance(r["ci95"], (list, tuple)) and len(r["ci95"]) == 2
        assert isinstance(r["p_value"], float) and 0 <= r["p_value"] <= 1
        assert r["scale"] == "odds_ratio"
        assert isinstance(r["artefacts"], dict)


# ---- 14. run_mcnemar ----
class TestRunMcNemarContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_run_mcnemar(SID, "raw/data.csv", endpoint="tol",
                                 baseline="D0", timepoint="D28")
        assert r["model"] == "mcnemar"
        assert isinstance(r["estimate"], float)
        assert isinstance(r["ci95"], (list, tuple)) and len(r["ci95"]) == 2
        assert isinstance(r["p_value"], float) and 0 <= r["p_value"] <= 1
        assert isinstance(r["n"], int) and r["n"] > 0
        assert isinstance(r["table"], dict)
        assert all(k in r["table"] for k in ("a", "b", "c", "d"))
        assert isinstance(r["artefacts"], dict)


# ---- 15. run_top2box ----
class TestRunTop2BoxContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_run_top2box(SID, "raw/consumer.csv", question_col="smooth",
                                 value_col="value", scale_max=5)
        assert isinstance(r["n"], int) and r["n"] > 0
        assert isinstance(r["top2_count"], int)
        assert isinstance(r["top2_pct"], float)
        assert 0 <= r["top2_pct"] <= 100
        assert isinstance(r["ci95_pct"], (list, tuple)) and len(r["ci95_pct"]) == 2
        assert r["ci95_pct"][0] <= r["ci95_pct"][1]
        assert r["ci_method"] == "wilson"
        assert isinstance(r["artefacts"], dict)


# ---- 16. run_tost ----
class TestRunTOSTContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_run_tost(SID, "clean/analysis_dataset.parquet",
                              endpoint="hyd", margin=20.0,
                              baseline="D0", timepoint="D28")
        assert r["model"] == "TOST"
        assert isinstance(r["mean_difference"], float)
        assert isinstance(r["margin"], float) and r["margin"] == 20.0
        assert isinstance(r["tost_p1"], float)
        assert isinstance(r["tost_p2"], float)
        assert isinstance(r["tost_p_max"], float)
        assert isinstance(r["ci90"], (list, tuple)) and len(r["ci90"]) == 2
        assert isinstance(r["equivalence_met"], bool)
        assert isinstance(r["n"], int) and r["n"] > 0
        assert isinstance(r["artefacts"], dict)


# ---- 17. record_package_versions ----
class TestRecordVersionsContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_record_package_versions(SID)
        assert isinstance(r["path"], str)
        assert isinstance(r["sha256"], str) and r["sha256"].startswith("sha256:")
        assert isinstance(r["n_packages"], int) and r["n_packages"] > 0


# ---- 18. request_human_approval ----
class TestRequestApprovalContract:
    def test_output_schema(self) -> None:
        r = t_mod._impl_request_human_approval(SID, "test_type", "test_obj", "test reason")
        assert isinstance(r["approval_id"], str) and r["approval_id"].startswith("APR-")
        assert r["status"] == "pending"
        assert r["object_type"] == "test_type"
        assert r["object_id"] == "test_obj"


# ---- 19. check_approval_status ----
class TestCheckApprovalContract:
    def test_output_schema(self) -> None:
        req = t_mod._impl_request_human_approval(SID, "sap", "test-sap", "lock SAP")
        r = t_mod._impl_check_approval_status(req["approval_id"])
        assert r["approval_id"] == req["approval_id"]
        assert r["status"] == "pending"
        assert r["reviewer"] is None
        assert r["decision_at"] is None

    def test_not_found(self) -> None:
        r = t_mod._impl_check_approval_status("APR-nonexistent")
        assert r["status"] == "not_found"
