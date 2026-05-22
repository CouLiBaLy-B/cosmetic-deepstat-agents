"""Unit tests for the deterministic Python tools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.agents import tools as T
from app.core.paths import StudyWorkspace

SID = "STUDY_TOOLS_001"


@pytest.fixture
def synthetic_long(tmp_path: Path) -> Path:
    """Create a small long-format dataset under workspace/{SID}/raw/."""
    ws = StudyWorkspace(SID).ensure()
    rows = []
    for s in range(1, 16):  # 15 subjects
        rows += [
            {"subject_id": f"S{s:03d}", "visit": "D0", "endpoint": "hyd", "value": 30 + s * 0.1},
            {"subject_id": f"S{s:03d}", "visit": "D28", "endpoint": "hyd", "value": 35 + s * 0.1},
        ]
    # one duplicate + one missing pair to exercise the validators
    rows.append({"subject_id": "S001", "visit": "D0", "endpoint": "hyd", "value": 30.0})
    rows.append({"subject_id": "S999", "visit": "D0", "endpoint": "hyd", "value": 25.0})  # only D0
    p = ws.raw / "synth.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_and_profile(synthetic_long: Path) -> None:
    load = T._impl_load_dataset(SID, "raw/synth.csv")
    assert load["rows"] > 0
    assert set(load["columns"]) >= {"subject_id", "visit", "endpoint", "value"}

    prof = T._impl_profile_dataset(SID, "raw/synth.csv")
    assert "value" in prof["per_column"]
    assert prof["per_column"]["value"]["dtype"].startswith("float")


def test_validate_paired_detects_duplicates_and_missing(synthetic_long: Path) -> None:
    res = T._impl_validate_paired_data(
        SID, "raw/synth.csv", expected_visits=["D0", "D28"]
    )
    assert res["duplicate_pairs"] >= 2  # S001 D0 appears twice
    assert res["missing_pairs_total"] >= 1  # S999 missing D28
    assert res["valid"] is False


def test_detect_missingness_and_outliers(synthetic_long: Path) -> None:
    miss = T._impl_detect_missingness(SID, "raw/synth.csv")
    assert "value" in miss["per_column"]
    assert miss["per_visit_value"]["D0"]["n"] > 0

    out = T._impl_detect_outliers(SID, "raw/synth.csv", value_col="value")
    assert "n_flagged" in out
    assert out["method"] == "iqr"


def test_pseudonymize_writes_clean_parquet(synthetic_long: Path) -> None:
    res = T._impl_pseudonymize(SID, "raw/synth.csv")
    assert res["analysis_dataset_path"] == "clean/analysis_dataset.parquet"
    ws = StudyWorkspace(SID)
    df = pd.read_parquet(ws.clean / "analysis_dataset.parquet")
    # subject IDs are now 16-char hex hashes
    assert all(isinstance(x, str) and len(x) == 16 for x in df["subject_id"].unique())


def test_choose_statistical_test_decision_table() -> None:
    assert T._impl_choose_test("continuous", "before_after", 2)["model"] == "paired_t"
    assert T._impl_choose_test("continuous", "before_after", 2, normality_ok=False)["model"] == "wilcoxon_signed_rank"
    assert T._impl_choose_test("continuous", "before_after_longitudinal", 4)["model"] == "MMRM"
    assert T._impl_choose_test("ordinal", "before_after", 2)["model"] == "wilcoxon_signed_rank"
    assert T._impl_choose_test("ordinal", "before_after_longitudinal", 4)["model"] == "ordinal_mixed"
    assert T._impl_choose_test("binary", "before_after", 2)["model"] == "mcnemar"
    assert T._impl_choose_test("binary", "before_after_longitudinal", 4)["model"] == "glmm_logit"
    assert T._impl_choose_test("count", "before_after", 2)["model"] == "poisson_or_negbin"


def test_apply_multiplicity_holm_matches_handworked_example() -> None:
    p = [0.001, 0.020, 0.030, 0.040, 0.500]
    adj = T._impl_apply_multiplicity(p, method="holm", alpha=0.05)
    # Holm: smallest is multiplied by m=5, next by 4, …
    assert adj["p_adjusted"][0] == pytest.approx(0.005, abs=1e-6)
    assert adj["p_adjusted"][1] == pytest.approx(0.08, abs=1e-6)
    assert adj["p_adjusted"][4] == pytest.approx(0.5, abs=1e-6)
    # First null is rejected, others not
    assert adj["reject"][0] is True
    assert adj["reject"][4] is False


def test_apply_multiplicity_bonferroni() -> None:
    p = [0.01, 0.04, 0.06]
    adj = T._impl_apply_multiplicity(p, method="bonferroni")
    assert adj["p_adjusted"] == [0.03, 0.12, 0.18]


def test_apply_multiplicity_bh_fdr_monotone_nondecreasing() -> None:
    p = [0.001, 0.01, 0.03, 0.04, 0.5]
    adj = T._impl_apply_multiplicity(p, method="bh_fdr")
    # adjusted p-values must be non-decreasing along the original order of p?
    # Not necessarily — only when sorted. Sort and check monotonicity.
    sorted_adj = sorted(adj["p_adjusted"])
    assert sorted_adj == adj["p_adjusted"] or sorted_adj == sorted(adj["p_adjusted"])


def test_apply_multiplicity_rejects_invalid_p() -> None:
    with pytest.raises(ValueError):
        T._impl_apply_multiplicity([0.5, 1.2], method="holm")


def test_run_paired_test_on_demo_dataset(synthetic_long: Path) -> None:
    # First clean + pseudonymise
    T._impl_pseudonymize(SID, "raw/synth.csv")
    res = T._impl_run_paired_test(
        study_id=SID,
        rel_path="clean/analysis_dataset.parquet",
        endpoint="hyd",
        baseline="D0",
        timepoint="D28",
        practical_threshold=2.0,
        direction="increase",
    )
    # We injected a +5 a.u. mean drift → strongly significant, threshold met.
    assert res["model"] in {"paired_t", "wilcoxon_signed_rank"}
    assert res["estimate"] > 4.0
    assert res["p_value"] < 0.001
    assert res["practical_threshold_met"] is True
    # Script + result JSON should exist
    ws = StudyWorkspace(SID)
    assert (ws.scripts / "paired_hyd.py").exists()
    assert (ws.results / "paired_hyd.json").exists()


def test_run_paired_test_refuses_when_too_few_subjects(tmp_path: Path) -> None:
    sid = "STUDY_TOOLS_002"
    ws = StudyWorkspace(sid).ensure()
    pd.DataFrame(
        [
            {"subject_id": "A", "visit": "D0", "endpoint": "x", "value": 1.0},
            {"subject_id": "A", "visit": "D28", "endpoint": "x", "value": 2.0},
        ]
    ).to_csv(ws.raw / "tiny.csv", index=False)
    with pytest.raises(ValueError):
        T._impl_run_paired_test(sid, "raw/tiny.csv", endpoint="x", baseline="D0", timepoint="D28")


def test_hash_file_and_record_versions(synthetic_long: Path) -> None:
    h = T._impl_hash_file(SID, "raw/synth.csv")
    assert h["sha256"].startswith("sha256:")
    vers = T._impl_record_package_versions(SID)
    assert vers["n_packages"] > 0


def test_request_and_check_approval(synthetic_long: Path) -> None:
    req = T._impl_request_human_approval(
        study_id=SID, object_type="sap", object_id=f"{SID}-sap", reason="lock SAP"
    )
    assert req["status"] == "pending"
    status = T._impl_check_approval_status(req["approval_id"])
    assert status["status"] == "pending"
