"""Tests for the Phase 5 tools: MMRM, GLMM, McNemar, top-2-box, TOST.

These test the _impl_* functions (filesystem-integrated), complementing
``test_statistics_runner.py`` which tests the pure compute functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents import tools as T
from app.core.paths import StudyWorkspace

SID_NEW = "STUDY_NEWTOOLS_001"


@pytest.fixture
def longitudinal_csv(tmp_path: Path) -> Path:
    """Longitudinal continuous dataset with 4 visits."""
    ws = StudyWorkspace(SID_NEW).ensure()
    rng = np.random.default_rng(42)
    rows = []
    for s in range(1, 26):
        base = float(rng.normal(35, 6))
        for i, v in enumerate(["D0", "D7", "D14", "D28"]):
            rows.append({
                "subject_id": f"S{s:03d}",
                "visit": v,
                "endpoint": "hyd",
                "value": round(base + 5 * (i / 3) + float(rng.normal(0, 1.5)), 3),
            })
    p = ws.raw / "longitudinal.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    # Also write a clean parquet
    pd.DataFrame(rows).to_parquet(ws.clean / "analysis_dataset.parquet", index=False)
    return p


@pytest.fixture
def binary_csv(tmp_path: Path) -> Path:
    """Binary paired dataset."""
    ws = StudyWorkspace(SID_NEW).ensure()
    rng = np.random.default_rng(99)
    rows = []
    for s in range(1, 31):
        rows.append({"subject_id": f"S{s:03d}", "visit": "D0", "endpoint": "tol", "value": int(rng.random() < 0.3)})
        rows.append({"subject_id": f"S{s:03d}", "visit": "D28", "endpoint": "tol", "value": int(rng.random() < 0.7)})
    p = ws.raw / "binary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture
def consumer_csv(tmp_path: Path) -> Path:
    """Consumer questionnaire dataset."""
    ws = StudyWorkspace(SID_NEW).ensure()
    rng = np.random.default_rng(123)
    rows = []
    for s in range(1, 51):
        rows.append({
            "subject_id": f"C{s:03d}",
            "question": "smoothness",
            "value": int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.15, 0.30, 0.40])),
        })
    p = ws.raw / "consumer.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


class TestMMRMTool:
    def test_run_mmrm_produces_artefacts(self, longitudinal_csv: Path) -> None:
        res = T._impl_run_mmrm(
            SID_NEW,
            "clean/analysis_dataset.parquet",
            endpoint="hyd",
            baseline="D0",
            primary_timepoint="D28",
            practical_threshold=0.5,
            direction="increase",
        )
        assert res["model"].startswith("MMRM")
        assert res["p_value"] < 0.05
        assert res["practical_threshold_met"] is True
        assert res["estimate"] > 0  # positive change expected
        ws = StudyWorkspace(SID_NEW)
        assert (ws.scripts / "mmrm_hyd.py").exists()
        assert (ws.results / "mmrm_hyd.json").exists()


class TestGLMMLogitTool:
    def test_glmm_logit_produces_artefacts(self, binary_csv: Path) -> None:
        # Build longitudinal binary data
        ws = StudyWorkspace(SID_NEW).ensure()
        rng = np.random.default_rng(77)
        rows = []
        for s in range(1, 26):
            for v in ["D0", "D7", "D14", "D28"]:
                p = 0.3 if v == "D0" else 0.5 + 0.1 * (["D0", "D7", "D14", "D28"].index(v))
                rows.append({
                    "subject_id": f"S{s:03d}", "visit": v, "endpoint": "tol",
                    "value": int(rng.random() < min(p, 0.95)),
                })
        pd.DataFrame(rows).to_csv(ws.raw / "binary_long.csv", index=False)

        res = T._impl_run_glmm_logit(
            SID_NEW, "raw/binary_long.csv", endpoint="tol",
            baseline="D0", primary_timepoint="D28",
        )
        assert "glmm" in res["model"] or "gee" in res["model"]
        ws = StudyWorkspace(SID_NEW)
        assert (ws.scripts / "glmm_logit_tol.py").exists()


class TestMcNemarTool:
    def test_mcnemar_produces_artefacts(self, binary_csv: Path) -> None:
        res = T._impl_run_mcnemar(
            SID_NEW, "raw/binary.csv", endpoint="tol",
            baseline="D0", timepoint="D28",
        )
        assert res["model"] == "mcnemar"
        assert res["n"] >= 25
        ws = StudyWorkspace(SID_NEW)
        assert (ws.scripts / "mcnemar_tol.py").exists()
        assert (ws.results / "mcnemar_tol.json").exists()


class TestTop2BoxTool:
    def test_top2box_produces_artefacts(self, consumer_csv: Path) -> None:
        res = T._impl_run_top2box(
            SID_NEW, "raw/consumer.csv",
            question_col="smoothness",
            value_col="value",
            scale_max=5,
        )
        assert res["n"] == 50
        assert res["top2_pct"] > 0
        assert "ci95_pct" in res
        ws = StudyWorkspace(SID_NEW)
        assert (ws.results / "top2box_smoothness.json").exists()


class TestTOSTTool:
    def test_tost_produces_artefacts(self, longitudinal_csv: Path) -> None:
        res = T._impl_run_tost(
            SID_NEW, "clean/analysis_dataset.parquet",
            endpoint="hyd", margin=10.0,
            baseline="D0", timepoint="D28",
        )
        assert res["model"] == "TOST"
        assert "equivalence_met" in res
        ws = StudyWorkspace(SID_NEW)
        assert (ws.scripts / "tost_hyd.py").exists()
        assert (ws.results / "tost_hyd.json").exists()


class TestSkillsExist:
    """Verify that all 14 SKILL.md files are present with frontmatter."""

    SKILL_PATHS = [
        "skills/statistics/paired_tests/SKILL.md",
        "skills/statistics/linear_mixed_models/SKILL.md",
        "skills/statistics/mmrm/SKILL.md",
        "skills/statistics/ordinal_models/SKILL.md",
        "skills/statistics/glmm_gee/SKILL.md",
        "skills/statistics/multiplicity/SKILL.md",
        "skills/statistics/equivalence_tost/SKILL.md",
        "skills/statistics/missing_data/SKILL.md",
        "skills/cosmetics/claims_eu/SKILL.md",
        "skills/cosmetics/claims_us/SKILL.md",
        "skills/cosmetics/tolerance/SKILL.md",
        "skills/reporting/statistical_report/SKILL.md",
        "skills/reporting/claim_substantiation/SKILL.md",
        "skills/postmarket/adverse_events/SKILL.md",
    ]

    @pytest.mark.parametrize("path", SKILL_PATHS)
    def test_skill_file_exists_with_frontmatter(self, path: str) -> None:
        import os
        full = os.path.join(os.path.dirname(__file__), "..", path)
        p = Path(full).resolve()
        assert p.exists(), f"Missing: {path}"
        content = p.read_text()
        assert content.startswith("---"), f"No YAML frontmatter in {path}"
        assert "name:" in content
        assert "description:" in content


class TestSubagentsFullyWired:
    """Verify that all 10 subagents have at least 1 tool."""

    def test_all_subagents_have_tools(self) -> None:
        from app.agents.subagents import build_subagents
        from app.agents.tools import build_langchain_tools

        try:
            tools = build_langchain_tools()
        except Exception:
            pytest.skip("langchain not installed")
            return

        if not tools:
            pytest.skip("langchain tools not available")
            return

        subs = build_subagents(tools)
        assert len(subs) == 10
        for s in subs:
            assert len(s["tools"]) >= 1, f"Subagent {s['name']!r} has no tools."
