"""Unit tests for app.services.statistics_runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.statistics_runner import (
    run_glmm_logit,
    run_lmm,
    run_mcnemar,
    run_mmrm,
    run_poisson_or_negbin,
    run_top2box,
    run_tost,
)

# ---------------------------------------------------------------------------
# Helpers: create small synthetic datasets
# ---------------------------------------------------------------------------

def _make_longitudinal(
    n: int = 25,
    visits: list[str] | None = None,
    drift: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Long-format continuous dataset with known effect."""
    rng = np.random.default_rng(seed)
    visits = visits or ["D0", "D7", "D14", "D28"]
    rows = []
    for s in range(1, n + 1):
        base = float(rng.normal(35.0, 6.0))
        for i, v in enumerate(visits):
            rows.append({
                "subject_id": f"S{s:03d}",
                "visit": v,
                "endpoint": "hyd",
                "value": round(base + drift * (i / (len(visits) - 1)) + float(rng.normal(0, 1.5)), 3),
            })
    return pd.DataFrame(rows)


def _make_binary(n: int = 30, p_pre: float = 0.3, p_post: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """Binary paired dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n + 1):
        rows.append({"subject_id": f"S{s:03d}", "visit": "D0", "endpoint": "tol", "value": int(rng.random() < p_pre)})
        rows.append({"subject_id": f"S{s:03d}", "visit": "D28", "endpoint": "tol", "value": int(rng.random() < p_post)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: MMRM
# ---------------------------------------------------------------------------

class TestMMRM:
    def test_mmrm_returns_significant_result(self) -> None:
        df = _make_longitudinal(n=25, drift=6.0)
        res = run_mmrm(df, "hyd", baseline="D0", primary_timepoint="D28")
        assert res["model"].startswith("MMRM")
        assert res["endpoint"] == "hyd"
        assert res["n"] >= 20
        assert res["p_value"] < 0.05
        assert "ci95" in res
        assert res["ci95"][0] < res["ci95"][1]

    def test_mmrm_too_few_raises(self) -> None:
        df = pd.DataFrame([
            {"subject_id": "S001", "visit": "D0", "endpoint": "hyd", "value": 30.0},
            {"subject_id": "S001", "visit": "D28", "endpoint": "hyd", "value": 35.0},
        ])
        with pytest.raises(ValueError, match="insufficient"):
            run_mmrm(df, "hyd", baseline="D0", primary_timepoint="D28")


# ---------------------------------------------------------------------------
# Tests: LMM
# ---------------------------------------------------------------------------

class TestLMM:
    def test_lmm_returns_result(self) -> None:
        df = _make_longitudinal(n=25, drift=5.0)
        res = run_lmm(df, "hyd", baseline="D0", primary_timepoint="D28")
        assert res["model"] == "LMM"
        assert res["n"] >= 20
        assert "ci95" in res
        assert res["converged"] is True


# ---------------------------------------------------------------------------
# Tests: McNemar
# ---------------------------------------------------------------------------

class TestMcNemar:
    def test_mcnemar_detects_change(self) -> None:
        df = _make_binary(n=40, p_pre=0.2, p_post=0.8, seed=99)
        res = run_mcnemar(df, "tol", baseline="D0", timepoint="D28")
        assert res["model"] == "mcnemar"
        assert res["n"] >= 30
        assert res["p_value"] < 0.05  # large shift should be detected
        assert "ci95" in res

    def test_mcnemar_too_few_raises(self) -> None:
        df = pd.DataFrame([
            {"subject_id": "S001", "visit": "D0", "endpoint": "tol", "value": 0},
            {"subject_id": "S001", "visit": "D28", "endpoint": "tol", "value": 1},
        ])
        with pytest.raises(ValueError, match="too few"):
            run_mcnemar(df, "tol", baseline="D0", timepoint="D28")


# ---------------------------------------------------------------------------
# Tests: GLMM logit
# ---------------------------------------------------------------------------

class TestGLMMLogit:
    def test_glmm_logit_runs(self) -> None:
        df = _make_binary(n=30, p_pre=0.3, p_post=0.7, seed=42)
        # Add more visits for longitudinal data
        rng = np.random.default_rng(123)
        extra_rows = []
        for s in range(1, 31):
            extra_rows.append({"subject_id": f"S{s:03d}", "visit": "D7", "endpoint": "tol",
                               "value": int(rng.random() < 0.5)})
            extra_rows.append({"subject_id": f"S{s:03d}", "visit": "D14", "endpoint": "tol",
                               "value": int(rng.random() < 0.6)})
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        res = run_glmm_logit(df, "tol", baseline="D0", primary_timepoint="D28")
        assert res["model"] == "glmm_logit_gee"
        assert res["n"] >= 25
        assert "ci95" in res
        assert res["scale"] == "odds_ratio"


# ---------------------------------------------------------------------------
# Tests: Top-2-box
# ---------------------------------------------------------------------------

class TestTop2Box:
    def test_top2box_basic(self) -> None:
        responses = [5, 4, 3, 5, 4, 4, 2, 1, 5, 3]
        res = run_top2box(responses, scale_max=5)
        assert res["n"] == 10
        assert res["top2_count"] == 6  # 5,4,5,4,4,5
        assert res["top2_pct"] == 60.0
        assert res["ci95_pct"][0] < 60.0
        assert res["ci95_pct"][1] > 60.0
        assert res["ci_method"] == "wilson"

    def test_top2box_all_top(self) -> None:
        responses = [5, 5, 5, 4, 4]
        res = run_top2box(responses, scale_max=5)
        assert res["top2_pct"] == 100.0
        assert res["ci95_pct"][1] <= 100.0

    def test_top2box_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No responses"):
            run_top2box([], scale_max=5)


# ---------------------------------------------------------------------------
# Tests: TOST
# ---------------------------------------------------------------------------

class TestTOST:
    def test_tost_equivalence_met(self) -> None:
        """Near-zero difference with large margin → equivalence declared."""
        rng = np.random.default_rng(42)
        x = rng.normal(50, 3, size=30)
        y = x + rng.normal(0, 0.5, size=30)  # very small difference
        res = run_tost(x, y, margin=5.0, paired=True)
        assert res["model"] == "TOST"
        assert res["equivalence_met"] is True
        assert res["tost_p_max"] < 0.05
        assert res["ci90"][0] > -5.0
        assert res["ci90"][1] < 5.0

    def test_tost_equivalence_not_met(self) -> None:
        """Large difference → equivalence NOT declared."""
        rng = np.random.default_rng(42)
        x = rng.normal(50, 3, size=30)
        y = x + rng.normal(10, 2, size=30)  # big offset
        res = run_tost(x, y, margin=2.0, paired=True)
        assert res["equivalence_met"] is False

    def test_tost_unpaired(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(50, 3, size=30)
        y = rng.normal(50.5, 3, size=35)
        res = run_tost(x, y, margin=5.0, paired=False)
        assert res["paired"] is False
        assert res["n"] == 65

    def test_tost_length_mismatch_paired_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            run_tost([1, 2, 3], [1, 2], margin=1.0, paired=True)


# ---------------------------------------------------------------------------
# Tests: Poisson / NegBin
# ---------------------------------------------------------------------------

class TestPoissonNegBin:
    def test_poisson_runs(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        for s in range(1, 26):
            rows.append({"subject_id": f"S{s:03d}", "visit": "D0", "endpoint": "count",
                          "value": int(rng.poisson(3))})
            rows.append({"subject_id": f"S{s:03d}", "visit": "D28", "endpoint": "count",
                          "value": int(rng.poisson(5))})
        df = pd.DataFrame(rows)
        res = run_poisson_or_negbin(df, "count", baseline="D0", primary_timepoint="D28")
        assert "poisson" in res["model"] or "negbin" in res["model"]
        assert res["scale"] == "rate_ratio"
        assert res["n"] >= 20
