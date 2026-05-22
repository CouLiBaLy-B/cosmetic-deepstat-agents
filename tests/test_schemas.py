"""Unit tests for Pydantic schemas (Study, Endpoint, Claim, StatisticalResult)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    Claim,
    ClaimType,
    DataType,
    Endpoint,
    Jurisdiction,
    MultiplicityMethod,
    StatisticalResult,
)
from app.schemas.statistics import AssumptionsCheck
from app.schemas.study import DesignType, StudyCreate


def test_endpoint_valid() -> None:
    ep = Endpoint(
        name="corneometer_hydration",
        data_type=DataType.CONTINUOUS,
        unit="a.u.",
        timepoints=["D0", "D7", "D14", "D28"],
        primary_or_secondary="primary",
        practical_threshold=5.0,
        multiplicity_family="hydration",
    )
    assert ep.data_type is DataType.CONTINUOUS
    assert ep.timepoints == ["D0", "D7", "D14", "D28"]


def test_study_create_rejects_invalid_id() -> None:
    with pytest.raises(ValidationError):
        StudyCreate(
            study_id="bad id with spaces",
            product_id="P",
            title="t",
            design_type=DesignType.BEFORE_AFTER,
            population="women 40-60",
        )


def test_statistical_result_orders_ci() -> None:
    with pytest.raises(ValidationError):
        StatisticalResult(
            endpoint="x",
            data_type=DataType.CONTINUOUS,
            model="paired_t",
            contrast="D28 - D0",
            estimate=0.5,
            ci95=(1.0, 0.2),  # inverted
            p_value=0.04,
            n=30,
            conclusion="significant",
        )


def test_statistical_result_happy_path() -> None:
    sr = StatisticalResult(
        endpoint="corneometer_hydration",
        data_type=DataType.CONTINUOUS,
        model="paired_t",
        contrast="D28 - D0",
        estimate=6.2,
        ci95=(3.5, 8.9),
        p_value=0.0002,
        p_adjusted=0.0008,
        p_adjustment_method=MultiplicityMethod.HOLM,
        effect_size=0.78,
        effect_size_metric="cohen_dz",
        practical_threshold=5.0,
        practical_threshold_met=True,
        n=30,
        n_complete=30,
        assumptions=AssumptionsCheck(normality_p=0.21, overall_ok=True),
        sensitivity_analysis="MMRM with MAR; consistent",
        conclusion="Statistically and practically relevant improvement at D28.",
    )
    assert sr.practical_threshold_met is True
    assert sr.p_adjustment_method is MultiplicityMethod.HOLM


def test_claim_minimal() -> None:
    c = Claim(claim_id="C001", text="Hydrate pendant 24h", claim_type=ClaimType.INSTRUMENTAL)
    assert c.jurisdiction is Jurisdiction.EU
    assert c.status == "proposed"
