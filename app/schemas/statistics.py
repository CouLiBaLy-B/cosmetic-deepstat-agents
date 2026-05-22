"""Schemas describing endpoints and statistical results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataType(StrEnum):
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    BINARY = "binary"
    COUNT = "count"
    CATEGORICAL = "categorical"
    TIME_TO_EVENT = "time_to_event"


class MultiplicityMethod(StrEnum):
    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    HOMMEL = "hommel"
    FIXED_SEQUENCE = "fixed_sequence"
    GATEKEEPING = "gatekeeping"
    BH_FDR = "bh_fdr"
    BY_FDR = "by_fdr"


class Endpoint(BaseModel):
    """A study endpoint (primary or secondary)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    data_type: DataType
    unit: str | None = Field(None, description='Physical unit (e.g. "a.u.", "mm", "%").')
    timepoints: list[str] = Field(default_factory=list)
    primary_or_secondary: Literal["primary", "secondary", "exploratory"] = "secondary"
    practical_threshold: float | None = Field(
        None,
        description=(
            "Minimum effect size considered practically relevant "
            "(on the natural scale of the endpoint). Sign matters: "
            "negative means a decrease is the desired direction."
        ),
    )
    multiplicity_family: str | None = Field(
        None,
        description="Family identifier used by the multiplicity sub-agent.",
    )
    direction: Literal["increase", "decrease", "two_sided"] = "two_sided"


class AssumptionsCheck(BaseModel):
    """Outcome of statistical-assumption checks."""

    model_config = ConfigDict(extra="forbid")

    normality_p: float | None = None
    homoscedasticity_p: float | None = None
    sphericity_p: float | None = None
    residual_normality_p: float | None = None
    notes: list[str] = Field(default_factory=list)
    overall_ok: bool = True


class StatisticalResult(BaseModel):
    """One row of statistical output for a single endpoint / contrast."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str
    data_type: DataType
    model: str = Field(..., description='e.g. "paired_t", "wilcoxon", "MMRM", "GEE_logit"')
    contrast: str = Field(..., description='e.g. "D28 - D0", "active vs vehicle"')

    estimate: float
    ci95: tuple[float, float] = Field(..., description="(low, high) 95% confidence interval")
    p_value: float = Field(..., ge=0.0, le=1.0)
    p_adjusted: float | None = Field(None, ge=0.0, le=1.0)
    p_adjustment_method: MultiplicityMethod = MultiplicityMethod.NONE

    effect_size: float | None = None
    effect_size_metric: str | None = Field(
        None, description='e.g. "cohen_d", "cohen_dz", "odds_ratio", "rate_ratio"'
    )

    practical_threshold: float | None = None
    practical_threshold_met: bool | None = None

    n: int = Field(..., ge=0)
    n_complete: int | None = Field(None, ge=0)

    assumptions: AssumptionsCheck = Field(default_factory=AssumptionsCheck)
    sensitivity_analysis: str | None = None
    conclusion: str = Field(..., min_length=3)

    artefacts: dict[str, str] = Field(
        default_factory=dict,
        description="Filesystem paths of figures/tables/scripts that produced this row.",
    )
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ci95")
    @classmethod
    def _ordered_ci(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo > hi:
            raise ValueError(f"CI95 lower bound {lo} > upper bound {hi}")
        return v
