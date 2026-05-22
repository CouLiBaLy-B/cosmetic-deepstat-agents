"""``Study`` and ``Endpoint`` Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.statistics import Endpoint


class DesignType(StrEnum):
    BEFORE_AFTER = "before_after"
    BEFORE_AFTER_LONGITUDINAL = "before_after_longitudinal"
    SPLIT_FACE = "split_face"
    SPLIT_BODY = "split_body"
    PARALLEL_GROUPS = "parallel_groups"
    CROSSOVER = "crossover"
    POST_MARKET_SURVEILLANCE = "post_market_surveillance"
    OTHER = "other"


class StudyStatus(StrEnum):
    DRAFT = "draft"
    DATA_UPLOADED = "data_uploaded"
    CLAIMS_MAPPED = "claims_mapped"
    QC_DONE = "qc_done"
    SAP_DRAFTED = "sap_drafted"
    SAP_LOCKED = "sap_locked"
    ANALYSED = "analysed"
    CLAIMS_DECIDED = "claims_decided"
    REPORT_DRAFTED = "report_drafted"
    REPORT_RELEASED = "report_released"
    FAILED = "failed"


class StudyArm(BaseModel):
    """An arm / treatment group."""

    name: str = Field(..., description='e.g. "active", "vehicle", "untreated"')
    description: str | None = None
    sample_size_target: int | None = Field(None, ge=1)


class _StudyBase(BaseModel):
    """Shared fields between Study and StudyCreate."""

    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Stable identifier, also used as workspace directory name.",
    )
    product_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=3, max_length=300)

    design_type: DesignType
    population: str = Field(..., min_length=3, description="Free-text inclusion summary.")
    visits: list[str] = Field(default_factory=list)
    arms: list[StudyArm] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    jurisdiction: Literal["EU", "US", "UK", "CN", "JP", "OTHER"] = "EU"

    @field_validator("visits")
    @classmethod
    def _no_empty_visit(cls, v: list[str]) -> list[str]:
        if any(not s or not s.strip() for s in v):
            raise ValueError("Visit labels must be non-empty strings.")
        return v


class Study(_StudyBase):
    """Top-level metadata for a single clinical study on a cosmetic product."""

    data_paths: list[str] = Field(
        default_factory=list,
        description="Relative paths (under workspace/{study_id}/raw/) of uploaded files.",
    )
    status: StudyStatus = StudyStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StudyCreate(_StudyBase):
    """Payload accepted by ``POST /api/studies`` (same constraints as Study)."""
