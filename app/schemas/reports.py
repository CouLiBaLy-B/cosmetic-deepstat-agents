"""Schemas describing generated reports."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReportType(StrEnum):
    STATISTICAL_ANALYSIS = "statistical_analysis_report"
    CLAIM_SUBSTANTIATION = "claim_substantiation_report"
    PIF_SUMMARY = "pif_summary"
    SAFETY = "safety_report"
    POSTMARKET = "postmarket_report"
    EXECUTIVE = "executive_summary"


class ReportArtifact(BaseModel):
    """A produced report file (markdown / PDF) referenced by its workspace path."""

    model_config = ConfigDict(extra="forbid")

    study_id: str
    report_type: ReportType
    path: str = Field(..., description="Path relative to workspace/{study_id}/reports/")
    sha256: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    template_id: str | None = None
    inputs: dict[str, object] = Field(default_factory=dict)
