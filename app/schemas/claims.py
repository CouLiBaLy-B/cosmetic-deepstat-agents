"""Schemas for marketing claims and their substantiation decisions."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Jurisdiction(StrEnum):
    EU = "EU"
    US = "US"
    UK = "UK"
    CN = "CN"
    JP = "JP"
    OTHER = "OTHER"


class ClaimType(StrEnum):
    INSTRUMENTAL = "instrumental"
    CONSUMER = "consumer"
    SAFETY = "safety"
    COMPARATIVE = "comparative"
    EQUIVALENCE = "equivalence"
    NON_INFERIORITY = "non_inferiority"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimSupportLevel(StrEnum):
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    EXPLORATORY = "exploratory"
    NOT_SUPPORTED = "not_supported"


class Claim(BaseModel):
    """A single marketing claim attached to a study/product."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    text: str = Field(..., min_length=3, max_length=500)
    jurisdiction: Jurisdiction = Jurisdiction.EU
    claim_type: ClaimType
    product_id: str | None = None
    study_id: str | None = None
    status: Literal["proposed", "mapped", "decided", "approved", "rejected"] = "proposed"


class ClaimEvidenceMap(BaseModel):
    """Output of ``regulatory_claim_mapper`` for a single claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    jurisdiction: Jurisdiction
    claim_type: ClaimType
    risk_level: RiskLevel = RiskLevel.MEDIUM
    required_evidence: list[str] = Field(default_factory=list)
    primary_endpoint: str | None = None
    secondary_endpoints: list[str] = Field(default_factory=list)
    forbidden_wording: list[str] = Field(default_factory=list)
    allowed_wording_conditions: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    rationale: str | None = None


class ClaimDecision(BaseModel):
    """Final decision on whether a claim is supported by the analysis."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    supported: bool
    support_level: ClaimSupportLevel
    statistical_basis: dict[str, object] = Field(default_factory=dict)
    allowed_wording: str | None = None
    forbidden_wording: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    human_approval_required: bool = True
    human_approval_id: str | None = None
