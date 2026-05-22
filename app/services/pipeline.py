"""Deterministic pipeline — Phase 6+7 (full orchestration + HITL).

This module implements the **same workflow** the master DeepAgent would
orchestrate, but as straight Python. It is used:

* by the ``mock`` LLM provider so the API and the demo work without a key,
* by integration tests as a ground-truth reference,
* as a fallback if an LLM call fails (defence in depth).

Steps follow the brief's "Pipeline fonctionnel" (§11):

    1. (already done at upload) study + claims + raw data
    2. map claims → claim_evidence_map.json
    3. data QC → qc_report.json + clean/analysis_dataset.parquet
    4. draft SAP → sap_draft.json
    5. request human approval of SAP   (PAUSES here)
    6. run statistical analyses — model selection via choose_test
    7. apply multiplicity + decide claims → claim_decisions.json
    8. safety analysis → safety_report.json
    9. write reports (stat report, claim substantiation, safety, exec summary)
   10. QA audit
   11. request human approval of final report

HITL gates (Phase 7):
   - SAP lock (step 5)
   - Claim wording (step 7)
   - Safety conclusion (step 8)
   - Final report release (step 9)

Every step is **idempotent**: re-running a completed step overwrites
artefacts but does not create duplicate approval requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson

from app.agents import tools as T
from app.core.audit import hash_file, write_audit_event
from app.core.paths import StudyWorkspace
from app.schemas.claims import (
    Claim,
    ClaimDecision,
    ClaimEvidenceMap,
    ClaimSupportLevel,
    ClaimType,
    Jurisdiction,
)
from app.schemas.claims import RiskLevel as ClaimRisk
from app.schemas.study import StudyStatus
from app.storage import db

# ---------------------------------------------------------------------------
# Approval helpers
# ---------------------------------------------------------------------------


def _has_approved(study_id: str, object_type: str) -> bool:
    for a in db.approvals().list():
        if a.study_id == study_id and a.object_type == object_type and a.status.value == "approved":
            return True
    return False


def _has_pending(study_id: str, object_type: str) -> bool:
    for a in db.approvals().list():
        if a.study_id == study_id and a.object_type == object_type and a.status.value == "pending":
            return True
    return False


def _sap_is_locked(study_id: str) -> bool:
    """Return True iff a `sap` approval has been APPROVED for this study."""
    return _has_approved(study_id, "sap")


def _request_approval_once(
    study_id: str,
    object_type: str,
    object_id: str,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Create an approval request only if one doesn't already exist."""
    if _has_approved(study_id, object_type) or _has_pending(study_id, object_type):
        return None
    result = T._impl_request_human_approval(
        study_id=study_id,
        object_type=object_type,
        object_id=object_id,
        reason=reason,
        payload=payload or {},
    )
    return result["approval_id"]


def _update_study_status(study_id: str, status: StudyStatus) -> None:
    """Update the study status in the in-memory repo."""
    study = db.studies().get(study_id)
    if study is not None:
        study.status = status
        db.studies().upsert(study.study_id, study)


# ---------------------------------------------------------------------------
# Step 2 — regulatory_claim_mapper (deterministic rule-based version)
# ---------------------------------------------------------------------------

_REQUIRED_EVIDENCE_BY_TYPE: dict[ClaimType, list[str]] = {
    ClaimType.INSTRUMENTAL: [
        "controlled clinical study",
        "validated instrument",
        "pre-specified primary endpoint",
        "statistical significance with multiplicity control",
        "practical/clinical effect size threshold",
    ],
    ClaimType.CONSUMER: [
        "representative consumer panel",
        "validated questionnaire",
        "top-2-box with 95% CI",
        "wording starts with 'consumers reported/perceived'",
    ],
    ClaimType.SAFETY: [
        "dermatological evaluation",
        "adverse-event collection",
        "exposure data",
    ],
    ClaimType.COMPARATIVE: [
        "controlled head-to-head study",
        "pre-specified comparator",
        "between-group statistical test",
    ],
    ClaimType.EQUIVALENCE: [
        "pre-specified equivalence margin",
        "TOST or two one-sided CIs",
    ],
    ClaimType.NON_INFERIORITY: [
        "pre-specified non-inferiority margin",
        "one-sided CI within margin",
    ],
}

_FORBIDDEN_BY_TYPE: dict[ClaimType, list[str]] = {
    ClaimType.INSTRUMENTAL: ["medical", "treats", "cures", "anti-inflammatory drug-like"],
    ClaimType.CONSUMER: ["proven by science", "clinically proven", "scientifically measured"],
    ClaimType.SAFETY: ["100% safe", "no side effect ever", "guaranteed"],
    ClaimType.COMPARATIVE: ["best in the world", "miracle"],
    ClaimType.EQUIVALENCE: ["equivalent (without margin)", "identical"],
    ClaimType.NON_INFERIORITY: ["non-inferior (without margin)"],
}


def _guess_primary_endpoint(claim: Claim, study_endpoints: list[Any]) -> str | None:
    """Heuristic mapping from a claim wording to one of the study endpoints."""
    txt = claim.text.lower()
    keywords: dict[str, list[str]] = {
        "hydration": ["hydrate", "hydration", "moisture", "moisturis"],
        "anti_age": ["ride", "wrinkle", "anti-age", "anti-aging", "fine line"],
        "firmness": ["ferm", "firm", "elasticity"],
        "radiance": ["radiance", "eclat", "glow"],
        "tolerance": ["tolerance", "tolere", "sensitiv"],
        "sebum": ["sebum", "oily", "shine", "brillance"],
    }
    for endpoint in study_endpoints:
        name = getattr(endpoint, "name", endpoint.get("name", "") if isinstance(endpoint, dict) else "")
        family = getattr(endpoint, "multiplicity_family", None) or (
            endpoint.get("multiplicity_family") if isinstance(endpoint, dict) else None
        )
        if family and any(kw in txt for kw in keywords.get(family, [])):
            return str(name)
        if any(kw in txt for kw in name.lower().split("_")):
            return str(name)
    return None


def step_map_claims(study_id: str) -> dict[str, Any]:
    """Deterministic regulatory_claim_mapper."""
    study = db.studies().get(study_id)
    claims = [c for c in db.claims().list() if c.study_id == study_id]
    if study is None:
        raise ValueError(f"Study {study_id} not found")

    mapped: list[ClaimEvidenceMap] = []
    for c in claims:
        primary_ep = _guess_primary_endpoint(c, study.endpoints)
        risk = ClaimRisk.HIGH if c.claim_type in {ClaimType.SAFETY, ClaimType.EQUIVALENCE} else (
            ClaimRisk.MEDIUM if c.claim_type == ClaimType.COMPARATIVE else ClaimRisk.LOW
        )
        mapped.append(
            ClaimEvidenceMap(
                claim_id=c.claim_id,
                claim_text=c.text,
                jurisdiction=Jurisdiction(c.jurisdiction.value if hasattr(c.jurisdiction, "value") else c.jurisdiction),
                claim_type=c.claim_type,
                risk_level=risk,
                required_evidence=_REQUIRED_EVIDENCE_BY_TYPE.get(c.claim_type, []),
                primary_endpoint=primary_ep,
                secondary_endpoints=[],
                forbidden_wording=_FORBIDDEN_BY_TYPE.get(c.claim_type, []),
                allowed_wording_conditions=[
                    "must reference the measurement instrument and timepoint",
                    "must include sample size and study population",
                    "must not extrapolate beyond tested duration",
                ],
                human_review_required=(risk != ClaimRisk.LOW),
                rationale=(
                    f"Mapped {c.claim_type.value} claim under "
                    f"{c.jurisdiction.value if hasattr(c.jurisdiction, 'value') else c.jurisdiction} "
                    f"requirements. Primary endpoint guess: {primary_ep or 'none'}."
                ),
            )
        )

    ws = StudyWorkspace(study_id).ensure()
    out_path = ws.results / "claim_evidence_map.json"
    out_path.write_bytes(
        orjson.dumps(
            [m.model_dump(mode="json") for m in mapped],
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    _update_study_status(study_id, StudyStatus.CLAIMS_MAPPED)
    write_audit_event(
        actor="pipeline:map_claims",
        action="claims.mapped",
        study_id=study_id,
        metadata={"n_claims": len(mapped)},
    )
    return {
        "step": "map_claims",
        "n_claims": len(mapped),
        "output_path": "results/claim_evidence_map.json",
    }


# ---------------------------------------------------------------------------
# Step 3 — data_quality_subagent
# ---------------------------------------------------------------------------

def step_qc_data(study_id: str) -> dict[str, Any]:
    study = db.studies().get(study_id)
    if study is None:
        raise ValueError(f"Study {study_id} not found")
    if not study.data_paths:
        raise ValueError(f"Study {study_id} has no uploaded data files")

    primary_file = study.data_paths[0]
    rel = f"raw/{primary_file}"

    load = T._impl_load_dataset(study_id, rel)
    T._impl_profile_dataset(study_id, rel)

    expected_visits = study.visits or None
    paired = T._impl_validate_paired_data(
        study_id, rel, expected_visits=expected_visits
    )
    missing = T._impl_detect_missingness(study_id, rel)

    outliers: dict[str, Any] = {}
    if "value" in load["columns"]:
        outliers = T._impl_detect_outliers(study_id, rel, value_col="value")

    pseudo = T._impl_pseudonymize(study_id, rel)

    # Check for missingness blockers
    blockers: list[str] = []
    if not paired.get("valid", False):
        blockers.append("paired validation failed")

    # Check >20% missingness at any primary timepoint
    primary_endpoints = [ep for ep in study.endpoints if ep.primary_or_secondary == "primary"]
    per_visit = missing.get("per_visit_value", {})
    for ep in primary_endpoints:
        for tp in ep.timepoints[-1:]:  # check last (primary) timepoint
            vinfo = per_visit.get(tp, {})
            if vinfo.get("pct_missing", 0) > 20:
                blockers.append(
                    f"Primary endpoint at {tp}: >{vinfo['pct_missing']:.0f}% missing "
                    f"(threshold: 20%). Human review required."
                )

    qc = {
        "files_checked": [primary_file],
        "n_subjects": int(paired.get("n_subjects", 0) or 0),
        "n_observations": int(load["rows"]),
        "visits_present": paired.get("visits_present", []),
        "duplicates": {
            "subject_visit_pairs": int(paired.get("duplicate_pairs", 0) or 0),
        },
        "missing_pairs": paired.get("missing_pairs", []),
        "missingness_per_column": missing["per_column"],
        "missingness_per_visit_value": missing.get("per_visit_value", {}),
        "outliers_per_column": {"value": int(outliers.get("n_flagged", 0))} if outliers else {},
        "pseudonymisation_applied": True,
        "analysis_dataset_path": pseudo["analysis_dataset_path"],
        "analysis_dataset_sha256": pseudo["analysis_dataset_sha256"],
        "ready_for_analysis": bool(paired.get("valid", False)) and len(blockers) == 0,
        "blockers": blockers,
    }

    ws = StudyWorkspace(study_id)
    out_path = ws.results / "qc_report.json"
    out_path.write_bytes(
        orjson.dumps(
            qc,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    _update_study_status(study_id, StudyStatus.QC_DONE)
    write_audit_event(
        actor="pipeline:qc_data",
        action="qc.report",
        study_id=study_id,
        metadata={"n_subjects": qc["n_subjects"], "ready": qc["ready_for_analysis"]},
    )
    return {"step": "qc_data", "ready_for_analysis": qc["ready_for_analysis"]}


# ---------------------------------------------------------------------------
# Step 4 — study_design_subagent (deterministic SAP draft)
# ---------------------------------------------------------------------------

def step_draft_sap(study_id: str) -> dict[str, Any]:
    study = db.studies().get(study_id)
    if study is None:
        raise ValueError(f"Study {study_id} not found")

    endpoints_in_sap: list[dict[str, Any]] = []
    for ep in study.endpoints:
        n_t = len(ep.timepoints)
        rec = T._impl_choose_test(ep.data_type.value, study.design_type.value, n_t)
        endpoints_in_sap.append(
            {
                "name": ep.name,
                "data_type": ep.data_type.value,
                "primary_or_secondary": ep.primary_or_secondary,
                "model": rec["model"],
                "contrast": f"{ep.timepoints[-1]} - {ep.timepoints[0]}" if n_t >= 2 else "N/A",
                "covariates": ["baseline_value"] if rec["model"] in {"MMRM", "LMM"} else [],
                "multiplicity_family": ep.multiplicity_family or ep.name,
                "practical_threshold": ep.practical_threshold,
                "direction": ep.direction,
                "timepoints": ep.timepoints,
            }
        )

    # Determine multiplicity strategy
    primaries = [e for e in endpoints_in_sap if e["primary_or_secondary"] == "primary"]
    mult_method = "holm" if len(primaries) > 1 else "none"

    sap = {
        "study_id": study_id,
        "population_inscope": study.population,
        "endpoints": endpoints_in_sap,
        "multiplicity_strategy": {
            "method": mult_method,
            "families": {
                "primary": [e["name"] for e in primaries],
                "secondary": [e["name"] for e in endpoints_in_sap if e["primary_or_secondary"] != "primary"],
            },
        },
        "missing_data_strategy": {
            "primary": "MMRM_MAR" if any(e["model"] in {"MMRM", "LMM"} for e in endpoints_in_sap) else "complete_case",
            "sensitivity_analyses": ["tipping_point"],
        },
        "equivalence_margins": {},
        "sample_size_justification": (
            f"Based on {study.population}, "
            f"targeting >=80% power for the primary endpoint(s)."
        ),
        "human_approval_required": True,
    }

    ws = StudyWorkspace(study_id).ensure()
    out_path = ws.results / "sap_draft.json"
    out_path.write_bytes(
        orjson.dumps(
            sap,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    _update_study_status(study_id, StudyStatus.SAP_DRAFTED)

    # HITL gate: SAP lock
    approval_id = _request_approval_once(
        study_id=study_id,
        object_type="sap",
        object_id=f"{study_id}-sap",
        reason="Statistical Analysis Plan draft requires human approval before any confirmatory analysis.",
        payload={"sap_path": "results/sap_draft.json"},
    )
    return {"step": "draft_sap", "approval_id": approval_id, "n_endpoints": len(endpoints_in_sap)}


# ---------------------------------------------------------------------------
# Step 6 — statistical analysis (model selection via choose_test)
# ---------------------------------------------------------------------------

def step_run_analyses(study_id: str) -> dict[str, Any]:
    """Run statistical analyses, selecting the appropriate model per endpoint."""
    if not _sap_is_locked(study_id):
        raise PermissionError(
            f"Refusing to run confirmatory analyses for study {study_id}: SAP is not locked. "
            "Approve the SAP via POST /api/approvals/{{id}} (decision=approved) first."
        )

    study = db.studies().get(study_id)
    assert study is not None
    ws = StudyWorkspace(study_id)
    clean_path = ws.clean / "analysis_dataset.parquet"
    if not clean_path.exists():
        raise FileNotFoundError("Clean analysis dataset not found. Run QC first.")

    # Read SAP for model recommendations
    sap_path = ws.results / "sap_draft.json"
    sap: dict[str, Any] = {}
    if sap_path.exists():
        sap = json.loads(sap_path.read_text())
    sap_endpoints = {e["name"]: e for e in sap.get("endpoints", [])}

    results: list[dict[str, Any]] = []
    rel_clean = "clean/analysis_dataset.parquet"

    for ep in study.endpoints:
        if len(ep.timepoints) < 2:
            continue

        baseline = ep.timepoints[0]
        timepoint = ep.timepoints[-1]
        n_timepoints = len(ep.timepoints)

        # Determine the model via the decision table
        sap_ep = sap_endpoints.get(ep.name, {})
        recommended_model = sap_ep.get("model") or T._impl_choose_test(
            ep.data_type.value, study.design_type.value, n_timepoints
        )["model"]

        try:
            if recommended_model in {"MMRM", "LMM"} and n_timepoints >= 3:
                # Use MMRM for longitudinal continuous data
                res = T._impl_run_mmrm(
                    study_id=study_id,
                    rel_path=rel_clean,
                    endpoint=ep.name,
                    baseline=baseline,
                    primary_timepoint=timepoint,
                    practical_threshold=ep.practical_threshold,
                    direction=ep.direction,
                )
            elif recommended_model == "mcnemar" and ep.data_type.value == "binary":
                res = T._impl_run_mcnemar(
                    study_id=study_id,
                    rel_path=rel_clean,
                    endpoint=ep.name,
                    baseline=baseline,
                    timepoint=timepoint,
                )
                # Add practical threshold info
                res["practical_threshold"] = ep.practical_threshold
                res["practical_threshold_met"] = None
            elif recommended_model == "glmm_logit" and ep.data_type.value == "binary":
                res = T._impl_run_glmm_logit(
                    study_id=study_id,
                    rel_path=rel_clean,
                    endpoint=ep.name,
                    baseline=baseline,
                    primary_timepoint=timepoint,
                )
                res["practical_threshold"] = ep.practical_threshold
                res["practical_threshold_met"] = None
            else:
                # Default: paired test (paired-t or Wilcoxon)
                res = T._impl_run_paired_test(
                    study_id=study_id,
                    rel_path=rel_clean,
                    endpoint=ep.name,
                    baseline=baseline,
                    timepoint=timepoint,
                    practical_threshold=ep.practical_threshold,
                    direction=ep.direction,
                )
            results.append(res)
        except Exception as exc:
            results.append({"endpoint": ep.name, "error": str(exc)})

    out_path = ws.results / "statistical_results.json"
    out_path.write_bytes(
        orjson.dumps(
            results,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    T._impl_record_package_versions(study_id)
    _update_study_status(study_id, StudyStatus.ANALYSED)
    write_audit_event(
        actor="pipeline:run_analyses",
        action="stats.batch",
        study_id=study_id,
        output_hash=hash_file(out_path),
        metadata={"n_endpoints": len(results), "models_used": [r.get("model", "?") for r in results]},
    )
    return {"step": "run_analyses", "n_endpoints": len(results)}


# ---------------------------------------------------------------------------
# Step 7 — multiplicity + claim decisions
# ---------------------------------------------------------------------------

def step_decide_claims(study_id: str) -> dict[str, Any]:
    ws = StudyWorkspace(study_id)
    map_path = ws.results / "claim_evidence_map.json"
    res_path = ws.results / "statistical_results.json"
    if not map_path.exists() or not res_path.exists():
        raise FileNotFoundError("claim_evidence_map.json or statistical_results.json missing")

    cmap = json.loads(map_path.read_text())
    sres: list[dict[str, Any]] = json.loads(res_path.read_text())

    # Group results by endpoint name
    by_endpoint: dict[str, dict[str, Any]] = {}
    for r in sres:
        if "endpoint" in r and "p_value" in r:
            by_endpoint[r["endpoint"]] = r

    # Apply Holm across all primary endpoints (default multiplicity strategy)
    p_values = [r["p_value"] for r in by_endpoint.values()]
    if len(p_values) > 1:
        adj = T._impl_apply_multiplicity(p_values, method="holm", alpha=0.05)
        for (_name, r), padj in zip(by_endpoint.items(), adj["p_adjusted"], strict=False):
            r["p_adjusted"] = padj
            r["p_adjustment_method"] = "holm"
    elif p_values:
        only = next(iter(by_endpoint.values()))
        only["p_adjusted"] = only["p_value"]
        only["p_adjustment_method"] = "none"

    decisions: list[ClaimDecision] = []
    for cm in cmap:
        ep_name = cm.get("primary_endpoint")
        result_for_ep: dict[str, Any] | None = by_endpoint.get(ep_name) if ep_name else None

        if result_for_ep is None:
            decisions.append(
                ClaimDecision(
                    claim_id=cm["claim_id"],
                    claim_text=cm["claim_text"],
                    supported=False,
                    support_level=ClaimSupportLevel.NOT_SUPPORTED,
                    statistical_basis={"reason": f"no analysis for endpoint {ep_name!r}"},
                    forbidden_wording=cm.get("forbidden_wording", []),
                    limitations=["No statistical result linked to this claim."],
                    human_approval_required=True,
                )
            )
            continue

        p_adj = result_for_ep.get("p_adjusted") or result_for_ep.get("p_value")
        practical_met = result_for_ep.get("practical_threshold_met")
        estimate = result_for_ep.get("estimate", 0)
        ci95 = result_for_ep.get("ci95", [0, 0])
        n = result_for_ep.get("n", 0)

        if p_adj is not None and p_adj < 0.05 and practical_met is True:
            level = ClaimSupportLevel.CONFIRMED
            direction_word = "decreased" if estimate < 0 else "increased"
            allowed = (
                f"After use, {ep_name} {direction_word} by {estimate:+.2g} "
                f"(95% CI [{ci95[0]:.2g}, {ci95[1]:.2g}]; "
                f"adjusted p={p_adj:.3g}; n={n}; instrument-based)."
            )
            supported = True
        elif p_adj is not None and p_adj < 0.05:
            level = ClaimSupportLevel.PARTIAL
            allowed = (
                f"Statistically significant change ({estimate:+.2g}, p={p_adj:.3g}) "
                f"but practical relevance threshold not met."
            )
            supported = False
        else:
            level = ClaimSupportLevel.NOT_SUPPORTED
            allowed = None
            supported = False

        decisions.append(
            ClaimDecision(
                claim_id=cm["claim_id"],
                claim_text=cm["claim_text"],
                supported=supported,
                support_level=level,
                statistical_basis={
                    "endpoint": ep_name,
                    "model": result_for_ep["model"],
                    "estimate": estimate,
                    "ci95": ci95,
                    "p_value": result_for_ep["p_value"],
                    "p_adjusted": p_adj,
                    "p_adjustment_method": result_for_ep.get("p_adjustment_method", "none"),
                    "n": n,
                },
                allowed_wording=allowed,
                forbidden_wording=cm.get("forbidden_wording", []),
                limitations=[] if supported else ["Insufficient or non-significant evidence."],
                human_approval_required=True,
            )
        )

    out_path = ws.results / "claim_decisions.json"
    out_path.write_bytes(
        orjson.dumps(
            [d.model_dump(mode="json") for d in decisions],
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    _update_study_status(study_id, StudyStatus.CLAIMS_DECIDED)

    # HITL gate: claim wording
    approval_id = _request_approval_once(
        study_id=study_id,
        object_type="claim_wording",
        object_id=f"{study_id}-claims",
        reason="Final claim wording requires human approval before any external release.",
        payload={"decisions_path": "results/claim_decisions.json", "n_claims": len(decisions)},
    )
    write_audit_event(
        actor="pipeline:decide_claims",
        action="claims.decided",
        study_id=study_id,
        output_hash=hash_file(out_path),
        metadata={"n_claims": len(decisions)},
    )
    return {"step": "decide_claims", "approval_id": approval_id, "n_claims": len(decisions)}


# ---------------------------------------------------------------------------
# Step 8 — safety analysis
# ---------------------------------------------------------------------------

def step_safety_analysis(study_id: str) -> dict[str, Any]:
    """Analyse safety/tolerance endpoints and produce safety_report.json.

    For the deterministic pipeline this is a rule-based stub that:
    - counts AE-like endpoints (if any),
    - flags safety claims for HITL approval.
    """
    study = db.studies().get(study_id)
    assert study is not None
    ws = StudyWorkspace(study_id)

    # Identify safety endpoints (claim_type == safety in the evidence map)
    map_path = ws.results / "claim_evidence_map.json"
    safety_claims: list[dict[str, Any]] = []
    if map_path.exists():
        cmap = json.loads(map_path.read_text())
        safety_claims = [c for c in cmap if c.get("claim_type") == "safety"]

    # Produce a minimal safety report
    safety_report: dict[str, Any] = {
        "study_id": study_id,
        "summary": (
            "No safety/tolerance endpoints detected in the current dataset. "
            if not safety_claims
            else f"{len(safety_claims)} safety claim(s) identified; requires dermatological review."
        ),
        "ae_by_severity": {},
        "discontinuations": 0,
        "by_subgroup": {},
        "signals": [],
        "safety_claims": [c["claim_id"] for c in safety_claims],
        "human_approval_required": len(safety_claims) > 0,
    }

    out_path = ws.results / "safety_report.json"
    out_path.write_bytes(
        orjson.dumps(
            safety_report,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )

    # HITL gate: safety conclusion (only if safety claims exist)
    approval_id = None
    if safety_claims:
        approval_id = _request_approval_once(
            study_id=study_id,
            object_type="safety_conclusion",
            object_id=f"{study_id}-safety",
            reason="Safety/tolerance conclusion requires human review before any safety claim.",
            payload={"safety_report_path": "results/safety_report.json"},
        )

    write_audit_event(
        actor="pipeline:safety_analysis",
        action="safety.report",
        study_id=study_id,
        metadata={"n_safety_claims": len(safety_claims)},
    )
    return {"step": "safety_analysis", "approval_id": approval_id, "n_safety_claims": len(safety_claims)}


# ---------------------------------------------------------------------------
# Step 9 — report writer (full markdown reports)
# ---------------------------------------------------------------------------

def step_write_reports(study_id: str) -> dict[str, Any]:
    ws = StudyWorkspace(study_id)
    study = db.studies().get(study_id)
    assert study is not None

    map_path = ws.results / "claim_evidence_map.json"
    res_path = ws.results / "statistical_results.json"
    dec_path = ws.results / "claim_decisions.json"
    safety_path = ws.results / "safety_report.json"

    cmap = json.loads(map_path.read_text()) if map_path.exists() else []
    sres = json.loads(res_path.read_text()) if res_path.exists() else []
    dec = json.loads(dec_path.read_text()) if dec_path.exists() else []
    safety = json.loads(safety_path.read_text()) if safety_path.exists() else {}

    reports_written: list[str] = []

    # --- Statistical Analysis Report ---
    sar_lines = [
        f"# Statistical Analysis Report -- {study.study_id}",
        "",
        f"- **Product:** {study.product_id}",
        f"- **Design:** {study.design_type.value}",
        f"- **Population:** {study.population}",
        f"- **Visits:** {', '.join(study.visits)}",
        f"- **Jurisdiction:** {study.jurisdiction}",
        "",
        "## Study design",
        "",
        f"This is a {study.design_type.value} study on {study.population}, "
        f"with {len(study.endpoints)} endpoint(s) measured at visits "
        f"{', '.join(study.visits)}.",
        "",
        "## SAP summary",
        "",
        "See `results/sap_draft.json` for the full Statistical Analysis Plan.",
        "",
        "## Endpoints analysed",
        "",
    ]
    for r in sres:
        if "error" in r:
            sar_lines.append(f"- **{r.get('endpoint', '?')}** -- ERROR: {r['error']}")
            continue
        p_adj_str = f", adj p = {r['p_adjusted']:.3g}" if r.get("p_adjusted") is not None else ""
        sar_lines.append(
            f"- **{r['endpoint']}** ({r['model']}, contrast `{r['contrast']}`): "
            f"estimate = {r['estimate']:+.3g}, "
            f"95% CI [{r['ci95'][0]:.3g}, {r['ci95'][1]:.3g}], "
            f"p = {r['p_value']:.3g}{p_adj_str}, n = {r['n']}."
        )
    sar_lines += [
        "",
        "## Claim substantiation",
        "",
    ]
    for d in dec:
        wording = f"_Allowed wording_: {d['allowed_wording']}" if d.get("allowed_wording") else "_No wording allowed_."
        sar_lines.append(f"- **{d['claim_id']}** -- {d['support_level'].upper()}. {wording}")

    sar_lines += [
        "",
        "## Sensitivity analyses",
        "",
        "- Tipping-point analysis: planned for future iteration.",
        "- Pattern-mixture model: planned for future iteration.",
        "",
        "## Limitations",
        "",
        "- Reported results are conditional on the locked SAP. "
        "Any deviation from the SAP must be documented in audit/.",
        "- Multiplicity correction: Holm across primary endpoints.",
        "- Missing data handled under MAR; sensitivity analyses pending.",
        "",
    ]
    sar_path = ws.reports / "statistical_analysis_report.md"
    sar_path.write_text("\n".join(sar_lines))
    reports_written.append("statistical_analysis_report.md")

    # --- Claim Substantiation Report ---
    csr_lines = [
        f"# Claim Substantiation Report -- {study.study_id}",
        "",
        "## Claim inventory",
        "",
        "| # | Claim ID | Claim text | Jurisdiction | Type | Risk |",
        "|---|----------|------------|-------------|------|------|",
    ]
    for i, cm in enumerate(cmap, 1):
        csr_lines.append(
            f"| {i} | {cm['claim_id']} | {cm['claim_text']} | "
            f"{cm['jurisdiction']} | {cm['claim_type']} | {cm['risk_level']} |"
        )
    csr_lines += [
        "",
        "## Evidence matrix",
        "",
    ]
    for d in dec:
        csr_lines.append(f"### {d['claim_id']}: {d['claim_text']}")
        csr_lines.append(f"- **Support level:** {d['support_level']}")
        sb = d.get("statistical_basis", {})
        if "endpoint" in sb:
            csr_lines.append(f"- **Endpoint:** {sb['endpoint']}")
            csr_lines.append(f"- **Model:** {sb.get('model', 'N/A')}")
            csr_lines.append(f"- **Estimate:** {sb.get('estimate', 'N/A')}")
            csr_lines.append(f"- **Adjusted p:** {sb.get('p_adjusted', 'N/A')}")
        if d.get("allowed_wording"):
            csr_lines.append(f"- **Allowed wording:** {d['allowed_wording']}")
        if d.get("forbidden_wording"):
            csr_lines.append(f"- **Forbidden:** {', '.join(d['forbidden_wording'])}")
        csr_lines.append("")

    csr_lines += [
        "## Summary decision table",
        "",
        "| Claim ID | Support | Allowed wording | Approval |",
        "|----------|---------|-----------------|----------|",
    ]
    for d in dec:
        status = "pending" if d.get("human_approval_required") else "N/A"
        aw = (d.get("allowed_wording") or "none")[:60]
        csr_lines.append(f"| {d['claim_id']} | {d['support_level']} | {aw}... | {status} |")
    csr_lines.append("")

    csr_path = ws.reports / "claim_substantiation_report.md"
    csr_path.write_text("\n".join(csr_lines))
    reports_written.append("claim_substantiation_report.md")

    # --- Safety Report ---
    sf_lines = [
        f"# Safety Report -- {study.study_id}",
        "",
        "## Summary",
        "",
        safety.get("summary", "No safety data analysed."),
        "",
        "## Adverse events by severity",
        "",
    ]
    ae = safety.get("ae_by_severity", {})
    if ae:
        for sev, count in ae.items():
            sf_lines.append(f"- {sev}: {count}")
    else:
        sf_lines.append("- No AE data available in current dataset.")
    sf_lines += [
        "",
        f"## Discontinuations: {safety.get('discontinuations', 0)}",
        "",
        f"## Signals: {len(safety.get('signals', []))} detected",
        "",
    ]
    sf_path = ws.reports / "safety_report.md"
    sf_path.write_text("\n".join(sf_lines))
    reports_written.append("safety_report.md")

    # --- Executive Summary ---
    n_confirmed = sum(1 for d in dec if d.get("support_level") == "confirmed")
    n_partial = sum(1 for d in dec if d.get("support_level") == "partial")
    n_ns = sum(1 for d in dec if d.get("support_level") == "not_supported")

    exec_md = ws.reports / "executive_summary.md"
    exec_md.write_text(
        f"# Executive summary -- {study.study_id}\n\n"
        f"- **Product:** {study.product_id}\n"
        f"- **Design:** {study.design_type.value}\n"
        f"- **Population:** {study.population}\n\n"
        f"## Results at a glance\n\n"
        f"- Claims analysed: {len(cmap)}\n"
        f"- Endpoints with results: {sum(1 for r in sres if 'error' not in r)}\n"
        f"- Claims confirmed: {n_confirmed}\n"
        f"- Claims partial: {n_partial}\n"
        f"- Claims not supported: {n_ns}\n\n"
        f"## Safety\n\n"
        f"{safety.get('summary', 'N/A')}\n"
    )
    reports_written.append("executive_summary.md")

    _update_study_status(study_id, StudyStatus.REPORT_DRAFTED)

    # HITL gate: final report release
    approval_id = _request_approval_once(
        study_id=study_id,
        object_type="final_report",
        object_id=f"{study_id}-final-report",
        reason="Final report release requires human approval.",
        payload={"report_paths": [f"reports/{r}" for r in reports_written]},
    )
    write_audit_event(
        actor="pipeline:write_reports",
        action="reports.written",
        study_id=study_id,
        metadata={"reports": reports_written},
    )
    return {
        "step": "write_reports",
        "approval_id": approval_id,
        "reports": reports_written,
    }


# ---------------------------------------------------------------------------
# Step 10 — QA audit
# ---------------------------------------------------------------------------

def step_qa_audit(study_id: str) -> dict[str, Any]:
    ws = StudyWorkspace(study_id)
    checks: dict[str, bool] = {}
    issues: list[str] = []

    # 1. Required artefacts exist
    must_exist: dict[str, Path] = {
        "claim_evidence_map.json": ws.results / "claim_evidence_map.json",
        "qc_report.json": ws.results / "qc_report.json",
        "sap_draft.json": ws.results / "sap_draft.json",
        "statistical_results.json": ws.results / "statistical_results.json",
        "claim_decisions.json": ws.results / "claim_decisions.json",
        "safety_report.json": ws.results / "safety_report.json",
        "package_versions.json": ws.audit / "package_versions.json",
        "analysis_dataset.parquet": ws.clean / "analysis_dataset.parquet",
    }
    for name, p in must_exist.items():
        checks[f"exists:{name}"] = p.exists()
        if not p.exists():
            issues.append(f"Missing artefact: {name}")

    # 2. Every result references a known endpoint
    study = db.studies().get(study_id)
    res_path = ws.results / "statistical_results.json"
    if study and res_path.exists():
        endpoints = {ep.name for ep in study.endpoints}
        sres: list[dict[str, Any]] = json.loads(res_path.read_text())
        for r in sres:
            ep = r.get("endpoint")
            ok = bool(ep) and ep in endpoints
            checks[f"endpoint_in_study:{ep}"] = ok
            if not ok:
                issues.append(f"Result references unknown endpoint: {ep!r}")

    # 3. Every script referenced exists
    scripts_dir = ws.scripts
    if scripts_dir.exists():
        for script in scripts_dir.iterdir():
            checks[f"script_exists:{script.name}"] = script.is_file()

    # 4. Audit trail exists and is non-empty
    audit_trail = ws.audit / "audit_trail.jsonl"
    checks["audit_trail_exists"] = audit_trail.exists()
    if audit_trail.exists():
        checks["audit_trail_non_empty"] = audit_trail.stat().st_size > 0
    else:
        issues.append("Missing audit trail")

    # 5. Hash integrity: clean dataset hash matches recorded hash
    qc_path = ws.results / "qc_report.json"
    if qc_path.exists() and (ws.clean / "analysis_dataset.parquet").exists():
        qc = json.loads(qc_path.read_text())
        recorded_hash = qc.get("analysis_dataset_sha256", "")
        actual_hash = hash_file(ws.clean / "analysis_dataset.parquet")
        checks["hash_integrity:analysis_dataset"] = recorded_hash == actual_hash
        if recorded_hash != actual_hash:
            issues.append(
                f"Hash mismatch for analysis_dataset.parquet: "
                f"recorded={recorded_hash[:20]}... actual={actual_hash[:20]}..."
            )

    # 6. No raw subject IDs in reports
    for report_file in ws.reports.iterdir():
        if report_file.is_file() and report_file.suffix == ".md":
            content = report_file.read_text()
            # Check for patterns like S001, S002, etc.
            import re
            raw_ids = re.findall(r"\bS\d{3}\b", content)
            no_raw = len(raw_ids) == 0
            checks[f"no_raw_ids:{report_file.name}"] = no_raw
            if not no_raw:
                issues.append(f"Raw subject IDs found in {report_file.name}: {raw_ids[:5]}")

    # 7. All approvals decided (no pending)
    pending_blocking: list[str] = []
    for a in db.approvals().list():
        if a.study_id == study_id and a.status.value == "pending":
            pending_blocking.append(a.approval_id)
    checks["all_approvals_decided"] = len(pending_blocking) == 0
    if pending_blocking:
        issues.append(f"Pending approvals: {pending_blocking}")

    # 8. Claim decisions reference approved SAP
    checks["sap_locked"] = _sap_is_locked(study_id)
    if not _sap_is_locked(study_id):
        issues.append("SAP was not locked before analysis (integrity violation)")

    qa = {
        "study_id": study_id,
        "passed": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "pending_approvals": pending_blocking,
        "n_checks": len(checks),
        "n_passed": sum(checks.values()),
        "n_failed": sum(1 for v in checks.values() if not v),
    }
    out_path = ws.audit / "qa_audit_report.json"
    out_path.write_bytes(
        orjson.dumps(
            qa,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    )

    write_audit_event(
        actor="pipeline:qa_audit",
        action="qa.audit",
        study_id=study_id,
        output_hash=hash_file(out_path),
        metadata={"passed": qa["passed"], "n_issues": len(issues), "n_checks": len(checks)},
    )
    return {"step": "qa_audit", "passed": qa["passed"], "issues": issues}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pipeline_deterministic(study_id: str) -> dict[str, Any]:
    """Run the full 11-step pipeline.

    The pipeline is **re-entrant**: it can be called multiple times. On each
    invocation it re-runs the pre-SAP steps (idempotent) and only proceeds
    past the SAP gate if the SAP has been approved. Similarly, the claim-
    wording and final-report gates are respected.

    Returns a structured summary of what was done and what is now pending.
    """
    summary: dict[str, Any] = {"study_id": study_id, "steps": []}

    # Pre-SAP steps (always run, idempotent)
    summary["steps"].append(step_map_claims(study_id))
    summary["steps"].append(step_qc_data(study_id))
    summary["steps"].append(step_draft_sap(study_id))

    if _sap_is_locked(study_id):
        summary["sap_locked"] = True
        summary["steps"].append(step_run_analyses(study_id))
        summary["steps"].append(step_decide_claims(study_id))
        summary["steps"].append(step_safety_analysis(study_id))
        summary["steps"].append(step_write_reports(study_id))
        summary["steps"].append(step_qa_audit(study_id))
    else:
        summary["sap_locked"] = False
        summary["next_action"] = (
            "Approve the SAP via POST /api/approvals/{approval_id} "
            "(decision=approved) then re-invoke the pipeline."
        )

    pending_list = [
        a.approval_id
        for a in db.approvals().list()
        if a.study_id == study_id and a.status.value == "pending"
    ]
    summary["pending_approvals"] = pending_list
    return summary
