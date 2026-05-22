"""Deterministic pipeline.

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
    6. run statistical analyses (per primary endpoint)
    7. apply multiplicity + decide claims → claim_decisions.json
    8. safety (stub for MVP)
    9. write reports
   10. QA audit
   11. request human approval of final report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson

from app.agents import tools as T
from app.core.audit import write_audit_event
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
from app.storage import db

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
        "radiance": ["radiance", "éclat", "eclat", "glow"],
        "tolerance": ["tolerance", "toléré", "tolere", "sensitiv"],
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
                    f"Mapped {c.claim_type.value} claim under {c.jurisdiction.value if hasattr(c.jurisdiction, 'value') else c.jurisdiction} "
                    f"requirements. Primary endpoint guess: {primary_ep or 'none'}."
                ),
            )
        )

    ws = StudyWorkspace(study_id).ensure()
    out_path = ws.results / "claim_evidence_map.json"
    out_path.write_bytes(
        orjson.dumps([m.model_dump(mode="json") for m in mapped], option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)
    )
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
        "ready_for_analysis": bool(paired.get("valid", False)) and bool(load["rows"] > 0),
        "blockers": [] if paired.get("valid", False) else ["paired validation failed"],
    }

    ws = StudyWorkspace(study_id)
    out_path = ws.results / "qc_report.json"
    out_path.write_bytes(orjson.dumps(qc, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))
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
                "rationale": rec["rationale"],
                "contrast": (
                    f"{ep.timepoints[-1]} - {ep.timepoints[0]}"
                    if len(ep.timepoints) >= 2
                    else "single timepoint"
                ),
                "covariates": [],
                "multiplicity_family": ep.multiplicity_family,
                "practical_threshold": ep.practical_threshold,
                "direction": ep.direction,
            }
        )

    n_primary = sum(1 for ep in endpoints_in_sap if ep["primary_or_secondary"] == "primary")
    sap = {
        "study_id": study_id,
        "population_inscope": study.population,
        "estimands": [
            {
                "endpoint": ep["name"],
                "treatment_condition": "active",
                "summary_measure": "mean change from baseline",
                "intercurrent_event_strategy": "treatment policy",
                "population": study.population,
            }
            for ep in endpoints_in_sap
        ],
        "endpoints": endpoints_in_sap,
        "multiplicity_strategy": {
            "method": "holm" if n_primary > 1 else "none",
            "families": {
                ep["multiplicity_family"] or "default": [ep["name"]] for ep in endpoints_in_sap
            },
        },
        "missing_data_strategy": {
            "primary": "MMRM_MAR" if any(len(ep.timepoints) > 2 for ep in study.endpoints) else "complete_case",
            "sensitivity_analyses": ["tipping_point"],
        },
        "equivalence_margins": {},
        "sample_size_justification": (
            "Sample size derived from historical effect sizes "
            "(see memories/products/{product_id}/historical_effect_sizes.md)."
        ),
        "human_approval_required": True,
    }

    ws = StudyWorkspace(study_id)
    out_path = ws.results / "sap_draft.json"
    out_path.write_bytes(orjson.dumps(sap, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))

    # Trigger HITL for SAP lock — but only once
    if not _has_approved(study_id, "sap") and not _has_pending(study_id, "sap"):
        approval = T._impl_request_human_approval(
            study_id=study_id,
            object_type="sap",
            object_id=f"{study_id}-sap",
            reason="Statistical Analysis Plan draft requires human approval before any confirmatory analysis.",
            payload={"sap_path": "results/sap_draft.json"},
        )
        approval_id = approval["approval_id"]
    else:
        approval_id = None
    return {"step": "draft_sap", "approval_id": approval_id, "n_endpoints": len(endpoints_in_sap)}


# ---------------------------------------------------------------------------
# Step 6 — statistical_analysis_subagent (only if SAP is locked)
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
    for a in db.approvals().list():
        if a.study_id == study_id and a.object_type == "sap" and a.status.value == "approved":
            return True
    return False


def step_run_analyses(study_id: str) -> dict[str, Any]:
    if not _sap_is_locked(study_id):
        raise PermissionError(
            f"Refusing to run confirmatory analyses for study {study_id}: SAP is not locked. "
            "Approve the SAP via POST /api/approvals/{id} (decision=approved) first."
        )

    study = db.studies().get(study_id)
    assert study is not None
    ws = StudyWorkspace(study_id)
    clean_path = ws.clean / "analysis_dataset.parquet"
    if not clean_path.exists():
        raise FileNotFoundError("Clean analysis dataset not found. Run QC first.")

    results: list[dict[str, Any]] = []
    rel_clean = "clean/analysis_dataset.parquet"
    for ep in study.endpoints:
        if len(ep.timepoints) < 2:
            continue
        baseline = ep.timepoints[0]
        timepoint = ep.timepoints[-1]
        try:
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
            results.append(
                {
                    "endpoint": ep.name,
                    "error": str(exc),
                }
            )

    out_path = ws.results / "statistical_results.json"
    out_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    T._impl_record_package_versions(study_id)
    write_audit_event(
        actor="pipeline:run_analyses",
        action="stats.batch",
        study_id=study_id,
        metadata={"n_endpoints": len(results)},
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

    # Group results by their endpoint name → take first occurrence for MVP
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
        if p_adj is not None and p_adj < 0.05 and practical_met is True:
            level = ClaimSupportLevel.CONFIRMED
            allowed = (
                f"After {r.get('extras', {}).get('timepoint', '')} of use, "
                f"{ep_name} {('decreased' if result_for_ep['estimate'] < 0 else 'increased')} "
                f"by {result_for_ep['estimate']:+.2g} (95% CI {result_for_ep['ci95'][0]:.2g}-{result_for_ep['ci95'][1]:.2g}; "
                f"adjusted p={p_adj:.3g}; n={result_for_ep['n']}; instrument-based)."
            )
            supported = True
        elif p_adj is not None and p_adj < 0.05:
            level = ClaimSupportLevel.PARTIAL
            allowed = "Statistically significant but practical relevance threshold not met."
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
                    "estimate": result_for_ep["estimate"],
                    "ci95": result_for_ep["ci95"],
                    "p_value": result_for_ep["p_value"],
                    "p_adjusted": p_adj,
                    "p_adjustment_method": result_for_ep.get("p_adjustment_method", "none"),
                    "n": result_for_ep["n"],
                },
                allowed_wording=allowed,
                forbidden_wording=cm.get("forbidden_wording", []),
                limitations=[] if supported else ["Insufficient or non-significant evidence."],
                human_approval_required=True,
            )
        )

    out_path = ws.results / "claim_decisions.json"
    out_path.write_bytes(
        orjson.dumps([d.model_dump(mode="json") for d in decisions], option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)
    )

    # Trigger HITL for the final claim wording — once
    if not _has_approved(study_id, "claim_wording") and not _has_pending(study_id, "claim_wording"):
        approval = T._impl_request_human_approval(
            study_id=study_id,
            object_type="claim_wording",
            object_id=f"{study_id}-claims",
            reason="Final claim wording requires human approval before any external release.",
            payload={"decisions_path": "results/claim_decisions.json", "n_claims": len(decisions)},
        )
        approval_id = approval["approval_id"]
    else:
        approval_id = None

    write_audit_event(
        actor="pipeline:decide_claims",
        action="claims.decided",
        study_id=study_id,
        metadata={"n_claims": len(decisions)},
    )
    return {
        "step": "decide_claims",
        "approval_id": approval_id,
        "n_claims": len(decisions),
    }


# ---------------------------------------------------------------------------
# Step 9 — report writer (minimal markdown)
# ---------------------------------------------------------------------------

def step_write_reports(study_id: str) -> dict[str, Any]:
    ws = StudyWorkspace(study_id)
    study = db.studies().get(study_id)
    assert study is not None

    map_path = ws.results / "claim_evidence_map.json"
    res_path = ws.results / "statistical_results.json"
    dec_path = ws.results / "claim_decisions.json"

    cmap = json.loads(map_path.read_text()) if map_path.exists() else []
    sres = json.loads(res_path.read_text()) if res_path.exists() else []
    dec = json.loads(dec_path.read_text()) if dec_path.exists() else []

    md_lines = [
        f"# Statistical Analysis Report — {study.study_id}",
        "",
        f"- **Product:** {study.product_id}",
        f"- **Design:** {study.design_type.value}",
        f"- **Population:** {study.population}",
        f"- **Visits:** {', '.join(study.visits)}",
        f"- **Jurisdiction:** {study.jurisdiction}",
        "",
        "## Endpoints analysed",
        "",
    ]
    for r in sres:
        if "error" in r:
            md_lines.append(f"- **{r.get('endpoint', '?')}** — ERROR: {r['error']}")
            continue
        md_lines.append(
            f"- **{r['endpoint']}** ({r['model']}, contrast `{r['contrast']}`): "
            f"estimate = {r['estimate']:+.3g}, 95% CI [{r['ci95'][0]:.3g}, {r['ci95'][1]:.3g}], "
            f"p = {r['p_value']:.3g}"
            + (f", adj p = {r['p_adjusted']:.3g}" if r.get('p_adjusted') is not None else "")
            + (f", n = {r['n']}.")
        )
    md_lines += [
        "",
        "## Claim substantiation",
        "",
    ]
    for d in dec:
        md_lines.append(
            f"- **{d['claim_id']}** — {d['support_level'].upper()}. "
            + (f"_Allowed wording_: {d['allowed_wording']}" if d.get("allowed_wording") else "_No wording allowed_.")
        )
    md_lines += [
        "",
        "## Limitations",
        "",
        "- Reported results are conditional on the locked SAP. "
        "Any deviation from the SAP must be documented in audit/.",
        "- Multiplicity correction: Holm across primary endpoints.",
        "- Missing data handled under MAR; sensitivity analyses pending in future iteration.",
        "",
    ]
    stat_md = ws.reports / "statistical_analysis_report.md"
    stat_md.write_text("\n".join(md_lines))

    exec_md = ws.reports / "executive_summary.md"
    exec_md.write_text(
        f"# Executive summary — {study.study_id}\n\n"
        f"- Claims analysed: {len(cmap)}\n"
        f"- Endpoints with results: {sum(1 for r in sres if 'error' not in r)}\n"
        f"- Claims confirmed: {sum(1 for d in dec if d.get('support_level') == 'confirmed')}\n"
        f"- Claims partial: {sum(1 for d in dec if d.get('support_level') == 'partial')}\n"
        f"- Claims not supported: {sum(1 for d in dec if d.get('support_level') == 'not_supported')}\n"
    )

    # Trigger HITL on final report release — once
    if not _has_approved(study_id, "final_report") and not _has_pending(study_id, "final_report"):
        approval = T._impl_request_human_approval(
            study_id=study_id,
            object_type="final_report",
            object_id=f"{study_id}-final-report",
            reason="Final report release requires human approval.",
            payload={"report_paths": ["reports/statistical_analysis_report.md", "reports/executive_summary.md"]},
        )
        approval_id = approval["approval_id"]
    else:
        approval_id = None

    write_audit_event(
        actor="pipeline:write_reports",
        action="reports.written",
        study_id=study_id,
        metadata={"reports": ["statistical_analysis_report.md", "executive_summary.md"]},
    )
    return {
        "step": "write_reports",
        "approval_id": approval_id,
        "reports": ["statistical_analysis_report.md", "executive_summary.md"],
    }


# ---------------------------------------------------------------------------
# Step 10 — QA audit
# ---------------------------------------------------------------------------

def step_qa_audit(study_id: str) -> dict[str, Any]:
    ws = StudyWorkspace(study_id)
    checks: dict[str, bool] = {}
    issues: list[str] = []

    must_exist: dict[str, Path] = {
        "claim_evidence_map.json": ws.results / "claim_evidence_map.json",
        "qc_report.json": ws.results / "qc_report.json",
        "sap_draft.json": ws.results / "sap_draft.json",
        "statistical_results.json": ws.results / "statistical_results.json",
        "claim_decisions.json": ws.results / "claim_decisions.json",
        "package_versions.json": ws.audit / "package_versions.json",
        "analysis_dataset.parquet": ws.clean / "analysis_dataset.parquet",
    }
    for name, p in must_exist.items():
        checks[f"exists:{name}"] = p.exists()
        if not p.exists():
            issues.append(f"Missing artefact: {name}")

    # Every recorded result must reference an endpoint that exists in the study
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

    # Every approval object must be either approved or still pending
    pending_blocking: list[str] = []
    for a in db.approvals().list():
        if a.study_id == study_id and a.status.value == "pending":
            pending_blocking.append(a.approval_id)
    checks["all_approvals_decided"] = len(pending_blocking) == 0
    if pending_blocking:
        issues.append(f"Pending approvals: {pending_blocking}")

    qa = {
        "study_id": study_id,
        "passed": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "pending_approvals": pending_blocking,
    }
    out_path = ws.audit / "qa_audit_report.json"
    out_path.write_bytes(orjson.dumps(qa, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))

    write_audit_event(
        actor="pipeline:qa_audit",
        action="qa.audit",
        study_id=study_id,
        metadata={"passed": qa["passed"], "n_issues": len(issues)},
    )
    return {"step": "qa_audit", "passed": qa["passed"], "issues": issues}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pipeline_deterministic(study_id: str) -> dict[str, Any]:
    """Run the pipeline up to (and including) the SAP-approval gate.

    If the SAP has been approved, continue with analyses, decisions, reports,
    QA. Returns a structured summary of what was done and what is now pending.
    """
    summary: dict[str, Any] = {"study_id": study_id, "steps": []}

    summary["steps"].append(step_map_claims(study_id))
    summary["steps"].append(step_qc_data(study_id))
    summary["steps"].append(step_draft_sap(study_id))

    if _sap_is_locked(study_id):
        summary["steps"].append(step_run_analyses(study_id))
        summary["steps"].append(step_decide_claims(study_id))
        summary["steps"].append(step_write_reports(study_id))
        summary["steps"].append(step_qa_audit(study_id))
        summary["sap_locked"] = True
    else:
        summary["sap_locked"] = False
        summary["next_action"] = (
            "Approve the SAP via POST /api/approvals/{approval_id} "
            "(decision=approved) then re-invoke the pipeline."
        )

    pending_list = [a.approval_id for a in db.approvals().list() if a.study_id == study_id and a.status.value == "pending"]
    summary["pending_approvals"] = pending_list
    return summary
