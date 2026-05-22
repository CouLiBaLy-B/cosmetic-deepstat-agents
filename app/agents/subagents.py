"""DeepAgents sub-agent specifications.

Each subagent is a dict matching deepagents 0.6.x's SubAgent schema:
``{"name", "description", "system_prompt", "tools", "model?", "middleware?",
"interrupt_on?"}``.

Phase 4-5: all 10 sub-agents are now fully wired with their tool subsets.
"""

from __future__ import annotations

from typing import Any

from app.agents import prompts


def _by_name(tools: list[Any], names: list[str]) -> list[Any]:
    by_name = {getattr(t, "name", getattr(t, "__name__", str(t))): t for t in tools}
    return [by_name[n] for n in names if n in by_name]


def build_subagents(
    tools: list[Any],
    *,
    default_model: str | None = None,
) -> list[dict[str, Any]]:
    """Build the list of sub-agent specs for create_deep_agent(subagents=...).

    The ``tools`` argument is the full langchain tool list built by
    ``app.agents.tools.build_langchain_tools()``; we hand each sub-agent the
    narrow subset it needs.
    """

    base: dict[str, Any] = {}
    if default_model:
        base["model"] = default_model

    regulatory = {
        **base,
        "name": "regulatory_claim_mapper",
        "description": (
            "Translates marketing claims into evidentiary requirements under the "
            "relevant jurisdiction (EU/US). Produces claim_evidence_map.json."
        ),
        "system_prompt": prompts.REGULATORY_CLAIM_MAPPER_PROMPT,
        "tools": _by_name(tools, ["write_audit_event_tool"]),
    }

    data_quality = {
        **base,
        "name": "data_quality_subagent",
        "description": (
            "Inspects raw uploaded datasets, validates paired/longitudinal "
            "structure, quantifies missingness and outliers, pseudonymises "
            "subjects, and produces a cleaned analysis dataset."
        ),
        "system_prompt": prompts.DATA_QUALITY_PROMPT,
        "tools": _by_name(
            tools,
            [
                "load_dataset_tool",
                "profile_dataset_tool",
                "validate_paired_data_tool",
                "detect_missingness_tool",
                "detect_outliers_tool",
                "pseudonymize_subjects_tool",
                "hash_file_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    statistical_analysis = {
        **base,
        "name": "statistical_analysis_subagent",
        "description": (
            "Executes the approved SAP on the cleaned dataset. Chooses the right "
            "test (paired t / Wilcoxon / MMRM / GLMM …), writes reproducible "
            "scripts, persists StatisticalResult JSON per endpoint."
        ),
        "system_prompt": prompts.STATISTICAL_ANALYSIS_PROMPT,
        "tools": _by_name(
            tools,
            [
                "load_dataset_tool",
                "choose_statistical_test_tool",
                "run_paired_test_tool",
                "run_mmrm_tool",
                "run_glmm_logit_tool",
                "run_mcnemar_tool",
                "run_tost_tool",
                "hash_file_tool",
                "record_package_versions_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    study_design = {
        **base,
        "name": "study_design_subagent",
        "description": (
            "Drafts the Statistical Analysis Plan (SAP) from claims + qc report. "
            "Requires HUMAN APPROVAL before unlocking confirmatory analysis."
        ),
        "system_prompt": prompts.STUDY_DESIGN_PROMPT,
        "tools": _by_name(
            tools, ["choose_statistical_test_tool", "request_human_approval_tool"]
        ),
    }

    multiplicity = {
        **base,
        "name": "multiplicity_claim_subagent",
        "description": (
            "Applies the SAP multiplicity correction to statistical results and "
            "decides whether each claim is confirmed / partial / exploratory / "
            "not supported. Proposes ALLOWED and FORBIDDEN wording."
        ),
        "system_prompt": prompts.MULTIPLICITY_CLAIM_PROMPT,
        "tools": _by_name(
            tools,
            [
                "apply_multiplicity_tool",
                "request_human_approval_tool",
                "hash_file_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    safety = {
        **base,
        "name": "safety_tolerability_subagent",
        "description": (
            "Pre- and post-market safety / tolerance analysis. Triggers HUMAN "
            "APPROVAL before any safety claim wording."
        ),
        "system_prompt": prompts.SAFETY_TOLERABILITY_PROMPT,
        "tools": _by_name(
            tools,
            [
                "load_dataset_tool",
                "run_paired_test_tool",
                "run_mcnemar_tool",
                "request_human_approval_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    consumer = {
        **base,
        "name": "consumer_insight_subagent",
        "description": (
            "Analyses consumer questionnaires (top-2-box + 95% Wilson CI, "
            "Likert distributions, Cronbach α). NEVER mixes perception with "
            "instrumental evidence."
        ),
        "system_prompt": prompts.CONSUMER_INSIGHT_PROMPT,
        "tools": _by_name(
            tools,
            [
                "load_dataset_tool",
                "run_top2box_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    postmarket = {
        **base,
        "name": "postmarket_monitoring_subagent",
        "description": (
            "Post-market surveillance: complaint / AE ingestion, temporal trend, "
            "per-lot/country/channel breakdown, rule-based signal alerts."
        ),
        "system_prompt": prompts.POSTMARKET_MONITORING_PROMPT,
        "tools": _by_name(
            tools,
            [
                "load_dataset_tool",
                "request_human_approval_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    report = {
        **base,
        "name": "report_writer_subagent",
        "description": (
            "Produces the markdown reports (statistical_analysis_report, "
            "claim_substantiation_report, pif_summary, safety_report, "
            "postmarket_report, executive_summary). Final report requires "
            "HUMAN APPROVAL."
        ),
        "system_prompt": prompts.REPORT_WRITER_PROMPT,
        "tools": _by_name(
            tools,
            [
                "hash_file_tool",
                "request_human_approval_tool",
                "write_audit_event_tool",
            ],
        ),
    }

    qa = {
        **base,
        "name": "qa_auditor_subagent",
        "description": (
            "Independent QA audit: reproducibility, script existence, hash "
            "checks, claim ↔ endpoint coherence, presence of every required "
            "approval, no raw subject ID in reports."
        ),
        "system_prompt": prompts.QA_AUDITOR_PROMPT,
        "tools": _by_name(tools, ["hash_file_tool", "write_audit_event_tool"]),
    }

    return [
        regulatory,
        study_design,
        data_quality,
        statistical_analysis,
        multiplicity,
        safety,
        consumer,
        postmarket,
        report,
        qa,
    ]
