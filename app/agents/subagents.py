"""DeepAgents sub-agent specifications.

Each subagent is a dict matching deepagents 0.6.x's SubAgent schema:
``{"name", "description", "system_prompt", "tools?", "model?", "middleware?",
"interrupt_on?", "skills?", "permissions?"}``.

IMPORTANT (C1 audit fix): when ``tools`` is specified on a sub-agent,
deepagents **replaces** the parent's tools entirely — including the built-in
filesystem tools (``write_file``, ``read_file``, ``edit_file``, ``ls``,
``glob``, ``grep``) and the planning tools (``write_todos``, ``read_todos``).

Strategy:
  - Sub-agents that need filesystem access MUST either omit ``tools``
    (inherit everything from the parent) or explicitly include the built-in
    tool names alongside our custom tools.
  - deepagents automatically injects built-in tools by name; we only need to
    list our *custom* tools if we also list built-in names.
  - Sub-agents that are purely LLM-reasoning (no custom tools needed beyond
    the built-in ones) omit ``tools`` entirely.
"""

from __future__ import annotations

from typing import Any

from app.agents import prompts

# The built-in tool names that deepagents 0.6.x injects automatically.
# When we override ``tools`` on a sub-agent we must re-include whichever
# built-ins the sub-agent needs.
_BUILTIN_FS_TOOLS = [
    "write_file",
    "read_file",
    "edit_file",
    "ls",
    "glob",
    "grep",
]
_BUILTIN_PLANNING_TOOLS = [
    "write_todos",
    "read_todos",
]
_BUILTIN_ALL = _BUILTIN_FS_TOOLS + _BUILTIN_PLANNING_TOOLS


def _by_name(tools: list[Any], names: list[str]) -> list[Any]:
    """Pick tools from the master tool list by their langchain name."""
    by_name = {getattr(t, "name", getattr(t, "__name__", str(t))): t for t in tools}
    return [by_name[n] for n in names if n in by_name]


def _custom_plus_builtins(
    tools: list[Any],
    custom_names: list[str],
    *,
    need_fs: bool = True,
    need_planning: bool = False,
) -> list[Any] | None:
    """Return custom tools + the built-in tool *objects* the sub-agent needs.

    deepagents injects built-in tools by creating its own tool objects, so
    we cannot look them up in our ``tools`` list. Instead, when a sub-agent
    needs built-ins AND custom tools, we pass our custom tools and let
    deepagents merge them with the built-ins.

    However, per deepagents 0.6.x semantics: specifying ``tools`` on a
    sub-agent **replaces** inherited tools. The built-in filesystem/planning
    tools are always available regardless of the ``tools`` field — they are
    injected by the harness, not inherited from the parent.

    So the correct approach is:
      - ``tools`` field = only our CUSTOM tools (not built-ins)
      - built-ins are always injected by the harness
      - parent's custom tools are NOT inherited when ``tools`` is specified

    This means we CAN safely specify a narrow ``tools`` list without losing
    built-in filesystem access. The audit C1 concern was based on an
    incorrect understanding — built-in tools are harness-level, not
    parent-inherited.

    UPDATE after deeper research: the deepagents source confirms that
    built-in tools (write_file, read_file, ls, etc.) are added by the
    harness in _create_task_tool() and are ALWAYS available to sub-agents.
    Only the parent's CUSTOM tools are inherited/overridden by the
    ``tools`` field.

    We keep this helper for documentation clarity.
    """
    return _by_name(tools, custom_names) or None


def build_subagents(
    tools: list[Any],
    *,
    default_model: str | None = None,
) -> list[dict[str, Any]]:
    """Build the list of sub-agent specs for create_deep_agent(subagents=...).

    The ``tools`` argument is the full langchain tool list built by
    ``app.agents.tools.build_langchain_tools()``; we hand each sub-agent the
    narrow subset of *custom* tools it needs. Built-in tools (filesystem,
    planning) are always available via the deepagents harness.
    """

    base: dict[str, Any] = {}
    if default_model:
        base["model"] = default_model

    # ---- 1. regulatory_claim_mapper ----
    # Needs: write_file (built-in) + write_audit_event (custom)
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

    # ---- 2. study_design_subagent ----
    # Needs: read_file (built-in) + choose_test + request_approval (custom)
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

    # ---- 3. data_quality_subagent ----
    # Needs: read_file + write_file (built-in) + many custom QC tools
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

    # ---- 4. statistical_analysis_subagent ----
    # Needs: read_file (built-in) + all stat tools (custom)
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

    # ---- 5. multiplicity_claim_subagent ----
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

    # ---- 6. safety_tolerability_subagent ----
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

    # ---- 7. consumer_insight_subagent ----
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

    # ---- 8. postmarket_monitoring_subagent ----
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

    # ---- 9. report_writer_subagent ----
    # Needs: write_file (built-in, always available) + custom tools
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

    # ---- 10. qa_auditor_subagent ----
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
