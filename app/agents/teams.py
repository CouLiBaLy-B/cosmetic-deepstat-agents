"""Nested (hierarchical) sub-agent topology for token optimisation.

Problem
-------
In the *flat* topology (``build_subagents``) the master ``create_deep_agent``
receives **all** custom tools (``tools=tools``) **and** the description of
**all 10** leaf sub-agents. Every master LLM turn therefore carries ~18 tool
JSON schemas + 10 sub-agent descriptions in its input — most of which are
irrelevant to the step at hand.

Solution (deepagents 0.6.x nested agents)
-----------------------------------------
deepagents lets any compiled graph — including another ``create_deep_agent``
graph — act as a sub-agent through ``CompiledSubAgent``
(``subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent]``).

We regroup the 10 specialists into **4 team-lead deep agents**. Each team lead:

* owns only the *union of its members'* tools (a narrow slice of the 18),
* declares only its 2-3 members as sub-agents,
* is exposed to the master as a single ``CompiledSubAgent``.

Resulting context budget per LLM turn:

* **master**: 0 custom tools + 4 team descriptions  (was 18 tools + 10 descs)
* **each team lead**: ~3-7 tools + 2-3 member descriptions
* **each leaf**: unchanged (its own narrow tools)

Because sub-agents run with an **isolated context window** and only their
*final summary* is returned to the parent as a ``ToolMessage``, the heavy
intermediate tool traffic (dataset profiles, stat dumps) never reaches the
master's context at all. This is context isolation *by construction*.

This module stays import-safe when ``deepagents`` is NOT installed: the team
*specifications* are plain data (testable in mock mode); the actual
``create_deep_agent`` compilation + ``CompiledSubAgent`` wrapping happens
lazily in :func:`build_nested_subagents`, which the master factory only calls
on the real-LLM path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents import prompts
from app.agents.subagents import build_subagents
from app.core.logging import get_logger

logger = get_logger("app.agents.teams")


@dataclass(frozen=True)
class TeamDefinition:
    """Declarative description of one team-lead deep agent."""

    name: str
    description: str
    system_prompt: str
    # Leaf sub-agent names (must match ``build_subagents`` names) coordinated
    # by this team, in execution order.
    members: tuple[str, ...]
    # Custom tool names owned by the team lead. Kept as the *union* of the
    # members' tool needs so the leaves can inherit / re-declare them.
    tool_names: tuple[str, ...]


# ---------------------------------------------------------------------------
# The 4-team partition of the 10 specialists.
# tool_names is the union of the member sub-agents' custom tools (see
# app/agents/subagents.py), so a team lead never carries a tool none of its
# members can use.
# ---------------------------------------------------------------------------
TEAM_DEFINITIONS: tuple[TeamDefinition, ...] = (
    TeamDefinition(
        name="protocol_team",
        description=(
            "Upstream protocol team: maps marketing claims to evidence "
            "requirements and drafts the Statistical Analysis Plan (SAP). "
            "Owns the SAP human-approval gate. Delegate here FIRST."
        ),
        system_prompt=prompts.PROTOCOL_TEAM_PROMPT,
        members=("regulatory_claim_mapper", "study_design_subagent"),
        tool_names=(
            "write_audit_event_tool",
            "choose_statistical_test_tool",
            "request_human_approval_tool",
        ),
    ),
    TeamDefinition(
        name="evidence_team",
        description=(
            "Evidence-generation team: data quality/cleaning, confirmatory "
            "statistical analysis of the approved SAP, and (separately) "
            "consumer perception analysis. Delegate here AFTER SAP approval."
        ),
        system_prompt=prompts.EVIDENCE_TEAM_PROMPT,
        members=(
            "data_quality_subagent",
            "statistical_analysis_subagent",
            "consumer_insight_subagent",
        ),
        tool_names=(
            "load_dataset_tool",
            "profile_dataset_tool",
            "validate_paired_data_tool",
            "detect_missingness_tool",
            "detect_outliers_tool",
            "pseudonymize_subjects_tool",
            "hash_file_tool",
            "choose_statistical_test_tool",
            "run_paired_test_tool",
            "run_mmrm_tool",
            "run_glmm_logit_tool",
            "run_mcnemar_tool",
            "run_tost_tool",
            "run_top2box_tool",
            "record_package_versions_tool",
            "write_audit_event_tool",
        ),
    ),
    TeamDefinition(
        name="decision_safety_team",
        description=(
            "Decision & safety team: applies multiplicity correction and "
            "decides each claim, runs pre/post-market safety and surveillance. "
            "Owns the claim-wording and safety-conclusion approval gates."
        ),
        system_prompt=prompts.DECISION_SAFETY_TEAM_PROMPT,
        members=(
            "multiplicity_claim_subagent",
            "safety_tolerability_subagent",
            "postmarket_monitoring_subagent",
        ),
        tool_names=(
            "apply_multiplicity_tool",
            "load_dataset_tool",
            "run_paired_test_tool",
            "run_mcnemar_tool",
            "hash_file_tool",
            "request_human_approval_tool",
            "write_audit_event_tool",
        ),
    ),
    TeamDefinition(
        name="reporting_team",
        description=(
            "Reporting & QA team: writes the markdown reports (owns the final "
            "report-release approval gate) and runs the independent QA audit. "
            "Delegate here LAST."
        ),
        system_prompt=prompts.REPORTING_TEAM_PROMPT,
        members=("report_writer_subagent", "qa_auditor_subagent"),
        tool_names=(
            "hash_file_tool",
            "request_human_approval_tool",
            "write_audit_event_tool",
        ),
    ),
)


@dataclass
class TeamSpec:
    """Resolved (data-only) spec for one team, used for tests + real build.

    ``lead`` is a plain dict shaped like a deepagents ``SubAgent`` (the team
    lead) and ``members`` is the list of leaf sub-agent dicts it coordinates.
    Nothing here touches deepagents, so it is fully testable in mock mode.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[Any]
    members: list[dict[str, Any]] = field(default_factory=list)


def _index_subagents(tools: list[Any]) -> dict[str, dict[str, Any]]:
    """Return the flat leaf sub-agents keyed by name."""
    return {s["name"]: s for s in build_subagents(tools)}


def _index_tools(tools: list[Any]) -> dict[str, Any]:
    return {getattr(t, "name", getattr(t, "__name__", str(t))): t for t in tools}


def build_team_specs(tools: list[Any]) -> list[TeamSpec]:
    """Resolve :data:`TEAM_DEFINITIONS` into concrete, data-only specs.

    Deterministic and deepagents-free — this is what the tests exercise and
    what :func:`build_nested_subagents` compiles on the real-LLM path.
    """
    leaves = _index_subagents(tools)
    tool_by_name = _index_tools(tools)

    specs: list[TeamSpec] = []
    for td in TEAM_DEFINITIONS:
        members = [leaves[name] for name in td.members if name in leaves]
        missing = [name for name in td.members if name not in leaves]
        if missing:
            logger.warning("team_missing_members", team=td.name, missing=missing)

        team_tools = [tool_by_name[n] for n in td.tool_names if n in tool_by_name]
        specs.append(
            TeamSpec(
                name=td.name,
                description=td.description,
                system_prompt=td.system_prompt,
                tools=team_tools,
                members=members,
            )
        )
    return specs


def _approx_tokens(text: str) -> int:
    """Cheap 4-chars-per-token approximation (no tokenizer dependency)."""
    return max(1, len(text) // 4)


def _tool_footprint_tokens(t: Any) -> int:
    """Approximate the input tokens a single tool schema costs per LLM turn.

    A tool is injected as name + description + a JSON schema of its args. We
    approximate the schema cost from the argument names/annotations.
    """
    name = getattr(t, "name", getattr(t, "__name__", ""))
    desc = getattr(t, "description", "") or ""
    schema = ""
    args = getattr(t, "args", None)
    if isinstance(args, dict):
        # rough JSON of the arg schema
        schema = "".join(f"{k}{v}" for k, v in args.items())
    return _approx_tokens(name + desc + schema) + 8  # +8 structural overhead


def _entry_footprint_tokens(name: str, description: str) -> int:
    """Approximate the input cost of one sub-agent/team entry in `task`."""
    return _approx_tokens(name + description) + 8


def estimate_context_footprint(tools: list[Any]) -> dict[str, Any]:
    """Estimate the *master-level* per-turn input-token footprint for both
    topologies, plus the per-team footprints for the nested one.

    This measures only the delegation surface (tools + sub-agent/team
    descriptions) carried in the master's context on **every** turn — the part
    the nested topology shrinks. It is an approximation, not a tokenizer.
    """
    tool_by_name = _index_tools(tools)
    leaves = _index_subagents(tools)

    # ---- Flat topology: master carries ALL tools + ALL 10 leaf descriptions.
    flat_tools = sum(_tool_footprint_tokens(t) for t in tools)
    flat_subs = sum(
        _entry_footprint_tokens(s["name"], s.get("description", ""))
        for s in leaves.values()
    )
    flat_master = flat_tools + flat_subs

    # ---- Nested topology: master carries 0 custom tools + 4 team descriptions.
    specs = build_team_specs(tools)
    nested_master = sum(
        _entry_footprint_tokens(td.name, td.description) for td in TEAM_DEFINITIONS
    )
    per_team = {}
    for spec in specs:
        t_tools = sum(_tool_footprint_tokens(t) for t in spec.tools)
        t_subs = sum(
            _entry_footprint_tokens(m["name"], m.get("description", ""))
            for m in spec.members
        )
        per_team[spec.name] = t_tools + t_subs

    saved = flat_master - nested_master
    return {
        "flat_master_tokens": flat_master,
        "flat_breakdown": {"tools": flat_tools, "subagents": flat_subs},
        "nested_master_tokens": nested_master,
        "nested_team_tokens": per_team,
        "master_tokens_saved": saved,
        "master_reduction_pct": round(100 * saved / flat_master, 1) if flat_master else 0.0,
        "n_tools": len(tool_by_name),
        "n_leaves": len(leaves),
        "n_teams": len(TEAM_DEFINITIONS),
    }


def build_nested_subagents(
    tools: list[Any],
    *,
    model: Any,
    backend: Any | None = None,
    skills: list[str] | None = None,
) -> list[Any]:
    """Build the list of ``CompiledSubAgent`` team leads for the master.

    Each team is compiled as its own ``create_deep_agent`` graph (with its
    narrow tool slice and its member sub-agents) and wrapped in a
    ``CompiledSubAgent`` so the master sees exactly one entry per team.

    Requires ``deepagents`` to be importable; raises ``ImportError`` otherwise
    so the caller can fall back to the flat topology.
    """
    from deepagents import create_deep_agent

    # CompiledSubAgent moved around across 0.6.x point releases; accept both.
    try:  # pragma: no cover - import shim
        from deepagents import CompiledSubAgent  # type: ignore
    except Exception:  # pragma: no cover - import shim
        from deepagents.middleware.subagents import CompiledSubAgent  # type: ignore

    specs = build_team_specs(tools)
    compiled: list[Any] = []
    for spec in specs:
        team_kwargs: dict[str, Any] = {
            "model": model,
            "tools": spec.tools,
            "system_prompt": spec.system_prompt,
            "subagents": spec.members,
            "name": spec.name,
        }
        if backend is not None:
            team_kwargs["backend"] = backend
        if skills:
            team_kwargs["skills"] = skills

        team_graph = create_deep_agent(**team_kwargs)
        compiled.append(
            CompiledSubAgent(
                name=spec.name,
                description=spec.description,
                runnable=team_graph,
            )
        )
        logger.info(
            "team_compiled",
            team=spec.name,
            n_tools=len(spec.tools),
            n_members=len(spec.members),
        )
    return compiled
