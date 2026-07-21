"""Master agent factory + a CompiledMasterAgent wrapper.

We support two execution modes selected by ``settings.llm_provider``:

* ``mock`` (default): no LLM call is made. ``invoke`` runs the deterministic
  pipeline in ``app.services.pipeline.run_pipeline_deterministic`` so the
  whole product (API + workspace + audit + reports) can be exercised end-to-
  end without an API key. Production code paths (tools, schemas, filesystem,
  audit) are identical to those used by the real-LLM mode.
* ``anthropic`` / ``openai`` / ``google`` / ``azure_openai`` / ``bedrock``:
  build a real ``deepagents.create_deep_agent(...)`` graph. The deterministic
  pipeline is still callable as a fallback / sanity-check.

Audit fixes applied:
  C1 - sub-agent tools field now documented (built-ins are harness-level,
       always available regardless of the ``tools`` override).
  C2 - ``FilesystemBackend`` attached so built-in filesystem tools persist
       files to the real workspace on disk.
  C3 - ``invoke`` passes ``thread_id`` in config; ``resume`` method added
       for HITL continuation via ``Command(resume=...)``.

The factory is deliberately the only place we touch ``deepagents`` / model
constructors so the rest of the codebase remains framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents import prompts
from app.agents.subagents import build_subagents
from app.agents.tools import build_langchain_tools
from app.core.logging import get_logger
from app.core.settings import Settings, get_settings

logger = get_logger("app.agents.master")

# Maximum total size of memory files loaded into the system prompt (bytes).
# deepagents injects memory file contents verbatim; guard against accidental
# context overflow (C5 audit recommendation).
_MAX_MEMORY_BYTES = 50_000  # ~12 500 tokens at 4 chars/token


@dataclass
class CompiledMasterAgent:
    """Wraps either the real deepagents graph or the deterministic fallback."""

    mode: str  # "mock" | "deepagents"
    graph: Any | None  # the LangGraph CompiledStateGraph, or None in mock mode
    tools: list[Any]
    subagents: list[dict[str, Any]]
    settings: Settings
    _checkpointer: Any | None = field(default=None, repr=False)

    # -----------------------------------------------------------------
    # Invoke (first call or full deterministic run)
    # -----------------------------------------------------------------
    def invoke(
        self,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one pipeline iteration.

        In ``mock`` mode, ``payload`` must contain ``{"study_id": "..."}``
        and the deterministic pipeline takes over.

        In ``deepagents`` mode, ``payload`` is passed through to the graph
        as ``{"messages": [{"role": "user", "content": ...}]}``.
        A ``thread_id`` is required for state persistence and HITL; if not
        provided, one is derived from ``payload["study_id"]``.
        """
        if self.mode == "mock":
            from app.services.pipeline import run_pipeline_deterministic

            study_id = payload.get("study_id")
            if not study_id:
                raise ValueError("payload must include 'study_id' in mock mode")
            return run_pipeline_deterministic(study_id)

        # ---- Real DeepAgents path (C3 fix) ----
        assert self.graph is not None
        tid = thread_id or payload.get("study_id") or "default"
        config = {"configurable": {"thread_id": f"study-{tid}"}}

        # Wrap the payload in the messages format expected by deepagents
        if "messages" not in payload:
            study_id = payload.get("study_id", "unknown")
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Run the full cosmetic evidence pipeline for "
                            f"study {study_id}. Follow the plan: map_claims, "
                            f"qc_data, draft_sap, run_analyses, decide_claims, "
                            f"safety_analysis, write_reports, qa_audit."
                        ),
                    }
                ]
            }

        result = self.graph.invoke(payload, config=config)
        return result

    # -----------------------------------------------------------------
    # Resume after HITL interrupt (C3 fix)
    # -----------------------------------------------------------------
    def resume(
        self,
        thread_id: str,
        decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resume a paused graph after a human-in-the-loop interrupt.

        ``decisions`` is a list of dicts, one per interrupted tool call,
        in the order they were returned by the interrupt. Each dict has
        at least ``{"type": "approve"}`` or ``{"type": "reject"}`` or
        ``{"type": "edit", "args": {...}}``.

        Example::

            agent.resume("STUDY_001", decisions=[{"type": "approve"}])
        """
        if self.mode == "mock":
            # In mock mode, HITL is handled by re-invoking the pipeline
            # after the approval has been persisted via the API endpoint.
            from app.services.pipeline import run_pipeline_deterministic

            return run_pipeline_deterministic(thread_id)

        assert self.graph is not None
        from langgraph.types import Command

        config = {"configurable": {"thread_id": f"study-{thread_id}"}}
        resume_payload: Any = Command(
            resume={"decisions": decisions or [{"type": "approve"}]}
        )
        return self.graph.invoke(resume_payload, config=config)

    # -----------------------------------------------------------------
    # Inspect graph state (for debugging / status endpoint)
    # -----------------------------------------------------------------
    def get_state(self, thread_id: str) -> Any | None:
        """Return the current LangGraph state for a thread (deepagents mode only)."""
        if self.mode == "mock" or self.graph is None:
            return None
        config = {"configurable": {"thread_id": f"study-{thread_id}"}}
        try:
            return self.graph.get_state(config)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_model(settings: Settings) -> str:
    """Return the model spec string passed to deepagents.create_deep_agent."""
    if settings.llm_provider == "mock":
        return "mock"
    if ":" in settings.llm_model:
        return settings.llm_model
    return f"{settings.llm_provider}:{settings.llm_model}"


def _collect_memory_files(settings: Settings) -> list[str]:
    """Collect memory .md files with a total size guard (C5 fix)."""
    if not settings.memory_root_abs.exists():
        return []

    candidates = sorted(settings.memory_root_abs.rglob("*.md"))
    selected: list[str] = []
    total = 0
    for p in candidates:
        if not p.is_file():
            continue
        size = p.stat().st_size
        if total + size > _MAX_MEMORY_BYTES:
            logger.warning(
                "memory_file_skipped",
                path=str(p),
                reason=f"total would exceed {_MAX_MEMORY_BYTES} bytes",
            )
            continue
        selected.append(str(p))
        total += size

    return selected


def build_master_agent(settings: Settings | None = None) -> CompiledMasterAgent:
    """Build the master agent (real or mock) according to settings."""
    settings = settings or get_settings()
    tools = build_langchain_tools()
    subagents = build_subagents(tools)

    if settings.llm_provider == "mock":
        logger.info(
            "build_master_agent.mock",
            n_tools=len(tools),
            n_subagents=len(subagents),
        )
        return CompiledMasterAgent(
            mode="mock",
            graph=None,
            tools=tools,
            subagents=subagents,
            settings=settings,
        )

    # ---- Real DeepAgents path ----
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except Exception as exc:  # pragma: no cover
        logger.error(
            "deepagents import failed; falling back to mock", error=str(exc)
        )
        return CompiledMasterAgent(
            mode="mock",
            graph=None,
            tools=tools,
            subagents=subagents,
            settings=settings,
        )

    # C2 fix: attach a FilesystemBackend so built-in tools
    # (write_file, read_file, etc.) persist to the real workspace.
    backend = FilesystemBackend(root_dir=str(settings.workspace_root_abs))

    # ---- Topology selection (token optimisation) ----
    # In "nested" mode the master delegates to 4 team-lead deep agents instead
    # of 10 flat sub-agents, and carries NO custom tools of its own — each team
    # owns its narrow tool slice. This shrinks the master's per-turn input
    # context (see docs/nested_agents_token_optimization.md).
    master_tools: list[Any] = tools
    master_subagents: list[Any] = subagents
    master_prompt = prompts.MASTER_SYSTEM_PROMPT
    topology = settings.agent_topology
    if topology == "nested":
        try:
            from app.agents.teams import build_nested_subagents

            master_subagents = build_nested_subagents(
                tools,
                model=_resolve_model(settings),
                backend=backend,
                skills=[str(settings.skills_root_abs)],
            )
            master_tools = []  # teams own the tools; master only plans + delegates
            master_prompt = (
                prompts.MASTER_SYSTEM_PROMPT + prompts.MASTER_NESTED_ADDENDUM
            )
        except Exception as exc:
            logger.warning(
                "nested_topology_unavailable; falling back to flat",
                error=str(exc),
            )
            topology = "flat"

    # HITL: interrupt when the agent calls request_human_approval_tool.
    # ``True`` is a shortcut for {"allowed_decisions": ["approve","edit","reject"]}.
    interrupt_on: dict[str, bool] = {
        "request_human_approval_tool": True,
    }

    checkpointer = MemorySaver()

    # Collect memory files with size guard (C5 fix)
    memory_files = _collect_memory_files(settings)

    create_kwargs: dict[str, Any] = {
        "model": _resolve_model(settings),
        "tools": master_tools,
        "system_prompt": master_prompt,
        "subagents": master_subagents,
        "backend": backend,  # C2 fix
        "skills": [str(settings.skills_root_abs)],
        "memory": memory_files or None,
        "interrupt_on": interrupt_on,
        "checkpointer": checkpointer,
        "name": "cosmetic_evidence_orchestrator",
    }

    graph = create_deep_agent(**create_kwargs)
    logger.info(
        "build_master_agent.real",
        provider=settings.llm_provider,
        model=create_kwargs["model"],
        topology=topology,
        n_master_tools=len(master_tools),
        n_subagents=len(master_subagents),
        n_memory_files=len(memory_files),
    )
    return CompiledMasterAgent(
        mode="deepagents",
        graph=graph,
        tools=tools,
        subagents=master_subagents,
        settings=settings,
        _checkpointer=checkpointer,
    )
