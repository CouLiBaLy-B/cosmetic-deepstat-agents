"""Master agent factory + a CompiledMasterAgent wrapper.

We support three execution modes selected by ``settings.llm_provider``:

* ``mock`` (default): no LLM call is made. ``invoke`` runs the deterministic
  pipeline in ``app.services.pipeline.run_pipeline_deterministic`` so the
  whole product (API + workspace + audit + reports) can be exercised end-to-
  end without an API key. Production code paths (tools, schemas, filesystem,
  audit) are identical to those used by the real-LLM mode.
* ``anthropic`` / ``openai`` / ``google`` / ``azure_openai`` / ``bedrock``:
  build a real ``deepagents.create_deep_agent(...)`` graph. The deterministic
  pipeline is still callable as a fallback / sanity-check.

The factory is deliberately the only place we touch ``deepagents`` / model
constructors so the rest of the codebase remains framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents import prompts
from app.agents.subagents import build_subagents
from app.agents.tools import build_langchain_tools
from app.core.logging import get_logger
from app.core.settings import Settings, get_settings

logger = get_logger("app.agents.master")


@dataclass
class CompiledMasterAgent:
    """Wraps either the real deepagents graph or the deterministic fallback."""

    mode: str  # "mock" | "deepagents"
    graph: Any | None  # the LangGraph CompiledStateGraph, or None in mock mode
    tools: list[Any]
    subagents: list[dict[str, Any]]
    settings: Settings

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one pipeline iteration.

        In ``mock`` mode, ``payload`` must contain ``{"study_id": "..."}``
        and the deterministic pipeline takes over.

        In ``deepagents`` mode, ``payload`` is passed through to the graph
        as ``{"messages": [{"role": "user", "content": ...}]}``.
        """
        if self.mode == "mock":
            from app.services.pipeline import run_pipeline_deterministic

            study_id = payload.get("study_id")
            if not study_id:
                raise ValueError("payload must include 'study_id' in mock mode")
            return run_pipeline_deterministic(study_id)

        # Real DeepAgents path
        assert self.graph is not None
        return self.graph.invoke(payload)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_model(settings: Settings) -> str:
    """Return the model spec string passed to deepagents.create_deep_agent."""
    # We always pass `provider:model` so deepagents calls init_chat_model.
    if settings.llm_provider == "mock":
        return "mock"  # not actually used; mock mode skips graph creation
    if ":" in settings.llm_model:
        return settings.llm_model
    return f"{settings.llm_provider}:{settings.llm_model}"


def build_master_agent(settings: Settings | None = None) -> CompiledMasterAgent:
    """Build the master agent (real or mock) according to settings."""
    settings = settings or get_settings()
    tools = build_langchain_tools()
    subagents = build_subagents(tools)

    if settings.llm_provider == "mock":
        logger.info("build_master_agent.mock", n_tools=len(tools), n_subagents=len(subagents))
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
        from langgraph.checkpoint.memory import MemorySaver
    except Exception as exc:  # pragma: no cover - happens only without deps
        logger.error("deepagents import failed; falling back to mock", error=str(exc))
        return CompiledMasterAgent(
            mode="mock",
            graph=None,
            tools=tools,
            subagents=subagents,
            settings=settings,
        )

    interrupt_on = {
        # Whenever the agent calls one of these tools, the run pauses for HITL.
        "request_human_approval_tool": True,
    }

    create_kwargs: dict[str, Any] = {
        "model": _resolve_model(settings),
        "tools": tools,
        "system_prompt": prompts.MASTER_SYSTEM_PROMPT,
        "subagents": subagents,
        "skills": [str(settings.skills_root_abs)],
        "interrupt_on": interrupt_on,
        "checkpointer": MemorySaver(),
        "name": "cosmetic_evidence_orchestrator",
    }

    # Optional: long-term memory files if present.
    memory_files = [
        p for p in settings.memory_root_abs.rglob("*.md") if p.is_file()
    ]
    if memory_files:
        create_kwargs["memory"] = [str(p) for p in memory_files]

    graph = create_deep_agent(**create_kwargs)
    logger.info(
        "build_master_agent.real",
        provider=settings.llm_provider,
        model=create_kwargs["model"],
        n_tools=len(tools),
        n_subagents=len(subagents),
        n_memory_files=len(memory_files),
    )
    return CompiledMasterAgent(
        mode="deepagents",
        graph=graph,
        tools=tools,
        subagents=subagents,
        settings=settings,
    )
