"""Tests for the nested (hierarchical) sub-agent topology.

All tests run in mock mode: they exercise the data-only team *specs* and the
token-footprint estimator without needing deepagents or a real LLM.
"""

from __future__ import annotations

import inspect

import pytest

from app.agents.teams import (
    TEAM_DEFINITIONS,
    build_nested_subagents,
    build_team_specs,
    estimate_context_footprint,
)


@pytest.fixture
def tools() -> list:
    try:
        from app.agents.tools import build_langchain_tools

        t = build_langchain_tools()
    except Exception:  # pragma: no cover
        pytest.skip("langchain tools not available")
    if not t:  # pragma: no cover
        pytest.skip("langchain tools not available")
    return t


# ----------------------------------------------------------------------
# Team partition covers every leaf, exactly once
# ----------------------------------------------------------------------
class TestTeamPartition:
    def test_four_teams(self) -> None:
        assert len(TEAM_DEFINITIONS) == 4

    def test_every_leaf_assigned_exactly_once(self, tools: list) -> None:
        from app.agents.subagents import build_subagents

        leaf_names = {s["name"] for s in build_subagents(tools)}
        assigned: list[str] = []
        for td in TEAM_DEFINITIONS:
            assigned.extend(td.members)
        # partition: no duplicates, full cover
        assert len(assigned) == len(set(assigned)), "a leaf is in two teams"
        assert set(assigned) == leaf_names, "teams must cover all 10 leaves"
        assert len(assigned) == 10

    def test_team_specs_resolve_members_and_tools(self, tools: list) -> None:
        specs = build_team_specs(tools)
        assert len(specs) == 4
        for spec in specs:
            assert spec.name
            assert spec.system_prompt
            assert len(spec.members) >= 2
            assert len(spec.tools) >= 1
            for m in spec.members:
                assert {"name", "description", "system_prompt"} <= set(m)

    def test_team_tools_are_union_of_member_tools(self, tools: list) -> None:
        """A team lead must own every custom tool its members declare."""
        specs = build_team_specs(tools)
        for spec in specs:
            team_tool_names = {getattr(t, "name", "") for t in spec.tools}
            for m in spec.members:
                member_tool_names = {getattr(t, "name", "") for t in m.get("tools", [])}
                missing = member_tool_names - team_tool_names
                assert not missing, (
                    f"team {spec.name!r} misses member tools {missing}"
                )


# ----------------------------------------------------------------------
# Token optimisation is real and measurable
# ----------------------------------------------------------------------
class TestTokenFootprint:
    def test_master_footprint_shrinks(self, tools: list) -> None:
        est = estimate_context_footprint(tools)
        assert est["nested_master_tokens"] < est["flat_master_tokens"]
        # The delegation surface at the master should shrink dramatically.
        assert est["master_reduction_pct"] >= 60.0
        assert est["master_tokens_saved"] > 0

    def test_no_single_team_exceeds_flat_master(self, tools: list) -> None:
        """Context isolation: no team lead carries more than the old flat
        master did — heavy work is spread across isolated contexts."""
        est = estimate_context_footprint(tools)
        for name, toks in est["nested_team_tokens"].items():
            assert toks <= est["flat_master_tokens"], name

    def test_counts(self, tools: list) -> None:
        est = estimate_context_footprint(tools)
        assert est["n_leaves"] == 10
        assert est["n_teams"] == 4


# ----------------------------------------------------------------------
# The real-build path is wired correctly (source-level, no deepagents needed)
# ----------------------------------------------------------------------
class TestNestedBuildWiring:
    def test_build_nested_uses_compiled_subagent(self) -> None:
        src = inspect.getsource(build_nested_subagents)
        assert "CompiledSubAgent" in src
        assert "create_deep_agent" in src
        assert "runnable" in src

    def test_master_factory_selects_topology(self) -> None:
        from app.agents.master_agent import build_master_agent

        src = inspect.getsource(build_master_agent)
        assert "agent_topology" in src
        assert "build_nested_subagents" in src
        # In nested mode the master must drop its own custom tools.
        assert "master_tools = []" in src

    def test_build_nested_raises_without_deepagents(self, tools: list) -> None:
        """Without deepagents installed, the builder raises so the factory can
        fall back to the flat topology."""
        try:
            import deepagents  # noqa: F401
        except Exception:
            with pytest.raises(ImportError):
                build_nested_subagents(tools, model="mock")
        else:  # pragma: no cover - deepagents present
            pytest.skip("deepagents installed; import-error path not exercised")

    def test_build_nested_compiles_four_teams(self, tools: list) -> None:
        """When deepagents IS installed, we get one CompiledSubAgent per team,
        each carrying name/description/runnable."""
        pytest.importorskip("deepagents")
        subs = build_nested_subagents(tools, model=None)
        assert len(subs) == 4
        names = set()
        for s in subs:
            # CompiledSubAgent is a TypedDict in deepagents 0.6.x
            assert s["name"] and s["description"]
            assert s["runnable"] is not None
            names.add(s["name"])
        assert names == {td.name for td in TEAM_DEFINITIONS}


# ----------------------------------------------------------------------
# Settings default
# ----------------------------------------------------------------------
def test_default_topology_is_nested() -> None:
    from app.core.settings import Settings

    assert Settings(_env_file=None).agent_topology == "nested"
