"""Tests for the DeepAgents audit fixes C1, C2, C3.

These tests verify the corrections without requiring a real LLM — they
test the factory, sub-agent wiring, and invoke/resume API paths in mock
mode, plus structural checks on the deepagents integration code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.master_agent import (
    CompiledMasterAgent,
    _collect_memory_files,
    build_master_agent,
)
from app.agents.subagents import build_subagents
from app.main import app

EX = Path(__file__).parent.parent / "examples" / "sample_study"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ======================================================================
# C1 — Sub-agent tools: built-in filesystem tools are always available
# ======================================================================


class TestC1SubagentTools:
    """Verify that sub-agent tool lists are correctly structured."""

    def test_all_subagents_have_custom_tools(self) -> None:
        """Every sub-agent must have at least 1 custom tool."""
        try:
            from app.agents.tools import build_langchain_tools
            tools = build_langchain_tools()
        except Exception:
            pytest.skip("langchain tools not available")
            return
        if not tools:
            pytest.skip("langchain tools not available")
            return

        subs = build_subagents(tools)
        assert len(subs) == 10

        for s in subs:
            assert "name" in s
            assert "description" in s
            assert "system_prompt" in s
            # Every sub-agent should have custom tools
            assert "tools" in s, f"Sub-agent {s['name']!r} missing 'tools' key"
            assert len(s["tools"]) >= 1, f"Sub-agent {s['name']!r} has no tools"

    def test_subagent_tool_names_are_strings(self) -> None:
        """Tool objects must have a .name attribute (langchain convention)."""
        try:
            from app.agents.tools import build_langchain_tools
            tools = build_langchain_tools()
        except Exception:
            pytest.skip("langchain tools not available")
            return
        if not tools:
            pytest.skip("langchain tools not available")
            return

        subs = build_subagents(tools)
        for s in subs:
            for t in s["tools"]:
                name = getattr(t, "name", None)
                assert name is not None, (
                    f"Tool in {s['name']!r} has no .name attribute: {t!r}"
                )
                assert isinstance(name, str) and len(name) > 0

    def test_subagent_tools_do_not_include_builtins(self) -> None:
        """Custom tool lists must NOT contain built-in names like
        'write_file', 'read_file' — those are injected by the harness."""
        try:
            from app.agents.tools import build_langchain_tools
            tools = build_langchain_tools()
        except Exception:
            pytest.skip("langchain tools not available")
            return
        if not tools:
            pytest.skip("langchain tools not available")
            return

        builtins = {"write_file", "read_file", "edit_file", "ls", "glob",
                     "grep", "write_todos", "read_todos", "task"}
        subs = build_subagents(tools)
        for s in subs:
            tool_names = {getattr(t, "name", "") for t in s["tools"]}
            overlap = tool_names & builtins
            assert not overlap, (
                f"Sub-agent {s['name']!r} includes built-in tools {overlap} "
                f"in its custom tools list — these are injected by the "
                f"deepagents harness and should NOT be listed."
            )

    def test_subagent_schema_keys_correct(self) -> None:
        """Sub-agent dicts must use 'system_prompt' (not 'prompt')."""
        try:
            from app.agents.tools import build_langchain_tools
            tools = build_langchain_tools()
        except Exception:
            pytest.skip("langchain tools not available")
            return
        if not tools:
            pytest.skip("langchain tools not available")
            return

        subs = build_subagents(tools)
        for s in subs:
            assert "system_prompt" in s, f"{s['name']!r}: uses wrong key"
            assert "prompt" not in s, f"{s['name']!r}: uses legacy 'prompt' key"


# ======================================================================
# C2 — FilesystemBackend must be passed in deepagents mode
# ======================================================================


class TestC2FilesystemBackend:
    """Verify that the factory passes a FilesystemBackend."""

    def test_mock_mode_does_not_need_backend(self) -> None:
        agent = build_master_agent()
        assert agent.mode == "mock"
        assert agent.graph is None

    def test_factory_code_references_filesystem_backend(self) -> None:
        """The source code of build_master_agent must import and use
        FilesystemBackend in the real path."""
        import inspect
        source = inspect.getsource(build_master_agent)
        assert "FilesystemBackend" in source, (
            "build_master_agent does not reference FilesystemBackend"
        )
        assert 'backend' in source, (
            "build_master_agent does not pass 'backend' to create_deep_agent"
        )

    def test_factory_passes_backend_kwarg(self) -> None:
        """When the factory builds create_kwargs, 'backend' must be present."""
        import inspect
        source = inspect.getsource(build_master_agent)
        # The key "backend" must appear in create_kwargs assignment
        assert '"backend"' in source or "'backend'" in source


# ======================================================================
# C3 — thread_id in config + resume method
# ======================================================================


class TestC3ThreadIdAndResume:
    """Verify invoke passes thread_id and resume method exists."""

    def test_invoke_accepts_thread_id(self) -> None:
        """CompiledMasterAgent.invoke must accept thread_id kwarg."""
        agent = build_master_agent()
        # In mock mode, thread_id is accepted but not used
        meta = json.loads((EX / "study_metadata.json").read_text())
        from app.schemas.study import Study
        from app.storage import db
        study = Study(**meta)
        db.studies().upsert(study.study_id, study)

        csv_path = EX / "data" / "measurements_long.csv"
        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study.study_id).ensure()
        import shutil
        shutil.copy(csv_path, ws.raw / csv_path.name)
        study.data_paths.append(csv_path.name)
        db.studies().upsert(study.study_id, study)

        claims = json.loads((EX / "claims.json").read_text())
        from app.schemas.claims import Claim
        for c in claims:
            claim = Claim(**c, study_id=study.study_id)
            db.claims().upsert(claim.claim_id, claim)

        result = agent.invoke(
            {"study_id": study.study_id},
            thread_id=study.study_id,
        )
        assert "study_id" in result

    def test_resume_method_exists(self) -> None:
        """CompiledMasterAgent must have a resume() method."""
        agent = build_master_agent()
        assert hasattr(agent, "resume")
        assert callable(agent.resume)

    def test_resume_in_mock_mode_reruns_pipeline(self) -> None:
        """In mock mode, resume() re-runs the deterministic pipeline."""
        agent = build_master_agent()
        assert agent.mode == "mock"

        # Setup a study
        meta = json.loads((EX / "study_metadata.json").read_text())
        meta["study_id"] = "STUDY_RESUME_TEST"
        from app.schemas.study import Study
        from app.storage import db
        study = Study(**meta)
        db.studies().upsert(study.study_id, study)

        csv_path = EX / "data" / "measurements_long.csv"
        from app.core.paths import StudyWorkspace
        ws = StudyWorkspace(study.study_id).ensure()
        import shutil
        shutil.copy(csv_path, ws.raw / csv_path.name)
        study.data_paths.append(csv_path.name)
        db.studies().upsert(study.study_id, study)

        claims = json.loads((EX / "claims.json").read_text())
        from app.schemas.claims import Claim
        for c in claims:
            claim = Claim(**c, study_id=study.study_id)
            db.claims().upsert(claim.claim_id, claim)

        result = agent.resume(study.study_id, decisions=[{"type": "approve"}])
        assert "study_id" in result

    def test_get_state_returns_none_in_mock(self) -> None:
        agent = build_master_agent()
        assert agent.get_state("any_thread") is None

    def test_invoke_source_references_thread_id(self) -> None:
        """The invoke method source must construct a config with thread_id."""
        import inspect
        source = inspect.getsource(CompiledMasterAgent.invoke)
        assert "thread_id" in source
        assert "configurable" in source

    def test_resume_source_uses_command(self) -> None:
        """The resume method must reference langgraph Command."""
        import inspect
        source = inspect.getsource(CompiledMasterAgent.resume)
        assert "Command" in source
        assert "resume" in source


# ======================================================================
# C3 — API endpoint /resume
# ======================================================================


class TestC3ResumeEndpoint:
    """Verify the new POST /api/analyses/{id}/resume endpoint."""

    def test_resume_endpoint_exists(self, client: TestClient) -> None:
        """The resume endpoint should return 404 for unknown study (not 405)."""
        r = client.post(
            "/api/analyses/NONEXISTENT/resume",
            json={"decisions": [{"type": "approve"}]},
        )
        assert r.status_code == 404  # study not found, not method not allowed

    def test_resume_full_flow(self, client: TestClient) -> None:
        """Launch → approve SAP → resume → should produce results."""
        meta = json.loads((EX / "study_metadata.json").read_text())
        meta["study_id"] = "STUDY_RESUME_API"
        r = client.post("/api/studies", json=meta)
        assert r.status_code in (201, 409)

        csv_path = EX / "data" / "measurements_long.csv"
        with csv_path.open("rb") as fh:
            r = client.post(
                f"/api/studies/{meta['study_id']}/data",
                files={"file": (csv_path.name, fh, "text/csv")},
            )
        assert r.status_code in (201, 409)

        claims = json.loads((EX / "claims.json").read_text())
        client.post(f"/api/studies/{meta['study_id']}/claims", json=claims)

        # Launch → pauses at SAP
        r = client.post(f"/api/analyses/{meta['study_id']}")
        assert r.status_code == 202

        # Approve all pending
        pending = client.get(
            f"/api/approvals?study_id={meta['study_id']}&status=pending"
        ).json()
        for a in pending:
            client.post(
                f"/api/approvals/{a['approval_id']}",
                json={"decision": "approved", "reviewer": "test"},
            )

        # Resume via the new endpoint
        r = client.post(
            f"/api/analyses/{meta['study_id']}/resume",
            json={"decisions": [{"type": "approve"}]},
        )
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "ok"


# ======================================================================
# C5 — Memory size guard
# ======================================================================


class TestC5MemoryGuard:
    """Verify that memory file collection has a size limit."""

    def test_collect_memory_files_skips_large_files(self, tmp_path: Path) -> None:
        """Files that would exceed the budget are skipped."""
        from app.core.settings import Settings

        mem_root = tmp_path / "memories"
        mem_root.mkdir()
        # Write a small file
        (mem_root / "small.md").write_text("# Small\nHello\n")
        # Write a huge file (> 50KB)
        (mem_root / "huge.md").write_text("x" * 60_000)

        settings = Settings(
            memory_root=mem_root,
            workspace_root=tmp_path / "ws",
        )
        files = _collect_memory_files(settings)
        # small.md should be included, huge.md skipped
        assert len(files) == 1
        assert "small.md" in files[0]

    def test_collect_memory_files_empty_dir(self, tmp_path: Path) -> None:
        from app.core.settings import Settings

        mem_root = tmp_path / "no_memories"
        # Don't create the dir
        settings = Settings(
            memory_root=mem_root,
            workspace_root=tmp_path / "ws",
        )
        files = _collect_memory_files(settings)
        assert files == []
