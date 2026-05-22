# CosmeticDeepStat Agents

> An agentic platform (built on **DeepAgents** / **LangChain 1.x** / **LangGraph 1.x**) to
> **design, analyse, audit and document** statistical studies on cosmetic products,
> both **pre-market** (efficacy & tolerance) and **post-market** (signal detection,
> complaints, adverse events).

The system orchestrates a **supervisor agent** and **specialised sub-agents**
(regulatory, statistical, data quality, safety, consumer insight, reporting, QA),
loads **skills** on demand (procedural knowledge), persists short- and long-term
**memory** (organisational, product, instrument, regulatory), writes all
intermediate artefacts to a structured **agentic filesystem** (`workspace/{study_id}/...`),
and **never** injects raw datasets into the LLM context.

It enforces hard guard-rails against the most common abuses of cosmetic-claim
statistics:

- turning an exploratory result into a confirmatory marketing claim
- concluding "equivalent" because *p > 0.05*
- ignoring multiplicity corrections
- using a non-paired test on paired data (J0/J7/J14/J28)
- confusing **consumer perception** with **instrumental evidence**
- producing a claim **wording** not supported by the data

---

## Status

**MVP scope (current iteration).** See [`docs/implementation_plan.md`](docs/implementation_plan.md).

| Phase | Item                                                  | Status         |
|------:|-------------------------------------------------------|----------------|
| 1     | Architecture, assumptions, doc DeepAgents 0.6.3 check | ✅ done        |
| 2     | Project skeleton, schemas, minimal API                | ✅ done        |
| 3     | Master agent + 3 priority sub-agents                  | 🟡 in progress |
| 4     | Skills (`SKILL.md` for stats + claims)                | 🟡 in progress |
| 5     | Statistical tools (paired/longitudinal/multiplicity)  | 🟡 in progress |
| 6     | End-to-end pipeline on `examples/sample_study`        | 🔜 next        |
| 7     | Human-in-the-loop (SAP lock, claim wording)           | 🔜 next        |
| 8–10  | Demo study, full tests, full docs                     | 🔜 next        |

The 3 priority sub-agents in MVP are:
`regulatory_claim_mapper`, `data_quality_subagent`, `statistical_analysis_subagent`.
The 7 others (multiplicity, safety, consumer, postmarket, report writer, QA,
study design) are scaffolded with their prompt, schema and `SKILL.md`, ready to
be wired into the pipeline.

---

## Quick start (mock LLM, no API key required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # default: LLM_PROVIDER=mock
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs>.

### With a real model

```bash
pip install -e '.[dev,anthropic]'   # or [...,openai] / [...,google]
# edit .env:
#   LLM_PROVIDER=anthropic
#   LLM_MODEL=anthropic:claude-sonnet-4-5-20250929
#   ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

### Run the demo study

```bash
python -m app.cli run-demo --study-id STUDY_DEMO_001
```

This:

1. creates `workspace/STUDY_DEMO_001/{raw,clean,scripts,results,figures,reports,audit,approvals}/`
2. ingests `examples/sample_study/{study_metadata.json, claims.json, data/*.csv}`
3. runs claim → endpoint mapping (`regulatory_claim_mapper`)
4. runs data QC (`data_quality_subagent`)
5. produces a draft SAP (`study_design_subagent`)
6. **pauses for human approval** of the SAP
7. on approval, runs paired / longitudinal analyses (`statistical_analysis_subagent`)
8. applies multiplicity, decides claims, writes reports.

### Tests

```bash
pytest -q
ruff check app tests
mypy app
```

---

## Architecture (one paragraph)

A **FastAPI** layer exposes REST endpoints (`/studies`, `/analyses`, `/approvals`,
`/reports`, `/postmarket`). Each endpoint validates input with **Pydantic** and
either records a study/approval in the metadata DB (SQLite/Postgres) or
**delegates** to the agentic core.

The agentic core is a `create_deep_agent(...)` graph (deepagents 0.6.3) whose
**main agent** ("Cosmetic Evidence Orchestrator") is configured with:

- a curated list of domain tools (`load_dataset_tool`, `profile_dataset_tool`,
  `validate_paired_data_tool`, `choose_statistical_test_tool`,
  `run_python_stats_tool`, `apply_multiplicity_tool`,
  `request_human_approval_tool`, `write_audit_event_tool`, `hash_file_tool`,
  `pseudonymize_subjects_tool`, …),
- a `FilesystemBackend(root_dir="./workspace")` so the agent only ever reads/writes
  files (never raw data in context),
- `skills=["./skills"]` for progressive disclosure of statistical / regulatory
  procedural knowledge,
- `memory=[...]` for long-term notes (org policies, per-product history),
- `interrupt_on={...}` to require **human-in-the-loop** approval before SAP
  lock, claim wording, equivalence margins, safety conclusions, final report,
- a list of **sub-agents** (each itself a `create_agent` graph) for specialised
  work.

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and
data flow.

---

## Repository layout

```text
cosmetic-deepstat-agents/
├── app/
│   ├── main.py                # FastAPI app entry-point
│   ├── cli.py                 # `cosmetic-deepstat ...` CLI
│   ├── core/                  # settings, logging, paths, audit
│   ├── api/                   # FastAPI routers
│   ├── agents/                # master agent + subagents + prompts + tools
│   ├── schemas/               # Pydantic models (Study/Claim/Endpoint/...)
│   ├── services/              # ingestion, data_quality, statistics_runner,
│   │                          # report_generation, memory, audit
│   ├── storage/               # SQLAlchemy db, object store, vector store
│   └── security/              # permissions (RBAC) + PII pseudonymisation
├── skills/                    # SKILL.md folders consumed by deepagents
│   ├── statistics/            # paired_tests, lmm, mmrm, ordinal, glmm_gee,
│   │                          # multiplicity, equivalence_tost, missing_data
│   ├── cosmetics/             # claims_eu, claims_us, tolerance
│   ├── reporting/             # statistical_report, claim_substantiation
│   └── postmarket/            # adverse_events
├── memories/                  # long-term markdown notes loaded by main agent
├── workspace/                 # agentic filesystem root (per-study sub-dirs)
├── examples/sample_study/     # demo study (metadata + claims + synthetic data)
├── tests/                     # pytest unit & integration tests
├── docs/                      # architecture, statistical_methods, agent_design,
│   │                          # validation_plan, assumptions
├── scripts/                   # ops helpers
├── docker/                    # Dockerfile(s)
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Hard rules enforced by the system

1. Raw datasets **never** enter the LLM context (only paths + structured summaries).
2. No confirmatory analysis without an **approved SAP**.
3. No final claim wording without **human approval**.
4. Every result is reproducible: scripts + seeds + dataset hashes + package
   versions are written to `workspace/{study_id}/audit/`.
5. Each claim decision must carry: jurisdiction, endpoint, estimand, model,
   estimate, CI95, raw p, adjusted p, practical significance, sensitivity
   analysis, limitations, allowed wording, forbidden wording.
6. Consumer perception data are never reported as instrumental evidence.

## License

Proprietary — for internal use.
