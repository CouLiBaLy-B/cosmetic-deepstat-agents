# Architecture — CosmeticDeepStat Agents

> Reference: **deepagents 0.6.3** / **langchain 1.3.x** / **langgraph 1.2.x**
> (verified against the LangChain OSS docs at <https://docs.langchain.com/oss/python/deepagents/>
> on the date the project was bootstrapped). API used:
> `create_deep_agent(model, tools, system_prompt, subagents, skills, memory,
> interrupt_on, backend, checkpointer, middleware, name, debug)`.

## 1. Goal

Build a production-grade backend platform that helps a cosmetic R&D /
biostatistics team **design**, **execute**, **interpret**, **audit** and
**document** clinical studies on cosmetic products, both pre-market (efficacy
and tolerance) and post-market (complaints, adverse events, signal detection).

The platform is **agentic** — it uses a supervisor LLM agent that plans,
delegates to specialised sub-agents, loads skills on demand, calls deterministic
Python statistical tools, and persists every intermediate artefact to a
versioned filesystem with an immutable audit trail.

## 2. Layered architecture

```text
 ┌───────────────────────────────────────────────────────────────────────┐
 │                          A. API LAYER (FastAPI)                        │
 │  /studies  /analyses  /approvals  /reports  /postmarket  /health       │
 └───────────────┬───────────────────────────────────────┬────────────────┘
                 │                                       │
 ┌───────────────▼───────────────┐       ┌───────────────▼────────────────┐
 │   B. AGENTIC CORE (DeepAgents) │       │  G. STORAGE                    │
 │  - master agent (supervisor)   │       │  - SQLAlchemy metadata DB      │
 │  - 10 sub-agents               │       │  - workspace/ (filesystem)     │
 │  - skills (SKILL.md)           │       │  - object store (S3-compat.)   │
 │  - memory (markdown notes)     │       │  - vector store (RAG)          │
 │  - interrupt_on (HITL)         │       │  - audit log (JSONL, append)   │
 │  - FilesystemBackend           │       └────────────────────────────────┘
 └───────┬───────────────────────┘
         │ calls
 ┌───────▼─────────────────────┐   ┌─────────────────────────────────────┐
 │  F. STATISTICAL TOOLS (Py)  │   │  H. SECURITY                        │
 │  - pandas / numpy / scipy   │   │  - RBAC on API                      │
 │  - statsmodels / pingouin   │   │  - pseudonymisation of subjects     │
 │  - matplotlib / seaborn     │   │  - raw/ folder is read-only         │
 │  - optional R via pymer4    │   │  - audit trail on every write       │
 └─────────────────────────────┘   └─────────────────────────────────────┘
```

### A. API layer

| Endpoint                                         | Method  | Purpose                                     |
|--------------------------------------------------|---------|---------------------------------------------|
| `/api/studies`                                   | POST    | Create a study, allocate workspace          |
| `/api/studies/{id}`                              | GET     | Fetch metadata + status                     |
| `/api/studies/{id}/data`                         | POST    | Upload raw dataset(s)                       |
| `/api/studies/{id}/claims`                       | POST    | Attach marketing claims                     |
| `/api/analyses/{study_id}`                       | POST    | Launch the full agentic pipeline            |
| `/api/analyses/{study_id}/status`                | GET     | Get current pipeline step + interrupts      |
| `/api/approvals`                                 | GET     | List pending human approvals                |
| `/api/approvals/{id}`                            | POST    | Approve / edit / reject (HITL decision)     |
| `/api/reports/{study_id}/{report_name}`          | GET     | Download a generated report                 |
| `/api/postmarket/{product_id}`                   | POST    | Ingest a post-market data batch             |
| `/api/postmarket/{product_id}/signals`           | GET     | Current signal-detection dashboard          |
| `/health`                                        | GET     | Liveness + provider readiness               |

### B. Agentic core

#### Master agent — `cosmetic_evidence_orchestrator`

Built with `deepagents.create_deep_agent(...)`. It carries the system prompt
described in §9 of the project brief and is the **only** entry-point for the
pipeline. It plans (via the built-in `write_todos`), delegates (via the
built-in `task` tool, populated by `SubAgentMiddleware`), reads/writes files
(via the built-in filesystem tools backed by `FilesystemBackend(root_dir=WORKSPACE_ROOT)`),
and uses our custom domain tools (see §F).

#### Sub-agents (10)

Each sub-agent is a dict `{name, description, system_prompt, tools, model?,
middleware?, interrupt_on?}` (per deepagents 0.6.3 schema). Sub-agents are
launched by the master through the `task(subagent, prompt)` tool.

| # | Name                                | Tools (initial)                                                       | HITL? |
|---|-------------------------------------|-----------------------------------------------------------------------|-------|
| 1 | `regulatory_claim_mapper`           | none (LLM-only with structured output)                                | no    |
| 2 | `study_design_subagent`             | `choose_statistical_test_tool`                                        | SAP   |
| 3 | `data_quality_subagent`             | `profile_dataset_tool`, `detect_missingness_tool`, `detect_outliers_tool`, `validate_paired_data_tool`, `pseudonymize_subjects_tool` | no |
| 4 | `statistical_analysis_subagent`     | `choose_statistical_test_tool`, `run_python_stats_tool`, `generate_plot_tool`, `hash_file_tool` | no |
| 5 | `multiplicity_claim_subagent`       | `apply_multiplicity_tool`                                             | claims|
| 6 | `safety_tolerability_subagent`      | `run_python_stats_tool`                                               | safety|
| 7 | `consumer_insight_subagent`         | `run_python_stats_tool`                                               | no    |
| 8 | `postmarket_monitoring_subagent`    | `run_python_stats_tool`                                               | signals|
| 9 | `report_writer_subagent`            | `create_report_tool`                                                  | report|
|10 | `qa_auditor_subagent`               | `hash_file_tool`, `write_audit_event_tool`                            | no    |

#### Skills (`SKILL.md`)

Loaded via `skills=["./skills"]`. The agent only sees the **frontmatter
(name + description)** of every `SKILL.md` at startup. When a task matches,
the agent reads the full skill (and any additional assets it references)
— this is *progressive disclosure*.

Skills currently scaffolded:

- `skills/statistics/paired_tests/SKILL.md`
- `skills/statistics/linear_mixed_models/SKILL.md`
- `skills/statistics/mmrm/SKILL.md`
- `skills/statistics/ordinal_models/SKILL.md`
- `skills/statistics/glmm_gee/SKILL.md`
- `skills/statistics/multiplicity/SKILL.md`
- `skills/statistics/equivalence_tost/SKILL.md`
- `skills/statistics/missing_data/SKILL.md`
- `skills/cosmetics/claims_eu/SKILL.md`
- `skills/cosmetics/claims_us/SKILL.md`
- `skills/cosmetics/tolerance/SKILL.md`
- `skills/reporting/statistical_report/SKILL.md`
- `skills/reporting/claim_substantiation/SKILL.md`
- `skills/postmarket/adverse_events/SKILL.md`

#### Memory

Long-term notes are passed via `memory=[...]` (a list of markdown files
loaded into the agent's working context). Per the brief:

- `memories/org/statistical_policy.md`
- `memories/org/claim_wording_policy.md`
- `memories/products/{product_id}/historical_effect_sizes.md`
- `memories/products/{product_id}/previous_claim_decisions.md`
- `memories/instruments/corneometer.md`
- `memories/postmarket/signal_history.md`

#### Filesystem (per-study workspace)

```text
workspace/{study_id}/
├── raw/         ← uploaded, READ-ONLY (hash recorded)
├── clean/       ← analysis dataset(s) produced by data_quality_subagent
├── scripts/     ← reproducible Python scripts written by statistical agent
├── results/     ← JSON results, tables (.csv)
├── figures/     ← .png / .svg
├── reports/     ← .md / .pdf
├── audit/       ← audit_trail.jsonl, package_versions.json, seeds.json
└── approvals/   ← approval_request_*.json + decisions
```

#### Human-in-the-loop

We use deepagents' native `interrupt_on={...}` mechanism (requires a
`Checkpointer`). The following tool names are wired with interrupts:

| Tool                                | Allowed decisions               | Triggered for           |
|-------------------------------------|---------------------------------|-------------------------|
| `lock_sap_tool`                     | approve, edit, reject           | SAP finalisation        |
| `lock_primary_endpoint_tool`        | approve, edit, reject           | Primary endpoint choice |
| `lock_multiplicity_strategy_tool`   | approve, edit, reject           | Multiplicity strategy   |
| `lock_equivalence_margin_tool`      | approve, edit, reject           | NI/equivalence margin   |
| `exclude_subjects_tool`             | approve, edit, reject           | Data exclusion          |
| `finalize_claim_wording_tool`       | approve, edit, reject           | Final claim wording     |
| `finalize_safety_conclusion_tool`   | approve, edit, reject           | Safety conclusion       |
| `release_final_report_tool`         | approve, reject                 | Final report release    |

Each call writes an `ApprovalRequest` row in the DB **and** an
`approval_request_{id}.json` file in `workspace/{study_id}/approvals/`.

### F. Statistical tools (deterministic Python)

Implemented in `app/agents/tools.py`. Each tool:

1. validates inputs via Pydantic,
2. executes deterministic Python (or shells out to R if `pymer4` is installed),
3. writes long outputs to `workspace/{study_id}/...`,
4. returns a compact JSON dict (suitable for the LLM context),
5. emits an `AuditEvent`.

The full list is in `docs/agent_design.md`.

### G. Storage

| Concern              | Backend (MVP)              | Backend (Prod)                |
|----------------------|----------------------------|-------------------------------|
| Metadata DB          | SQLite (file)              | PostgreSQL                    |
| Workspace files      | local FS                   | local FS + S3 mirror          |
| Vector store (RAG)   | none (stub)                | Chroma / Qdrant / pgvector    |
| Audit log            | JSONL append-only          | JSONL + WORM object store     |

### H. Security

- `raw/` folder is enforced read-only at the service layer.
  `ALLOW_DATA_MUTATION_IN_RAW=false` must remain `false`.
- `pseudonymize_subjects_tool` replaces subject identifiers with hashes
  and keeps a salt-pepper mapping in an encrypted side-file.
- Every tool writes an `AuditEvent` (`actor`, `action`, `input_hash`,
  `output_hash`, `timestamp`, `metadata`).
- RBAC scaffolded via FastAPI dependencies (roles: `viewer`, `analyst`,
  `reviewer`, `qa`, `admin`).

## 3. Why DeepAgents (vs. plain LangGraph)?

`deepagents` already provides — out of the box — the four capabilities we
would otherwise re-implement:

1. **Planning** (`write_todos` / `read_todos` middleware): the master agent
   maintains an explicit checklist for every study.
2. **Filesystem** (`FilesystemBackend` + `ls / read_file / write_file /
   edit_file / glob / grep`): we keep raw data out of the prompt and have
   a real working directory.
3. **Subagents** (`SubAgentMiddleware` + `task` tool): isolated context per
   specialisation (no cross-contamination, smaller per-call prompt size).
4. **Skills** (progressive disclosure of `SKILL.md` frontmatter): we can
   ship dozens of statistical procedures without bloating the system prompt.

It also gives us `interrupt_on` (HITL) and `memory` (long-term notes) for
free.

## 4. Data flow (happy path)

```text
POST /studies                  → metadata DB row, mkdir workspace/{id}/...
POST /studies/{id}/data        → save to raw/, hash, audit
POST /studies/{id}/claims      → save claims.json
POST /analyses/{id}            → master agent invoked (async)

master.write_todos([
  "map_claims", "qc_data", "draft_sap", "wait_sap_approval",
  "run_analyses", "apply_multiplicity", "decide_claims",
  "safety", "write_reports", "qa_audit", "wait_final_approval"
])

master.task(regulatory_claim_mapper, claims.json)   → claim_evidence_map.json
master.task(data_quality_subagent, raw/)            → qc_report.json + clean/
master.task(study_design_subagent, claim_evidence_map.json + qc_report.json)
                                                    → sap_draft.json
lock_sap_tool(sap_draft.json)                       → INTERRUPT (HITL)
…on approve…
master.task(statistical_analysis_subagent, sap.json + clean/)
                                                    → results/*.json, figures/, scripts/
master.task(multiplicity_claim_subagent, results/ + claim_evidence_map)
                                                    → claim_decisions.json
finalize_claim_wording_tool(...)                    → INTERRUPT (HITL)
master.task(safety_tolerability_subagent, ...)
master.task(report_writer_subagent, ...)            → reports/*.md
master.task(qa_auditor_subagent, workspace/)        → qa_audit_report.json
release_final_report_tool(...)                      → INTERRUPT (HITL)
```

## 5. Hard rules (non-negotiable)

1. **No raw data in the prompt.** All tools take paths, all summaries are
   structured.
2. **No confirmatory analysis without an approved SAP.** The
   `statistical_analysis_subagent` checks for `workspace/{id}/approvals/sap_locked.json`
   before running anything labelled *confirmatory*.
3. **No final claim wording without human approval.** Enforced by
   `interrupt_on["finalize_claim_wording_tool"]`.
4. **Effect + CI95 + adjusted p + practical threshold** for every endpoint.
   `StatisticalResult` is a Pydantic model with these as required fields.
5. **Consumer ≠ instrumental.** Two distinct sub-agents, two distinct
   schemas, two distinct claim-wording catalogues.
6. **Audit trail is immutable.** Append-only JSONL with input/output hashes.
