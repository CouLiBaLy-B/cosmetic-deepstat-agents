# CosmeticDeepStat Agents

> An agentic platform (built on **DeepAgents 0.6** / **LangChain 1.x** / **LangGraph 1.x**) to
> **design, analyse, audit and document** statistical studies on cosmetic products,
> both **pre-market** (efficacy & tolerance) and **post-market** (signal detection,
> complaints, adverse events).

The system orchestrates a **supervisor agent** and **10 specialised sub-agents**
(regulatory, statistical, data quality, safety, consumer insight, reporting, QA),
loads **14 skills** on demand, persists long-term **memory**, writes all
intermediate artefacts to a structured **workspace**, and **never** injects
raw datasets into the LLM context.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/CouLiBaLy-B/cosmetic-deepstat-agents.git
cd cosmetic-deepstat-agents

# 2. Install (Python 3.11+)
pip install -e ".[dev]"

# 3. Generate the demo dataset
python examples/sample_study/generate_synthetic_data.py

# 4. Run the demo end-to-end (no API key needed — uses mock mode)
cosmetic-deepstat run-demo --auto-approve

# 5. Run tests (116 tests)
pytest tests/ -v
```

### With a real LLM

```bash
cp .env.example .env
# Edit .env → set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-...
pip install -e ".[anthropic]"   # or [openai], [google]
uvicorn app.main:app --reload
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    FastAPI (API layer)                         │
│  /studies  /analyses  /approvals  /reports  /postmarket        │
└───────────────────────┬───────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   Master agent (mock/LLM) │
          └─────────────┬─────────────┘
                        │ delegates
          ┌─────────────▼─────────────────────────┐
          │         10 sub-agents                  │
          │   regulatory · study_design · data_qc  │
          │   statistical · multiplicity · safety  │
          │   consumer · postmarket · report · qa  │
          └─────────────┬─────────────────────────┘
                        │ calls
          ┌─────────────▼─────────────┐
          │  19 deterministic tools   │
          │  + 14 skills (SKILL.md)   │
          │  + 4 memory files         │
          └───────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full architecture and
[`docs/agent_design.md`](docs/agent_design.md) for agent/tool/skill details.

---

## Pipeline (11 steps)

```
POST /api/analyses/{study_id}
  │
  ├─ 1. Map claims        → claim_evidence_map.json
  ├─ 2. QC data            → qc_report.json + clean/
  ├─ 3. Draft SAP          → sap_draft.json
  │      └─ 🔒 HITL: SAP approval ──────── PAUSE
  ├─ 4. Run analyses       → statistical_results.json (MMRM / paired-t / McNemar / GLMM)
  ├─ 5. Decide claims      → claim_decisions.json (Holm multiplicity)
  │      └─ 🔒 HITL: claim wording ─────── PAUSE
  ├─ 6. Safety analysis    → safety_report.json
  │      └─ 🔒 HITL: safety conclusion ─── PAUSE
  ├─ 7. Write reports      → 4 markdown reports
  │      └─ 🔒 HITL: final report ──────── PAUSE
  └─ 8. QA audit           → qa_audit_report.json
```

The pipeline is **re-entrant**: call it again after approving a gate, and it
resumes from where it paused. Approval requests are idempotent.

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/studies` | Create a study + workspace |
| `GET`  | `/api/studies/{id}` | Fetch metadata |
| `POST` | `/api/studies/{id}/data` | Upload raw dataset |
| `POST` | `/api/studies/{id}/claims` | Attach marketing claims |
| `POST` | `/api/analyses/{id}` | Launch pipeline |
| `GET`  | `/api/analyses/{id}/status` | Pipeline status + artefacts |
| `GET`  | `/api/approvals` | List pending approvals |
| `POST` | `/api/approvals/{id}` | Approve / reject |
| `GET`  | `/api/reports/{id}` | List reports |
| `GET`  | `/api/reports/{id}/{name}` | Download a report |
| `GET`  | `/health` | Health check |

---

## Statistical methods

| Data type   | 2 timepoints     | ≥ 3 timepoints        |
|-------------|------------------|-----------------------|
| Continuous  | paired-t / Wilcoxon | MMRM (approx.)     |
| Binary      | McNemar          | Logistic GEE          |
| Count       | Poisson/NegBin   | Poisson GEE           |
| Ordinal     | Wilcoxon         | Ordinal mixed (planned) |
| Consumer    | Top-2-box + Wilson CI | —                |
| Equivalence | TOST             | —                     |

**Multiplicity:** Holm (confirmatory), BH-FDR (exploratory).
**Missing data:** MMRM under MAR (no imputation).

See [`docs/statistical_methods.md`](docs/statistical_methods.md) for full details.

---

## Hard rules (non-negotiable)

1. **No raw data in the LLM prompt** — tools return summaries only
2. **No confirmatory analysis without an approved SAP**
3. **No claim wording without human approval**
4. **Effect + 95% CI + adjusted p + practical threshold** for every result
5. **Consumer ≠ instrumental** — separate sub-agents, separate wording
6. **Immutable audit trail** — append-only JSONL with SHA-256 hashes
7. **Multiplicity always applied** when ≥ 2 confirmatory endpoints
8. **Equivalence needs a pre-specified margin** — no "not significant = equivalent"
9. **No exploratory → confirmatory promotion**
10. **Every result is reproducible** — scripts + package versions + seeds

All 10 rules are verified by automated tests (`tests/test_brief_10.py`).

---

## Project structure

```
cosmetic-deepstat-agents/
├── app/
│   ├── agents/          # master agent, sub-agents, prompts, tools
│   ├── api/             # FastAPI routers
│   ├── core/            # settings, audit, logging, paths
│   ├── schemas/         # Pydantic models (Study, Endpoint, Claim, ...)
│   ├── services/        # pipeline, statistics_runner
│   ├── storage/         # in-memory repos (→ SQLAlchemy later)
│   └── security/        # RBAC scaffold
├── skills/              # 14 SKILL.md files (statistics, cosmetics, reporting)
├── memories/            # 4 long-term memory files
├── examples/            # 2 demo studies with generation scripts
├── tests/               # 116 tests (unit, contract, integration, brief)
├── docs/                # architecture, assumptions, methods, agent design, validation
├── workspace/           # runtime agentic filesystem (per study)
└── pyproject.toml       # hatch build, ruff, mypy, pytest config
```

---

## Implementation progress

| Phase | Title | Status |
|:-----:|-------|:------:|
| 1 | Architecture + analysis | ✅ |
| 2 | Skeleton + schemas + API | ✅ |
| 3 | Master agent + 3 sub-agents | ✅ |
| 4 | 14 skills (SKILL.md) | ✅ |
| 5 | MMRM/GLMM/McNemar/TOST tools | ✅ |
| 6 | Full pipeline orchestration | ✅ |
| 7 | Human-in-the-loop + audit | ✅ |
| 8 | Demo studies | ✅ |
| 9 | 116 tests (brief + contracts) | ✅ |
| 10 | Documentation + roadmap | ✅ |

---

## Tests

```bash
pytest tests/ -v         # 116 tests
ruff check app/ tests/   # linting
mypy app/                # type checking
```

| Suite | Tests | What it covers |
|-------|------:|----------------|
| `test_brief_10` | 14 | The 10 non-negotiable rules |
| `test_tool_contracts` | 20 | JSON output schema of every tool |
| `test_pipeline_phase67` | 9 | Pipeline + HITL + QA audit |
| `test_statistics_runner` | 15 | Pure model runners |
| `test_new_tools` | 22 | Phase 5 tools + skills + subagents |
| `test_pipeline_e2e` | 3 | End-to-end happy path |
| `test_tools` | 13 | Phase 3 original tools |
| Other | 20 | API, schemas, settings, audit |

---

## Roadmap

| Phase | Planned feature |
|-------|-----------------|
| 11 | Sensitivity analyses (tipping-point, pattern-mixture, MI) |
| 11 | Ordinal CLMM tool |
| 11 | Cross-validation vs SAS/R MMRM |
| 12 | Post-market ML signal detection |
| 12 | Figure generation (matplotlib/seaborn) |
| 12 | PDF report export (via WeasyPrint) |
| 13 | SQLAlchemy persistence (replace in-memory repos) |
| 13 | S3 workspace mirror |
| 14 | RBAC enforcement on API |
| 14 | Vector store (RAG over SOPs/regulatory docs) |

---

## License

Proprietary. See `pyproject.toml`.
