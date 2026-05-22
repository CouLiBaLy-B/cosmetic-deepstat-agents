# Assumptions

This file records every assumption that was made (without an explicit
specification from the requester) while building **CosmeticDeepStat Agents**.
Every assumption is documented, justified, and reversible.

## 1. Versions / runtime

- **Python 3.11+** (matches `pyproject.toml`). DeepAgents 0.6.3 supports it.
- **DeepAgents 0.6.3 / LangChain 1.3.x / LangGraph 1.2.x.** API confirmed
  against the official docs at <https://docs.langchain.com/oss/python/deepagents/>.
  In particular:
  - `create_deep_agent(model, tools, system_prompt, subagents, skills, memory,
    interrupt_on, backend, checkpointer, middleware, name, debug)`
  - Sub-agents are dicts with keys `name, description, system_prompt, tools,
    model?, middleware?, interrupt_on?`. The legacy `prompt=` key is no
    longer used in 0.5+; we use `system_prompt`.
  - `interrupt_on` requires a `Checkpointer`. We default to `MemorySaver` in
    dev and to a SQLite/Postgres checkpointer in prod.
  - Skills are auto-discovered from directories passed to `skills=[...]`.
- **Provider-agnostic.** Default `LLM_PROVIDER=mock` so the project starts
  with no API key. Switching to Anthropic / OpenAI / Google is a single
  `.env` change + an `extras` install.

## 2. Scope of the MVP

The first iteration implements **end-to-end** the three priority sub-agents
identified by the requester (Phase 1 question 3 — *MVP fonctionnel d'abord*):

1. `regulatory_claim_mapper`
2. `data_quality_subagent`
3. `statistical_analysis_subagent`

The seven remaining sub-agents are scaffolded (prompt + Pydantic output
schema + `SKILL.md` files referenced) so that wiring them later only requires
adding them to the `SUBAGENTS` list in `app/agents/subagents.py`.

## 3. Data model

- **Pseudonymisation by default** (`PSEUDONYMIZE_SUBJECTS=true`). The raw
  `subject_id` is replaced by `sha256(salt || subject_id)[:16]` before any
  agent ever sees the data. The salt is per-study and kept in the audit
  folder.
- **Wide vs. long format.** The MVP expects datasets in **long format**
  (`subject_id, visit, endpoint_name, value`) with one row per measurement.
  A helper (`app.services.ingestion.normalize_to_long`) converts common wide
  formats automatically.
- **Visit labels.** Visits are stored as strings (`"D0", "D7", "D14", "D28"`)
  and parsed via a regex `^D(\d+)$` to obtain a numeric day for longitudinal
  models. Non-standard labels fall back to lexical order with a warning.

## 4. Statistics

- We use **`statsmodels`** as the reference engine for LMM / MMRM / GLMM
  (MixedLM, GEE, OrdinalGEE, Logit). `pingouin` is used as a convenience
  layer for paired tests / effect sizes. R via `pymer4` is **optional** and
  only loaded when the user installs `[r-stats]`.
- **Default multiplicity policy:**
  - Confirmatory claims → Holm.
  - Multiple primary endpoints with a hierarchy → fixed-sequence /
    gatekeeping.
  - Exploratory claims → Benjamini–Hochberg FDR.
  - This default is overridden by the SAP if specified.
- **Default equivalence margins:** none. A claim of equivalence /
  non-inferiority is **rejected by default** unless the SAP supplies a
  pre-specified margin, and the margin is approved via HITL
  (`lock_equivalence_margin_tool`).
- **Missing data:** by default the system reports % missing per
  endpoint/visit and runs the primary analysis under MAR (MMRM). At least
  one sensitivity analysis (tipping-point or pattern-mixture) is required
  before any confirmatory claim is allowed.

## 5. Reports

- **Markdown first**, PDF later. The `report_writer_subagent` produces
  Markdown via Jinja2 templates. PDF rendering (WeasyPrint or Pandoc) is
  out of MVP scope.
- **Report templates** live in `skills/reporting/*/templates/` so they are
  versioned with the skills themselves.

## 6. Security

- The MVP does not implement authentication. RBAC dependencies are
  scaffolded in `app/security/permissions.py` so that an OAuth2 / JWT
  layer can be added without touching the routers.
- `raw/` enforcement is at the service level (`StorageService`) — a Python
  call cannot overwrite a file in `raw/` once it exists. Tightening this
  at the OS level (read-only bind mount) is recommended in production.

## 7. Post-market

- The MVP scaffolds the `postmarket_monitoring_subagent` and ingestion
  endpoint but only ships a **rule-based** signal detector
  (rate change > 50 % w/w on a 4-week window). Semantic clustering of
  complaints (embeddings + HDBSCAN) is planned for the next iteration.

## 8. Observability

- Structured JSON logs via `structlog`.
- Optional LangSmith tracing (`LANGSMITH_TRACING=true`).
- No Prometheus / Sentry in the MVP — to be added in a later phase.

## 9. Open questions for the user

- Which jurisdictions are in scope long-term? MVP supports `EU` and `US`
  catalogues. China / Japan / UK can be added as extra `skills/cosmetics/claims_{xx}/`.
- Do we need batch-traceability fields (`batch_id`, `lot_id`) in the
  `Study` schema right away, or only in post-market data?
- Multilingual reports — should the report templates support fr/en at
  the same time?
