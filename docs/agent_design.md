# Agent design

> This document describes the agentic architecture of CosmeticDeepStat:
> the master agent, 10 sub-agents, skills, memory, tools, and the
> human-in-the-loop mechanism.

## 1. Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI (API layer)                       │
│  /studies  /analyses  /approvals  /reports  /postmarket      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   CompiledMasterAgent       │
        │   mode: mock | deepagents   │
        └──────────────┬──────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   Deterministic pipeline  │  (mock mode)
         │   OR                      │
         │   DeepAgents graph        │  (LLM mode)
         └─────────────┬─────────────┘
                       │ delegates via task(subagent, prompt)
         ┌─────────────▼─────────────────────────────┐
         │           10 sub-agents                    │
         │  regulatory | study_design | data_quality  │
         │  statistical | multiplicity | safety       │
         │  consumer | postmarket | report | qa       │
         └─────────────┬─────────────────────────────┘
                       │ calls
         ┌─────────────▼─────────────┐
         │   19 deterministic tools  │
         │   (Python, no LLM)        │
         └───────────────────────────┘
```

## 2. Master agent

**Name:** `cosmetic_evidence_orchestrator`

**Role:** plan, delegate, and coordinate. Never does statistics itself.

**System prompt:** `app/agents/prompts.MASTER_SYSTEM_PROMPT` — 10 core
rules including "never paste raw data", "require SAP before analysis",
"require human approval for claims/safety/reports".

**Execution modes:**

| Mode        | When                   | Behaviour                                      |
|-------------|------------------------|-------------------------------------------------|
| `mock`      | `LLM_PROVIDER=mock`   | Runs `pipeline.run_pipeline_deterministic()`    |
| `deepagents`| Any real provider      | Builds a LangGraph with `create_deep_agent()`   |

**Factory:** `app/agents/master_agent.build_master_agent()`

## 3. Sub-agents (10)

Each sub-agent is a dict `{name, description, system_prompt, tools}` passed
to `create_deep_agent(subagents=...)`.

| # | Name                            | Tools                                                    | HITL gate                |
|---|---------------------------------|----------------------------------------------------------|--------------------------|
| 1 | `regulatory_claim_mapper`       | write_audit_event                                        | —                        |
| 2 | `study_design_subagent`         | choose_statistical_test, request_human_approval          | SAP lock                 |
| 3 | `data_quality_subagent`         | load_dataset, profile_dataset, validate_paired_data, detect_missingness, detect_outliers, pseudonymize_subjects, hash_file, write_audit_event | — |
| 4 | `statistical_analysis_subagent` | load_dataset, choose_statistical_test, run_paired_test, run_mmrm, run_glmm_logit, run_mcnemar, run_tost, hash_file, record_package_versions, write_audit_event | — |
| 5 | `multiplicity_claim_subagent`   | apply_multiplicity, request_human_approval, hash_file, write_audit_event | Claim wording |
| 6 | `safety_tolerability_subagent`  | load_dataset, run_paired_test, run_mcnemar, request_human_approval, write_audit_event | Safety conclusion |
| 7 | `consumer_insight_subagent`     | load_dataset, run_top2box, write_audit_event             | —                        |
| 8 | `postmarket_monitoring_subagent`| load_dataset, request_human_approval, write_audit_event  | Post-market signal       |
| 9 | `report_writer_subagent`        | hash_file, request_human_approval, write_audit_event     | Final report release     |
|10 | `qa_auditor_subagent`           | hash_file, write_audit_event                             | —                        |

**Implementation:** `app/agents/subagents.build_subagents()`

## 4. Tools (19)

Every tool follows the dual-function pattern:

1. `_impl_xxx(...)` — pure Python, no LLM, used by tests and the
   deterministic pipeline.
2. `xxx_tool` — `@langchain.tool` wrapper, same logic, used by the LLM
   agent via function calling.

### Tool inventory

| #  | Tool name                    | Category     | Writes to                         |
|----|------------------------------|-------------|-----------------------------------|
| 1  | `load_dataset`               | Data        | —                                 |
| 2  | `profile_dataset`            | Data        | results/profile.json              |
| 3  | `validate_paired_data`       | Data        | results/paired_validation.json    |
| 4  | `detect_missingness`         | Data        | results/missingness_summary.json  |
| 5  | `detect_outliers`            | Data        | results/outlier_report.csv        |
| 6  | `pseudonymize_subjects`      | Security    | clean/analysis_dataset.parquet    |
| 7  | `hash_file`                  | Audit       | —                                 |
| 8  | `write_audit_event`          | Audit       | audit/audit_trail.jsonl           |
| 9  | `choose_statistical_test`    | Statistics  | —                                 |
| 10 | `apply_multiplicity`         | Statistics  | —                                 |
| 11 | `run_paired_test`            | Statistics  | results/ + scripts/               |
| 12 | `run_mmrm`                   | Statistics  | results/ + scripts/               |
| 13 | `run_glmm_logit`             | Statistics  | results/ + scripts/               |
| 14 | `run_mcnemar`                | Statistics  | results/ + scripts/               |
| 15 | `run_top2box`                | Statistics  | results/                          |
| 16 | `run_tost`                   | Statistics  | results/ + scripts/               |
| 17 | `record_package_versions`    | Audit       | audit/package_versions.json       |
| 18 | `request_human_approval`     | HITL        | approvals/                        |
| 19 | `check_approval_status`      | HITL        | —                                 |

**Implementation:** `app/agents/tools.py`

## 5. Skills (14)

Skills are `SKILL.md` files with YAML frontmatter loaded by DeepAgents'
progressive-disclosure mechanism. The agent only sees the name +
description at startup; the full content is loaded on demand.

```
skills/
├── statistics/
│   ├── paired_tests/SKILL.md
│   ├── linear_mixed_models/SKILL.md
│   ├── mmrm/SKILL.md
│   ├── ordinal_models/SKILL.md
│   ├── glmm_gee/SKILL.md
│   ├── multiplicity/SKILL.md
│   ├── equivalence_tost/SKILL.md
│   └── missing_data/SKILL.md
├── cosmetics/
│   ├── claims_eu/SKILL.md
│   ├── claims_us/SKILL.md
│   └── tolerance/SKILL.md
├── reporting/
│   ├── statistical_report/SKILL.md
│   └── claim_substantiation/SKILL.md
└── postmarket/
    └── adverse_events/SKILL.md
```

Each skill contains:
- **When to use** — data type / design / claim type
- **Procedure** — step-by-step instructions
- **Output schema** — JSON structure
- **Hard rules** — non-negotiable constraints
- **References** — regulatory / statistical sources

## 6. Memory (4 files)

Long-term notes passed to the agent via `memory=[...]`:

| File | Content |
|------|---------|
| `memories/org/statistical_policy.md` | Default models, multiplicity, forbidden practices |
| `memories/org/claim_wording_policy.md` | Templates by support level, forbidden terms |
| `memories/instruments/corneometer.md` | Normal ranges, practical thresholds, protocol |
| `memories/postmarket/signal_history.md` | Signal log (initially empty) |

## 7. Human-in-the-loop (4 gates)

The pipeline pauses at 4 mandatory HITL gates:

| Gate                 | Object type        | Triggered by          | Consequence if not approved |
|----------------------|--------------------|-----------------------|-----------------------------|
| **SAP lock**         | `sap`              | `step_draft_sap()`    | No confirmatory analysis    |
| **Claim wording**    | `claim_wording`    | `step_decide_claims()`| No claim release            |
| **Safety conclusion**| `safety_conclusion`| `step_safety_analysis()` | No safety claim          |
| **Final report**     | `final_report`     | `step_write_reports()`| No report release           |

Each gate:
1. Creates an `ApprovalRequest` in the in-memory repo
2. Persists a copy as JSON in `workspace/{study_id}/approvals/`
3. Logs an audit event
4. Blocks pipeline progression until `POST /api/approvals/{id}`

The gate is **idempotent**: re-running the pipeline does not create
duplicate approval requests (`_request_approval_once` guard).

## 8. Pipeline steps (11)

```
POST /analyses/{study_id}
  │
  ├─ 1. step_map_claims()       → claim_evidence_map.json
  ├─ 2. step_qc_data()          → qc_report.json + clean/
  ├─ 3. step_draft_sap()        → sap_draft.json
  │       └─ HITL: SAP lock ──────── PAUSE
  ├─ 4. step_run_analyses()     → statistical_results.json + scripts/
  ├─ 5. step_decide_claims()    → claim_decisions.json
  │       └─ HITL: claim wording ─── PAUSE
  ├─ 6. step_safety_analysis()  → safety_report.json
  │       └─ HITL: safety ────────── PAUSE (if safety claims)
  ├─ 7. step_write_reports()    → 4 markdown reports
  │       └─ HITL: final report ──── PAUSE
  └─ 8. step_qa_audit()         → qa_audit_report.json
```

## 9. Workspace layout

```
workspace/{study_id}/
├── raw/         ← uploaded, immutable (hash recorded)
├── clean/       ← analysis_dataset.parquet
├── scripts/     ← reproducible Python scripts
├── results/     ← JSON results, CSV tables
├── figures/     ← PNG / SVG (future)
├── reports/     ← markdown reports (SAR, CSR, safety, exec)
├── audit/       ← audit_trail.jsonl, package_versions.json, qa_audit_report.json
└── approvals/   ← approval_request_*.json
```

## 10. Guard-rails enforced

| Rule | Enforced by | Verified by test |
|------|-------------|-----------------|
| No raw data in LLM context | Tools return summaries only | `test_brief_10::TestNoRawDataInContext` |
| No analysis without SAP | `_sap_is_locked()` check | `test_brief_10::TestNoAnalysisWithoutSAP` |
| No claim without human approval | HITL gate | `test_brief_10::TestNoClaimWithoutApproval` |
| Effect + CI95 + p_adj + threshold | `StatisticalResult` schema | `test_brief_10::TestCompleteResultSchema` |
| Consumer ≠ instrumental | Separate sub-agents/wording | `test_brief_10::TestConsumerInstrumentalSeparation` |
| Immutable audit trail | Append-only JSONL | `test_brief_10::TestAuditTrailImmutability` |
| Multiplicity for ≥2 primaries | `_impl_apply_multiplicity` | `test_brief_10::TestMultiplicityApplied` |
| Equivalence needs margin | TOST requires margin > 0 | `test_brief_10::TestEquivalenceRequiresMargin` |
| No exploratory→confirmatory | SAP family check | `test_brief_10::TestNoExploratoryPromotion` |
| Reproducible scripts + versions | Every tool writes scripts | `test_brief_10::TestReproducibility` |
