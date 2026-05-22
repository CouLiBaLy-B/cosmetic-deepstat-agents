# Validation plan

> This document defines the validation strategy for CosmeticDeepStat Agents,
> ensuring the platform produces correct, reproducible, and auditable
> results suitable for regulatory submission.

## 1. Scope

The validation covers:
- **Statistical correctness** of all 10+ implemented models
- **Pipeline integrity** — the right model is selected for each endpoint
- **Human-in-the-loop** — no output without required approvals
- **Data integrity** — hashes, audit trail, no raw data leakage
- **Claim substantiation** — correct mapping from evidence to wording

## 2. Risk classification

| Component | Risk | Rationale | Mitigation |
|-----------|------|-----------|------------|
| Paired t-test / Wilcoxon | Medium | Wrong test → wrong conclusion | Shapiro-Wilk auto-switch, unit tested |
| MMRM | High | Complex model, convergence issues | Fallback to LMM, convergence flag, residual diagnostics |
| Multiplicity | High | Missing correction → inflated claims | Holm default, pipeline refuses single p-values for ≥2 primaries |
| HITL gates | Critical | Bypass → unapproved claims released | 4 gates tested individually, idempotence verified |
| Audit trail | Critical | Tampering → regulatory non-compliance | Append-only JSONL, SHA-256 hashes, per-study + global logs |
| Pseudonymisation | High | Leak → privacy violation | Per-study salt, SHA-256 truncated to 16 chars |

## 3. Test strategy

### 3.1. Unit tests (52 tests)

| Suite | Count | Coverage |
|-------|-------|----------|
| `test_tools.py` | 13 | All Phase-3 tools |
| `test_statistics_runner.py` | 15 | Pure model runners (MMRM, LMM, McNemar, GLMM, Poisson, top-2-box, TOST) |
| `test_new_tools.py` | 7+14+1 | Phase-5 tool wrappers, SKILL.md existence, subagent wiring |
| `test_schemas.py` | 5 | Pydantic schemas (CI ordering, study_id pattern) |
| `test_settings_and_paths.py` | 8 | Path traversal, study_id validation |
| `test_audit_trail.py` | 2 | Audit event writing, file hashing |
| `test_api.py` | 5 | API endpoints (health, CRUD, 404s) |

### 3.2. Contract tests (20 tests)

`test_tool_contracts.py` — verifies every `_impl_*` function returns
a dict with:
- Exactly the documented keys
- Correct types (str, int, float, list, dict, bool)
- Valid value ranges (0 ≤ p ≤ 1, CI ordered, sha256: prefix)

### 3.3. Integration tests (12 tests)

| Suite | Count | Scope |
|-------|-------|-------|
| `test_pipeline_e2e.py` | 3 | Full pipeline happy path, SAP gate, QA audit |
| `test_pipeline_phase67.py` | 9 | Model selection, safety step, 4 HITL gates, QA enriched, idempotence, status endpoint, report content |

### 3.4. Brief tests (14 tests)

`test_brief_10.py` — the 10 non-negotiable rules from the project brief:

1. No raw data in LLM context
2. No analysis without approved SAP
3. No claim wording without human approval
4. Effect + CI95 + adjusted p + practical threshold for every endpoint
5. Consumer ≠ instrumental separation
6. Immutable audit trail with hashes
7. Multiplicity always applied (Holm p_adj ≥ p_raw)
8. Equivalence requires pre-specified margin
9. No exploratory-to-confirmatory promotion
10. Reproducible scripts + package versions

### 3.5. Total

**116 tests**, all green (as of Phase 9 commit).

## 4. Hand-verified reference values

| Test | Reference value | Source |
|------|-----------------|--------|
| Holm correction | p=[0.001, 0.02, 0.03, 0.04, 0.5] → p_adj=[0.005, 0.08, 0.09, 0.08, 0.5] | Hand calculation |
| Bonferroni | p=[0.01, 0.04, 0.06] → p_adj=[0.03, 0.12, 0.18] | p × m |
| TOST equivalence | x=y+noise(0,0.5), margin=5 → met | Mathematical certainty |
| TOST non-equivalence | x=y+10, margin=2 → not met | Obvious large shift |
| Top-2-box | [5,4,3,5,4,4,2,1,5,3] scale=5 → 60% (6/10) | Manual count |

## 5. Regression testing

Every commit triggers the full test suite (116 tests). The CI pipeline
(to be set up) should:

1. Install `pip install -e ".[dev]"`
2. Run `ruff check app/ tests/`
3. Run `mypy app/`
4. Run `pytest tests/ -v --tb=short`
5. Fail on any error

## 6. Known limitations

| Limitation | Impact | Mitigation plan |
|------------|--------|-----------------|
| MMRM uses random intercept, not true unstructured covariance | Slightly different estimates from SAS PROC MIXED | Document as MMRM_approx; add pymer4 R bridge in future phase |
| Ordinal CLMM not yet implemented as a tool | Cannot model ordinal longitudinal endpoints | Wilcoxon fallback; CLMM planned for Phase 11 |
| No sensitivity analyses (tipping-point, pattern-mixture) | Confirmatory claims less robust | Planned for Phase 11 |
| Consumer insight sub-agent only runs top-2-box | Limited consumer analysis | Add Cronbach α, Likert distributions in Phase 11 |
| Post-market signal detection is rule-based only | No ML-based anomaly detection | Planned for Phase 12 |

## 7. Regulatory alignment

| Standard | Status | Notes |
|----------|--------|-------|
| EU Reg. 1223/2009 | ✅ Implemented | Claims EU skill, safety requirements |
| EU Reg. 655/2013 (Common Criteria) | ✅ Implemented | 6 criteria in regulatory_claim_mapper |
| FDA cosmetics guidance | ✅ Implemented | Claims US skill, cosmetic/drug boundary |
| MoCRA (2022) | ✅ Referenced | AE reporting in post-market skill |
| ICH E9(R1) | ✅ Referenced | Estimands, sensitivity analyses in SAP |
| ISO 16128 | ✅ Referenced | Natural/organic definitions |
| SCCS Notes of Guidance | ✅ Referenced | Safety testing, tolerance |

## 8. Future validation activities

| Activity | Phase | Deliverable |
|----------|-------|-------------|
| Cross-validation vs SAS/R MMRM | 11 | Reference comparison table |
| Formal IQ/OQ/PQ protocol | 11 | Qualification report |
| 21 CFR Part 11 gap analysis | 12 | Compliance matrix |
| Penetration testing | 12 | Security report |
| Performance benchmarking | 12 | Latency/throughput report |
