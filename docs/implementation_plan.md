# Implementation plan

The 10 phases agreed with the requester. Each phase ends with a green
`pytest` run and a git commit on `main`.

| Phase | Title                                | Deliverables                                                                                                              | Status |
|:----:|:--------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|:------:|
| 1    | Analysis + architecture               | `docs/architecture.md`, `docs/assumptions.md`, decisions log                                                              | ✅ |
| 2    | Skeleton + schemas + minimal API      | `pyproject.toml`, `app/main.py`, `app/api/*`, `app/schemas/*`, healthcheck                                                | ✅ |
| 3    | Master agent + 3 priority sub-agents  | `app/agents/master_agent.py`, `app/agents/subagents.py`, `app/agents/prompts.py`, mock LLM works without API key          | 🟡 |
| 4    | Skills                                | `skills/**/SKILL.md` for the 14 listed skills, with usable instructions and references                                    | 🟡 |
| 5    | Statistical & QC tools                | `app/agents/tools.py` + `app/services/statistics_runner.py`, unit-tested                                                  | 🟡 |
| 6    | Full pipeline orchestration           | `app/services/pipeline.py`, demo study runnable end-to-end                                                                | 🔜 |
| 7    | Human-in-the-loop + audit trail       | `interrupt_on` config, `ApprovalRequest` table, `audit_trail.jsonl` writer                                                | 🔜 |
| 8    | Demo study                            | `examples/sample_study/{study_metadata.json, claims.json, data/*.csv}` + generation script                                | 🔜 |
| 9    | Tests                                 | 10 tests listed in the brief, plus contract tests on every tool's JSON output                                             | 🔜 |
| 10   | Documentation + roadmap               | `docs/statistical_methods.md`, `docs/agent_design.md`, `docs/validation_plan.md`, README quick-start refreshed            | 🔜 |
