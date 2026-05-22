# `workspace/` — agentic filesystem root

This directory is the **only** place the agents (and statistical tools) write to.
Each study has its own sub-folder named after its `study_id`.

```
workspace/{study_id}/
├── raw/         ← uploaded, READ-ONLY after first write
├── clean/       ← analysis datasets produced by data_quality_subagent
├── scripts/     ← reproducible Python scripts
├── results/     ← JSON results, tables (.csv)
├── figures/     ← .png / .svg
├── reports/     ← .md / .pdf
├── audit/       ← audit_trail.jsonl, package_versions.json, seeds.json
└── approvals/   ← approval_request_*.json + decisions
```

Everything under `workspace/` is git-ignored (see `.gitignore`). Only this
`README.md` and `.gitkeep` are tracked.
