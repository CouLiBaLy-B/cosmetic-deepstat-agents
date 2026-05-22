---
name: paired_tests
description: >
  Run paired statistical tests (paired-t, Wilcoxon signed-rank) for
  before/after cosmetic studies with two timepoints. Includes assumption
  checks, effect-size estimation (Cohen's dz), and reproducible-script
  generation.
tags: [statistics, paired, t-test, wilcoxon, before_after]
---

# Paired tests for cosmetic efficacy studies

## When to use

- **Design:** before/after (single-group) with exactly **2 timepoints**
  (e.g. D0 vs D28).
- **Endpoint data type:** continuous (e.g. corneometer a.u., TEWL g/m²/h,
  wrinkle depth mm).
- **Comparator:** each subject is their own control (paired design).

## Decision logic

```text
Is the endpoint continuous?
├─ YES → Check Shapiro-Wilk on paired differences:
│   ├─ p ≥ 0.05 (normality acceptable) → paired t-test
│   └─ p < 0.05 (non-normal)           → Wilcoxon signed-rank test
└─ NO → see ordinal_models, glmm_gee, or multiplicity skills
```

## Procedure

1. **Subset** the cleaned dataset to the target endpoint and the two visits
   (baseline, timepoint).
2. **Pivot** to wide format: one column per visit, one row per subject.
3. **Compute differences** `d = value_timepoint − value_baseline`.
4. **Shapiro-Wilk test** on `d`:
   - If *p* ≥ 0.05 → proceed with paired t-test.
   - If *p* < 0.05 → switch to Wilcoxon signed-rank.
5. **Paired t-test** (`scipy.stats.ttest_rel`):
   - Extract: *t* statistic, two-sided *p*-value.
   - Mean difference ± 95% CI: `mean_d ± t_crit × SE_d`.
6. **Effect size:** Cohen's dz = mean_d / sd_d.
7. **Practical significance:** compare `|mean_d|` (or signed mean_d
   depending on `direction`) to `practical_threshold`.
8. **Write reproducible script** to `workspace/{study_id}/scripts/paired_{endpoint}.py`.
9. **Write result JSON** to `workspace/{study_id}/results/paired_{endpoint}.json`.
10. **Emit audit event** with SHA-256 hashes of inputs and outputs.

## Output schema (StatisticalResult)

```json
{
  "endpoint": "corneometer_hydration",
  "data_type": "continuous",
  "model": "paired_t",
  "contrast": "D28 − D0",
  "estimate": 7.23,
  "ci95": [5.12, 9.34],
  "p_value": 0.00012,
  "p_adjusted": null,
  "p_adjustment_method": "none",
  "effect_size": 1.42,
  "effect_size_metric": "cohen_dz",
  "practical_threshold": 5.0,
  "practical_threshold_met": true,
  "n": 30,
  "n_complete": 28,
  "assumptions": {
    "normality_p": 0.34,
    "overall_ok": true
  },
  "conclusion": "Observed increase of +7.23 a.u.; effect is statistically significant and the practical threshold is met.",
  "artefacts": {
    "script": "scripts/paired_corneometer_hydration.py",
    "result_json": "results/paired_corneometer_hydration.json"
  }
}
```

## Hard rules

1. **Always report** effect estimate + 95% CI + p-value + effect size.
   A p-value alone is never sufficient.
2. If the difference distribution is **heavily skewed** (Shapiro p < 0.05),
   report the Wilcoxon p-value AND the Hodges-Lehmann estimator.
3. **Do not apply multiplicity** here. Multiplicity is the
   `multiplicity_claim_subagent`'s job. Leave `p_adjusted = null`.
4. If n < 5 paired observations, **refuse to run** and return an error.
5. The `practical_threshold_met` field must respect the `direction`:
   - direction = "increase" → met if mean_d ≥ threshold
   - direction = "decrease" → met if mean_d ≤ threshold (threshold is negative)
   - direction = "two_sided" → met if |mean_d| ≥ |threshold|

## References

- ISO 16128 (cosmetic ingredient definitions)
- Regulation (EC) No 1223/2009 on cosmetic products
- Commission Regulation (EU) No 655/2013 — Common Criteria for claims
- Fay, M. P. & Proschan, M. A. (2010). "Wilcoxon–Mann–Whitney or t-test?"
  *Statistics in Medicine*, 29(19), 2120–2127.
