---
name: missing_data
description: >
  Strategies for handling missing data in cosmetic clinical studies. Covers
  MCAR/MAR/MNAR mechanisms, complete-case analysis, MMRM under MAR,
  multiple imputation, tipping-point analysis, and pattern-mixture models.
tags: [statistics, missing-data, imputation, sensitivity, MAR, MCAR, MNAR]
---

# Missing data handling

## Classification of missingness

| Mechanism | Abbreviation | Definition                                        | Implication                        |
|-----------|-------------|----------------------------------------------------|------------------------------------|
| Missing completely at random | MCAR | Missingness unrelated to observed or unobserved data | Complete-case analysis unbiased    |
| Missing at random           | MAR  | Missingness related to observed data but not to unobserved | MMRM / MI unbiased               |
| Missing not at random       | MNAR | Missingness related to the unobserved value itself | No model is unbiased; sensitivity needed |

## Decision tree

```text
Is missingness > 20% at the primary timepoint?
├─ YES → BLOCKER. Flag for human review. Possible causes: dropout, protocol
│        violation, data-entry error. Investigate root cause before proceeding.
└─ NO
   ├─ Little's MCAR test p ≥ 0.05?
   │    ├─ YES → Assume MCAR. Complete-case is acceptable for primary
   │    │        analysis. Still run MMRM as sensitivity.
   │    └─ NO → Assume MAR.
   │             Primary: MMRM (unstructured covariance).
   │             Sensitivity 1: Multiple imputation (MI) with m ≥ 20.
   │             Sensitivity 2: Tipping-point analysis.
   └─ If pattern suggests MNAR → Pattern-mixture model as additional
      sensitivity. Document reasoning.
```

## Primary analysis: MMRM under MAR

See `skills/statistics/mmrm/SKILL.md`. Key points:

- Do **not** impute before fitting MMRM.
- Use REML estimation.
- Include baseline value as covariate.
- Unstructured covariance by default.

## Sensitivity: Multiple imputation (MI)

1. Impute m ≥ 20 datasets using `mice` (R) or `sklearn.impute.IterativeImputer`.
2. Fit the analysis model on each imputed dataset.
3. Pool using Rubin's rules.
4. Report: pooled estimate, pooled 95% CI, fraction of missing information.

## Sensitivity: Tipping-point analysis

1. Systematically shift the imputed values for missing observations by
   increments δ₁, δ₂, … (e.g. 0, −1, −2, −3, −5 for a hydration endpoint).
2. Re-run the primary analysis for each δ.
3. Identify the δ at which the conclusion changes (p crosses 0.05 or
   practical threshold is no longer met).
4. Report: "The result is robust unless missing values are shifted by
   ≥ X units in the unfavourable direction."

## Sensitivity: Pattern-mixture model

1. Group subjects by their dropout pattern.
2. Fit the model within each pattern.
3. Average estimates across patterns, weighted by pattern frequency.
4. Variant: **jump-to-reference** — assume dropouts revert to baseline.

## Hard rules

1. **Never** silently drop missing data. Always document the mechanism and
   the chosen strategy.
2. **Never** use LOCF (Last Observation Carried Forward) as primary; it is
   not a valid estimand-aligned approach.
3. If a primary endpoint has > 20% missingness at the primary timepoint,
   the pipeline MUST pause and request human review.
4. At least one sensitivity analysis is mandatory for confirmatory studies.

## References

- National Research Council (2010). *The Prevention and Treatment of Missing
  Data in Clinical Trials*. National Academies Press.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed.
  CRC Press.
- ICH E9(R1) Addendum on estimands and sensitivity analysis.
