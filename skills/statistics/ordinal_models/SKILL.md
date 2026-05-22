---
name: ordinal_models
description: >
  Ordinal mixed models and non-parametric tests for endpoints measured on
  ordered scales (Likert, clinical grading 0–4, visual analogue scale
  categories). Includes Wilcoxon signed-rank for 2 timepoints and cumulative
  link mixed models (CLMM) for longitudinal data.
tags: [statistics, ordinal, likert, wilcoxon, clmm, longitudinal]
---

# Ordinal models for graded cosmetic endpoints

## When to use

- **Endpoint data type:** ordinal (e.g. dermatologist grading 0–4, Likert
  scale 1–5, visual analogue scale binned into categories).
- **Design:** before/after or longitudinal.

## Decision logic

```text
Ordinal endpoint?
├─ 2 timepoints → Wilcoxon signed-rank test on paired differences
└─ ≥ 3 timepoints → Cumulative Link Mixed Model (CLMM / ordinal mixed)
```

## Wilcoxon signed-rank (2 timepoints)

1. Compute paired differences (post − pre) on the ordinal scale.
2. Apply `scipy.stats.wilcoxon(d, alternative=...)`.
3. Report the Hodges-Lehmann pseudo-median and its 95% CI.
4. Effect size: rank-biserial correlation *r* = Z / √n.

## Cumulative Link Mixed Model (≥ 3 timepoints)

A CLMM models the log-odds of being at or below each ordinal category:

```
logit P(Y ≤ k) = θ_k − (β_visit × visit + u_subject)
```

- **Link:** logit (default), probit, or complementary log-log.
- **Fixed:** visit (categorical).
- **Random:** intercept per subject.
- **Threshold parameters** θ_k estimate the cut-points.
- Fit via R's `ordinal::clmm()` (bridged through `pymer4`) or a Python
  fallback using `statsmodels.MixedLM` on a latent-variable approximation.

### Reporting

- Odds ratio for a one-category improvement at D28 vs D0.
- 95% CI on the OR.
- Proportional-odds assumption test (Brant test).

## Hard rules

1. **Never** treat ordinal data as continuous (no mean ± SD). Report medians,
   IQR, and proportions per category.
2. If the proportional-odds assumption is violated (Brant p < 0.05), fit a
   partial-proportional-odds model or fall back to Wilcoxon and document the
   deviation.
3. Do not apply multiplicity here.

## References

- Agresti, A. (2010). *Analysis of Ordinal Categorical Data*, 2nd ed. Wiley.
- Christensen, R. H. B. (2019). "ordinal — Regression Models for Ordinal
  Data." R package.
