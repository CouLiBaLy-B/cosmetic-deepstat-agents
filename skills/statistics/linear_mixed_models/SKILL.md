---
name: linear_mixed_models
description: >
  Fit linear mixed-effects models (LMM) for longitudinal cosmetic studies
  with ≥3 timepoints and continuous endpoints. Random intercepts/slopes per
  subject, fixed effects for visit, optional covariates.
tags: [statistics, lmm, mixed-model, longitudinal, continuous]
---

# Linear mixed models (LMM) for longitudinal efficacy

## When to use

- **Design:** before/after longitudinal with **≥ 3 timepoints** (e.g. D0,
  D7, D14, D28).
- **Endpoint data type:** continuous.
- **Single-group** or **parallel-group** (add treatment arm as fixed effect).
- When interest lies in the **trajectory over time**, not just a single
  pairwise contrast.

## Default model specification

```
value ~ visit + (1 | subject_id)
```

- **Fixed effect:** visit (categorical) — estimates mean change at each
  timepoint relative to baseline.
- **Random effect:** random intercept per subject (accounts for
  within-subject correlation).
- **Optional covariates:** age, baseline value, skin type, site (when
  multi-center).
- **Optional random slope:** `(visit | subject_id)` when there is enough
  data (≥ 20 subjects × 4 visits).

## Procedure

1. **Fit** via `statsmodels.MixedLM` (Python) or `pymer4.Lmer` (R bridge).
2. **Extract** the fixed-effect contrasts of interest (e.g. D28 − D0).
3. **95% CI** from the mixed-model standard errors (Satterthwaite or
   Kenward-Roger d.f. when available).
4. **Assumptions:**
   - Residual normality: Shapiro-Wilk on level-1 residuals.
   - Homoscedasticity: visual (residuals vs fitted) or Levene's test by visit.
   - Linearity: not assumed if visit is categorical.
5. **Effect size:** standardised mean difference using the pooled
   within-subject SD.
6. **Write** reproducible script and result JSON.

## Output schema

Same `StatisticalResult` as paired_tests, with `model = "LMM"`.

## Hard rules

1. **Never** drop subjects with partial data — mixed models handle MAR
   missingness via REML.
2. The primary contrast MUST be pre-specified in the SAP. Do not fish for
   the "best" timepoint.
3. **Never** apply multiplicity inside this skill. Leave `p_adjusted = null`.
4. If the random-effects covariance matrix fails to converge, simplify to
   random-intercept only and note this in `assumptions.notes`.

## Relation to MMRM

LMM uses random effects; MMRM uses an unstructured (co)variance for the
repeated factor. In cosmetic studies the practical difference is small; MMRM
is preferred by some regulatory guidelines. See `skills/statistics/mmrm/`.

## References

- Fitzmaurice, G. M., Laird, N. M., & Ware, J. H. (2011). *Applied
  Longitudinal Analysis*, 2nd ed. Wiley.
- Bates, D., Mächler, M., Bolker, B., Walker, S. (2015). "Fitting Linear
  Mixed-Effects Models Using lme4." *J. Stat. Software*, 67(1).
