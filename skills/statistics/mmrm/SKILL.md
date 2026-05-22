---
name: mmrm
description: >
  Mixed Model for Repeated Measures (MMRM) — the gold-standard longitudinal
  model for clinical studies with continuous endpoints. Uses unstructured
  covariance, no random effects, Satterthwaite/Kenward-Roger degrees of
  freedom. Preferred under MAR assumption.
tags: [statistics, mmrm, longitudinal, continuous, regulatory]
---

# MMRM — Mixed Model for Repeated Measures

## When to use

- Longitudinal study with **≥ 3 timepoints**, continuous endpoint.
- Preferred over simple LMM when regulatory guidance recommends it (e.g.
  ICH E9(R1)).
- Default for **confirmatory primary analysis** under MAR.

## Model specification

```
value ~ baseline_value + visit + baseline_value:visit + (visit | subject_id)
```

In statsmodels / R this becomes a `MixedLM` or `nlme::lme` with:

- **Fixed effects:** baseline value (covariate), visit (categorical),
  baseline × visit interaction (optional).
- **Repeated factor:** visit within subject, with **unstructured**
  covariance (UN). Fall back to compound-symmetry (CS) if UN fails to
  converge with a note.
- **No random effects** in the strict MMRM sense — the within-subject
  covariance is modelled through the repeated statement.

## Procedure

1. **Prepare** dataset in long format: `subject_id`, `visit`, `value`,
   `baseline_value` (value at D0 for each subject).
2. **Fit** using `statsmodels.MixedLM` with `groups="subject_id"` and
   `re_formula="0"` (no random effects), plus a structured covariance for
   visit within subject.
   - *Alternative:* use `formulaic` + `pymer4.Lmer` with
     `family='gaussian'` and `correlation=corSymm(form=~1|subject_id)`.
3. **Extract** least-squares means (LS-means) at the primary timepoint.
4. **Contrast:** LS-mean at Tx (e.g. D28) minus LS-mean at baseline
   (or active − placebo in a parallel design).
5. **95% CI** using Kenward-Roger or Satterthwaite d.f.
6. **Assumptions:**
   - Residual normality (QQ-plot + Shapiro-Wilk on residuals).
   - Homoscedasticity by visit.
   - Missing data mechanism: MAR assumed; sensitivity via tipping-point or
     pattern-mixture model.
7. **Write** script, result JSON, residual-diagnostic figures.

## Covariance structures (preference order)

| Abbreviation | Name                  | # params (T visits) | When to use                |
|--------------|-----------------------|---------------------|-----------------------------|
| UN           | Unstructured          | T(T+1)/2            | Default — most flexible     |
| AR(1)        | Autoregressive(1)     | 2                   | Equally-spaced visits       |
| CS           | Compound symmetry     | 2                   | Fallback if UN doesn't converge |
| Toeplitz     | Banded Toeplitz       | T                   | Equally-spaced, declining corr. |

## Hard rules

1. **Always** include baseline value as a covariate.
2. **Always** start with UN covariance; document the switch if you fall back.
3. **Never** impute missing values before fitting — MMRM handles MAR
   internally via restricted maximum likelihood (REML).
4. Report the **LS-mean difference**, not just the raw visit mean difference.
5. **Pre-specify** the primary contrast in the SAP.

## Sensitivity analyses

- Tipping-point analysis for missing data.
- Pattern-mixture model (reference-based, jump-to-reference).
- Robust sandwich estimator for misspecified covariance.

## References

- Mallinckrodt, C. H., Lane, P. W., Schnell, D., et al. (2008).
  "Recommendations for the Primary Analysis of Continuous Endpoints in
  Longitudinal Clinical Trials." *Drug Information Journal*, 42(4), 303–319.
- ICH E9(R1) Addendum on estimands.
- National Research Council (2010). *The Prevention and Treatment of Missing
  Data in Clinical Trials*. National Academies Press.
