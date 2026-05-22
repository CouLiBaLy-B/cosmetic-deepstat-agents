# Organisational statistical policy

## Default analysis choices

- **Primary analysis model:** MMRM under MAR for longitudinal studies;
  paired t-test (or Wilcoxon if non-normal) for 2-timepoint designs.
- **Multiplicity correction:** Holm for confirmatory endpoints;
  BH-FDR for exploratory/secondary.
- **Missing data primary strategy:** MMRM (no imputation).
- **Sensitivity analysis:** at least one of: tipping-point, pattern-mixture,
  multiple imputation.
- **Significance level:** α = 0.05 (two-sided) for confirmatory;
  α = 0.10 for equivalence (TOST).
- **Effect size metric:** Cohen's dz for paired designs; standardised mean
  difference for parallel.
- **Practical threshold:** mandatory for every primary endpoint (no purely
  statistical significance claims).

## Reporting requirements

- Every result must include: estimate, 95% CI, p-value, effect size.
- Multiplicity-adjusted p-values for all confirmatory endpoints.
- No result without a reproducible script in `scripts/`.
- Package versions recorded in `audit/package_versions.json`.

## Forbidden practices

- LOCF (Last Observation Carried Forward) as primary analysis.
- p-hacking: running multiple models and reporting only the best.
- Converting exploratory findings into confirmatory claims.
- Reporting p-values without confidence intervals.
