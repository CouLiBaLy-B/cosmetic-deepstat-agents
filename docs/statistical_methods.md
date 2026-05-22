# Statistical methods reference

> This document describes every statistical method implemented in
> CosmeticDeepStat Agents, when each is selected, and the assumptions
> checked automatically.

## 1. Method selection decision table

The pipeline selects the model based on **endpoint data type**, **study
design**, and **number of timepoints**. The decision table is implemented
in `app/agents/tools._impl_choose_test`.

| Data type   | Design              | Timepoints | Normality OK | Model selected            |
|-------------|---------------------|:----------:|:------------:|---------------------------|
| continuous  | before_after        | 2          | yes          | `paired_t`                |
| continuous  | before_after        | 2          | no           | `wilcoxon_signed_rank`    |
| continuous  | before_after_longitudinal | ≥3   | —            | `MMRM` (approx.)         |
| ordinal     | before_after        | 2          | —            | `wilcoxon_signed_rank`    |
| ordinal     | before_after_longitudinal | ≥3   | —            | `ordinal_mixed` (CLMM)   |
| binary      | before_after        | 2          | —            | `mcnemar`                 |
| binary      | before_after_longitudinal | ≥3   | —            | `glmm_logit` (GEE)       |
| count       | before_after        | 2          | —            | `poisson_or_negbin`       |
| count       | before_after_longitudinal | ≥3   | —            | `poisson_or_negbin` (GEE) |

## 2. Paired t-test

**When:** continuous endpoint, 2 timepoints, Shapiro-Wilk p ≥ 0.05 on
paired differences.

**Implementation:** `scipy.stats.ttest_rel` on d = value(T) − value(T0).

**Output:**
- Mean difference ± 95% CI (t-distribution)
- Cohen's dz = mean(d) / SD(d)
- p-value (two-sided or one-sided per SAP `direction`)
- Practical-threshold assessment

**Assumptions checked:**
- Shapiro-Wilk on paired differences → if p < 0.05, switch to Wilcoxon

**Reference:** `app/agents/tools._impl_run_paired_test`

## 3. Wilcoxon signed-rank test

**When:** continuous or ordinal endpoint, 2 timepoints, non-normal
paired differences.

**Implementation:** `scipy.stats.wilcoxon` with exact or approximated
p-value. Effect size: rank-biserial *r* = Z / √n.

**Reference:** `app/agents/tools._impl_run_paired_test` (automatic
fallback from paired-t).

## 4. MMRM (Mixed Model for Repeated Measures)

**When:** continuous endpoint, ≥ 3 timepoints, longitudinal design.

**Implementation:** `statsmodels.MixedLM` with random intercept per
subject and baseline value as covariate. This is a practical
approximation of the true MMRM with unstructured covariance (the latter
requires R/SAS).

**Model:** `value ~ C(visit) + baseline_value`, `groups = subject_id`,
`re_formula = "1"`.

**Output:**
- Fixed-effect coefficient for the primary timepoint
- 95% CI from model standard errors
- Residual normality check (Shapiro-Wilk on residuals)
- Convergence flag

**Fallback:** if the primary timepoint is the reference level, the
estimate is computed as marginal mean − baseline mean.

**Reference:** `app/services/statistics_runner.run_mmrm`

## 5. Linear Mixed Model (LMM)

**When:** continuous endpoint, ≥ 3 timepoints (alternative to MMRM,
without the baseline covariate).

**Model:** `value ~ C(visit)`, random intercept per subject.

**Reference:** `app/services/statistics_runner.run_lmm`

## 6. McNemar's exact test

**When:** binary endpoint, 2 timepoints.

**Implementation:** `scipy.stats.binomtest` on discordant pairs.

**Output:**
- Proportion difference (post − pre)
- 95% CI (approximate)
- Exact p-value
- Cohen's g = b/(b+c) − 0.5
- 2×2 contingency table (a, b, c, d)

**Reference:** `app/services/statistics_runner.run_mcnemar`

## 7. Logistic GLMM (GEE)

**When:** binary endpoint, ≥ 3 timepoints.

**Implementation:** `statsmodels.GEE` with binomial family and
exchangeable correlation structure.

**Output:**
- Odds ratio (exponentiated coefficient)
- 95% CI on OR scale
- Wald p-value
- Log-odds ratio as effect size

**Reference:** `app/services/statistics_runner.run_glmm_logit`

## 8. Poisson / Negative-Binomial GEE

**When:** count endpoint, any design.

**Implementation:** `statsmodels.GEE` with Poisson family. If the
Pearson χ²/df > 1.5 (overdispersion), the model is labelled
`negbin_approx_gee`.

**Output:**
- Rate ratio (exponentiated coefficient)
- 95% CI on RR scale
- p-value
- Overdispersion diagnostics

**Reference:** `app/services/statistics_runner.run_poisson_or_negbin`

## 9. Top-2-box analysis

**When:** consumer perception endpoint (Likert scale).

**Implementation:** proportion of respondents selecting the top 2 values
on the scale (e.g. 4 or 5 on a 1–5 scale).

**Output:**
- Top-2-box count and percentage
- 95% Wilson confidence interval
- n

**Reference:** `app/services/statistics_runner.run_top2box`

## 10. TOST (Two One-Sided Tests)

**When:** equivalence claim with a pre-specified margin.

**Implementation:** paired or unpaired TOST. Equivalence declared if
max(p₁, p₂) < α AND the 90% CI is entirely within [−δ, +δ].

**Output:**
- Mean difference
- tost_p1, tost_p2, tost_p_max
- 90% CI
- equivalence_met (bool)

**Reference:** `app/services/statistics_runner.run_tost`

## 11. Multiplicity correction

**Methods implemented** (`app/agents/tools._impl_apply_multiplicity`):

| Method      | Description                                        | Default for     |
|-------------|----------------------------------------------------|-----------------|
| `holm`      | Holm step-down (FWER control)                      | Confirmatory    |
| `bonferroni`| Bonferroni (most conservative FWER)                | Fallback        |
| `hochberg`  | Hochberg step-up (less conservative than Bonferroni)| Alternative    |
| `bh_fdr`    | Benjamini-Hochberg (FDR control)                   | Exploratory     |
| `by_fdr`    | Benjamini-Yekutieli (FDR, dependent tests)         | Dependent tests |

The pipeline default for ≥ 2 primary endpoints is **Holm**.

## 12. Missing data strategy

| Primary analysis | Sensitivity analyses (planned) |
|------------------|-------------------------------|
| MMRM under MAR (no imputation) | Tipping-point analysis |
| Complete-case (if MCAR) | Pattern-mixture model |
| — | Multiple imputation (m ≥ 20) |

See `skills/statistics/missing_data/SKILL.md` for full decision tree.

## 13. Practical significance

Every endpoint has a `practical_threshold` defined in the study metadata.
A result is considered practically significant if:

- `direction = "increase"` → estimate ≥ threshold
- `direction = "decrease"` → estimate ≤ threshold (threshold is negative)
- `direction = "two_sided"` → |estimate| ≥ |threshold|

Claim substantiation requires **both** statistical significance
(adjusted p < 0.05) **and** practical significance to reach the
`confirmed` support level.

## 14. Packages and versions

All statistical computations are performed by:

| Package       | Role                                    |
|---------------|-----------------------------------------|
| `scipy`       | t-test, Wilcoxon, binomtest, Shapiro    |
| `statsmodels` | MixedLM, GEE, formula interface         |
| `numpy`       | Array operations                         |
| `pandas`      | Data manipulation                        |
| `pingouin`    | Effect sizes, convenience wrappers       |

Package versions are recorded per study in
`workspace/{study_id}/audit/package_versions.json`.
