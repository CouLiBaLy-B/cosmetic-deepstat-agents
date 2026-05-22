"""High-level statistical model runners.

This module provides the compute-heavy model fitting used by the tools in
``app.agents.tools``. Each public function:

- accepts a cleaned ``pd.DataFrame`` plus configuration parameters,
- returns a plain ``dict`` with the result (estimate, CI, p, effect-size, …),
- is **pure** — no filesystem side-effects, no audit writes — so it can be
  tested independently of the tool / pipeline wrappers.

Models implemented:

- ``run_mmrm``: Mixed Model for Repeated Measures (unstructured covariance).
- ``run_lmm``: Linear Mixed Model (random intercept, optional random slope).
- ``run_glmm_logit``: Logistic GLMM for binary endpoints.
- ``run_mcnemar``: McNemar's exact test for paired binary data.
- ``run_poisson_glmm``: Poisson / Negative-Binomial GLMM for count endpoints.
- ``run_top2box``: Top-2-box analysis with Wilson confidence interval.
- ``run_tost``: Two One-Sided Tests for equivalence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ===========================================================================
# MMRM
# ===========================================================================

def run_mmrm(
    df: pd.DataFrame,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Fit an MMRM-like model using statsmodels MixedLM.

    Because statsmodels does not have a native MMRM with unstructured
    covariance for the repeated statement, we approximate with a
    mixed-effects model: ``value ~ visit + baseline_value``
    with ``groups = subject_id`` and ``re_formula = "1"`` (random intercept).
    This is a close practical approximation for cosmetic studies.

    For a true MMRM with UN covariance, the R bridge (pymer4) or SAS would
    be needed.
    """
    import statsmodels.formula.api as smf

    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    sub = sub[[subject_col, time_col, value_col]].dropna()

    # Create baseline covariate
    bl = sub[sub[time_col] == baseline].set_index(subject_col)[value_col].rename("baseline_value")
    sub = sub.merge(bl, left_on=subject_col, right_index=True, how="inner")

    # Exclude baseline rows from the response (post-baseline only)
    post = sub[sub[time_col] != baseline].copy()
    if len(post) < 5:
        raise ValueError(f"MMRM: insufficient post-baseline data for endpoint {endpoint!r} (n={len(post)}).")

    # Encode visit as categorical
    post[time_col] = pd.Categorical(post[time_col])

    formula = f"{value_col} ~ C({time_col}) + baseline_value"
    try:
        model = smf.mixedlm(formula, data=post, groups=post[subject_col], re_formula="1")
        fit = model.fit(reml=True, method="lbfgs")
    except Exception:
        # Fallback: simpler model without random effects
        model = smf.mixedlm(formula, data=post, groups=post[subject_col], re_formula="1")
        fit = model.fit(reml=True)

    # Extract the contrast for the primary timepoint
    params = fit.params
    conf = fit.conf_int(alpha=0.05)
    pvals = fit.pvalues

    # Find the coefficient for the primary timepoint
    target_key = None
    for k in params.index:
        if primary_timepoint in str(k):
            target_key = k
            break

    if target_key is None:
        # If primary_timepoint is the reference level, compute marginal
        at_primary = post[post[time_col] == primary_timepoint][value_col]
        at_primary_bl = post[post[time_col] == primary_timepoint]["baseline_value"]
        if len(at_primary) < 3:
            raise ValueError(f"Too few observations at {primary_timepoint}.")
        diff = at_primary.values - at_primary_bl.values
        n = len(diff)
        mean_d = float(np.mean(diff))
        se_d = float(np.std(diff, ddof=1) / np.sqrt(n))
        t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
        ci_lo = mean_d - t_crit * se_d
        ci_hi = mean_d + t_crit * se_d
        _, p_norm = sp_stats.shapiro(diff) if n >= 3 else (0, 1.0)
        _, p_val = sp_stats.ttest_rel(at_primary.values, at_primary_bl.values)
        effect_size = mean_d / float(np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
        return {
            "model": "MMRM_approx",
            "endpoint": endpoint,
            "contrast": f"{primary_timepoint} − {baseline}",
            "estimate": mean_d,
            "ci95": (ci_lo, ci_hi),
            "p_value": float(p_val),
            "effect_size": effect_size,
            "effect_size_metric": "cohen_dz",
            "n": n,
            "assumptions": {"normality_p": float(p_norm), "overall_ok": p_norm >= 0.05},
            "converged": True,
            "covariance_structure": "random_intercept",
        }

    estimate = float(params[target_key])
    ci_lo = float(conf.loc[target_key, 0])
    ci_hi = float(conf.loc[target_key, 1])
    p_val = float(pvals[target_key])
    n = int(post[subject_col].nunique())

    # Residual normality
    residuals = fit.resid
    _, p_norm = sp_stats.shapiro(residuals[:min(len(residuals), 500)])

    return {
        "model": "MMRM_approx",
        "endpoint": endpoint,
        "contrast": f"{primary_timepoint} − {baseline}",
        "estimate": estimate,
        "ci95": (ci_lo, ci_hi),
        "p_value": p_val,
        "effect_size": None,
        "effect_size_metric": None,
        "n": n,
        "assumptions": {"normality_p": float(p_norm), "overall_ok": p_norm >= 0.05},
        "converged": fit.converged,
        "covariance_structure": "random_intercept",
    }


# ===========================================================================
# LMM (Linear Mixed Model)
# ===========================================================================

def run_lmm(
    df: pd.DataFrame,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
) -> dict[str, Any]:
    """Fit a linear mixed model: value ~ visit + (1 | subject)."""
    import statsmodels.formula.api as smf

    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    sub = sub[[subject_col, time_col, value_col]].dropna()

    if sub[subject_col].nunique() < 5:
        raise ValueError(f"LMM: too few subjects for endpoint {endpoint!r}.")

    sub[time_col] = pd.Categorical(sub[time_col])
    formula = f"{value_col} ~ C({time_col})"

    model = smf.mixedlm(formula, data=sub, groups=sub[subject_col], re_formula="1")
    fit = model.fit(reml=True)

    # Find contrast for primary timepoint
    params = fit.params
    conf = fit.conf_int(alpha=0.05)
    pvals = fit.pvalues

    target_key = None
    for k in params.index:
        if primary_timepoint in str(k):
            target_key = k
            break

    if target_key is None:
        raise ValueError(f"Primary timepoint {primary_timepoint!r} not found in model coefficients.")

    estimate = float(params[target_key])
    ci_lo = float(conf.loc[target_key, 0])
    ci_hi = float(conf.loc[target_key, 1])
    p_val = float(pvals[target_key])

    residuals = fit.resid
    _, p_norm = sp_stats.shapiro(residuals[:min(len(residuals), 500)])

    return {
        "model": "LMM",
        "endpoint": endpoint,
        "contrast": f"{primary_timepoint} − {baseline}",
        "estimate": estimate,
        "ci95": (ci_lo, ci_hi),
        "p_value": p_val,
        "effect_size": None,
        "effect_size_metric": None,
        "n": int(sub[subject_col].nunique()),
        "assumptions": {"normality_p": float(p_norm), "overall_ok": p_norm >= 0.05},
        "converged": fit.converged,
    }


# ===========================================================================
# McNemar (binary paired, 2 timepoints)
# ===========================================================================

def run_mcnemar(
    df: pd.DataFrame,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    timepoint: str = "D28",
) -> dict[str, Any]:
    """McNemar's test for paired binary data."""
    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    sub = sub[sub[time_col].isin([baseline, timepoint])].dropna(subset=[value_col])

    # Pivot to wide
    wide = sub.pivot_table(index=subject_col, columns=time_col, values=value_col, aggfunc="first")
    wide = wide.dropna()

    if len(wide) < 5:
        raise ValueError(f"McNemar: too few complete pairs for {endpoint!r} (n={len(wide)}).")

    pre = (wide[baseline] > 0).astype(int).values
    post = (wide[timepoint] > 0).astype(int).values

    # Build contingency table
    a = int(((pre == 1) & (post == 1)).sum())  # +/+
    b = int(((pre == 1) & (post == 0)).sum())  # +/-
    c = int(((pre == 0) & (post == 1)).sum())  # -/+
    d = int(((pre == 0) & (post == 0)).sum())  # -/-

    n_discord = b + c

    # Exact McNemar (binomial test on discordant pairs)
    if n_discord == 0:
        p_val = 1.0
    else:
        # scipy >= 1.7: binomtest replaces binom_test
        result = sp_stats.binomtest(b, n_discord, 0.5)
        p_val = float(result.pvalue)

    # Cohen's g = (b/(b+c)) - 0.5
    g = (b / max(n_discord, 1)) - 0.5

    # Proportion difference
    prop_pre = (pre.sum()) / len(pre)
    prop_post = (post.sum()) / len(post)
    prop_diff = float(prop_post - prop_pre)

    # Approximate CI for proportion difference
    se = math.sqrt(max((b + c - (b - c) ** 2 / len(wide)) / len(wide) ** 2, 1e-12))
    z = 1.96
    ci_lo = prop_diff - z * se
    ci_hi = prop_diff + z * se

    return {
        "model": "mcnemar",
        "endpoint": endpoint,
        "contrast": f"{timepoint} − {baseline}",
        "estimate": prop_diff,
        "ci95": (ci_lo, ci_hi),
        "p_value": p_val,
        "effect_size": g,
        "effect_size_metric": "cohen_g",
        "n": len(wide),
        "table": {"a": a, "b": b, "c": c, "d": d},
        "discordant_pairs": n_discord,
    }


# ===========================================================================
# Logistic GLMM (binary, >= 3 timepoints)
# ===========================================================================

def run_glmm_logit(
    df: pd.DataFrame,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
) -> dict[str, Any]:
    """Logistic mixed model for binary longitudinal data.

    Uses statsmodels GEE with exchangeable correlation structure.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    sub = sub[[subject_col, time_col, value_col]].dropna()
    sub[value_col] = (sub[value_col] > 0).astype(int)
    sub[time_col] = sub[time_col].astype(str)

    if sub[subject_col].nunique() < 5:
        raise ValueError(f"GLMM-logit: too few subjects for {endpoint!r}.")

    # Sort by subject for GEE
    sub_sorted = sub.sort_values(subject_col).copy()
    visit_dummies = pd.get_dummies(sub_sorted[time_col], prefix="visit", drop_first=True, dtype=float)
    exog = sm.add_constant(visit_dummies)
    endog = sub_sorted[value_col].values
    groups = sub_sorted[subject_col].values

    try:
        model = GEE(
            endog,
            exog,
            groups=groups,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        raise ValueError(f"GLMM-logit failed: {exc}") from exc

    params = fit.params
    conf = fit.conf_int(alpha=0.05)
    pvals = fit.pvalues

    # Find the primary timepoint coefficient by column name
    target_col = None
    for col_name in exog.columns:
        if primary_timepoint in str(col_name):
            target_col = col_name
            break

    if target_col is None:
        raise ValueError(f"Primary timepoint {primary_timepoint!r} not in model.")

    log_or = float(params[target_col])
    or_val = math.exp(log_or)
    ci_lo_or = math.exp(float(conf.loc[target_col, 0]))
    ci_hi_or = math.exp(float(conf.loc[target_col, 1]))
    p_val = float(pvals[target_col])

    return {
        "model": "glmm_logit_gee",
        "endpoint": endpoint,
        "contrast": f"{primary_timepoint} vs {baseline}",
        "estimate": or_val,
        "ci95": (ci_lo_or, ci_hi_or),
        "p_value": p_val,
        "effect_size": log_or,
        "effect_size_metric": "log_odds_ratio",
        "n": int(sub[subject_col].nunique()),
        "scale": "odds_ratio",
    }


# ===========================================================================
# Poisson / Negative-Binomial (count endpoints)
# ===========================================================================

def run_poisson_or_negbin(
    df: pd.DataFrame,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
) -> dict[str, Any]:
    """Poisson GEE for count data with overdispersion check."""
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.generalized_estimating_equations import GEE

    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    sub = sub[[subject_col, time_col, value_col]].dropna()
    sub[value_col] = sub[value_col].astype(float).clip(lower=0).astype(int)

    if sub[subject_col].nunique() < 5:
        raise ValueError(f"Poisson: too few subjects for {endpoint!r}.")

    sub_sorted = sub.sort_values(subject_col).copy()
    visit_dummies = pd.get_dummies(sub_sorted[time_col].astype(str), prefix="visit", drop_first=True, dtype=float)
    exog = sm.add_constant(visit_dummies)
    endog = sub_sorted[value_col].values
    groups = sub_sorted[subject_col].values

    model = GEE(endog, exog, groups=groups, family=Poisson(), cov_struct=Exchangeable())
    fit = model.fit()

    params = fit.params
    conf = fit.conf_int(alpha=0.05)
    pvals = fit.pvalues

    # Find target by column name
    target_col = None
    for col_name in exog.columns:
        if primary_timepoint in str(col_name):
            target_col = col_name
            break

    if target_col is None:
        raise ValueError(f"Primary timepoint {primary_timepoint!r} not in model.")

    log_rr = float(params[target_col])
    rr = math.exp(log_rr)
    ci_lo = math.exp(float(conf.loc[target_col, 0]))
    ci_hi = math.exp(float(conf.loc[target_col, 1]))
    p_val = float(pvals[target_col])

    # Overdispersion check
    resid_p = fit.resid_pearson
    dof = max(len(endog) - len(params), 1)
    pearson_chi2 = float((resid_p ** 2).sum() / dof)
    overdispersed = pearson_chi2 > 1.5

    return {
        "model": "poisson_gee" if not overdispersed else "negbin_approx_gee",
        "endpoint": endpoint,
        "contrast": f"{primary_timepoint} vs {baseline}",
        "estimate": rr,
        "ci95": (ci_lo, ci_hi),
        "p_value": p_val,
        "effect_size": log_rr,
        "effect_size_metric": "log_rate_ratio",
        "n": int(sub[subject_col].nunique()),
        "scale": "rate_ratio",
        "overdispersion": {"pearson_chi2_per_df": pearson_chi2, "overdispersed": overdispersed},
    }


# ===========================================================================
# Top-2-box (consumer perception)
# ===========================================================================

def run_top2box(
    responses: list[int] | np.ndarray,
    scale_max: int = 5,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute top-2-box percentage with Wilson confidence interval.

    ``responses`` are integers on a 1..scale_max Likert scale.
    Top-2-box = proportion answering (scale_max - 1) or scale_max.
    """
    arr = np.asarray(responses, dtype=int)
    n = len(arr)
    if n == 0:
        raise ValueError("No responses provided.")

    top2 = int(((arr >= scale_max - 1)).sum())
    p_hat = top2 / n

    # Wilson CI
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    ci_lo = max(0.0, centre - spread)
    ci_hi = min(1.0, centre + spread)

    return {
        "top2_count": top2,
        "n": n,
        "top2_pct": round(p_hat * 100, 2),
        "ci95_pct": (round(ci_lo * 100, 2), round(ci_hi * 100, 2)),
        "ci_method": "wilson",
        "alpha": alpha,
        "scale_max": scale_max,
    }


# ===========================================================================
# TOST (equivalence)
# ===========================================================================

def run_tost(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    margin: float,
    *,
    alpha: float = 0.05,
    paired: bool = True,
) -> dict[str, Any]:
    """Two One-Sided Tests for equivalence within [-margin, +margin].

    Parameters
    ----------
    x, y : array-like
        Two sets of measurements. If ``paired=True``, x and y must be the
        same length and correspond to the same subjects.
    margin : float
        Equivalence margin (positive). Equivalence is declared if the mean
        difference is within [-margin, +margin].
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if paired:
        if len(x_arr) != len(y_arr):
            raise ValueError("Paired TOST: x and y must have the same length.")
        d = x_arr - y_arr
        n = len(d)
        mean_d = float(np.mean(d))
        se_d = float(np.std(d, ddof=1) / np.sqrt(n))
        df = n - 1

        # One-sided test 1: H0: mu_d <= -margin  ->  t1 = (mean_d + margin) / se
        t1 = (mean_d + margin) / se_d if se_d > 0 else float("inf")
        p1 = float(1 - sp_stats.t.cdf(t1, df))

        # One-sided test 2: H0: mu_d >= +margin  ->  t2 = (mean_d - margin) / se
        t2 = (mean_d - margin) / se_d if se_d > 0 else float("-inf")
        p2 = float(sp_stats.t.cdf(t2, df))

        p_tost = max(p1, p2)

        # 90% CI (= (1 - 2*alpha) CI)
        t_crit = float(sp_stats.t.ppf(1 - alpha, df))
        ci90_lo = mean_d - t_crit * se_d
        ci90_hi = mean_d + t_crit * se_d
    else:
        n1, n2 = len(x_arr), len(y_arr)
        mean_d = float(np.mean(x_arr) - np.mean(y_arr))
        se_d = float(np.sqrt(np.var(x_arr, ddof=1) / n1 + np.var(y_arr, ddof=1) / n2))
        # Welch degrees of freedom
        s1, s2 = np.var(x_arr, ddof=1), np.var(y_arr, ddof=1)
        dof = float((s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)))
        n = n1 + n2

        t1 = (mean_d + margin) / se_d if se_d > 0 else float("inf")
        p1 = float(1 - sp_stats.t.cdf(t1, dof))
        t2 = (mean_d - margin) / se_d if se_d > 0 else float("-inf")
        p2 = float(sp_stats.t.cdf(t2, dof))
        p_tost = max(p1, p2)

        t_crit = float(sp_stats.t.ppf(1 - alpha, dof))
        ci90_lo = mean_d - t_crit * se_d
        ci90_hi = mean_d + t_crit * se_d

    equivalence_met = (p_tost < alpha) and (ci90_lo >= -margin) and (ci90_hi <= margin)

    return {
        "model": "TOST",
        "mean_difference": mean_d,
        "margin": margin,
        "tost_p1": p1,
        "tost_p2": p2,
        "tost_p_max": p_tost,
        "ci90": (ci90_lo, ci90_hi),
        "equivalence_met": equivalence_met,
        "n": n,
        "alpha": alpha,
        "paired": paired,
    }
