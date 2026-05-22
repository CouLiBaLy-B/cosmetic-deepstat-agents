"""Deterministic Python tools exposed to (a) the LLM agents via `@tool`,
(b) the deterministic fallback pipeline via plain function calls, and
(c) the unit tests.

Design rules common to every tool here:

- the underlying logic lives in a private ``_impl_xxx`` function returning a
  dict — it is plain Python, no LLM, no side effect other than the documented
  filesystem write(s) under ``workspace/{study_id}/...``;
- the public name ``xxx_tool`` is a langchain ``@tool`` wrapper, so the agent
  can call it. The wrapper just delegates to ``_impl_xxx``;
- every tool validates its inputs (study_id pattern, file existence),
- every tool emits an ``AuditEvent`` via ``app.core.audit.write_audit_event``,
- long outputs (parquet / json / png) are written to disk and only the **path
  + a compact summary** is returned to the LLM.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
import pandas as pd

from app.core.audit import hash_file, write_audit_event
from app.core.paths import StudyWorkspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_DATA_REQUIRED_COLS = {"subject_id", "visit"}


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read CSV / TSV / Parquet / Excel based on file extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset extension: {ext}")


def _resolve(study_id: str, rel: str) -> Path:
    """Resolve a path relative to ``workspace/{study_id}/`` safely."""
    ws = StudyWorkspace(study_id).ensure()
    return ws.safe_join(*rel.split("/"))


def _write_json(p: Path, obj: Any) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    return hash_file(p)


# ===========================================================================
# 1. load_dataset_tool
# ===========================================================================
def _impl_load_dataset(study_id: str, rel_path: str) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")
    df = _read_dataset(src)
    out = {
        "study_id": study_id,
        "path": rel_path,
        "rows": len(df),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "sha256": hash_file(src),
    }
    write_audit_event(
        actor="tool:load_dataset",
        action="dataset.load",
        study_id=study_id,
        output_hash=str(out["sha256"]),
        metadata={"path": rel_path, "rows": out["rows"], "cols": out["cols"]},
    )
    return out


# ===========================================================================
# 2. profile_dataset_tool
# ===========================================================================
def _impl_profile_dataset(study_id: str, rel_path: str) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    profile: dict[str, Any] = {
        "study_id": study_id,
        "path": rel_path,
        "rows": len(df),
        "cols": int(df.shape[1]),
        "per_column": {},
    }
    for c in df.columns:
        s = df[c]
        col: dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            if len(d):
                col["min"] = float(d.min())
                col["q1"] = float(d.quantile(0.25))
                col["median"] = float(d.median())
                col["mean"] = float(d.mean())
                col["q3"] = float(d.quantile(0.75))
                col["max"] = float(d.max())
                col["std"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        else:
            col["sample_values"] = [str(v) for v in s.dropna().unique()[:5]]
        profile["per_column"][str(c)] = col

    out_path = _resolve(study_id, "results/profile.json")
    out_hash = _write_json(out_path, profile)
    profile["output_path"] = "results/profile.json"
    profile["output_sha256"] = out_hash
    write_audit_event(
        actor="tool:profile_dataset",
        action="dataset.profile",
        study_id=study_id,
        output_hash=out_hash,
        metadata={"path": rel_path},
    )
    return profile


# ===========================================================================
# 3. validate_paired_data_tool
# ===========================================================================
def _impl_validate_paired_data(
    study_id: str,
    rel_path: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    expected_visits: list[str] | None = None,
) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)

    issues: list[str] = []
    for col in (subject_col, time_col):
        if col not in df.columns:
            issues.append(f"Missing required column: {col!r}")

    if issues:
        result = {"valid": False, "issues": issues}
    else:
        # Duplicates on (subject, visit)
        dup_mask = df.duplicated(subset=[subject_col, time_col], keep=False)
        n_dup_pairs = int(dup_mask.sum())

        visits_present = sorted(map(str, df[time_col].dropna().unique()))
        expected = visits_present if expected_visits is None else list(expected_visits)

        per_subject = df.groupby(subject_col)[time_col].apply(
            lambda s: sorted({str(v) for v in s.dropna()})
        )
        missing_pairs = []
        for subject, visits in per_subject.items():
            missing = [v for v in expected if v not in visits]
            if missing:
                missing_pairs.append({"subject_id": str(subject), "visits_missing": missing})

        n_subjects = int(df[subject_col].nunique())
        complete_subjects = int(
            (per_subject.apply(lambda v: all(e in v for e in expected))).sum()
        )

        result = {
            "valid": (n_dup_pairs == 0) and (len(missing_pairs) == 0),
            "n_subjects": n_subjects,
            "complete_subjects": complete_subjects,
            "expected_visits": expected,
            "visits_present": visits_present,
            "duplicate_pairs": n_dup_pairs,
            "missing_pairs": missing_pairs[:50],  # cap to keep response small
            "missing_pairs_total": len(missing_pairs),
        }

    out_path = _resolve(study_id, "results/paired_validation.json")
    out_hash = _write_json(out_path, result)
    result["output_path"] = "results/paired_validation.json"
    result["output_sha256"] = out_hash
    write_audit_event(
        actor="tool:validate_paired_data",
        action="dataset.validate_paired",
        study_id=study_id,
        output_hash=out_hash,
        metadata={"path": rel_path},
    )
    return result


# ===========================================================================
# 4. detect_missingness_tool
# ===========================================================================
def _impl_detect_missingness(study_id: str, rel_path: str) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)

    per_col = {
        str(c): {
            "missing": int(df[c].isna().sum()),
            "pct_missing": float(df[c].isna().mean() * 100.0),
        }
        for c in df.columns
    }
    per_visit_value: dict[str, Any] = {}
    if "visit" in df.columns and "value" in df.columns:
        g = df.groupby("visit")["value"].agg(
            n=lambda s: len(s),
            missing=lambda s: int(s.isna().sum()),
        )
        per_visit_value = {
            str(v): {
                "n": int(row["n"]),
                "missing": int(row["missing"]),
                "pct_missing": float(row["missing"] / max(row["n"], 1) * 100.0),
            }
            for v, row in g.iterrows()
        }

    result: dict[str, Any] = {
        "rows": len(df),
        "per_column": per_col,
        "per_visit_value": per_visit_value,
    }
    out_path = _resolve(study_id, "results/missingness_summary.json")
    out_hash = _write_json(out_path, result)
    result["output_path"] = "results/missingness_summary.json"
    result["output_sha256"] = out_hash

    # Also a CSV for human-readable QC summary.
    csv_path = _resolve(study_id, "results/missingness_summary.csv")
    pd.DataFrame.from_dict(per_col, orient="index").to_csv(csv_path)
    result["csv_path"] = "results/missingness_summary.csv"

    write_audit_event(
        actor="tool:detect_missingness",
        action="dataset.detect_missingness",
        study_id=study_id,
        output_hash=out_hash,
        metadata={"path": rel_path},
    )
    return result


# ===========================================================================
# 5. detect_outliers_tool
# ===========================================================================
def _impl_detect_outliers(
    study_id: str,
    rel_path: str,
    value_col: str = "value",
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    if value_col not in df.columns:
        raise KeyError(f"value column {value_col!r} not in dataset")

    s = df[value_col].dropna().astype(float)
    flags: list[dict[str, Any]] = []
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        out_mask = (df[value_col] < lo) | (df[value_col] > hi)
    else:
        mu, sigma = s.mean(), s.std(ddof=1) or 1.0
        z = (df[value_col] - mu) / sigma
        out_mask = z.abs() > threshold

    flagged = df[out_mask.fillna(False)]
    for idx, row in flagged.head(50).iterrows():
        rec: dict[str, Any] = {"row": int(idx), value_col: float(row[value_col])}
        for k in ("subject_id", "visit", "endpoint"):
            if k in df.columns:
                rec[k] = str(row[k])
        flags.append(rec)

    result = {
        "value_col": value_col,
        "method": method,
        "threshold": threshold,
        "n_flagged": int(out_mask.fillna(False).sum()),
        "flagged_sample": flags,
    }
    out_path = _resolve(study_id, "results/outliers.json")
    out_hash = _write_json(out_path, result)
    result["output_path"] = "results/outliers.json"
    result["output_sha256"] = out_hash

    csv_path = _resolve(study_id, "results/outlier_report.csv")
    flagged.to_csv(csv_path, index=False)
    result["csv_path"] = "results/outlier_report.csv"

    write_audit_event(
        actor="tool:detect_outliers",
        action="dataset.detect_outliers",
        study_id=study_id,
        output_hash=out_hash,
        metadata={"path": rel_path, "value_col": value_col, "n_flagged": result["n_flagged"]},
    )
    return result


# ===========================================================================
# 6. pseudonymize_subjects_tool
# ===========================================================================
def _impl_pseudonymize(
    study_id: str,
    rel_path: str,
    subject_col: str = "subject_id",
    salt: str | None = None,
) -> dict[str, Any]:
    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    if subject_col not in df.columns:
        raise KeyError(f"subject column {subject_col!r} not in dataset")

    ws = StudyWorkspace(study_id)
    salt_path = ws.audit / "pseudonym_salt.txt"
    salt_value: str
    if salt is None:
        if salt_path.exists():
            salt_value = salt_path.read_text().strip()
        else:
            salt_value = secrets.token_hex(16)
            salt_path.parent.mkdir(parents=True, exist_ok=True)
            salt_path.write_text(salt_value)
    else:
        salt_value = salt

    def _h(v: object) -> str:
        return hashlib.sha256(f"{salt_value}|{v}".encode()).hexdigest()[:16]

    mapping = {str(v): _h(v) for v in df[subject_col].astype(str).unique()}
    df[subject_col] = df[subject_col].astype(str).map(mapping)

    out_path = _resolve(study_id, "clean/analysis_dataset.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    out_hash = hash_file(out_path)

    mapping_path = ws.audit / "pseudonym_map.json"
    mapping_path.write_bytes(orjson.dumps(mapping, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY))
    # Restrict perms when possible.
    try:
        mapping_path.chmod(0o600)
        salt_path.chmod(0o600)
    except OSError:
        pass

    result = {
        "study_id": study_id,
        "rows": len(df),
        "n_subjects": len(mapping),
        "analysis_dataset_path": "clean/analysis_dataset.parquet",
        "analysis_dataset_sha256": out_hash,
        "pseudonym_map_path": "audit/pseudonym_map.json",
    }
    write_audit_event(
        actor="tool:pseudonymize_subjects",
        action="dataset.pseudonymize",
        study_id=study_id,
        output_hash=out_hash,
        metadata={"n_subjects": len(mapping)},
    )
    return result


# ===========================================================================
# 7. hash_file_tool
# ===========================================================================
def _impl_hash_file(study_id: str, rel_path: str) -> dict[str, Any]:
    p = _resolve(study_id, rel_path)
    h = hash_file(p)
    write_audit_event(
        actor="tool:hash_file",
        action="file.hash",
        study_id=study_id,
        output_hash=h,
        metadata={"path": rel_path},
    )
    return {"path": rel_path, "sha256": h, "size": int(p.stat().st_size)}


# ===========================================================================
# 8. write_audit_event_tool (allows the agent to log its own milestones)
# ===========================================================================
def _impl_write_audit_event(
    study_id: str | None,
    actor: str,
    action: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return write_audit_event(
        actor=actor, action=action, study_id=study_id, metadata=metadata or {}
    )


# ===========================================================================
# 9. choose_statistical_test_tool — DETERMINISTIC DECISION TABLE
# ===========================================================================
# Maps (data_type, design, n_timepoints) → recommended model, with rationale.
# This is the core of the "right test for the right data" logic; the LLM
# delegates to it instead of guessing.
def _impl_choose_test(
    data_type: str,
    design: str,
    n_timepoints: int,
    n_groups: int = 1,
    normality_ok: bool = True,
) -> dict[str, Any]:
    dt = data_type.lower()
    ds = design.lower()
    rec: dict[str, Any] = {
        "data_type": dt,
        "design": ds,
        "n_timepoints": n_timepoints,
        "n_groups": n_groups,
        "normality_ok": normality_ok,
    }

    if dt == "continuous":
        if n_timepoints == 2 and n_groups == 1:
            rec["model"] = "paired_t" if normality_ok else "wilcoxon_signed_rank"
            rec["rationale"] = (
                "Two paired timepoints on the same subjects → paired t-test "
                "when differences ~ Normal, else Wilcoxon signed-rank."
            )
        elif n_timepoints > 2:
            rec["model"] = "MMRM"
            rec["rationale"] = (
                "Longitudinal continuous data on the same subjects → MMRM "
                "(or LMM with random subject intercept) with time × treatment "
                "interaction. Handles missing-at-random."
            )
        elif n_groups >= 2 and n_timepoints == 1:
            rec["model"] = "welch_t" if normality_ok else "mann_whitney_u"
            rec["rationale"] = "Independent groups, single timepoint."
        else:
            rec["model"] = "LMM"
            rec["rationale"] = "Default to LMM with subject random effect."

    elif dt == "ordinal":
        if n_timepoints == 2:
            rec["model"] = "wilcoxon_signed_rank"
            rec["rationale"] = "Paired ordinal, two timepoints."
        else:
            rec["model"] = "ordinal_mixed"
            rec["rationale"] = "Multiple timepoints → ordinal mixed model (CLMM)."

    elif dt == "binary":
        if n_timepoints == 2:
            rec["model"] = "mcnemar"
            rec["rationale"] = "Paired binary, two timepoints."
        else:
            rec["model"] = "glmm_logit"
            rec["rationale"] = "Longitudinal binary → GLMM logit or GEE."

    elif dt == "count":
        rec["model"] = "poisson_or_negbin"
        rec["rationale"] = (
            "Count data → Poisson; switch to NB if overdispersion; "
            "zero-inflated if excess zeros."
        )

    else:
        rec["model"] = "unsupported"
        rec["rationale"] = f"No default model for data_type={data_type!r}."

    return rec


# ===========================================================================
# 10. apply_multiplicity_tool
# ===========================================================================
def _impl_apply_multiplicity(
    p_values: list[float],
    method: Literal["bonferroni", "holm", "hochberg", "bh_fdr", "by_fdr"] = "holm",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply a multiplicity correction. Returns adjusted p-values + reject decisions."""
    if not p_values:
        return {"method": method, "alpha": alpha, "p_adjusted": [], "reject": []}

    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("All p-values must be in [0, 1]")

    method_l = method.lower()
    if method_l == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method_l == "holm":
        order = np.argsort(p)
        sorted_p = p[order]
        adj_sorted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * sorted_p), 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted
    elif method_l == "hochberg":
        order = np.argsort(p)
        sorted_p = p[order]
        # Hochberg step-up
        steps = (m - np.arange(m)) * sorted_p
        adj_sorted = np.minimum.accumulate(steps[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted
    elif method_l == "bh_fdr":
        order = np.argsort(p)
        sorted_p = p[order]
        ranks = np.arange(1, m + 1)
        steps = sorted_p * m / ranks
        adj_sorted = np.minimum.accumulate(steps[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted
    elif method_l == "by_fdr":
        order = np.argsort(p)
        sorted_p = p[order]
        ranks = np.arange(1, m + 1)
        c_m = float(np.sum(1.0 / ranks))
        steps = sorted_p * m * c_m / ranks
        adj_sorted = np.minimum.accumulate(steps[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted
    else:
        raise ValueError(f"Unsupported multiplicity method: {method}")

    reject = (adj < alpha).tolist()
    return {
        "method": method_l,
        "alpha": alpha,
        "n_tests": m,
        "p_values": p.tolist(),
        "p_adjusted": [round(float(x), 6) for x in adj],
        "reject": reject,
    }


# ===========================================================================
# 11. run_paired_test_tool (writes a reproducible script + returns result)
# ===========================================================================
def _impl_run_paired_test(
    study_id: str,
    rel_path: str,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    timepoint: str = "D28",
    practical_threshold: float | None = None,
    direction: Literal["increase", "decrease", "two_sided"] = "two_sided",
    seed: int = 1729,
) -> dict[str, Any]:
    """Paired-t or Wilcoxon on (timepoint − baseline) for a single endpoint."""
    from scipy import stats as sps

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    if "endpoint" in df.columns:
        df = df[df["endpoint"] == endpoint].copy()
    wide = (
        df[df[time_col].isin([baseline, timepoint])]
        .pivot_table(index=subject_col, columns=time_col, values=value_col, aggfunc="mean")
        .dropna(subset=[baseline, timepoint])
    )
    paired = wide[timepoint].astype(float) - wide[baseline].astype(float)
    n = len(paired)
    if n < 3:
        raise ValueError(f"Not enough paired observations (n={n}) for endpoint {endpoint!r}")

    diff_mean = float(paired.mean())
    diff_sd = float(paired.std(ddof=1)) if n > 1 else 0.0
    se = diff_sd / math.sqrt(n) if n else float("nan")

    # Shapiro–Wilk on differences
    shapiro_p = float(sps.shapiro(paired).pvalue) if 3 <= n <= 5000 else float("nan")
    normality_ok = (shapiro_p > 0.05) if not math.isnan(shapiro_p) else True

    if normality_ok:
        model = "paired_t"
        t_stat, p_two = sps.ttest_rel(wide[timepoint], wide[baseline])
        t_stat = float(t_stat)
        p_two = float(p_two)
        # 95% CI on mean difference
        tcrit = float(sps.t.ppf(0.975, df=n - 1))
        ci_lo = diff_mean - tcrit * se
        ci_hi = diff_mean + tcrit * se
        effect_size = diff_mean / diff_sd if diff_sd > 0 else float("nan")
        effect_size_metric = "cohen_dz"
    else:
        model = "wilcoxon_signed_rank"
        res = sps.wilcoxon(wide[timepoint], wide[baseline])
        p_two = float(res.pvalue)
        # CI on median difference via Hodges–Lehmann; use bootstrap fallback
        rng = np.random.default_rng(seed)
        boots = rng.choice(paired.values, size=(2000, n), replace=True).mean(axis=1)
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        # Rank-biserial as effect size (computed from W)
        try:
            w = float(res.statistic)
            n_eff = (paired != 0).sum()
            effect_size = 1.0 - 2.0 * w / (n_eff * (n_eff + 1) / 2.0) if n_eff > 0 else float("nan")
            effect_size_metric = "rank_biserial"
        except Exception:
            effect_size = float("nan")
            effect_size_metric = "rank_biserial"

    # One-sided p if direction is specified
    if direction == "two_sided":
        p_value = p_two
    elif direction == "increase":
        p_value = p_two / 2.0 if diff_mean > 0 else 1.0 - p_two / 2.0
    else:  # decrease
        p_value = p_two / 2.0 if diff_mean < 0 else 1.0 - p_two / 2.0

    # Practical threshold check
    practical_met: bool | None = None
    if practical_threshold is not None:
        if direction == "decrease":
            practical_met = diff_mean <= practical_threshold  # threshold typically negative
        elif direction == "increase":
            practical_met = diff_mean >= practical_threshold
        else:
            practical_met = abs(diff_mean) >= abs(practical_threshold)

    # Write a reproducible Python script
    ws = StudyWorkspace(study_id)
    script_path = ws.scripts / f"paired_{endpoint}.py"
    script = f"""\
# Auto-generated by run_paired_test_tool — DO NOT EDIT BY HAND
import pandas as pd
from scipy import stats as sps

df = pd.read_parquet({str(_resolve(study_id, rel_path))!r})
df = df[df['endpoint'] == {endpoint!r}] if 'endpoint' in df.columns else df
wide = (df[df[{time_col!r}].isin([{baseline!r}, {timepoint!r}])]
          .pivot_table(index={subject_col!r}, columns={time_col!r}, values={value_col!r})
          .dropna(subset=[{baseline!r}, {timepoint!r}]))
print('n =', len(wide))
print('diff_mean =', float((wide[{timepoint!r}] - wide[{baseline!r}]).mean()))
print('shapiro p =', float(sps.shapiro(wide[{timepoint!r}] - wide[{baseline!r}]).pvalue))
print('t-test  p =', float(sps.ttest_rel(wide[{timepoint!r}], wide[{baseline!r}]).pvalue))
print('Wilcoxon p =', float(sps.wilcoxon(wide[{timepoint!r}], wide[{baseline!r}]).pvalue))
"""
    script_path.write_text(script)

    out: dict[str, Any] = {
        "endpoint": endpoint,
        "data_type": "continuous",
        "model": model,
        "contrast": f"{timepoint} - {baseline}",
        "estimate": round(diff_mean, 6),
        "ci95": [round(ci_lo, 6), round(ci_hi, 6)],
        "p_value": round(float(p_value), 6),
        "p_adjusted": None,
        "p_adjustment_method": "none",
        "effect_size": (None if math.isnan(effect_size) else round(effect_size, 4)),
        "effect_size_metric": effect_size_metric,
        "practical_threshold": practical_threshold,
        "practical_threshold_met": practical_met,
        "n": n,
        "n_complete": n,
        "assumptions": {
            "normality_p": (None if math.isnan(shapiro_p) else round(shapiro_p, 4)),
            "overall_ok": True,
        },
        "sensitivity_analysis": None,
        "conclusion": _short_conclusion(diff_mean, float(p_value), practical_met, direction),
        "artefacts": {"script": str(script_path.relative_to(ws.root))},
        "extras": {"baseline": baseline, "timepoint": timepoint},
    }

    res_path = ws.results / f"paired_{endpoint}.json"
    res_hash = _write_json(res_path, out)
    out["artefacts"]["result_json"] = str(res_path.relative_to(ws.root))
    out["artefacts"]["result_sha256"] = res_hash

    write_audit_event(
        actor="tool:run_paired_test",
        action="stats.paired_test",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"endpoint": endpoint, "n": n, "model": model},
    )
    return out


def _short_conclusion(
    diff: float, p: float, practical: bool | None, direction: str
) -> str:
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    prac = ""
    if practical is True:
        prac = " and the practical threshold is met"
    elif practical is False:
        prac = " but the practical threshold is NOT met"
    dirw = {"increase": "increase", "decrease": "decrease", "two_sided": "change"}[direction]
    return f"Observed {dirw} of {diff:+.3g}; effect is {sig}{prac}."


# ===========================================================================
# 12. record_package_versions_tool
# ===========================================================================
def _impl_record_package_versions(study_id: str) -> dict[str, Any]:
    import importlib.metadata as md

    pkgs = sorted(
        {dist.metadata["Name"]: dist.version for dist in md.distributions()}.items()
    )
    info = {
        "python": sys.version,
        "packages": dict(pkgs),
    }
    out_path = _resolve(study_id, "audit/package_versions.json")
    out_hash = _write_json(out_path, info)
    write_audit_event(
        actor="tool:record_package_versions",
        action="audit.record_versions",
        study_id=study_id,
        output_hash=out_hash,
    )
    return {"path": "audit/package_versions.json", "sha256": out_hash, "n_packages": len(pkgs)}


# ===========================================================================
# 13. request_human_approval_tool
# ===========================================================================
def _impl_request_human_approval(
    study_id: str,
    object_type: str,
    object_id: str,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pending ApprovalRequest in the in-memory store + file."""
    import uuid as _uuid

    from app.schemas.approvals import ApprovalRequest, ApprovalStatus
    from app.storage import db

    approval = ApprovalRequest(
        approval_id=f"APR-{_uuid.uuid4().hex[:8]}",
        study_id=study_id,
        object_type=object_type,
        object_id=object_id,
        reason=reason,
        payload=payload or {},
        status=ApprovalStatus.PENDING,
    )
    db.approvals().upsert(approval.approval_id, approval)

    # Persist a copy in the workspace approvals dir
    ws = StudyWorkspace(study_id)
    p = ws.approvals / f"{approval.approval_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(approval.model_dump_json(indent=2).encode())

    write_audit_event(
        actor="tool:request_human_approval",
        action="approval.requested",
        study_id=study_id,
        metadata={
            "approval_id": approval.approval_id,
            "object_type": object_type,
            "object_id": object_id,
        },
    )
    return {
        "approval_id": approval.approval_id,
        "status": "pending",
        "object_type": object_type,
        "object_id": object_id,
    }


# ===========================================================================
# 14. check_approval_status_tool
# ===========================================================================
def _impl_check_approval_status(approval_id: str) -> dict[str, Any]:
    from app.storage import db

    item = db.approvals().get(approval_id)
    if item is None:
        return {"approval_id": approval_id, "status": "not_found"}
    return {
        "approval_id": item.approval_id,
        "status": item.status.value,
        "reviewer": item.reviewer,
        "decision_at": item.decision_at.isoformat() if item.decision_at else None,
        "comment": item.comment,
        "edited_payload": item.edited_payload,
    }


# ===========================================================================


# ===========================================================================
# 15. run_mmrm_tool
# ===========================================================================
def _impl_run_mmrm(
    study_id: str,
    rel_path: str,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
    practical_threshold: float | None = None,
    direction: Literal["increase", "decrease", "two_sided"] = "two_sided",
) -> dict[str, Any]:
    """Run MMRM and persist the result."""
    from app.services.statistics_runner import run_mmrm

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    ws = StudyWorkspace(study_id).ensure()

    res = run_mmrm(
        df, endpoint, subject_col, time_col, value_col, baseline, primary_timepoint,
    )

    # Practical threshold
    prac_met: bool | None = None
    if practical_threshold is not None:
        if direction == "increase":
            prac_met = res["estimate"] >= practical_threshold
        elif direction == "decrease":
            prac_met = res["estimate"] <= practical_threshold
        else:
            prac_met = abs(res["estimate"]) >= abs(practical_threshold)

    out: dict[str, Any] = {
        **res,
        "practical_threshold": practical_threshold,
        "practical_threshold_met": prac_met,
        "conclusion": _short_conclusion(res["estimate"], res["p_value"], prac_met, direction),
        "artefacts": {},
    }

    # Write reproducible script
    script = (
        f"# Reproducible MMRM script for {endpoint}\n"
        f"import pandas as pd\n"
        f"from app.services.statistics_runner import run_mmrm\n\n"
        f"df = pd.read_parquet('{src}')\n"
        f"result = run_mmrm(\n"
        f"    df, endpoint={endpoint!r},\n"
        f"    subject_col={subject_col!r}, time_col={time_col!r},\n"
        f"    value_col={value_col!r}, baseline={baseline!r},\n"
        f"    primary_timepoint={primary_timepoint!r},\n"
        f")\n"
        f"print(result)\n"
    )
    script_path = ws.scripts / f"mmrm_{endpoint}.py"
    script_path.write_text(script)
    out["artefacts"]["script"] = str(script_path.relative_to(ws.root))

    # Write result JSON
    res_path = ws.results / f"mmrm_{endpoint}.json"
    res_hash = _write_json(res_path, out)
    out["artefacts"]["result_json"] = str(res_path.relative_to(ws.root))
    out["artefacts"]["result_sha256"] = res_hash

    write_audit_event(
        actor="tool:run_mmrm",
        action="stats.mmrm",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"endpoint": endpoint, "n": res["n"], "model": res["model"]},
    )
    return out


# ===========================================================================
# 16. run_glmm_logit_tool
# ===========================================================================
def _impl_run_glmm_logit(
    study_id: str,
    rel_path: str,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    primary_timepoint: str = "D28",
) -> dict[str, Any]:
    """Run logistic GLMM (GEE) for a binary endpoint and persist."""
    from app.services.statistics_runner import run_glmm_logit

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    ws = StudyWorkspace(study_id).ensure()

    res = run_glmm_logit(
        df, endpoint, subject_col, time_col, value_col, baseline, primary_timepoint,
    )

    script_path = ws.scripts / f"glmm_logit_{endpoint}.py"
    script_path.write_text(
        f"# Reproducible GLMM-logit script for {endpoint}\n"
        f"import pandas as pd\n"
        f"from app.services.statistics_runner import run_glmm_logit\n\n"
        f"df = pd.read_parquet('{src}')\n"
        f"result = run_glmm_logit(df, endpoint={endpoint!r})\n"
        f"print(result)\n"
    )
    res["artefacts"] = {"script": str(script_path.relative_to(ws.root))}

    res_path = ws.results / f"glmm_logit_{endpoint}.json"
    res_hash = _write_json(res_path, res)
    res["artefacts"]["result_json"] = str(res_path.relative_to(ws.root))
    res["artefacts"]["result_sha256"] = res_hash

    write_audit_event(
        actor="tool:run_glmm_logit",
        action="stats.glmm_logit",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"endpoint": endpoint, "n": res["n"]},
    )
    return res


# ===========================================================================
# 17. run_mcnemar_tool
# ===========================================================================
def _impl_run_mcnemar(
    study_id: str,
    rel_path: str,
    endpoint: str,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    timepoint: str = "D28",
) -> dict[str, Any]:
    """Run McNemar's test for a paired binary endpoint and persist."""
    from app.services.statistics_runner import run_mcnemar

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    ws = StudyWorkspace(study_id).ensure()

    res = run_mcnemar(df, endpoint, subject_col, time_col, value_col, baseline, timepoint)

    script_path = ws.scripts / f"mcnemar_{endpoint}.py"
    script_path.write_text(
        f"# Reproducible McNemar script for {endpoint}\n"
        f"import pandas as pd\n"
        f"from app.services.statistics_runner import run_mcnemar\n\n"
        f"df = pd.read_parquet('{src}')\n"
        f"result = run_mcnemar(df, endpoint={endpoint!r})\n"
        f"print(result)\n"
    )
    res["artefacts"] = {"script": str(script_path.relative_to(ws.root))}

    res_path = ws.results / f"mcnemar_{endpoint}.json"
    res_hash = _write_json(res_path, res)
    res["artefacts"]["result_json"] = str(res_path.relative_to(ws.root))
    res["artefacts"]["result_sha256"] = res_hash

    write_audit_event(
        actor="tool:run_mcnemar",
        action="stats.mcnemar",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"endpoint": endpoint, "n": res["n"]},
    )
    return res


# ===========================================================================
# 18. run_top2box_tool
# ===========================================================================
def _impl_run_top2box(
    study_id: str,
    rel_path: str,
    question_col: str,
    value_col: str = "value",
    scale_max: int = 5,
) -> dict[str, Any]:
    """Compute top-2-box percentage with Wilson CI for a consumer question."""
    from app.services.statistics_runner import run_top2box

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    ws = StudyWorkspace(study_id).ensure()

    # Filter to the specific question:
    # If the dataset has a 'question' column, filter by question_col value.
    # Otherwise, question_col must be a column name directly.
    if "question" in df.columns:
        sub = df[df["question"] == question_col]
        if sub.empty:
            raise KeyError(f"No rows matching question={question_col!r} in dataset.")
    elif question_col in df.columns:
        sub = df
        value_col = question_col  # The column itself contains the values
    else:
        raise KeyError(f"Column {question_col!r} not in dataset and no 'question' column found.")

    responses = sub[value_col].dropna().astype(int).tolist()
    res = run_top2box(responses, scale_max=scale_max)
    res["question"] = question_col

    res_path = ws.results / f"top2box_{question_col}.json"
    res_hash = _write_json(res_path, res)
    res["artefacts"] = {
        "result_json": str(res_path.relative_to(ws.root)),
        "result_sha256": res_hash,
    }

    write_audit_event(
        actor="tool:run_top2box",
        action="stats.top2box",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"question": question_col, "n": res["n"]},
    )
    return res


# ===========================================================================
# 19. run_tost_tool
# ===========================================================================
def _impl_run_tost(
    study_id: str,
    rel_path: str,
    endpoint: str,
    margin: float,
    subject_col: str = "subject_id",
    time_col: str = "visit",
    value_col: str = "value",
    baseline: str = "D0",
    timepoint: str = "D28",
) -> dict[str, Any]:
    """Run TOST equivalence test on a paired endpoint."""
    from app.services.statistics_runner import run_tost

    src = _resolve(study_id, rel_path)
    df = _read_dataset(src)
    ws = StudyWorkspace(study_id).ensure()

    sub = df[df["endpoint"] == endpoint].copy() if "endpoint" in df.columns else df.copy()
    pre = sub[sub[time_col] == baseline].set_index(subject_col)[value_col].dropna()
    post = sub[sub[time_col] == timepoint].set_index(subject_col)[value_col].dropna()
    common = pre.index.intersection(post.index)
    if len(common) < 5:
        raise ValueError(f"TOST: too few pairs for {endpoint!r} (n={len(common)}).")

    res = run_tost(post.loc[common].values, pre.loc[common].values, margin=margin, paired=True)
    res["endpoint"] = endpoint
    res["contrast"] = f"{timepoint} − {baseline}"

    script_path = ws.scripts / f"tost_{endpoint}.py"
    script_path.write_text(
        f"# Reproducible TOST script for {endpoint}\n"
        f"import pandas as pd\n"
        f"from app.services.statistics_runner import run_tost\n\n"
        f"df = pd.read_parquet('{src}')\n"
        f"# ... pivot and run_tost(..., margin={margin})\n"
    )
    res["artefacts"] = {"script": str(script_path.relative_to(ws.root))}

    res_path = ws.results / f"tost_{endpoint}.json"
    res_hash = _write_json(res_path, res)
    res["artefacts"]["result_json"] = str(res_path.relative_to(ws.root))
    res["artefacts"]["result_sha256"] = res_hash

    write_audit_event(
        actor="tool:run_tost",
        action="stats.tost",
        study_id=study_id,
        output_hash=res_hash,
        metadata={"endpoint": endpoint, "margin": margin, "n": res["n"]},
    )
    return res



# ===========================================================================
# LangChain @tool wrappers
# ---------------------------------------------------------------------------
# We import `tool` lazily so this module can be used in tests without the
# heavy langchain import being mandatory. The wrappers just delegate to the
# pure `_impl_*` functions above.
# ===========================================================================

def _get_tool_decorator() -> Any:
    try:
        from langchain_core.tools import tool

        return tool
    except Exception:  # pragma: no cover
        return None


def build_langchain_tools() -> list[Any]:
    """Return the list of langchain `@tool`-decorated callables."""
    tool = _get_tool_decorator()
    if tool is None:  # pragma: no cover
        return []

    @tool
    def load_dataset_tool(study_id: str, rel_path: str) -> dict[str, Any]:
        """Load a dataset under workspace/{study_id}/{rel_path} and return its shape + dtypes (no data)."""
        return _impl_load_dataset(study_id, rel_path)

    @tool
    def profile_dataset_tool(study_id: str, rel_path: str) -> dict[str, Any]:
        """Compute descriptive stats per column and persist them to results/profile.json."""
        return _impl_profile_dataset(study_id, rel_path)

    @tool
    def validate_paired_data_tool(
        study_id: str,
        rel_path: str,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        expected_visits: list[str] | None = None,
    ) -> dict[str, Any]:
        """Check for duplicate (subject, visit) pairs and missing visits per subject."""
        return _impl_validate_paired_data(
            study_id, rel_path, subject_col, time_col, expected_visits
        )

    @tool
    def detect_missingness_tool(study_id: str, rel_path: str) -> dict[str, Any]:
        """Quantify missingness per column (and per visit when applicable)."""
        return _impl_detect_missingness(study_id, rel_path)

    @tool
    def detect_outliers_tool(
        study_id: str,
        rel_path: str,
        value_col: str = "value",
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> dict[str, Any]:
        """Flag outliers using IQR or z-score and write outlier_report.csv."""
        return _impl_detect_outliers(study_id, rel_path, value_col, method, threshold)  # type: ignore[arg-type]

    @tool
    def pseudonymize_subjects_tool(
        study_id: str, rel_path: str, subject_col: str = "subject_id"
    ) -> dict[str, Any]:
        """Hash subject identifiers with a per-study salt; write clean parquet."""
        return _impl_pseudonymize(study_id, rel_path, subject_col)

    @tool
    def hash_file_tool(study_id: str, rel_path: str) -> dict[str, Any]:
        """Return the SHA-256 of a file inside the study workspace."""
        return _impl_hash_file(study_id, rel_path)

    @tool
    def write_audit_event_tool(
        study_id: str | None, actor: str, action: str, metadata: dict | None = None
    ) -> dict[str, Any]:
        """Append an event to the audit trail."""
        return _impl_write_audit_event(study_id, actor, action, metadata)

    @tool
    def choose_statistical_test_tool(
        data_type: str,
        design: str,
        n_timepoints: int,
        n_groups: int = 1,
        normality_ok: bool = True,
    ) -> dict[str, Any]:
        """Recommend a statistical model based on data type and design (deterministic)."""
        return _impl_choose_test(data_type, design, n_timepoints, n_groups, normality_ok)

    @tool
    def apply_multiplicity_tool(
        p_values: list[float], method: str = "holm", alpha: float = 0.05
    ) -> dict[str, Any]:
        """Apply a multiplicity correction (holm/bonferroni/hochberg/bh_fdr/by_fdr)."""
        return _impl_apply_multiplicity(p_values, method=method, alpha=alpha)  # type: ignore[arg-type]

    @tool
    def run_paired_test_tool(
        study_id: str,
        rel_path: str,
        endpoint: str,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        value_col: str = "value",
        baseline: str = "D0",
        timepoint: str = "D28",
        practical_threshold: float | None = None,
        direction: str = "two_sided",
    ) -> dict[str, Any]:
        """Run paired-t (or Wilcoxon if non-normal) for a single endpoint and persist the result."""
        return _impl_run_paired_test(
            study_id, rel_path, endpoint, subject_col, time_col, value_col,
            baseline, timepoint, practical_threshold, direction,  # type: ignore[arg-type]
        )

    @tool
    def run_mmrm_tool(
        study_id: str,
        rel_path: str,
        endpoint: str,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        value_col: str = "value",
        baseline: str = "D0",
        primary_timepoint: str = "D28",
        practical_threshold: float | None = None,
        direction: str = "two_sided",
    ) -> dict[str, Any]:
        """Run MMRM (mixed model for repeated measures) on a continuous longitudinal endpoint."""
        return _impl_run_mmrm(
            study_id, rel_path, endpoint, subject_col, time_col, value_col,
            baseline, primary_timepoint, practical_threshold, direction,  # type: ignore[arg-type]
        )

    @tool
    def run_glmm_logit_tool(
        study_id: str,
        rel_path: str,
        endpoint: str,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        value_col: str = "value",
        baseline: str = "D0",
        primary_timepoint: str = "D28",
    ) -> dict[str, Any]:
        """Run logistic GLMM (GEE) for a binary longitudinal endpoint."""
        return _impl_run_glmm_logit(
            study_id, rel_path, endpoint, subject_col, time_col, value_col,
            baseline, primary_timepoint,
        )

    @tool
    def run_mcnemar_tool(
        study_id: str,
        rel_path: str,
        endpoint: str,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        value_col: str = "value",
        baseline: str = "D0",
        timepoint: str = "D28",
    ) -> dict[str, Any]:
        """Run McNemar's exact test for a paired binary endpoint (2 timepoints)."""
        return _impl_run_mcnemar(
            study_id, rel_path, endpoint, subject_col, time_col, value_col,
            baseline, timepoint,
        )

    @tool
    def run_top2box_tool(
        study_id: str,
        rel_path: str,
        question_col: str,
        value_col: str = "value",
        scale_max: int = 5,
    ) -> dict[str, Any]:
        """Compute top-2-box % with Wilson CI for a consumer-perception question."""
        return _impl_run_top2box(study_id, rel_path, question_col, value_col, scale_max)

    @tool
    def run_tost_tool(
        study_id: str,
        rel_path: str,
        endpoint: str,
        margin: float,
        subject_col: str = "subject_id",
        time_col: str = "visit",
        value_col: str = "value",
        baseline: str = "D0",
        timepoint: str = "D28",
    ) -> dict[str, Any]:
        """Run TOST equivalence test on a paired endpoint with a pre-specified margin."""
        return _impl_run_tost(
            study_id, rel_path, endpoint, margin, subject_col, time_col, value_col,
            baseline, timepoint,
        )

    @tool
    def record_package_versions_tool(study_id: str) -> dict[str, Any]:
        """Record Python + installed package versions for reproducibility."""
        return _impl_record_package_versions(study_id)

    @tool
    def request_human_approval_tool(
        study_id: str,
        object_type: str,
        object_id: str,
        reason: str,
        payload: dict | None = None,
    ) -> dict[str, Any]:
        """Create a pending human-approval request and pause until it's decided."""
        return _impl_request_human_approval(study_id, object_type, object_id, reason, payload)

    @tool
    def check_approval_status_tool(approval_id: str) -> dict[str, Any]:
        """Return the current status of a pending approval request."""
        return _impl_check_approval_status(approval_id)

    return [
        load_dataset_tool,
        profile_dataset_tool,
        validate_paired_data_tool,
        detect_missingness_tool,
        detect_outliers_tool,
        pseudonymize_subjects_tool,
        hash_file_tool,
        write_audit_event_tool,
        choose_statistical_test_tool,
        apply_multiplicity_tool,
        run_paired_test_tool,
        run_mmrm_tool,
        run_glmm_logit_tool,
        run_mcnemar_tool,
        run_top2box_tool,
        run_tost_tool,
        record_package_versions_tool,
        request_human_approval_tool,
        check_approval_status_tool,
    ]
