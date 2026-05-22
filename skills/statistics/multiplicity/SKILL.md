---
name: multiplicity
description: >
  Multiplicity correction for multiple testing in cosmetic claim
  substantiation. Covers Holm, Bonferroni, Hochberg, BH-FDR, BY-FDR,
  fixed-sequence, and gatekeeping procedures. Maps corrected p-values to
  claim support levels.
tags: [statistics, multiplicity, holm, bonferroni, fdr, gatekeeping, claims]
---

# Multiplicity correction for cosmetic claims

## When to use

Whenever a study tests **more than one hypothesis** that feeds into a
marketing claim. This includes:

- Multiple primary endpoints (e.g. hydration AND wrinkle depth).
- Multiple timepoints declared as co-primary.
- Multiple claims from a single study.
- Confirmatory secondary endpoints.

## Method selection (from SAP)

| Context                              | Default method | Alternative          |
|--------------------------------------|----------------|----------------------|
| Confirmatory primaries               | **Holm**       | Bonferroni, Hochberg |
| Confirmatory + ordered hierarchy     | Fixed-sequence | Gatekeeping (Bretz)  |
| Exploratory / secondary              | BH-FDR         | BY-FDR (dependent)   |
| Consumer perception endpoints        | BH-FDR         | —                    |

## Procedure

1. **Collect** raw p-values from `statistical_results.json`.
2. **Group** them into **families** defined in `sap_locked.json →
   multiplicity_strategy.families`.
3. **Apply** the SAP-specified method via `apply_multiplicity_tool`:
   - **Bonferroni:** `p_adj_i = min(p_i × m, 1)`.
   - **Holm:** sort p ascending; `p_adj_(i) = min(max_{j≤i}(p_(j) × (m-j+1)), 1)`.
   - **Hochberg:** sort p descending; `p_adj_(i) = min(min_{j≥i}(p_(j) × (m-j+1)), 1)`.
   - **BH-FDR:** sort p ascending; `p_adj_(i) = min(min_{j≥i}(p_(j) × m / j), 1)`.
   - **BY-FDR:** like BH but with the harmonic correction `c(m)`.
4. **Compare** each adjusted p to α (default 0.05).
5. **Map** to claim support level:

```text
adjusted_p < α AND practical_threshold_met → CONFIRMED
adjusted_p < α BUT practical_threshold NOT met → PARTIAL
unadjusted_p < α but adjusted_p ≥ α → EXPLORATORY
adjusted_p ≥ α → NOT_SUPPORTED
```

## Output schema (ClaimDecision)

```json
{
  "claim_id": "C001",
  "claim_text": "Reduces wrinkles visibly in 28 days",
  "supported": true,
  "support_level": "confirmed",
  "statistical_basis": {
    "endpoint": "wrinkle_depth",
    "model": "paired_t",
    "estimate": -0.072,
    "ci95": [-0.095, -0.049],
    "p_value": 0.00003,
    "p_adjusted": 0.00006,
    "p_adjustment_method": "holm",
    "n": 30
  },
  "allowed_wording": "Wrinkle depth reduced by 0.07 mm (p<0.001, Holm-adjusted) after 28 days of use in 30 women aged 40-60.",
  "forbidden_wording": ["anti-wrinkle treatment", "eliminates wrinkles"],
  "limitations": [],
  "human_approval_required": true
}
```

## Hard rules

1. **Never** skip multiplicity when ≥ 2 confirmatory hypotheses exist.
2. **Never** let the statistical_analysis_subagent apply multiplicity —
   it's this skill's exclusive domain.
3. The SAP must specify the multiplicity method and family groupings
   **before** unblinding / analysis.
4. If the method in the SAP is not in {holm, bonferroni, hochberg, bh_fdr,
   by_fdr, fixed_sequence, gatekeeping}, refuse and request clarification.
5. For equivalence/non-inferiority claims, multiplicity is applied to the
   TOST p-values (both one-sided tests must pass).

## References

- Hochberg, Y. (1988). "A sharper Bonferroni procedure for multiple tests
  of significance." *Biometrika*, 75(4), 800–802.
- Holm, S. (1979). "A simple sequentially rejective multiple test
  procedure." *Scandinavian J. Statistics*, 6, 65–70.
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery
  rate." *J. Royal Statistical Society B*, 57(1), 289–300.
- Bretz, F. et al. (2009). *Multiple Comparisons Using R*. CRC Press.
