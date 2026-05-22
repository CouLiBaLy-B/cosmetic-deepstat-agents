---
name: equivalence_tost
description: >
  Two One-Sided Tests (TOST) procedure for equivalence and non-inferiority
  claims in cosmetic studies. Requires a pre-specified margin. Used when a
  cosmetic product must demonstrate it is 'as good as' a comparator.
tags: [statistics, equivalence, non-inferiority, tost, margin]
---

# Equivalence / Non-inferiority testing (TOST)

## When to use

- **Claim type:** equivalence or non-inferiority.
- The study must prove the product is **not worse** (NI) or **equivalent**
  (EQ) to a reference, within a **pre-specified margin** (δ).
- Common in cosmetic reformulation studies, generic/biosimilar cosmetic
  actives, and comparative advertising claims.

## Pre-requisites

1. A **margin δ** specified in the SAP (e.g. ±2 a.u. for corneometer).
   Without this, TOST MUST NOT be run.
2. A **comparator** (could be the same product before reformulation, or
   a market leader).

## Procedure — TOST for equivalence

1. **Compute** mean difference `d = mean_test − mean_reference`.
2. **First one-sided test:** H₀: d ≤ −δ  vs  H₁: d > −δ  → p₁.
3. **Second one-sided test:** H₀: d ≥ +δ  vs  H₁: d < +δ  → p₂.
4. **Equivalence p-value:** max(p₁, p₂).
5. **Equivalence declared** if max(p₁, p₂) < α AND the 90% CI for d is
   entirely within [−δ, +δ].

## Procedure — Non-inferiority

1. **Compute** mean difference `d = mean_test − mean_reference`.
2. **One-sided test:** H₀: d ≤ −δ  vs  H₁: d > −δ  → p.
3. **NI declared** if p < α AND the lower bound of the 95% CI (one-sided)
   is > −δ.

## Confidence interval approach (preferred)

- Compute the **(1 − 2α)% CI** (i.e. 90% CI for α = 0.05).
- If the entire 90% CI lies within [−δ, +δ], equivalence is established.
- For NI: if the lower bound of the 95% CI lies above −δ.

## Hard rules

1. **NEVER** declare equivalence from a non-significant superiority test.
   "Not significantly different" ≠ "equivalent."
2. **NEVER** run TOST without a pre-specified margin in the SAP.
3. The margin must be **clinically/perceptually meaningful** — not just
   statistically convenient.
4. Report the 90% CI and the margin alongside the p-value.
5. Multiplicity: if multiple equivalence claims, apply Bonferroni to the
   TOST p-values.

## Output additions

Add to `StatisticalResult.extras`:
- `"tost_p1"`, `"tost_p2"`, `"tost_p_max"`.
- `"equivalence_margin"`, `"ci90"`.
- `"equivalence_met": true/false`.

## References

- Schuirmann, D. J. (1987). "A comparison of the two one-sided tests
  procedure and the power approach for assessing the equivalence of average
  bioavailability." *J. Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Lakens, D. (2017). "Equivalence tests: a practical primer for t-tests,
  correlations, and meta-analyses." *Social Psych. & Personality Science*.
