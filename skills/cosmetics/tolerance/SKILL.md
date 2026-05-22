---
name: tolerance
description: >
  Tolerance and safety evaluation for cosmetic products. Covers patch testing,
  repeat open application test (ROAT), clinical tolerance studies, AE grading,
  and the statistical analysis of tolerance endpoints (binary/ordinal).
tags: [cosmetics, safety, tolerance, patch-test, AE, dermatology]
---

# Tolerance & safety evaluation

## Overview

Tolerance/safety claims are **high-risk** under EU Common Criteria and
require robust clinical evidence. Claims like "hypoallergenic", "suitable
for sensitive skin", or "dermatologically tested" each have specific
evidentiary requirements.

## Study types

### 1. Single-application patch test (48h / 72h)

- Semi-occlusive patch on the back or forearm.
- Dermatologist reading at 48h and 72h.
- Grading: 0 (no reaction) to 4 (strong positive reaction).
- Minimum 30 subjects (including 10 with sensitive/atopic skin if
  "sensitive skin" is in the claim).

### 2. Repeat open application test (ROAT)

- Product applied on the forearm 2×/day for 14-28 days.
- Dermatologist + subject self-assessment.
- More clinically relevant than patch test.

### 3. Full tolerance study (28 days)

- Product used under normal conditions for 28+ days.
- Dermatologist visits at D0, D14, D28.
- Endpoints: erythema score, dryness score, stinging/burning VAS,
  overall tolerance grading, AE incidence.

## Statistical analysis of tolerance endpoints

| Endpoint              | Data type | Model                        |
|-----------------------|-----------|------------------------------|
| AE incidence (yes/no) | Binary    | Proportion + 95% Wilson CI   |
| Erythema grade (0-4)  | Ordinal   | Wilcoxon signed-rank or CLMM |
| Stinging VAS (0-100)  | Continuous| Paired t-test / MMRM         |
| Discontinuation rate  | Binary    | Proportion + exact CI        |
| Overall tolerance     | Ordinal   | Frequency table + CI         |

## Reporting requirements

1. **AE listing:** every adverse event with severity, duration, relationship
   to product, and outcome.
2. **Serious AE:** any SAE must be reported separately and trigger a safety
   review.
3. **Proportion tolerating:** "X% of subjects with sensitive skin rated
   tolerance as 'good' or 'excellent' (95% CI: [lo, hi])."
4. **Dermatologist assessment:** must be blinded to subject self-report
   when both are collected.

## Hard rules

1. **Safety claims always require human approval** before wording is
   finalised (`human_review_required = true`).
2. **Never claim "100% safe"** — use "well tolerated under study conditions."
3. If any SAE is possibly related to the product, the safety claim is
   **blocked** until a causality assessment is complete.
4. "Hypoallergenic" requires an HRIPT (Human Repeat Insult Patch Test)
   with ≥ 100 subjects and zero positive reactions.

## References

- SCCS Notes of Guidance for Testing of Cosmetic Ingredients (11th Rev.)
- Regulation (EC) No 1223/2009, Art. 10-11 (safety assessment)
- ISO 10993-10:2021 — Biological evaluation of medical devices — Irritation
- Cosmetics Europe: Guidelines for Evaluation of Efficacy of Cosmetic
  Products, Section 6 "Tolerance Testing"
