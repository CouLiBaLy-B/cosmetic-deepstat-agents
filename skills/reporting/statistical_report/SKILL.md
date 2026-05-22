---
name: statistical_report
description: >
  Template and guidelines for writing the Statistical Analysis Report (SAR)
  for a cosmetic clinical study. Covers section structure, required content
  per section, tables, figures, and regulatory compliance.
tags: [reporting, statistical-report, SAR, template, regulatory]
---

# Statistical Analysis Report (SAR) — template

## Section structure

The SAR must follow this order:

### 1. Title page
- Study ID, product ID, report version.
- Author(s), date, signatures.
- Confidentiality notice.

### 2. Synopsis (1 page)
- Study objective, design, population, primary endpoint.
- Key result: estimate, 95% CI, p-value, conclusion (1 sentence).
- Claim substantiation outcome (confirmed / partial / not supported).

### 3. Study design
- Study type, visits, arms.
- Endpoints with data type, instrument, unit.
- Sample size justification.
- Randomisation / allocation (if applicable).

### 4. Statistical Analysis Plan (SAP) summary
- Link to the approved SAP file.
- Primary analysis model (e.g. MMRM, paired t).
- Multiplicity strategy.
- Missing-data strategy.
- Pre-specified practical thresholds.
- Sensitivity analyses planned.

### 5. Data overview
- Subjects enrolled / analysed / excluded (CONSORT-like flow).
- Missing data summary (per visit, per endpoint).
- Baseline characteristics (demographics, baseline endpoint values).
- Data quality summary (outliers flagged, duplicates resolved).

### 6. Results — Primary endpoint(s)
For each primary endpoint:
- Descriptive statistics by visit (mean, SD, median, IQR, n).
- Model results: estimate, 95% CI, p-value, effect size.
- Multiplicity-adjusted p-value.
- Practical threshold assessment.
- Figure: mean ± SE over visits (or box plot for ordinal).
- Assumption checks: normality, homoscedasticity, convergence.

### 7. Results — Secondary endpoints
Same structure as §6, clearly labelled "secondary".

### 8. Sensitivity analyses
- Re-analysis under alternative missing-data assumptions.
- Robustness to outlier exclusion.
- Tipping-point analysis (if applicable).

### 9. Claim substantiation summary table

| Claim ID | Claim text | Endpoint | Adjusted p | Practical threshold | Support level |
|----------|------------|----------|------------|---------------------|---------------|

### 10. Limitations
- Study design limitations.
- Missing data and impact.
- Generalisability.

### 11. Appendices
- Reproducible scripts (paths).
- Audit trail excerpts.
- Package versions.

## Figures required

1. Mean ± SE over visits (per primary endpoint).
2. Individual-subject spaghetti plot (per primary endpoint).
3. Box plot of change from baseline at primary timepoint.
4. QQ-plot of residuals (if MMRM/LMM).

## Tables required

1. Baseline demographics.
2. Descriptive statistics by visit.
3. Primary analysis results.
4. Multiplicity-adjusted p-values.
5. Sensitivity analysis results.
6. Claim substantiation summary.

## Hard rules

1. **Every numerical result** must reference its source file path in
   `workspace/{study_id}/results/`.
2. **No raw subject IDs** in the report.
3. **No result without 95% CI.**
4. The report must be **self-contained**: a reviewer reading only the SAR
   must be able to evaluate the claim.
