---
name: claim_substantiation
description: >
  Template for the Claim Substantiation Report — the document that links
  marketing claims to statistical evidence. Used by regulatory affairs and
  legal teams to validate claim wording before product launch.
tags: [reporting, claims, substantiation, regulatory, marketing]
---

# Claim Substantiation Report — template

## Purpose

This report is the bridge between the statistical analysis and the marketing
department. It is reviewed by regulatory affairs, legal, and (for high-risk
claims) external authorities.

## Section structure

### 1. Title page
- Product ID, study ID, report version.
- Author, date, regulatory contact.

### 2. Claim inventory

| # | Claim ID | Claim text (proposed) | Jurisdiction | Claim type | Risk level |
|---|----------|----------------------|--------------|------------|------------|

### 3. Evidence matrix (per claim)

For each claim:

#### 3.a. Evidentiary requirements
- List from `claim_evidence_map.json`.
- Status: MET / NOT MET / PARTIALLY MET.

#### 3.b. Statistical evidence
- Endpoint, model, contrast.
- Estimate, 95% CI, raw p, adjusted p.
- Effect size, practical threshold assessment.
- Reference to SAR section.

#### 3.c. Support level determination
- Decision logic:
  - **CONFIRMED:** adjusted p < α, practical threshold met, SAP-aligned.
  - **PARTIAL:** significant but practical threshold not met.
  - **EXPLORATORY:** only unadjusted significance.
  - **NOT SUPPORTED:** no significance or wrong direction.

#### 3.d. Allowed wording
- Proposed claim text (validated).
- Conditions (e.g. "must mention 30 women, 28 days, Corneometer®").

#### 3.e. Forbidden wording
- Examples of non-compliant variations.
- Rationale for each forbidden variant.

### 4. Summary decision table

| Claim ID | Support level | Allowed wording | Human approval status |
|----------|--------------|-----------------|----------------------|

### 5. Regulatory considerations
- Jurisdiction-specific notes.
- Pending external reviews.
- Comparability to prior claims on the same product family.

### 6. Sign-off
- Biostatistician, regulatory affairs, marketing, legal.
- Date and version.

## Hard rules

1. **No claim** may be released without a row in this report.
2. **No allowed wording** without human approval (status = approved).
3. Consumer perception and instrumental claims are in **separate sections**.
4. Every claim must have a **clear trail** back to the SAP, the endpoint,
   and the statistical result.
