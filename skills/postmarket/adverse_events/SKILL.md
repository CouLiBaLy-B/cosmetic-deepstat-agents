---
name: adverse_events
description: >
  Post-market adverse event and complaint monitoring for cosmetic products.
  Covers data ingestion, temporal trend analysis, signal detection rules,
  per-lot/country/channel breakdown, and regulatory reporting thresholds.
tags: [postmarket, adverse-events, complaints, signal-detection, cosmetovigilance]
---

# Post-market adverse event monitoring

## Overview

Under EU Regulation 1223/2009 (Art. 23) and US MoCRA (2022), cosmetic
manufacturers must report serious adverse events (SAEs) to the competent
authority. Beyond regulatory obligations, systematic monitoring helps
detect safety signals early.

## Data model

Each post-market record should contain:

```json
{
  "event_id": "EVT-20260101-001",
  "product_id": "CREAM_ANTIAGE_001",
  "lot_number": "L2025-0042",
  "country": "FR",
  "channel": "pharmacy",
  "report_date": "2026-01-15",
  "event_type": "complaint" | "adverse_event",
  "severity": "mild" | "moderate" | "severe" | "serious",
  "body_area": "face",
  "symptom_text": "erythema and itching after 3 days of use",
  "outcome": "resolved" | "resolving" | "not_resolved" | "unknown",
  "reported_by": "consumer" | "hcp" | "distributor"
}
```

## Signal detection rules

### Rule 1: Rate-change alert

```
IF complaint_rate(current_4_weeks) / complaint_rate(previous_4_weeks) > 1.5
   AND current_4_weeks.count ≥ 5
THEN ALERT
```

### Rule 2: Cluster alert (per lot)

```
IF complaints_per_lot(lot_X) / median_complaints_per_lot(all_lots) > 3.0
THEN ALERT on lot_X
```

### Rule 3: Geographic cluster

```
IF complaint_rate(country_X) / complaint_rate(all_countries) > 2.0
   AND country_X.count ≥ 3
THEN ALERT on country_X
```

### Rule 4: Severity escalation

```
IF any event.severity == "serious"
THEN IMMEDIATE ALERT + mandatory regulatory notification within 20 days
```

## Analysis pipeline

1. **Ingest** batch of events via `POST /api/postmarket/{product_id}`.
2. **Validate** schema (event_id unique, required fields present).
3. **Compute** descriptive statistics:
   - Total complaints, rate per 1000 units sold (if sales data available).
   - Breakdown by severity, body_area, country, channel, lot.
4. **Trend analysis:** weekly complaint rate with 4-week rolling average.
5. **Apply signal-detection rules** 1–4 above.
6. **Generate dashboard JSON** at
   `workspace/postmarket/{product_id}/dashboard.json`.
7. If any alert fires → `request_human_approval_tool(object_type="postmarket_signal")`.

## Dashboard JSON schema

```json
{
  "product_id": "...",
  "period": {"from": "2026-01-01", "to": "2026-03-31"},
  "total_events": 42,
  "by_severity": {"mild": 30, "moderate": 10, "severe": 2, "serious": 0},
  "by_country": {"FR": 20, "DE": 12, "IT": 10},
  "by_lot": {"L2025-0042": 8, "L2025-0043": 5, ...},
  "weekly_rate": [{"week": "2026-W01", "count": 3, "rate_per_1k": 0.12}, ...],
  "alerts": [
    {"rule": "rate_change", "value": 1.8, "threshold": 1.5, "detail": "..."}
  ],
  "human_review_required": true
}
```

## Hard rules

1. **Serious AEs** trigger an immediate alert and must be reported to the
   competent authority within the legal timeframe (20 days EU, 15 days US).
2. **Never suppress** a signal. Even if it is likely a false positive, it
   must be documented and reviewed.
3. **Every alert** requires human approval before downstream actions
   (recall, reformulation, communication).
4. Raw consumer data (name, address) must be **pseudonymised** before
   analysis.

## References

- Regulation (EC) No 1223/2009, Art. 23 (Serious undesirable effects)
- MoCRA, Pub. L. 117-328 (2022), mandatory AE reporting
- SCCS Notes of Guidance, Chapter 3.6 (Post-market surveillance)
- EU Commission Guidance on Cosmetovigilance (2019)
