"""Generate a comprehensive synthetic dataset for STUDY_DEMO_001.

Produces deterministic (seeded) data for the demo study:

Endpoints:
  1. corneometer_hydration (continuous, 4 visits: D0/D7/D14/D28)
     → strong positive drift → MMRM, confirmed claim
  2. wrinkle_depth (continuous, 2 visits: D0/D28)
     → moderate negative drift → paired-t, confirmed claim
  3. tolerance_ok (binary, 2 visits: D0/D28)
     → high tolerance rate → McNemar (safety claim)

Also generates a consumer perception questionnaire:
  4. consumer_smoothness (Likert 1-5, single visit post-use)
     → top-2-box analysis

30 subjects, 4 intentionally-missing cells for QC testing.

Run manually:

    python examples/sample_study/generate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "data"
SEED = 1729


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = 30
    subjects = [f"S{i:03d}" for i in range(1, n + 1)]

    # -------------------------------------------------------------------
    # 1. Instrumental measurements (long format)
    # -------------------------------------------------------------------
    rows: list[dict[str, object]] = []

    # Hydration: baseline ~35 ± 6 a.u., +0/+3/+5/+7.5 drift
    drift_h = {"D0": 0.0, "D7": 3.0, "D14": 5.0, "D28": 7.5}
    base_h = rng.normal(35.0, 6.0, size=n)
    for i, s in enumerate(subjects):
        for v, d in drift_h.items():
            val = base_h[i] + d + rng.normal(0.0, 1.5)
            rows.append({
                "subject_id": s,
                "visit": v,
                "endpoint": "corneometer_hydration",
                "value": round(float(val), 3),
            })

    # Wrinkle depth: baseline ~0.22 ± 0.05 mm, decrease ~0.07 mm
    base_w = rng.normal(0.22, 0.05, size=n)
    delta_w = rng.normal(-0.07, 0.025, size=n)
    for i, s in enumerate(subjects):
        rows.append({
            "subject_id": s, "visit": "D0",
            "endpoint": "wrinkle_depth",
            "value": round(float(base_w[i]), 4),
        })
        rows.append({
            "subject_id": s, "visit": "D28",
            "endpoint": "wrinkle_depth",
            "value": round(float(base_w[i] + delta_w[i]), 4),
        })

    # Tolerance (binary): D0 → 80% ok, D28 → 95% ok
    for i, s in enumerate(subjects):
        rows.append({
            "subject_id": s, "visit": "D0",
            "endpoint": "tolerance_ok",
            "value": int(rng.random() < 0.80),
        })
        rows.append({
            "subject_id": s, "visit": "D28",
            "endpoint": "tolerance_ok",
            "value": int(rng.random() < 0.95),
        })

    df = pd.DataFrame(rows)

    # Drop 4 random cells to exercise missingness handling
    drop_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[drop_idx, "value"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "measurements_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows, {df['subject_id'].nunique()} subjects)")

    # -------------------------------------------------------------------
    # 2. Consumer perception questionnaire
    # -------------------------------------------------------------------
    consumer_rows: list[dict[str, object]] = []
    consumer_subjects = [f"C{i:03d}" for i in range(1, 51)]  # 50 consumers
    questions = ["smoothness", "hydration_feel", "overall_satisfaction"]

    for s in consumer_subjects:
        for q in questions:
            # Skew towards positive responses (4 and 5)
            val = int(rng.choice([1, 2, 3, 4, 5], p=[0.03, 0.07, 0.15, 0.35, 0.40]))
            consumer_rows.append({
                "subject_id": s,
                "question": q,
                "value": val,
            })

    cdf = pd.DataFrame(consumer_rows)
    consumer_path = OUT_DIR / "consumer_questionnaire.csv"
    cdf.to_csv(consumer_path, index=False)
    print(f"Wrote {consumer_path} ({len(cdf)} rows, {cdf['subject_id'].nunique()} consumers)")

    # -------------------------------------------------------------------
    # 3. Summary
    # -------------------------------------------------------------------
    print(f"\nDataset summary:")
    print(f"  Instrumental: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    print(f"    Endpoints: {sorted(df['endpoint'].unique())}")
    print(f"    Visits: {sorted(df['visit'].unique())}")
    print(f"    Missing values: {int(df['value'].isna().sum())}")
    print(f"  Consumer: {len(cdf)} rows, {cdf['subject_id'].nunique()} consumers")
    print(f"    Questions: {sorted(cdf['question'].unique())}")


if __name__ == "__main__":
    main()
