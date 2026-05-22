"""Generate a small synthetic dataset for STUDY_DEMO_001 (deterministic seed).

Two endpoints, 30 subjects, 4 visits for hydration, 2 for wrinkle depth.
The data are seeded so the demo always shows the SAME conclusions:
- hydration increases significantly (clinically + practically),
- wrinkle depth decreases significantly (practically met).

Run it manually:

    python examples/sample_study/generate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "measurements_long.csv"
SEED = 1729


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = 30
    subjects = [f"S{i:03d}" for i in range(1, n + 1)]

    rows: list[dict[str, object]] = []

    # Hydration: baseline ~ 35 ± 6, +0, +3, +5, +7 a.u. mean drift, individual noise
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

    # Wrinkle depth: baseline ~ 0.22 ± 0.05 mm, decrease of ~0.07 mm at D28
    base_w = rng.normal(0.22, 0.05, size=n)
    delta_w = rng.normal(-0.07, 0.025, size=n)
    for i, s in enumerate(subjects):
        rows.append({"subject_id": s, "visit": "D0", "endpoint": "wrinkle_depth", "value": round(float(base_w[i]), 4)})
        rows.append({"subject_id": s, "visit": "D28", "endpoint": "wrinkle_depth", "value": round(float(base_w[i] + delta_w[i]), 4)})

    # Intentionally drop a few cells to exercise missingness handling.
    df = pd.DataFrame(rows)
    drop_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[drop_idx, "value"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df['subject_id'].nunique()} subjects)")


if __name__ == "__main__":
    main()
