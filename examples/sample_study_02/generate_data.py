"""Generate data for STUDY_DEMO_002 — simple 2-timepoint study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "measurements.csv"
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    rows = []
    for s in range(1, n + 1):
        base = float(rng.normal(32.0, 5.0))
        rows.append({
            "subject_id": f"S{s:03d}", "visit": "D0",
            "endpoint": "corneometer_24h",
            "value": round(base, 3),
        })
        rows.append({
            "subject_id": f"S{s:03d}", "visit": "D28",
            "endpoint": "corneometer_24h",
            "value": round(base + 6.0 + float(rng.normal(0, 2.0)), 3),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
