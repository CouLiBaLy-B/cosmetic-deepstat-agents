"""Tiny CLI: ``cosmetic-deepstat {version, run-demo}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import __version__

EXAMPLES = Path(__file__).parent.parent / "examples" / "sample_study"


def _ingest_demo(study_id: str) -> dict[str, object]:
    """Create the demo study + upload the synthetic CSV + attach the demo claims.

    Returns a dict with the artefacts that were created.
    """
    from fastapi.testclient import TestClient

    from app.main import app
    from app.schemas.study import StudyCreate

    client = TestClient(app)

    meta = json.loads((EXAMPLES / "study_metadata.json").read_text())
    meta["study_id"] = study_id
    # Pydantic validates strictness
    StudyCreate.model_validate(meta)
    r = client.post("/api/studies", json=meta)
    if r.status_code == 409:
        print(f"[run-demo] study {study_id!r} already exists, re-using.")
    else:
        r.raise_for_status()

    csv_path = EXAMPLES / "data" / "measurements_long.csv"
    if not csv_path.exists():
        raise SystemExit(
            "Demo CSV not found. Run: python examples/sample_study/generate_synthetic_data.py"
        )
    files = {"file": (csv_path.name, csv_path.read_bytes(), "text/csv")}
    r = client.post(f"/api/studies/{study_id}/data", files=files)
    if r.status_code not in (201, 409):
        r.raise_for_status()

    claims = json.loads((EXAMPLES / "claims.json").read_text())
    r = client.post(f"/api/studies/{study_id}/claims", json=claims)
    r.raise_for_status()
    return {"study_id": study_id, "claims": [c["claim_id"] for c in claims]}


def _run_pipeline(study_id: str, auto_approve: bool) -> dict[str, object]:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.storage import db

    client = TestClient(app)

    # 1st launch — will pause at SAP approval gate.
    r = client.post(f"/api/analyses/{study_id}")
    out = r.json()
    print(f"[run-demo] first launch → status={out.get('status')!r}")

    pending = [a for a in db.approvals().list() if a.study_id == study_id and a.status.value == "pending"]
    if not pending:
        print("[run-demo] no pending approvals (already approved?)")
    elif not auto_approve:
        print(f"[run-demo] {len(pending)} pending approval(s):")
        for a in pending:
            print(f"  - {a.approval_id} ({a.object_type}) — {a.reason}")
        print(
            "[run-demo] approve with:\n"
            "  curl -X POST localhost:8000/api/approvals/<id> "
            '-H "content-type: application/json" '
            '-d \'{"decision":"approved","reviewer":"demo"}\''
        )
        return out

    # Auto-approve every pending request, in order, until the pipeline ends.
    for round_idx in range(1, 6):
        pending = [a for a in db.approvals().list() if a.study_id == study_id and a.status.value == "pending"]
        if not pending:
            break
        for a in pending:
            print(f"[run-demo] auto-approving {a.approval_id} ({a.object_type})")
            client.post(
                f"/api/approvals/{a.approval_id}",
                json={"decision": "approved", "reviewer": "demo-auto", "comment": "auto-approve"},
            )
        r = client.post(f"/api/analyses/{study_id}")
        out = r.json()
        print(f"[run-demo] round {round_idx} → status={out.get('status')!r}")

    # Show what we produced
    r = client.get(f"/api/reports/{study_id}")
    print(f"[run-demo] reports: {r.json()}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cosmetic-deepstat")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("version", help="Print version and exit.")
    run_demo = sub.add_parser("run-demo", help="Run the bundled demo study end-to-end.")
    run_demo.add_argument("--study-id", default="STUDY_DEMO_001")
    run_demo.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve every HITL gate so the pipeline runs to completion.",
    )

    args = parser.parse_args(argv)

    if args.cmd in (None, "version"):
        print(__version__)
        return 0

    if args.cmd == "run-demo":
        _ingest_demo(args.study_id)
        _run_pipeline(args.study_id, auto_approve=args.auto_approve)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
