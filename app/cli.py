"""Tiny CLI entry-point used by ``cosmetic-deepstat`` (see pyproject scripts)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cosmetic-deepstat")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("version", help="Print version and exit.")
    run_demo = sub.add_parser("run-demo", help="Run the bundled demo study end-to-end.")
    run_demo.add_argument("--study-id", default="STUDY_DEMO_001")

    args = parser.parse_args(argv)

    if args.cmd in (None, "version"):
        from app import __version__

        print(__version__)
        return 0

    if args.cmd == "run-demo":
        # Wired in a later phase (Phase 6 — full pipeline orchestration).
        print(
            f"[run-demo] study_id={args.study_id} — not yet wired. "
            "Run `uvicorn app.main:app --reload` and POST /api/studies in the meantime."
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
