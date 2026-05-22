"""Post-market surveillance endpoints (scaffold)."""

from __future__ import annotations

from fastapi import APIRouter, status

router = APIRouter(prefix="/api/postmarket", tags=["postmarket"])


@router.post("/{product_id}", status_code=status.HTTP_202_ACCEPTED)
def ingest_postmarket(product_id: str, payload: dict) -> dict:
    """Ingest a batch of post-market data (complaints, AE). Stub for MVP."""
    return {
        "product_id": product_id,
        "accepted": True,
        "received_keys": sorted(payload.keys()),
    }


@router.get("/{product_id}/signals")
def get_signals(product_id: str) -> dict:
    """Return the current signal-detection dashboard. Stub for MVP."""
    return {"product_id": product_id, "signals": []}
