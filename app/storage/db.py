"""Lightweight in-process repositories.

The MVP keeps studies / approvals / claims in process memory so the API
endpoints work end-to-end without provisioning a database. The repository
classes have **the exact same interface** as the future SQLAlchemy-backed
implementations — switching is a single import change in
``app/services/registry.py``.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

from app.schemas.approvals import ApprovalRequest
from app.schemas.claims import Claim
from app.schemas.study import Study

T = TypeVar("T")


class _InMemoryRepo(Generic[T]):
    """Thread-safe `dict[id, model]` repository."""

    def __init__(self) -> None:
        self._store: dict[str, T] = {}
        self._lock = threading.RLock()

    def upsert(self, key: str, value: T) -> T:
        with self._lock:
            self._store[key] = value
            return value

    def get(self, key: str) -> T | None:
        with self._lock:
            return self._store.get(key)

    def list(self) -> list[T]:
        with self._lock:
            return list(self._store.values())

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# --------- Concrete singletons for the MVP ---------

class StudyRepository(_InMemoryRepo[Study]):
    pass


class ClaimRepository(_InMemoryRepo[Claim]):
    pass


class ApprovalRepository(_InMemoryRepo[ApprovalRequest]):
    pass


_studies = StudyRepository()
_claims = ClaimRepository()
_approvals = ApprovalRepository()


def studies() -> StudyRepository:
    return _studies


def claims() -> ClaimRepository:
    return _claims


def approvals() -> ApprovalRepository:
    return _approvals
