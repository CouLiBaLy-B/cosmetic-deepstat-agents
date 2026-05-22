"""FastAPI application entry-point for CosmeticDeepStat Agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import analyses, approvals, postmarket, reports, studies
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    log = get_logger("app.main")
    log.info(
        "starting",
        version=__version__,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        workspace=str(settings.workspace_root_abs),
    )
    yield
    log.info("stopping")


app = FastAPI(
    title="CosmeticDeepStat Agents",
    version=__version__,
    summary=(
        "Agentic platform for cosmetic clinical-study design, analysis, "
        "claim substantiation and post-market monitoring."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


# Routers
app.include_router(studies.router)
app.include_router(analyses.router)
app.include_router(approvals.router)
app.include_router(reports.router)
app.include_router(postmarket.router)
