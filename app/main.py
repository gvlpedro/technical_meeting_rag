import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.config import settings
from app.routers.completions import router as completions_router
from app.routers.ingestion import router as ingestion_router

logging.basicConfig(level=settings.log_level, format="%(message)s")
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("app_startup", environment=settings.environment)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(completions_router)
app.include_router(ingestion_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
