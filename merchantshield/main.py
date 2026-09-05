"""MerchantShield application entrypoint."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.case_routes import router as case_router
from .api.demo_routes import router as demo_router
from .api.routes import router as legacy_router
from .config import get_settings
from .db.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)
    mode = "full (LLM enabled)" if settings.llm_enabled else "heuristic-only (no API key)"
    print(f"  MerchantShield ready — analysis mode: {mode}")
    yield


app = FastAPI(
    title="MerchantShield",
    description=(
        "Human-centred merchant-ring investigation. Deterministic policy owns "
        "risk and permitted actions; high-risk cases stay with a reviewer. "
        "All demonstration data is synthetic."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(case_router)
app.include_router(demo_router)

# The original automatic decision API is retained only as an isolated test
# fixture for backwards compatibility. It is deliberately not mounted on the
# reviewer application because it can emit automatic rejection outcomes.
legacy_app = FastAPI(
    title="MerchantShield legacy API",
    description="Deprecated compatibility surface; not part of the safe demo.",
    version="0.legacy",
    lifespan=lifespan,
)
legacy_app.include_router(legacy_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "healthy",
        "service": "MerchantShield",
        "version": "1.0.0",
        "analysis_mode": settings.mode,
        "llm_enabled": settings.llm_enabled,
    }


legacy_app.get("/health", tags=["meta"])(health)
