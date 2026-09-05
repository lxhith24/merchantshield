"""Streamlit Community Cloud entrypoint for the self-contained demo.

Community Cloud starts one Streamlit process, while the reviewer surface is
normally backed by the local FastAPI service started by ``run.sh``.  This
entrypoint keeps that boundary intact by running the same FastAPI application
on loopback in a daemon thread.  It is suitable for a disposable synthetic
demonstration only; production would deploy the API and UI separately.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import streamlit as st
import uvicorn


API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"


def _seed_synthetic_cases() -> None:
    """Populate the ephemeral Cloud filesystem with the normal demo queue."""
    from merchantshield.db.database import init_db
    from merchantshield.runtime import build_runtime

    async def seed() -> None:
        init_db()
        runtime = await build_runtime(force_offline_investigator=True)
        for candidate in runtime.world.candidates:
            await runtime.service.open_case(candidate.candidate_id)

    asyncio.run(seed())


@st.cache_resource(show_spinner=False)
def start_local_api() -> Any:
    """Start the real FastAPI app once per Streamlit process."""
    from merchantshield.main import app

    config = uvicorn.Config(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="merchantshield-api", daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{API_URL}/health", timeout=1.0)
            if response.is_success:
                _seed_synthetic_cases()
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.15)
    raise RuntimeError("MerchantShield API did not become healthy on loopback")


def main() -> None:
    # streamlit_app reads this value at import time.
    import os

    os.environ["MERCHANTSHIELD_API_URL"] = API_URL
    start_local_api()
    from merchantshield.ui.streamlit_app import main as render_app

    render_app()


if __name__ == "__main__":
    main()
