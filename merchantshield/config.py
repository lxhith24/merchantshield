"""Runtime configuration for MerchantShield."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Environment-backed settings.

    Only ANTHROPIC_API_KEY is optional-with-consequence: without it the service
    runs in heuristic-only mode (deterministic rules still score every
    application, but document vision analysis and LLM explanations are skipped).
    """

    def __init__(self) -> None:
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model: str = os.getenv(
            "MERCHANTSHIELD_MODEL", "claude-sonnet-4-20250514"
        ).strip()
        self.database_url: str = os.getenv(
            "DATABASE_URL", "sqlite:///./merchantshield.db"
        ).strip()
        self.upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads").strip()
        # The Phase 3 investigator is configured separately from the legacy
        # document-vision model so the two can move independently.
        self.investigator_model: str = os.getenv(
            "MERCHANTSHIELD_INVESTIGATOR_MODEL", "claude-opus-5"
        ).strip()
        self.investigator_effort: str = os.getenv(
            "MERCHANTSHIELD_INVESTIGATOR_EFFORT", "high"
        ).strip()

    @property
    def llm_enabled(self) -> bool:
        """True when an Anthropic key is present and LLM calls may be made."""
        return bool(self.anthropic_api_key)

    @property
    def mode(self) -> str:
        return "full" if self.llm_enabled else "heuristic_only"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
