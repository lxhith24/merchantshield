"""Database engine, session factory, and FastAPI dependency."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

_settings = get_settings()

# check_same_thread=False is required for SQLite under FastAPI's threadpool.
engine = create_engine(
    _settings.database_url,
    connect_args={"check_same_thread": False}
    if _settings.database_url.startswith("sqlite")
    else {},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def init_db() -> None:
    """Create all tables. Idempotent."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Iterator[Session]:
    """Context-managed session for scripts and internal use."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency. The route owns commit/rollback semantics."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
