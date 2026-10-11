"""Pytest configuration and global fixtures with database isolation."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.main import create_app
from src.core.config import Settings, get_settings
from src.database.models import Base
from src.database.session import get_db


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Return test-isolated application settings."""
    return Settings(
        app_name="Solar PV AI Diagnostic Assistant (Test)",
        app_env="testing",
        debug=True,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "data",
    )


@pytest.fixture
def test_engine(test_settings):
    """Create test engine with foreign key constraints enabled and schema built."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        finally:
            cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """Yield a clean test database session."""
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(test_settings, db_session) -> Generator[TestClient, None, None]:
    """Provide a TestClient with database session override."""
    app = create_app()

    def override_get_settings():
        return test_settings

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
