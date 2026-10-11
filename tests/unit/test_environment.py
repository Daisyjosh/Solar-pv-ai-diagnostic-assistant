"""Unit tests verifying environment setup, configuration safety, and foundational health."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.core.config import Settings, get_settings
from src.database.session import get_db

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def test_settings_loading():
    """Verify application settings initialize with expected defaults."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "Solar PV AI Diagnostic Assistant"
    assert settings.app_version == "0.1.0"
    assert settings.api_port == 8000
    assert settings.min_irradiance_threshold_w_m2 >= 600.0


def test_env_example_safety():
    """Verify .env.example contains no real credentials or secret keys."""
    env_example_path = WORKSPACE_ROOT / ".env.example"
    assert env_example_path.exists(), ".env.example must exist in the project root"

    content = env_example_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        # Validate that sensitive keys only contain documented placeholder values
        if "SECRET_KEY" in key:
            assert "change-in-production" in val or val == ""
        if "API_KEY" in key or "PASSWORD" in key:
            assert val == "" or "placeholder" in val.lower()


def test_directory_structure_integrity():
    """Verify that all foundational directories exist in the project workspace."""
    required_directories = [
        WORKSPACE_ROOT / "frontend",
        WORKSPACE_ROOT / "src" / "api" / "routes",
        WORKSPACE_ROOT / "src" / "core",
        WORKSPACE_ROOT / "src" / "database" / "models",
        WORKSPACE_ROOT / "src" / "services",
        WORKSPACE_ROOT / "src" / "vision",
        WORKSPACE_ROOT / "src" / "diagnosis",
        WORKSPACE_ROOT / "src" / "telemetry",
        WORKSPACE_ROOT / "src" / "knowledge",
        WORKSPACE_ROOT / "src" / "reports",
        WORKSPACE_ROOT / "data" / "raw",
        WORKSPACE_ROOT / "data" / "processed",
        WORKSPACE_ROOT / "data" / "external",
        WORKSPACE_ROOT / "tests" / "unit",
        WORKSPACE_ROOT / "tests" / "integration",
        WORKSPACE_ROOT / "docs",
        WORKSPACE_ROOT / "scripts",
    ]

    for directory in required_directories:
        assert directory.is_dir(), f"Expected directory does not exist: {directory}"


def test_gitignore_coverage():
    """Verify that .gitignore properly covers sensitive files, local databases, and raw media."""
    gitignore_path = WORKSPACE_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"

    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert "*.db" in content
    assert "__pycache__/" in content
    assert "data/raw/*" in content
    assert "*.pdf" in content


def test_health_endpoints(client: TestClient):
    """Verify that the FastAPI health endpoints return 200 OK with valid schema."""
    # Root /health
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp_utc" in data

    # Prefixed /api/v1/health
    response_prefixed = client.get("/api/v1/health")
    assert response_prefixed.status_code == 200
    data_prefixed = response_prefixed.json()
    assert data_prefixed["status"] == "healthy"


def test_database_session_generator():
    """Verify that database session dependency yields a valid SQLAlchemy session."""
    session_gen = get_db()
    session = next(session_gen)
    try:
        assert session is not None
        # Verify basic session query capability
        assert hasattr(session, "execute")
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
