"""Application configuration loader using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the Solar PV AI Diagnostic Assistant."""

    # Application
    app_name: str = "Solar PV AI Diagnostic Assistant"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = "dev-insecure-secret-key-change-in-production"

    # API Server
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    # Frontend
    frontend_port: int = 8501
    backend_api_url: str = "http://127.0.0.1:8000/api/v1"

    # Database
    database_url: str = "sqlite:///./data/solar_pv_diagnostics.db"

    # File Storage
    storage_dir: Path = Path("./data")
    max_upload_size_mb: int = 50
    allowed_image_extensions: List[str] = ["jpg", "jpeg", "png", "tif", "tiff"]

    # Engineering / Domain Thresholds (IEC 62446-3)
    # Outdoor thermography requires >= 600 W/m^2 (ideally > 700 W/m^2) irradiance for reliable diagnosis
    min_irradiance_threshold_w_m2: float = 600.0

    # Optional Future Extensions (Phase 8)
    llm_provider: str = "none"

    # Gemini Multimodal Vision API (Free-Tier)
    gemini_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-3.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
