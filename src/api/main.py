"""FastAPI application entry point and factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.assets import router as assets_router
from src.api.routes.batch import router as batch_router
from src.api.routes.chat import router as chat_router
from src.api.routes.diagnoses import router as diagnoses_router
from src.api.routes.health import router as health_router
from src.api.routes.inspections import router as inspections_router
from src.api.routes.plants import router as plants_router
from src.api.routes.reports import router as reports_router
from src.api.routes.telemetry import router as telemetry_router
from src.api.routes.vision import router as vision_router
from src.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "REST API for Solar PV AI Diagnostic Assistant. "
            "Provides inspection workflows, thermal/optical analysis, "
            "and electrical telemetry diagnostic evaluation."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoints
    app.include_router(health_router)
    app.include_router(health_router, prefix=settings.api_prefix)

    # Core Asset & Inspection endpoints
    app.include_router(plants_router, prefix=settings.api_prefix)
    app.include_router(assets_router, prefix=settings.api_prefix)
    app.include_router(inspections_router, prefix=settings.api_prefix)

    # AI Vision, Diagnostic Reasoning, Telemetry, and Assistant endpoints
    app.include_router(vision_router, prefix=settings.api_prefix)
    app.include_router(diagnoses_router, prefix=settings.api_prefix)
    app.include_router(telemetry_router, prefix=settings.api_prefix)
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(batch_router, prefix=settings.api_prefix)
    app.include_router(reports_router, prefix=settings.api_prefix)

    return app


app = create_app()
