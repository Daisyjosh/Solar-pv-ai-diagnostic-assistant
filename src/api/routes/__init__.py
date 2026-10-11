"""API route definitions and endpoints."""

from src.api.routes.assets import router as assets_router
from src.api.routes.health import router as health_router
from src.api.routes.inspections import router as inspections_router
from src.api.routes.plants import router as plants_router

__all__ = ["health_router", "plants_router", "inspections_router", "assets_router"]
