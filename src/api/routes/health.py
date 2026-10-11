"""System and database health check endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Response, status

from src.core.config import get_settings
from src.database.session import check_db_health

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic and database health check")
async def health_check(response: Response) -> Dict[str, Any]:
    """Return runtime service health status, live database connectivity, and metadata."""
    settings = get_settings()
    db_connected = check_db_health()

    if not db_connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        service_status = "degraded"
    else:
        response.status_code = status.HTTP_200_OK
        service_status = "healthy"

    return {
        "status": service_status,
        "database": "connected" if db_connected else "disconnected",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
