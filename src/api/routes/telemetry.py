"""Electrical telemetry ingestion, analytics, and SCADA gateway REST endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from src.database.models.plant import Plant
from src.database.session import get_db
from src.telemetry.analytics import TelemetryAnalyticsService
from src.telemetry.importer import TelemetryImporter
from src.telemetry.providers.scada_adapter import SCADAAdapter

router = APIRouter(tags=["Electrical Telemetry"])


@router.post("/plants/{plant_id}/telemetry/preview")
def preview_telemetry_upload(
    plant_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Validate uploaded CSV/Excel telemetry file and return column mappings and preview rows."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with id {plant_id} not found.",
        )

    importer = TelemetryImporter(db)
    try:
        content = file.file.read()
        preview = importer.preview_import(content, file.filename)
        return {
            "filename": preview.filename,
            "total_rows": preview.total_rows,
            "valid_rows_count": preview.valid_rows_count,
            "column_mapping": preview.column_mapping,
            "validation_warnings": preview.validation_warnings,
            "detected_time_range": preview.detected_time_range,
            "preview_records": preview.preview_records,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Telemetry preview failed: {str(exc)}",
        )


@router.post("/plants/{plant_id}/telemetry/import")
def import_telemetry_file(
    plant_id: int,
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Commit telemetry file ingestion into database records."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with id {plant_id} not found.",
        )

    importer = TelemetryImporter(db)
    try:
        content = file.file.read()
        batch = importer.commit_import(plant_id, content, file.filename, notes)
        return {
            "message": "Telemetry batch committed successfully.",
            "batch": batch.to_dict(),
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Telemetry import failed: {str(exc)}",
        )


@router.get("/plants/{plant_id}/telemetry/batches")
def list_telemetry_batches(
    plant_id: int,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all telemetry import batches for a plant."""
    analytics = TelemetryAnalyticsService(db)
    return analytics.get_batches(plant_id)


@router.get("/plants/{plant_id}/telemetry/records")
def get_telemetry_records(
    plant_id: int,
    inverter_id: Optional[int] = Query(None),
    string_id: Optional[int] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(500, le=2000),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Query time-series measurement points."""
    analytics = TelemetryAnalyticsService(db)
    return analytics.get_records(plant_id, inverter_id, string_id, start_time, end_time, limit)


@router.get("/plants/{plant_id}/telemetry/trends")
def get_telemetry_trends(
    plant_id: int,
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve aggregated curve points and statistical summary for dashboard charts."""
    analytics = TelemetryAnalyticsService(db)
    return analytics.get_trend_summary(plant_id, hours)


@router.get("/telemetry/scada-status")
def get_scada_status() -> Dict[str, Any]:
    """Retrieve connection status of external SCADA/Modbus provider adapter."""
    adapter = SCADAAdapter()
    return adapter.get_status()
