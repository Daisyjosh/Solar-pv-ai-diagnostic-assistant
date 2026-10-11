"""PDF engineering report streaming and high-level dashboard aggregation endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.diagnosis import Diagnosis
from src.database.models.image_asset import ImageAsset
from src.database.models.inspection import Inspection
from src.database.models.inverter import Inverter
from src.database.models.module import PVModule
from src.database.models.plant import Plant
from src.database.models.string import PVString
from src.database.models.telemetry import TelemetryBatch, TelemetryRecord
from src.database.session import get_db
from src.reports.pdf_generator import InspectionReportPDFGenerator

router = APIRouter(tags=["Reports & Dashboard"])


@router.get("/inspections/{inspection_id}/report/pdf")
def download_inspection_pdf_report(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """Compile and stream audit-ready PDF engineering dossier for an inspection."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection {inspection_id} not found.",
        )

    plant = db.get(Plant, inspection.plant_id)
    inverter = db.get(Inverter, inspection.inverter_id) if inspection.inverter_id else None
    pv_string = db.get(PVString, inspection.string_id) if inspection.string_id else None
    module = db.get(PVModule, inspection.module_id) if inspection.module_id else None

    # Diagnosis
    stmt_diag = select(Diagnosis).where(Diagnosis.inspection_id == inspection.id)
    diag = db.scalars(stmt_diag).first()

    # Images
    stmt_imgs = select(ImageAsset).where(ImageAsset.inspection_id == inspection.id)
    images = db.scalars(stmt_imgs).all()

    # Telemetry
    stmt_tel = (
        select(TelemetryRecord)
        .where(TelemetryRecord.plant_id == inspection.plant_id)
        .order_by(TelemetryRecord.timestamp.desc())
        .limit(10)
    )
    telemetry_records = db.scalars(stmt_tel).all()

    insp_data = {
        "code": inspection.code,
        "status": inspection.status,
        "inspector_name": inspection.inspector_name,
        "inspection_date": inspection.inspection_date.strftime("%Y-%m-%d %H:%M UTC"),
        "ambient_temp_c": inspection.ambient_temp_c,
        "irradiance_w_m2": inspection.irradiance_w_m2,
        "asset_info": {
            "plant_name": plant.name if plant else "N/A",
            "capacity_dc_mw": plant.capacity_dc_mw if plant else "N/A",
            "inverter_code": inverter.code if inverter else "Unspecified",
            "string_code": pv_string.code if pv_string else "Unspecified",
            "module_code": module.code if module else "Unspecified",
        },
    }

    diag_data = diag.to_dict() if diag else None
    imgs_data = [
        {
            "id": img.id,
            "image_type": img.image_type,
            "original_filename": img.original_filename,
            "file_path": img.file_path,
            "notes": img.notes,
        }
        for img in images
    ]
    tel_data = [t.to_dict() for t in telemetry_records]

    pdf_gen = InspectionReportPDFGenerator()
    try:
        pdf_bytes = pdf_gen.generate(insp_data, diag_data, imgs_data, tel_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(exc)}",
        )

    filename = f"report_{inspection.code}_{inspection.id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/pdf",
        },
    )


@router.get("/dashboard/summary")
def get_dashboard_summary(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Aggregate high-level KPIs and operational statistics across all plants."""
    total_plants = db.scalar(select(func.count(Plant.id))) or 0
    total_inspections = db.scalar(select(func.count(Inspection.id))) or 0
    total_images = db.scalar(select(func.count(ImageAsset.id))) or 0

    pending_reviews = (
        db.scalar(select(func.count(Diagnosis.id)).where(Diagnosis.review_status == "PENDING_REVIEW"))
        or 0
    )
    critical_findings = (
        db.scalar(select(func.count(Diagnosis.id)).where(Diagnosis.severity.in_(["CRITICAL", "MAJOR"])))
        or 0
    )

    # Fault Category distribution
    fault_counts_query = (
        select(Diagnosis.fault_category, func.count(Diagnosis.id))
        .group_by(Diagnosis.fault_category)
    )
    fault_distribution = {row[0]: row[1] for row in db.execute(fault_counts_query).all()}

    # Severity distribution
    sev_counts_query = (
        select(Diagnosis.severity, func.count(Diagnosis.id))
        .group_by(Diagnosis.severity)
    )
    severity_distribution = {row[0]: row[1] for row in db.execute(sev_counts_query).all()}

    # Recent inspections
    recent_insps = (
        db.scalars(select(Inspection).order_by(Inspection.inspection_date.desc()).limit(5)).all()
    )
    recent_list = []
    for insp in recent_insps:
        plant = db.get(Plant, insp.plant_id)
        diag = db.scalars(select(Diagnosis).where(Diagnosis.inspection_id == insp.id)).first()
        recent_list.append({
            "id": insp.id,
            "code": insp.code,
            "plant_name": plant.name if plant else "N/A",
            "date": insp.inspection_date.strftime("%Y-%m-%d %H:%M"),
            "status": insp.status,
            "severity": diag.severity if diag else "UNASSESSED",
            "fault_category": diag.fault_category if diag else "Pending Diagnosis",
        })

    # Telemetry summary
    total_telemetry_records = db.scalar(select(func.count(TelemetryRecord.id))) or 0
    total_telemetry_batches = db.scalar(select(func.count(TelemetryBatch.id))) or 0

    return {
        "kpis": {
            "total_plants": total_plants,
            "total_inspections": total_inspections,
            "total_images": total_images,
            "pending_reviews": pending_reviews,
            "open_critical_findings": critical_findings,
            "total_telemetry_records": total_telemetry_records,
            "total_telemetry_batches": total_telemetry_batches,
        },
        "fault_distribution": fault_distribution,
        "severity_distribution": severity_distribution,
        "recent_inspections": recent_list,
    }
