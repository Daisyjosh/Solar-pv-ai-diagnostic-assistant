"""Diagnostic reasoning and engineer review REST endpoints."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.diagnosis import Diagnosis, DiagnosticAuditLog
from src.database.models.image_asset import ImageAsset
from src.database.models.inspection import Inspection
from src.database.models.inverter import Inverter
from src.database.models.module import PVModule
from src.database.models.plant import Plant
from src.database.models.string import PVString
from src.database.session import get_db
from src.diagnosis.engine import DiagnosticEngine, get_diagnostic_engine
from src.vision.inference import VisionInferenceService, get_inference_service

router = APIRouter(tags=["Diagnostic Engine"])


class ReviewUpdateRequest(BaseModel):
    """Schema for human-in-the-loop review update."""

    new_status: str  # CONFIRMED, MODIFIED, REJECTED
    new_severity: Optional[str] = None
    reviewed_by: str
    engineer_notes: Optional[str] = None


@router.post("/inspections/{inspection_id}/diagnose")
def trigger_inspection_diagnosis(
    inspection_id: int,
    db: Session = Depends(get_db),
    diag_engine: DiagnosticEngine = Depends(get_diagnostic_engine),
    vision_service: VisionInferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    """Synthesize multi-modal evidence and execute automated diagnostic assessment."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with id {inspection_id} not found.",
        )

    plant = db.get(Plant, inspection.plant_id)
    inverter = db.get(Inverter, inspection.inverter_id) if inspection.inverter_id else None
    pv_string = db.get(PVString, inspection.string_id) if inspection.string_id else None
    module = db.get(PVModule, inspection.module_id) if inspection.module_id else None

    asset_info = {
        "plant_name": plant.name if plant else "N/A",
        "plant_code": plant.code if plant else "N/A",
        "capacity_dc_mw": plant.capacity_dc_mw if plant else None,
        "inverter_code": inverter.code if inverter else None,
        "string_code": pv_string.code if pv_string else None,
        "module_code": module.code if module else None,
        "rated_voc_v": pv_string.rated_voc_v if pv_string else None,
    }

    # Retrieve linked images
    stmt_imgs = select(ImageAsset).where(ImageAsset.inspection_id == inspection.id)
    images = db.scalars(stmt_imgs).all()

    # Look for thermal image to pass through vision pipeline
    thermal_img = next((img for img in images if img.image_type == "THERMAL_IR"), None)
    pred_class = None
    pred_conf = None
    pred_probs = {}
    unc_tier = None
    inp_val = None
    tax_note = None

    if thermal_img:
        try:
            resolved_img_path = thermal_img.resolve_absolute_path()
            pred_res = vision_service.predict_path(resolved_img_path)
            if pred_res.status.value == "MODEL_READY":
                pred_class = pred_res.predicted_class
                pred_conf = pred_res.confidence
                pred_probs = pred_res.probabilities
                unc_tier = pred_res.uncertainty_tier
                inp_val = pred_res.input_validation
                tax_note = pred_res.taxonomy_note
        except Exception as exc:
            logger.warning("Could not run vision inference for thermal image %s: %s", thermal_img.id, exc)

    # Run Diagnostic Engine
    assessment = diag_engine.diagnose(
        inspection_id=inspection.id,
        asset_info=asset_info,
        image_modality="THERMAL_IR" if thermal_img else "OPTICAL_RGB",
        model_prediction=pred_class,
        model_confidence=pred_conf,
        model_probabilities=pred_probs,
        irradiance_w_m2=inspection.irradiance_w_m2,
        ambient_temp_c=inspection.ambient_temp_c,
        notes=inspection.general_notes,
        uncertainty_tier=unc_tier,
        input_validation=inp_val,
        taxonomy_note=tax_note,
    )

    # Persist or update Diagnosis entity
    stmt_diag = select(Diagnosis).where(Diagnosis.inspection_id == inspection.id)
    diagnosis_record = db.scalars(stmt_diag).first()

    if not diagnosis_record:
        diagnosis_record = Diagnosis(
            inspection_id=inspection.id,
            assessment_id=assessment.assessment_id,
            fault_category=assessment.fault_category_name,
            canonical_fault_code=assessment.canonical_fault_code,
            severity=assessment.severity.value,
            severity_justification=assessment.severity_justification,
            model_prediction=assessment.model_prediction,
            model_confidence=assessment.prediction_confidence,
            power_loss_percent=assessment.estimated_power_loss_percent,
            power_loss_justification=assessment.power_loss_justification,
            root_causes_json=json.dumps(assessment.probable_root_causes),
            corrective_actions_json=json.dumps(assessment.corrective_actions),
            preventive_actions_json=json.dumps(assessment.preventive_actions),
            safety_precautions_json=json.dumps(assessment.safety_precautions),
            required_measurements_json=json.dumps(assessment.required_additional_measurements),
            diagnostic_limitations_json=json.dumps(assessment.diagnostic_limitations),
            raw_assessment_json=json.dumps(assessment.to_dict()),
            review_status="PENDING_REVIEW",
        )
        db.add(diagnosis_record)
    else:
        # Update existing
        diagnosis_record.assessment_id = assessment.assessment_id
        diagnosis_record.fault_category = assessment.fault_category_name
        diagnosis_record.canonical_fault_code = assessment.canonical_fault_code
        diagnosis_record.severity = assessment.severity.value
        diagnosis_record.severity_justification = assessment.severity_justification
        diagnosis_record.model_prediction = assessment.model_prediction
        diagnosis_record.model_confidence = assessment.prediction_confidence
        diagnosis_record.power_loss_percent = assessment.estimated_power_loss_percent
        diagnosis_record.power_loss_justification = assessment.power_loss_justification
        diagnosis_record.root_causes_json = json.dumps(assessment.probable_root_causes)
        diagnosis_record.corrective_actions_json = json.dumps(assessment.corrective_actions)
        diagnosis_record.preventive_actions_json = json.dumps(assessment.preventive_actions)
        diagnosis_record.safety_precautions_json = json.dumps(assessment.safety_precautions)
        diagnosis_record.required_measurements_json = json.dumps(assessment.required_additional_measurements)
        diagnosis_record.diagnostic_limitations_json = json.dumps(assessment.diagnostic_limitations)
        diagnosis_record.raw_assessment_json = json.dumps(assessment.to_dict())

    if inspection.status == "SCHEDULED":
        inspection.status = "IN_PROGRESS"
        db.add(inspection)

    db.commit()
    db.refresh(diagnosis_record)
    return diagnosis_record.to_dict()


@router.get("/inspections/{inspection_id}/diagnosis")
def get_inspection_diagnosis(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve saved diagnostic assessment for an inspection."""
    stmt = select(Diagnosis).where(Diagnosis.inspection_id == inspection_id)
    diag = db.scalars(stmt).first()
    if not diag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No diagnosis found for inspection {inspection_id}.",
        )
    return diag.to_dict()


@router.put("/diagnoses/{diagnosis_id}/review")
def update_diagnosis_review(
    diagnosis_id: int,
    req: ReviewUpdateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Record human engineer sign-off, override severity, and log audit trail."""
    diag = db.get(Diagnosis, diagnosis_id)
    if not diag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnosis with id {diagnosis_id} not found.",
        )

    prev_status = diag.review_status
    prev_sev = diag.severity
    new_sev = req.new_severity or prev_sev

    diag.review_status = req.new_status
    if req.new_severity:
        diag.severity = req.new_severity
    diag.reviewed_by = req.reviewed_by
    diag.reviewed_at = datetime.now(timezone.utc)
    if req.engineer_notes:
        diag.engineer_notes = req.engineer_notes

    # Create audit log entry
    audit = DiagnosticAuditLog(
        diagnosis_id=diag.id,
        changed_by=req.reviewed_by,
        previous_status=prev_status,
        new_status=req.new_status,
        previous_severity=prev_sev,
        new_severity=new_sev,
        notes=req.engineer_notes,
    )
    db.add(audit)
    db.commit()
    db.refresh(diag)
    return diag.to_dict()
