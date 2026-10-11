"""Batch inspection processing service for automated multi-module ingestion."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.batch_inspection import BatchInspection
from src.database.models.plant import Plant
from src.diagnosis.engine import DiagnosticEngine, get_diagnostic_engine
from src.vision.inference import VisionInferenceService, get_inference_service

logger = logging.getLogger(__name__)


class BatchInspectionService:
    """Coordinates multi-image ingestion, batch inference, and diagnostic triage."""

    def __init__(
        self,
        db_session: Session,
        inference_service: Optional[VisionInferenceService] = None,
        diagnostic_engine: Optional[DiagnosticEngine] = None,
    ):
        self.db = db_session
        self.inference = inference_service or get_inference_service()
        self.engine = diagnostic_engine or get_diagnostic_engine()

    def create_batch(self, plant_id: int, title: str, expected_count: int = 0) -> BatchInspection:
        """Initialize a new batch inspection tracking record."""
        code = f"BATCH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        batch = BatchInspection(
            code=code,
            plant_id=plant_id,
            title=title,
            total_images=expected_count,
            processed_images=0,
            status="PENDING",
        )
        self.db.add(batch)
        self.db.commit()
        self.db.refresh(batch)
        return batch

    def process_image_crops(
        self,
        batch_id: int,
        images_data: List[Dict[str, Any]],  # list of dicts with 'filename', 'bytes'
    ) -> Dict[str, Any]:
        """Run vision inference and diagnostic classification on batch images."""
        batch = self.db.get(BatchInspection, batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")

        plant = self.db.get(Plant, batch.plant_id)
        plant_info = {
            "plant_name": plant.name,
            "plant_code": plant.code,
            "capacity_dc_mw": plant.capacity_dc_mw,
        } if plant else {}

        batch.status = "PROCESSING"
        batch.total_images = len(images_data)
        self.db.commit()

        results = []
        fault_distribution: Dict[str, int] = {}
        severity_counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "NORMAL": 0, "INCONCLUSIVE": 0}

        for idx, item in enumerate(images_data):
            fname = item["filename"]
            raw_bytes = item["bytes"]

            try:
                img = Image.open(io_bytes := Path(fname) if isinstance(raw_bytes, Path) else None) if isinstance(raw_bytes, Path) else Image.open(io.BytesIO(raw_bytes))
            except Exception as e:
                import io
                try:
                    img = Image.open(io.BytesIO(raw_bytes))
                except Exception as ex:
                    logger.error("Failed to decode image %s: %s", fname, ex)
                    continue

            # Run Vision Inference
            pred_res = self.inference.predict_image(img)

            # Run Diagnostic Engine
            diag = self.engine.diagnose(
                inspection_id=0,  # Batch preview mode
                asset_info=plant_info,
                image_modality="THERMAL_IR",
                model_prediction=pred_res.predicted_class,
                model_confidence=pred_res.confidence,
                model_probabilities=pred_res.probabilities,
                uncertainty_tier=pred_res.uncertainty_tier,
                input_validation=pred_res.input_validation,
                taxonomy_note=pred_res.taxonomy_note,
            )

            pred_class = pred_res.predicted_class or "Unknown"
            fault_distribution[pred_class] = fault_distribution.get(pred_class, 0) + 1
            severity_counts[diag.severity.value] = severity_counts.get(diag.severity.value, 0) + 1

            results.append({
                "item_index": idx + 1,
                "filename": fname,
                "predicted_class": pred_res.predicted_class,
                "confidence": pred_res.confidence,
                "fault_category": diag.fault_category_name,
                "severity": diag.severity.value,
                "power_loss_percent": diag.estimated_power_loss_percent,
                "primary_hypothesis": diag.probable_root_causes[0] if diag.probable_root_causes else "N/A",
            })

            batch.processed_images += 1

        batch.status = "COMPLETED"
        self.db.commit()

        return {
            "batch_id": batch.id,
            "batch_code": batch.code,
            "plant_id": batch.plant_id,
            "total_images": batch.total_images,
            "processed_images": batch.processed_images,
            "status": batch.status,
            "fault_distribution": fault_distribution,
            "severity_counts": severity_counts,
            "critical_anomalies_count": severity_counts.get("CRITICAL", 0) + severity_counts.get("MAJOR", 0),
            "results": results,
        }

    def list_batches(self, plant_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all batch inspection sessions."""
        stmt = select(BatchInspection)
        if plant_id:
            stmt = stmt.where(BatchInspection.plant_id == plant_id)
        stmt = stmt.order_by(BatchInspection.created_at.desc())
        batches = self.db.scalars(stmt).all()
        return [b.to_dict() for b in batches]
