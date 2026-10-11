"""Batch inspection processing endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.services.batch_inspection import BatchInspectionService

router = APIRouter(tags=["Batch Inspections"])


class CreateBatchRequest(BaseModel):
    title: str
    expected_count: Optional[int] = 0


@router.post("/plants/{plant_id}/batch-inspections")
def create_batch_inspection(
    plant_id: int,
    req: CreateBatchRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Initialize a batch inspection tracking session."""
    service = BatchInspectionService(db)
    batch = service.create_batch(plant_id, req.title, req.expected_count or 0)
    return batch.to_dict()


@router.post("/batch-inspections/{batch_id}/process")
def process_batch_images(
    batch_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Execute automated vision classification and diagnostic assessment across uploaded image crops."""
    service = BatchInspectionService(db)
    images_data = []
    for f in files:
        images_data.append({
            "filename": f.filename,
            "bytes": f.file.read(),
        })

    try:
        summary = service.process_image_crops(batch_id, images_data)
        return summary
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch processing failed: {str(exc)}",
        )


@router.get("/plants/{plant_id}/batch-inspections")
def list_batch_inspections(
    plant_id: int,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all batch sessions for a solar plant."""
    service = BatchInspectionService(db)
    return service.list_batches(plant_id)
