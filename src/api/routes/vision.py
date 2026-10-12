"""Vision model lifecycle and inference REST endpoints."""

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from src.database.models.image_asset import ImageAsset
from src.database.session import get_db
from src.services.storage import DirectoryTraversalError
from src.vision.inference import (
    InferenceStatus,
    PredictionResult,
    VisionInferenceService,
    get_inference_service,
)

router = APIRouter(prefix="/vision", tags=["Vision Pipeline"])


@router.get("/model-status")
def get_model_status(
    service: VisionInferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    """Retrieve operational lifecycle status, version, and metadata of the active vision classifier."""
    return service.get_status()


@router.get("/evaluation-metrics")
def get_evaluation_metrics(
    service: VisionInferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    """Retrieve empirical test and validation metrics, including accuracy and confusion matrix."""
    metrics = service.get_evaluation_metrics()
    if metrics["status"] != "SUCCESS":
        return {"status": metrics["status"], "metrics": None, "message": "No evaluation report available on disk."}
    return metrics


@router.post("/predict/{image_asset_id}", response_model=PredictionResult)
def predict_image_asset(
    image_asset_id: int,
    db: Session = Depends(get_db),
    service: VisionInferenceService = Depends(get_inference_service),
) -> PredictionResult:
    """Execute anomaly classification against a previously ingested image asset."""
    asset = db.get(ImageAsset, image_asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset with id {image_asset_id} not found.",
        )

    try:
        resolved_path = asset.resolve_absolute_path()
    except (FileNotFoundError, DirectoryTraversalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored image file not found on disk at {asset.file_path}",
        )

    return service.predict_path(resolved_path)


@router.post("/predict-upload", response_model=PredictionResult)
def predict_uploaded_image(
    file: UploadFile = File(...),
    service: VisionInferenceService = Depends(get_inference_service),
) -> PredictionResult:
    """Execute anomaly classification directly on an uploaded image file."""
    try:
        content = file.file.read()
        import io
        img = Image.open(io.BytesIO(content))
        return service.predict_image(img)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode uploaded image: {str(exc)}",
        )


# --- Gemini Vision Endpoints (Free-Tier) ---

from src.vision.gemini_service import (
    GeminiAnalysisResult,
    GeminiVisionService,
    get_gemini_service,
)


@router.get("/gemini/status")
def get_gemini_status(
    gemini_svc: GeminiVisionService = Depends(get_gemini_service),
) -> Dict[str, Any]:
    """Return operational status and model name of the Gemini vision service."""
    return gemini_svc.get_status()


@router.post("/gemini/analyze/{image_asset_id}", response_model=GeminiAnalysisResult)
def analyze_image_with_gemini(
    image_asset_id: int,
    db: Session = Depends(get_db),
    gemini_svc: GeminiVisionService = Depends(get_gemini_service),
) -> GeminiAnalysisResult:
    """Analyze an existing image asset using Gemini multimodal vision."""
    asset = db.get(ImageAsset, image_asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset with id {image_asset_id} not found.",
        )

    try:
        resolved_path = asset.resolve_absolute_path()
    except (FileNotFoundError, DirectoryTraversalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored image file not found on disk at {asset.file_path}",
        )

    return gemini_svc.analyze_asset(resolved_path)


@router.post("/gemini/analyze-upload", response_model=GeminiAnalysisResult)
def analyze_uploaded_image_with_gemini(
    file: UploadFile = File(...),
    gemini_svc: GeminiVisionService = Depends(get_gemini_service),
) -> GeminiAnalysisResult:
    """Analyze directly uploaded image bytes using Gemini multimodal vision."""
    try:
        content = file.file.read()
        mime = file.content_type or "image/jpeg"
        return gemini_svc.analyze_image(content, mime_type=mime)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image bytes: {str(exc)}",
        )


# --- YOLO Object Detection Endpoints ---

from src.vision.yolo_service import (
    YOLOAnalysisResult,
    YOLOVisionService,
    get_yolo_service,
)


@router.get("/yolo/status")
def get_yolo_status(
    yolo_svc: YOLOVisionService = Depends(get_yolo_service),
) -> Dict[str, Any]:
    """Return operational status, weights path, and readiness of the YOLO vision service."""
    return yolo_svc.get_status()


@router.post("/yolo/detect/{image_asset_id}", response_model=YOLOAnalysisResult)
def detect_image_with_yolo(
    image_asset_id: int,
    db: Session = Depends(get_db),
    yolo_svc: YOLOVisionService = Depends(get_yolo_service),
) -> YOLOAnalysisResult:
    """Run YOLO object detection and anomaly localization on an existing stored image asset."""
    asset = db.get(ImageAsset, image_asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset with id {image_asset_id} not found.",
        )

    try:
        resolved_path = asset.resolve_absolute_path()
    except (FileNotFoundError, DirectoryTraversalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored image file not found on disk at {asset.file_path}",
        )

    return yolo_svc.detect_asset(resolved_path)


@router.post("/yolo/detect-upload", response_model=YOLOAnalysisResult)
def detect_uploaded_image_with_yolo(
    file: UploadFile = File(...),
    yolo_svc: YOLOVisionService = Depends(get_yolo_service),
) -> YOLOAnalysisResult:
    """Run YOLO object detection directly on uploaded image bytes."""
    try:
        content = file.file.read()
        return yolo_svc.detect_image(content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image bytes for YOLO inference: {str(exc)}",
        )

