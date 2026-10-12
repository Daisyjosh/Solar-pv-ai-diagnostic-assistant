"""YOLO Vision Service for Solar PV Thermal and Optical Anomaly Detection.

Provides dedicated, isolated object detection and spatial defect localization
using YOLO models (Ultralytics / ONNX), operating independently alongside the
HistGradientBoosting benchmark classifier and Gemini multimodal vision service.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from pydantic import BaseModel, Field

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    """Bounding box coordinates for a localized visual anomaly."""

    xmin: float = Field(description="Top-left X coordinate (pixels or normalized)")
    ymin: float = Field(description="Top-left Y coordinate (pixels or normalized)")
    xmax: float = Field(description="Bottom-right X coordinate (pixels or normalized)")
    ymax: float = Field(description="Bottom-right Y coordinate (pixels or normalized)")
    is_normalized: bool = Field(default=False, description="Whether coordinates are normalized [0.0, 1.0]")


class YOLODetection(BaseModel):
    """Individual object detection or anomaly localization candidate."""

    class_id: int = Field(description="Numeric class identifier")
    class_name: str = Field(description="Detected anomaly category label")
    confidence: float = Field(description="Detection confidence score [0.0, 1.0]")
    bbox: BoundingBox = Field(description="Spatial bounding box")


class YOLOAnalysisResult(BaseModel):
    """Structured response from YOLO object detection and anomaly localization."""

    status: str = Field(description="Operational status: SUCCESS, WEIGHTS_UNAVAILABLE, DEPENDENCY_UNAVAILABLE, etc.")
    is_configured: bool = Field(default=False, description="Whether YOLO weights and runtime are available")
    model_path: Optional[str] = Field(default=None, description="Path to active YOLO model weights")
    detections: List[YOLODetection] = Field(default_factory=list, description="List of localized anomaly detections")
    total_detections: int = Field(default=0, description="Total count of detected anomalies")
    annotated_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded JPEG with rendered bounding boxes"
    )
    message: str = Field(default="", description="User-facing status or guidance message")
    disclaimer: str = Field(
        default="Automated YOLO detection hypothesis for engineering review — requires physical verification.",
        description="Mandatory engineering disclaimer",
    )


class YOLOVisionService:
    """Manages YOLO model lifecycle, weights validation, and object detection inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ):
        settings = get_settings()
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = settings.yolo_model_path or os.getenv(
                "YOLO_MODEL_PATH", "models/yolo/active/best.pt"
            )

        eff_conf = conf_threshold if conf_threshold is not None else confidence_threshold
        if eff_conf is not None:
            self.conf_threshold = eff_conf
        else:
            self.conf_threshold = settings.yolo_confidence_threshold or 0.25

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        else:
            self.iou_threshold = settings.yolo_iou_threshold or 0.45

        self._model = None

    def _check_dependencies(self) -> bool:
        """Check if YOLO inference runtime libraries (ultralytics / torch) are importable."""
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    def _has_weights(self) -> bool:
        """Check whether the configured weights file exists on disk and is non-empty."""
        if not self.model_path or not self.model_path.strip():
            return False
        p = Path(self.model_path)
        return p.is_file() and p.stat().st_size > 0

    def get_status(self) -> Dict[str, Any]:
        """Return operational configuration and readiness status of the YOLO vision service."""
        has_deps = self._check_dependencies()
        has_w = self._has_weights()
        is_ready = has_deps and has_w

        if not has_w:
            status_val = "WEIGHTS_UNAVAILABLE"
            msg = (
                f"No YOLO model weights found at '{self.model_path}'. "
                "The benchmark dataset (InfraredSolarModules) provides cropped classification labels "
                "without bounding-box ground truth. To enable YOLO object detection, place trained weights "
                "(e.g. best.pt or best.onnx) at the designated path or configure YOLO_MODEL_PATH."
            )
        elif not has_deps:
            status_val = "DEPENDENCY_UNAVAILABLE"
            msg = (
                "YOLO model weights are present, but the 'ultralytics' runtime package is not installed. "
                "Install ultralytics ('pip install ultralytics') to enable YOLO inference."
            )
        else:
            status_val = "READY"
            msg = f"YOLO vision service ready with weights '{self.model_path}'."

        return {
            "status": status_val,
            "is_configured": is_ready,
            "has_weights": has_w,
            "has_dependencies": has_deps,
            "model_path": self.model_path,
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "provider": "Ultralytics YOLO Anomaly Localization",
            "message": msg,
        }

    def _load_model(self):
        """Lazy-load the YOLO model instance."""
        if self._model is not None:
            return self._model

        if not self._has_weights() or not self._check_dependencies():
            return None

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info("Loaded YOLO model from %s", self.model_path)
            return self._model
        except Exception as exc:
            logger.exception("Failed to load YOLO model from %s: %s", self.model_path, exc)
            return None

    def detect_image(self, image_bytes: bytes) -> YOLOAnalysisResult:
        """Execute YOLO anomaly detection against raw image bytes."""
        if not image_bytes or len(image_bytes) == 0:
            return YOLOAnalysisResult(
                status="IMAGE_ERROR",
                is_configured=False,
                model_path=self.model_path,
                message="Cannot analyze empty image bytes.",
            )

        # 1. Validate weights availability
        if not self._has_weights():
            return YOLOAnalysisResult(
                status="WEIGHTS_UNAVAILABLE",
                is_configured=False,
                model_path=self.model_path,
                detections=[],
                total_detections=0,
                message=(
                    f"YOLO weights file not found at '{self.model_path}'. "
                    "The benchmark dataset (InfraredSolarModules) provides cropped classification labels "
                    "without bounding-box annotations. To run YOLO defect localization, provide trained "
                    "weights at this path or configure YOLO_MODEL_PATH in .env."
                ),
            )

        # 2. Validate runtime dependencies
        if not self._check_dependencies():
            return YOLOAnalysisResult(
                status="DEPENDENCY_UNAVAILABLE",
                is_configured=False,
                model_path=self.model_path,
                detections=[],
                total_detections=0,
                message=(
                    "The 'ultralytics' library is not installed in the Python environment. "
                    "Install ultralytics to execute YOLO inference."
                ),
            )

        # 3. Load model
        model = self._load_model()
        if model is None:
            return YOLOAnalysisResult(
                status="LOAD_ERROR",
                is_configured=False,
                model_path=self.model_path,
                detections=[],
                total_detections=0,
                message=f"Failed to load YOLO model from '{self.model_path}'.",
            )

        # 4. Decode image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            return YOLOAnalysisResult(
                status="IMAGE_DECODE_ERROR",
                is_configured=True,
                model_path=self.model_path,
                message=f"Failed to decode image: {exc}",
            )

        # 5. Run prediction
        try:
            results = model.predict(
                source=img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections: List[YOLODetection] = []
            annotated_base64 = None

            if results and len(results) > 0:
                res0 = results[0]
                boxes = getattr(res0, "boxes", None)
                names = getattr(res0, "names", {}) or {}

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, f"Class {cls_id}")

                        detections.append(
                            YOLODetection(
                                class_id=cls_id,
                                class_name=str(cls_name),
                                confidence=conf,
                                bbox=BoundingBox(
                                    xmin=round(coords[0], 1),
                                    ymin=round(coords[1], 1),
                                    xmax=round(coords[2], 1),
                                    ymax=round(coords[3], 1),
                                    is_normalized=False,
                                ),
                            )
                        )

                # Render annotated visualization
                try:
                    annotated_bgr = res0.plot()
                    if annotated_bgr is not None:
                        import cv2
                        _, buffer = cv2.imencode(".jpg", annotated_bgr)
                        annotated_base64 = base64.b64encode(buffer).decode("utf-8")
                except Exception as render_exc:
                    logger.warning("Failed to render YOLO annotated visualization: %s", render_exc)

            return YOLOAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_path=self.model_path,
                detections=detections,
                total_detections=len(detections),
                annotated_image_base64=annotated_base64,
                message=(
                    f"YOLO detected {len(detections)} anomaly candidate(s)."
                    if len(detections) > 0
                    else "YOLO inference executed successfully — no anomalies detected above confidence threshold."
                ),
            )

        except Exception as exc:
            logger.exception("YOLO inference failed: %s", exc)
            return YOLOAnalysisResult(
                status="ERROR",
                is_configured=True,
                model_path=self.model_path,
                message=f"YOLO inference error: {exc}",
            )

    def detect_asset(self, image_path: Union[str, Path]) -> YOLOAnalysisResult:
        """Resolve image asset from disk and execute YOLO anomaly detection."""
        path_obj = Path(image_path)
        if not path_obj.exists():
            try:
                from src.services.storage import StorageService
                resolved = StorageService().get_image_file_path(path_obj)
                if resolved.exists():
                    path_obj = resolved
            except Exception:
                pass

        if not path_obj.exists():
            return YOLOAnalysisResult(
                status="FILE_NOT_FOUND",
                is_configured=self._has_weights(),
                model_path=self.model_path,
                message=f"Image file does not exist on disk: {image_path}",
            )

        try:
            with open(path_obj, "rb") as f:
                img_bytes = f.read()
            return self.detect_image(img_bytes)
        except Exception as exc:
            return YOLOAnalysisResult(
                status="FILE_READ_ERROR",
                is_configured=self._has_weights(),
                model_path=self.model_path,
                message=f"Unable to read image file from disk: {exc}",
            )


# Global singleton instance
_yolo_service: Optional[YOLOVisionService] = None


def get_yolo_service() -> YOLOVisionService:
    """Retrieve global YOLOVisionService singleton."""
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOVisionService()
    return _yolo_service
