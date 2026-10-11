"""Solar PV Vision Inference Service.

Provides a unified prediction interface, explicit inference statuses,
and evaluation metric reporting for backend and diagnostic consumers.
"""

from dataclasses import asdict
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from src.vision.classifier import (
    SolarPVClassifier,
    ThermalInputValidation,
    extract_features,
    validate_thermal_input,
)

logger = logging.getLogger(__name__)


class InferenceStatus(str, Enum):
    """Explicit lifecycle status for model inference operations."""

    MODEL_READY = "MODEL_READY"
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"


class PredictionResult(BaseModel):
    """Structured inference response returned by the vision pipeline."""

    status: InferenceStatus
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_version: Optional[str] = None
    algorithm: Optional[str] = None
    is_low_confidence: bool = False
    uncertainty_tier: str = "ACCEPTED_HYPOTHESIS"
    is_calibrated: bool = False
    taxonomy_note: Optional[str] = None
    input_validation: Optional[Dict[str, Any]] = None
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = ""
    evaluation_disclaimer: str = (
        "Model prediction is an image-level hypothesis based on 24x40 thermal crops. "
        "It does not constitute confirmed physical root-cause diagnosis without electrical corroboration."
    )


class VisionInferenceService:
    """Manages model loading, caching, inference execution, and metric retrieval."""

    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = models_root or Path("models/classifier")
        self._classifier: Optional[SolarPVClassifier] = None
        self._calibrator: Optional[Any] = None
        self._model_version: Optional[str] = None
        self._load_active_model()

    def _resolve_model_dir(self) -> Optional[Path]:
        """Locate active or latest trained model directory."""
        if not self.models_root.exists():
            return None

        # 1. Check for dedicated 'active' link/folder
        active_dir = self.models_root / "active"
        if active_dir.exists() and (active_dir / "model.joblib").exists():
            return active_dir

        # 2. Check for versioned directories (e.g. v1_*, 2026-*)
        subdirs = [
            d for d in self.models_root.iterdir()
            if d.is_dir() and (d / "model.joblib").exists() and (d / "metadata.json").exists()
        ]
        if subdirs:
            # Sort by modification time descending
            subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return subdirs[0]

        return None

    def _load_active_model(self) -> None:
        """Attempt to load the model into memory."""
        model_dir = self._resolve_model_dir()
        if not model_dir:
            logger.info("No trained model found in %s", self.models_root)
            self._classifier = None
            self._calibrator = None
            self._model_version = None
            return

        try:
            self._classifier = SolarPVClassifier.load(model_dir)
            self._model_version = (
                self._classifier.metadata.model_version
                if self._classifier.metadata
                else model_dir.name
            )
            logger.info("Successfully loaded vision model version: %s", self._model_version)

            # Optional validation-fitted calibrator sidecar
            cal_path = model_dir / "calibrator.joblib"
            if cal_path.exists():
                try:
                    self._calibrator = joblib.load(cal_path)
                    logger.info("Successfully loaded probability calibrator sidecar from %s", cal_path)
                except Exception as cal_err:
                    logger.warning("Failed to load calibrator sidecar (%s); operating in raw mode: %s", cal_path, cal_err)
                    self._calibrator = None
            else:
                self._calibrator = None

        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_dir, exc)
            self._classifier = None
            self._calibrator = None
            self._model_version = None

    def reload(self) -> None:
        """Force reloading model from disk."""
        self._load_active_model()

    def get_status(self) -> Dict[str, Any]:
        """Return operational status and metadata of the inference engine."""
        if self._classifier is None:
            model_dir = self._resolve_model_dir()
            if model_dir is None:
                return {
                    "status": InferenceStatus.MODEL_NOT_TRAINED,
                    "message": "No trained model found on disk. Run `python scripts/train_classifier.py` to train.",
                    "model_version": None,
                    "class_names": [],
                    "is_calibrated": False,
                }
            return {
                "status": InferenceStatus.MODEL_UNAVAILABLE,
                "message": "Model files present on disk but failed to initialize.",
                "model_version": None,
                "class_names": [],
                "is_calibrated": False,
            }

        meta = self._classifier.metadata
        return {
            "status": InferenceStatus.MODEL_READY,
            "message": "Vision classifier is loaded and ready for inference.",
            "model_version": self._model_version,
            "algorithm": meta.algorithm if meta else "HistGradientBoostingClassifier",
            "class_names": self._classifier.class_names,
            "training_samples": meta.training_sample_count if meta else None,
            "created_at_utc": meta.created_at_utc if meta else None,
            "is_calibrated": self._calibrator is not None,
        }

    def predict_image(self, image: Image.Image) -> PredictionResult:
        """Execute inference against an in-memory PIL Image."""
        if self._classifier is None:
            return PredictionResult(
                status=InferenceStatus.MODEL_NOT_TRAINED,
                message="Cannot perform inference: model is not trained or loaded.",
            )

        try:
            # 1. Explicit input validation diagnostics
            val_res = validate_thermal_input(image)

            # 2. Domain feature extraction
            feats = extract_features(image)
            X = feats.reshape(1, -1)

            # 3. Probability estimation: use calibrator if loaded, otherwise raw model
            if self._calibrator is not None:
                probs = self._calibrator.predict_proba(X)[0]
                is_calibrated = True
            else:
                probs = self._classifier.predict_proba(X)[0]
                is_calibrated = False

            class_probs = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self._classifier.class_names, probs)
            }

            sorted_indices = np.argsort(probs)[::-1]
            top1_idx = int(sorted_indices[0])
            top2_idx = int(sorted_indices[1])
            top1_class = self._classifier.class_names[top1_idx]
            top2_class = self._classifier.class_names[top2_idx]
            top1_prob = float(probs[top1_idx])
            top2_prob = float(probs[top2_idx])
            margin = top1_prob - top2_prob

            predicted_class = top1_class
            confidence = round(float(top1_prob), 4)

            # 4. Dual-tier empirical uncertainty policy based on validation calibration study
            if top1_prob < 0.40 or margin < 0.05:
                uncertainty_tier = "AMBIGUOUS_HYPOTHESIS"
                is_low_conf = True
            elif top1_prob < 0.50 or margin < 0.08:
                uncertainty_tier = "LOW_CONFIDENCE_HYPOTHESIS"
                is_low_conf = True
            else:
                uncertainty_tier = "ACCEPTED_HYPOTHESIS"
                is_low_conf = False

            top_candidates = [
                {"class_name": self._classifier.class_names[int(idx)], "probability": round(float(probs[idx]), 4)}
                for idx in sorted_indices[:3]
            ]

            # 5. Domain benchmark taxonomy context
            if predicted_class in ("Cell", "Cell-Multi"):
                taxonomy_note = (
                    "In the InfraredSolarModules benchmark, 'Cell' and 'Cell-Multi' designate "
                    "localized hot spots on crystalline silicon (c-Si) modules."
                )
            elif predicted_class in ("Hot-Spot", "Hot-Spot-Multi"):
                taxonomy_note = (
                    "In the InfraredSolarModules benchmark, 'Hot-Spot' specifically designates "
                    "thin-film modules (CdTe/CIGS monolithic scribes)."
                )
            elif predicted_class == "Cracking":
                taxonomy_note = (
                    "Thermal infrared alone cannot confirm microcracking without dark-field "
                    "Electroluminescence (EL) and string I-V curve corroboration."
                )
            elif predicted_class == "Diode":
                taxonomy_note = "Characteristic 1/3 substring bypass diode conduction under IEC 62446-3 Clause 5.2.3."
            else:
                taxonomy_note = None

            # 6. Structured human-readable message
            if uncertainty_tier == "AMBIGUOUS_HYPOTHESIS":
                msg = (
                    f"Ambiguous hypothesis: '{top1_class}' ({top1_prob:.1%}) closely contested by "
                    f"'{top2_class}' ({top2_prob:.1%}, margin: {margin:.1%}). "
                    "Prediction is ambiguous and requires physical electrical / EL corroboration."
                )
            elif uncertainty_tier == "LOW_CONFIDENCE_HYPOTHESIS":
                msg = (
                    f"Low-confidence hypothesis: '{top1_class}' ({top1_prob:.1%}) contested by "
                    f"'{top2_class}' ({top2_prob:.1%}, margin: {margin:.1%}). "
                    "Corroborating electrical measurements required."
                )
            else:
                msg = f"Predicted '{predicted_class}' with confidence {confidence:.2%}."

            if val_res.warnings:
                msg += f" [Input note: {'; '.join(val_res.warnings)}]"

            return PredictionResult(
                status=InferenceStatus.MODEL_READY,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=class_probs,
                is_low_confidence=is_low_conf,
                uncertainty_tier=uncertainty_tier,
                is_calibrated=is_calibrated,
                taxonomy_note=taxonomy_note,
                input_validation=asdict(val_res),
                top_candidates=top_candidates,
                model_version=self._model_version,
                algorithm="HistGradientBoostingClassifier",
                message=msg,
            )
        except Exception as exc:
            logger.exception("Inference execution failed: %s", exc)
            return PredictionResult(
                status=InferenceStatus.INFERENCE_FAILED,
                message=f"Inference computation error: {str(exc)}",
            )

    def predict_path(self, file_path: Union[Path, str]) -> PredictionResult:
        """Execute inference against an image file on disk."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            # Attempt central storage resolution as fallback
            try:
                from src.services.storage import StorageService
                resolved = StorageService().get_image_file_path(path_obj)
                if resolved.exists():
                    path_obj = resolved
            except Exception:
                pass

        if not path_obj.exists():
            return PredictionResult(
                status=InferenceStatus.INFERENCE_FAILED,
                message=f"Image file does not exist: {file_path}",
            )

        try:
            with Image.open(path_obj) as img:
                return self.predict_image(img)
        except Exception as exc:
            return PredictionResult(
                status=InferenceStatus.INFERENCE_FAILED,
                message=f"Unable to read or decode image file: {str(exc)}",
            )

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Retrieve stored evaluation report for the active model."""
        model_dir = self._resolve_model_dir()
        if not model_dir:
            return {"status": "NO_MODEL_AVAILABLE", "metrics": None}

        eval_path = model_dir / "evaluation_report.json"
        if not eval_path.exists():
            return {"status": "NO_EVALUATION_REPORT", "metrics": None}

        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return {"status": "SUCCESS", "metrics": report}
        except Exception as exc:
            return {"status": "ERROR_READING_REPORT", "error": str(exc), "metrics": None}


# Global singleton instance
_inference_service: Optional[VisionInferenceService] = None


def get_inference_service() -> VisionInferenceService:
    """Retrieve global VisionInferenceService singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = VisionInferenceService()
    return _inference_service
