"""Solar PV Anomaly Image Classification Pipeline.

Implements feature extraction, model definition, training, evaluation,
and persistence for the 12-class InfraredSolarModules thermal anomaly dataset.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Canonical 12 upstream InfraredSolarModules classes preserved without merging
UPSTREAM_CLASSES: List[str] = [
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "No-Anomaly",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]

TARGET_IMAGE_SIZE: Tuple[int, int] = (24, 40)  # width x height


@dataclass
class ThermalInputValidation:
    """Explicit validation diagnostics for input images prior to feature extraction."""

    is_valid: bool
    original_dimensions: Tuple[int, int]
    aspect_ratio: float
    benchmark_aspect_ratio: float = 0.60
    aspect_ratio_deviation: bool = False
    is_uncropped_scene: bool = False
    is_native_grayscale: bool = False
    color_mode: str = "L"
    modality_description: str = ""
    warnings: List[str] = field(default_factory=list)


def validate_thermal_input(image: Image.Image) -> ThermalInputValidation:
    """Explicitly validate input geometry, aspect ratio, and color mode.

    Engineering principles:
    - Preserves native grayscale without alteration.
    - Strictly avoids auto-rotating landscape images to preserve directional gradient semantics.
    - Flags aspect-ratio deviation (> 0.15 from 0.60) and uncropped scene advisories (> 100x100).
    - Handles RGB/false-color captures explicitly without guessing uncalibrated radiometry.
    """
    w, h = image.size
    aspect_ratio = round(w / float(h), 4) if h > 0 else 1.0
    benchmark_ar = 0.60  # 24 / 40
    warnings = []

    # 1. Native Grayscale vs. Rendered False-Color
    is_native_gray = image.mode in ("L", "I", "I;16", "I;16L", "I;16B", "1")
    if is_native_gray:
        modality_desc = f"Native single-channel grayscale ({image.mode})"
    else:
        modality_desc = (
            f"Rendered false-color capture ({image.mode}) without sensor calibration metadata"
        )
        warnings.append(
            "Image is an 8-bit rendered false-color capture. Converted to grayscale deterministically; "
            "pixel gradients may not reflect calibrated radiometry."
        )

    # 2. Aspect Ratio & Geometry Checks
    ar_dev = abs(aspect_ratio - benchmark_ar) > 0.15
    if ar_dev:
        warnings.append(
            f"Input aspect ratio {aspect_ratio:.2f} deviates from benchmark {benchmark_ar:.2f} (24x40). "
            "Bilinear resizing causes spatial geometric distortion."
        )

    # 3. Scene Crop Check
    is_uncropped = (w > 100 or h > 100)
    if is_uncropped:
        warnings.append(
            f"Dimensions ({w}x{h}) exceed benchmark crop dimensions (24x40). "
            "Capture appears to be a multi-module scene or uncropped frame."
        )

    return ThermalInputValidation(
        is_valid=True,
        original_dimensions=(w, h),
        aspect_ratio=aspect_ratio,
        benchmark_aspect_ratio=benchmark_ar,
        aspect_ratio_deviation=ar_dev,
        is_uncropped_scene=is_uncropped,
        is_native_grayscale=is_native_gray,
        color_mode=image.mode,
        modality_description=modality_desc,
        warnings=warnings,
    )


def extract_features(image: Image.Image) -> np.ndarray:
    """Extract domain-aligned feature vector from a PV module thermal crop.

    Contract:
    - If image is already native grayscale, it is preserved directly.
    - If image is RGB/RGBA, deterministic grayscale conversion is applied without
      guessing arbitrary palette mappings.
    - Resized to standard TARGET_IMAGE_SIZE (24, 40) via bilinear interpolation without auto-rotation.
    - Computes 2,950 deterministic features: 960 raw pixels, 920 dx, 936 dy,
      40 row means, 40 row stds, 24 col means, 24 col stds, and 6 global moments.
    """
    # 1. Normalize image geometry and mode without auto-rotation
    if image.size != TARGET_IMAGE_SIZE:
        image = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_gray = image.convert("L")
    arr = np.array(img_gray, dtype=np.float32) / 255.0  # shape: (40, 24)

    # 2. Raw pixel intensities (40 x 24 = 960 features)
    raw_pixels = arr.flatten()

    # 3. Horizontal & vertical thermal gradients
    # dx captures row-wise hot spots / diode band transitions
    dx = np.abs(arr[:, 1:] - arr[:, :-1]).flatten()  # 40 x 23 = 920 features
    # dy captures vertical temperature gradients / shading edges
    dy = np.abs(arr[1:, :] - arr[:-1, :]).flatten()  # 39 x 24 = 936 features

    # 4. Summary profiles: row means/stds and column means/stds
    row_means = arr.mean(axis=1)  # 40 features
    row_stds = arr.std(axis=1)    # 40 features
    col_means = arr.mean(axis=0)  # 24 features
    col_stds = arr.std(axis=0)    # 24 features

    # 5. Global contrast and distribution moments
    global_stats = np.array([
        arr.mean(),
        arr.std(),
        arr.min(),
        arr.max(),
        np.percentile(arr, 95) - np.percentile(arr, 5),  # robust delta-T proxy
        np.median(arr),
    ], dtype=np.float32)

    # Concatenate all features
    features = np.concatenate([
        raw_pixels,
        dx,
        dy,
        row_means,
        row_stds,
        col_means,
        col_stds,
        global_stats,
    ])
    return features.astype(np.float32)


@dataclass
class ModelMetadata:
    """Metadata sidecar describing model version, parameters, and training provenance."""

    model_version: str
    created_at_utc: str
    algorithm: str
    feature_dim: int
    num_classes: int
    class_names: List[str]
    training_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    split_manifest_source: str
    dataset_name: str
    hyperparameters: Dict[str, Any]


class SolarPVClassifier:
    """Production classifier for solar PV thermal anomaly classification."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        learning_rate: float = 0.08,
        random_state: int = 42,
    ):
        self.class_names = class_names or sorted(UPSTREAM_CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_names)}
        self.hyperparameters = {
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "class_weight": "balanced",
        }
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            random_state=random_state,
            class_weight="balanced",
        )
        self.metadata: Optional[ModelMetadata] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train_labels: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val_labels: Optional[List[str]] = None,
    ) -> "SolarPVClassifier":
        """Train the classifier on extracted feature matrices."""
        y_train_idx = np.array([self.class_to_idx[lbl] for lbl in y_train_labels], dtype=np.int32)
        self.model.fit(X_train, y_train_idx)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict class labels for a feature matrix."""
        preds = self.model.predict(X)
        return [self.idx_to_class[int(idx)] for idx in preds]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probability distributions."""
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_labels: List[str]) -> Dict[str, Any]:
        """Compute rigorous classification metrics on evaluation data."""
        y_true_idx = [self.class_to_idx[lbl] for lbl in y_labels]
        preds_idx = self.model.predict(X)
        preds_labels = [self.idx_to_class[int(idx)] for idx in preds_idx]

        acc = float(accuracy_score(y_true_idx, preds_idx))
        macro_f1 = float(f1_score(y_true_idx, preds_idx, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true_idx, preds_idx, average="weighted", zero_division=0))

        report_dict = classification_report(
            y_labels,
            preds_labels,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true_idx, preds_idx, labels=list(range(len(self.class_names))))

        return {
            "overall_accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "sample_count": len(y_labels),
            "per_class_metrics": {
                c: {
                    "precision": round(report_dict.get(c, {}).get("precision", 0.0), 4),
                    "recall": round(report_dict.get(c, {}).get("recall", 0.0), 4),
                    "f1_score": round(report_dict.get(c, {}).get("f1-score", 0.0), 4),
                    "support": report_dict.get(c, {}).get("support", 0),
                }
                for c in self.class_names
            },
            "confusion_matrix": cm.tolist(),
            "class_names": self.class_names,
        }

    def save(self, output_dir: Path, metadata: ModelMetadata) -> None:
        """Save model weights and metadata sidecar."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.joblib"
        meta_path = output_dir / "metadata.json"

        joblib.dump(self.model, model_path, compress=3)
        self.metadata = metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    @classmethod
    def load(cls, model_dir: Path) -> "SolarPVClassifier":
        """Load trained model from artifact directory."""
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        metadata = ModelMetadata(**meta_dict)
        classifier = cls(
            class_names=metadata.class_names,
            max_iter=metadata.hyperparameters.get("max_iter", 150),
            max_leaf_nodes=metadata.hyperparameters.get("max_leaf_nodes", 31),
            learning_rate=metadata.hyperparameters.get("learning_rate", 0.08),
            random_state=metadata.hyperparameters.get("random_state", 42),
        )
        classifier.model = joblib.load(model_path)
        classifier.metadata = metadata
        return classifier
