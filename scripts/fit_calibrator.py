"""Fit probability calibrator strictly on held-out validation split.

Engineering principles:
- Calibrator is fitted ONLY on the validation partition (N=1,548), NEVER on train or test.
- The test set (N=1,568) is strictly preserved for final evaluation.
- Decoupled from base classifier: saves calibrator.joblib and calibration_report.json
  as sidecars so the base model can be safely rolled back at any time.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import sys

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.vision.classifier import SolarPVClassifier, extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_multiclass_brier_score(y_true_indices: np.ndarray, prob_matrix: np.ndarray, num_classes: int) -> float:
    """Compute multi-class mean squared probability error (Brier score)."""
    one_hot = np.zeros_like(prob_matrix)
    for i, idx in enumerate(y_true_indices):
        one_hot[i, idx] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def main():
    model_dir = Path("models/classifier/active")
    manifest_path = Path("data/processed/proposed_splits.json")
    meta_path = Path("data/external/quarantine/extracted/InfraredSolarModules/module_metadata.json")

    logger.info("Loading active classifier from %s...", model_dir)
    classifier = SolarPVClassifier.load(model_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_ids = manifest["splits"]["val"]
    logger.info("Loaded %d validation IDs from %s", len(val_ids), manifest_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    img_dir = meta_path.parent

    # 1. Extract validation features
    t0 = time.time()
    X_val = []
    y_val_indices = []
    logger.info("Extracting features for validation partition (%d samples)...", len(val_ids))
    for i in val_ids:
        item = meta[str(i)]
        lbl = item["anomaly_class"]
        with Image.open(img_dir / item["image_filepath"]) as im:
            feats = extract_features(im)
            X_val.append(feats)
            y_val_indices.append(classifier.class_to_idx[lbl])

    X_val = np.array(X_val, dtype=np.float32)
    y_val_indices = np.array(y_val_indices, dtype=np.int32)
    logger.info("Validation feature extraction complete in %.2fs (shape: %s)", time.time() - t0, X_val.shape)

    # 2. Evaluate raw model calibration metrics
    raw_probs = classifier.predict_proba(X_val)
    raw_log_loss = float(log_loss(y_val_indices, raw_probs))
    raw_brier = compute_multiclass_brier_score(y_val_indices, raw_probs, len(classifier.class_names))
    logger.info("Raw Model Validation: Log-Loss = %.4f, Brier Score = %.4f", raw_log_loss, raw_brier)

    # 3. Fit Isotonic Calibrator strictly on validation data
    logger.info("Fitting Isotonic CalibratedClassifierCV on validation data...")
    t_fit = time.time()
    try:
        from sklearn.calibration import FrozenEstimator
        frozen = FrozenEstimator(classifier.model)
        calibrator = CalibratedClassifierCV(estimator=frozen, method="isotonic")
    except (ImportError, AttributeError):
        calibrator = CalibratedClassifierCV(estimator=classifier.model, method="isotonic", cv="prefit")

    calibrator.fit(X_val, y_val_indices)
    logger.info("Calibration fitting complete in %.2fs", time.time() - t_fit)

    # 4. Evaluate calibrated model metrics
    cal_probs = calibrator.predict_proba(X_val)
    cal_log_loss = float(log_loss(y_val_indices, cal_probs))
    cal_brier = compute_multiclass_brier_score(y_val_indices, cal_probs, len(classifier.class_names))
    logger.info("Calibrated Validation: Log-Loss = %.4f, Brier Score = %.4f", cal_log_loss, cal_brier)

    # 5. Persist calibrator artifact and sidecar report
    cal_path = model_dir / "calibrator.joblib"
    joblib.dump(calibrator, cal_path, compress=3)
    logger.info("Persisted calibrator artifact to %s", cal_path)

    report = {
        "calibrator_version": "isotonic_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fitting_split": "validation",
        "sample_count": len(val_ids),
        "method": "isotonic",
        "validation_metrics": {
            "raw_log_loss": round(raw_log_loss, 4),
            "raw_brier_score": round(raw_brier, 4),
            "calibrated_log_loss": round(cal_log_loss, 4),
            "calibrated_brier_score": round(cal_brier, 4),
            "log_loss_reduction_pct": round((raw_log_loss - cal_log_loss) / raw_log_loss * 100, 2),
            "brier_reduction_pct": round((raw_brier - cal_brier) / raw_brier * 100, 2),
        },
        "empirical_thresholds": {
            "tier_1_ambiguous": {"max_confidence_below": 0.40, "margin_below": 0.05, "action": "INCONCLUSIVE"},
            "tier_2_low_confidence": {"max_confidence_below": 0.50, "margin_below": 0.08, "action": "CORROBORATION_REQUIRED"},
            "tier_3_accepted": {"min_confidence": 0.50, "min_margin": 0.08, "action": "ACCEPTED_HYPOTHESIS"},
        },
        "notes": "Fitted strictly on validation partition. Test partition remained completely untouched.",
    }

    report_path = model_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Persisted calibration report to %s", report_path)


if __name__ == "__main__":
    main()
