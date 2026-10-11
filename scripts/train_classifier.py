"""CLI runner to train, evaluate, and persist the Solar PV Anomaly Classifier.

Reads the quarantined InfraredSolarModules dataset and the approved cluster-aware
split manifest, trains HistGradientBoostingClassifier, evaluates on validation/test sets,
and saves the model along with evaluation artifacts to models/classifier/active.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.vision.classifier import (
    ModelMetadata,
    SolarPVClassifier,
    UPSTREAM_CLASSES,
    extract_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_split(
    meta_path: Path,
    manifest_path: Path,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]], Path]:
    """Load metadata mapping and split allocations."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Dict[str, str]] = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits: Dict[str, List[str]] = manifest["splits"]
    img_dir = meta_path.parent
    return splits, meta, img_dir


def extract_features_for_ids(
    ids: List[str],
    meta: Dict[str, Dict[str, str]],
    img_dir: Path,
    desc: str = "split",
) -> Tuple[np.ndarray, List[str]]:
    """Extract features and target labels for a list of image IDs."""
    logger.info("Extracting features for %s (%d images)...", desc, len(ids))
    t0 = time.time()
    feature_list = []
    labels = []

    for img_id in ids:
        item = meta[str(img_id)]
        img_path = img_dir / item["image_filepath"]
        label = item["anomaly_class"]

        with Image.open(img_path) as img:
            feats = extract_features(img)
            feature_list.append(feats)
            labels.append(label)

    X = np.array(feature_list, dtype=np.float32)
    elapsed = time.time() - t0
    logger.info("Finished %s extraction in %.2fs (shape: %s).", desc, elapsed, X.shape)
    return X, labels


def main():
    parser = argparse.ArgumentParser(description="Train Solar PV Anomaly Classifier")
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/external/quarantine/extracted/InfraredSolarModules/module_metadata.json"),
        help="Path to module_metadata.json",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/processed/proposed_splits.json"),
        help="Path to proposed_splits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/classifier/active"),
        help="Output directory for trained model artifacts",
    )
    parser.add_argument("--max-iter", type=int, default=150, help="Max iterations for boosting")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Learning rate")
    args = parser.parse_args()

    if not args.meta_path.exists():
        logger.error("Metadata path does not exist: %s", args.meta_path)
        return 1
    if not args.manifest_path.exists():
        logger.error("Manifest path does not exist: %s", args.manifest_path)
        return 1

    logger.info("Loading dataset splits...")
    splits, meta, img_dir = load_dataset_split(args.meta_path, args.manifest_path)

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    logger.info("Splits: Train=%d, Val=%d, Test=%d", len(train_ids), len(val_ids), len(test_ids))

    # 1. Feature Extraction
    X_train, y_train = extract_features_for_ids(train_ids, meta, img_dir, "Train")
    X_val, y_val = extract_features_for_ids(val_ids, meta, img_dir, "Validation")
    X_test, y_test = extract_features_for_ids(test_ids, meta, img_dir, "Test")

    # 2. Model Initialization and Training
    logger.info("Initializing SolarPVClassifier (HistGradientBoosting, 12 classes)...")
    classifier = SolarPVClassifier(
        class_names=sorted(UPSTREAM_CLASSES),
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        learning_rate=args.learning_rate,
        random_state=42,
    )

    t_train_start = time.time()
    logger.info("Fitting model on %d training samples...", len(train_ids))
    classifier.fit(X_train, y_train)
    train_duration = time.time() - t_train_start
    logger.info("Training completed in %.2f seconds.", train_duration)

    # 3. Evaluation on Validation and Test Sets
    logger.info("Evaluating on Validation set (%d samples)...", len(val_ids))
    val_metrics = classifier.evaluate(X_val, y_val)
    logger.info(
        "Validation Results -> Accuracy: %.2f%%, Macro-F1: %.4f, Weighted-F1: %.4f",
        val_metrics["overall_accuracy"] * 100,
        val_metrics["macro_f1"],
        val_metrics["weighted_f1"],
    )

    logger.info("Evaluating on Test set (%d samples)...", len(test_ids))
    test_metrics = classifier.evaluate(X_test, y_test)
    logger.info(
        "Test Results -> Accuracy: %.2f%%, Macro-F1: %.4f, Weighted-F1: %.4f",
        test_metrics["overall_accuracy"] * 100,
        test_metrics["macro_f1"],
        test_metrics["weighted_f1"],
    )

    # 4. Save Artifacts
    created_at = datetime.now(timezone.utc).isoformat()
    version_str = f"v1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    metadata = ModelMetadata(
        model_version=version_str,
        created_at_utc=created_at,
        algorithm="HistGradientBoostingClassifier",
        feature_dim=int(X_train.shape[1]),
        num_classes=len(classifier.class_names),
        class_names=classifier.class_names,
        training_sample_count=len(train_ids),
        validation_sample_count=len(val_ids),
        test_sample_count=len(test_ids),
        split_manifest_source=str(args.manifest_path),
        dataset_name="InfraredSolarModules",
        hyperparameters={
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "train_duration_seconds": round(train_duration, 2),
        },
    )

    logger.info("Saving model and metadata to %s...", args.output_dir)
    classifier.save(args.output_dir, metadata)

    eval_report = {
        "model_version": version_str,
        "evaluation_timestamp_utc": created_at,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_manifest": str(args.manifest_path),
        "dataset_name": "InfraredSolarModules",
        "training_time_seconds": round(train_duration, 2),
    }

    eval_report_path = args.output_dir / "evaluation_report.json"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    logger.info("Evaluation report saved to %s", eval_report_path)
    logger.info("Vision Pipeline Training & Evaluation Successfully Completed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
