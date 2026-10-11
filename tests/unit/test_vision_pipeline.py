"""Unit tests for Phase 5.2 AI Vision Classifier and Inference Service."""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from src.vision.classifier import (
    SolarPVClassifier,
    UPSTREAM_CLASSES,
    extract_features,
)
from src.vision.inference import (
    InferenceStatus,
    VisionInferenceService,
    get_inference_service,
)


def test_extract_features_shape_and_type():
    """Verify feature extraction produces deterministic vectors of expected dimension."""
    im = Image.new("L", (24, 40), color=128)
    feats = extract_features(im)

    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32
    assert len(feats) == 2950  # 960 raw + 920 dx + 936 dy + 40 row_m + 40 row_std + 24 col_m + 24 col_std + 6 global


def test_classifier_fit_predict_and_metrics():
    """Test model training and evaluation on synthetic feature vectors."""
    num_samples = 48
    feature_dim = 2950
    np.random.seed(42)

    X = np.random.rand(num_samples, feature_dim).astype(np.float32)
    # Assign labels cycling through the 12 upstream classes
    y = [UPSTREAM_CLASSES[i % len(UPSTREAM_CLASSES)] for i in range(num_samples)]

    classifier = SolarPVClassifier(max_iter=10)
    classifier.fit(X, y)

    preds = classifier.predict(X[:5])
    assert len(preds) == 5
    assert all(p in UPSTREAM_CLASSES for p in preds)

    probs = classifier.predict_proba(X[:5])
    assert probs.shape == (5, 12)
    assert np.allclose(probs.sum(axis=1), 1.0)

    eval_results = classifier.evaluate(X, y)
    assert "overall_accuracy" in eval_results
    assert "macro_f1" in eval_results
    assert "per_class_metrics" in eval_results
    assert len(eval_results["confusion_matrix"]) == 12


def test_inference_service_status_and_prediction():
    """Test global VisionInferenceService against active model."""
    service = get_inference_service()
    status = service.get_status()

    # Active model was trained during Phase 5.2
    assert status["status"] == InferenceStatus.MODEL_READY
    assert status["model_version"] is not None
    assert len(status["class_names"]) == 12

    # Predict test image
    test_img = Image.new("L", (24, 40), color=50)
    res = service.predict_image(test_img)
    assert res.status == InferenceStatus.MODEL_READY
    assert res.predicted_class in UPSTREAM_CLASSES
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 12


def test_inference_service_uncertainty_detection():
    """Verify uncertainty detection flags low confidence or narrow top margin."""
    service = get_inference_service()
    # A flat color image or complex RGB image often produces distributed probabilities
    test_img = Image.new("RGB", (200, 200), color=(255, 100, 20))
    res = service.predict_image(test_img)

    assert res.status == InferenceStatus.MODEL_READY
    assert hasattr(res, "is_low_confidence")
    assert hasattr(res, "top_candidates")
    assert len(res.top_candidates) == 3
    assert res.top_candidates[0]["probability"] >= res.top_candidates[1]["probability"]
    if res.is_low_confidence:
        assert ("Low-confidence hypothesis" in res.message) or ("Ambiguous hypothesis" in res.message)
        assert "requires physical electrical / EL corroboration" in res.message


def test_validate_thermal_input_native_and_deviations():
    """Verify input validation correctly screens native grayscale vs RGB and flags geometric deviations."""
    from src.vision.classifier import validate_thermal_input

    # 1. Benchmark standard: 24x40 native grayscale
    bench_img = Image.new("L", (24, 40), color=100)
    val_bench = validate_thermal_input(bench_img)
    assert val_bench.is_valid is True
    assert val_bench.is_native_grayscale is True
    assert val_bench.aspect_ratio == 0.60
    assert val_bench.aspect_ratio_deviation is False
    assert val_bench.is_uncropped_scene is False
    assert len(val_bench.warnings) == 0

    # 2. Arbitrary landscape false-color capture (e.g. 583x421)
    scene_img = Image.new("RGB", (583, 421), color=(200, 50, 30))
    val_scene = validate_thermal_input(scene_img)
    assert val_scene.is_valid is True
    assert val_scene.is_native_grayscale is False
    assert val_scene.aspect_ratio > 1.0  # Landscape
    assert val_scene.aspect_ratio_deviation is True
    assert val_scene.is_uncropped_scene is True
    assert len(val_scene.warnings) >= 3
    # Verify dimensions are not silently swapped
    assert val_scene.original_dimensions == (583, 421)


def test_calibrator_loading_and_rollback_safety(tmp_path):
    """Verify that probability calibrator loads cleanly and falls back seamlessly if absent."""
    service = get_inference_service()
    service.reload()
    status = service.get_status()

    # Active model directory contains the fitted calibrator
    assert status["is_calibrated"] is True

    # Test rollback fallback when calibrator is absent
    backup_root = Path("models/classifier/backup_v1_baseline")
    assert backup_root.exists(), "Rollback baseline backup must exist"

    # Initialize a service pointing to backup_v1_baseline which does not have calibrator.joblib
    rollback_service = VisionInferenceService(models_root=backup_root.parent)
    # Even if calibrator is missing, inference must succeed
    test_img = Image.new("L", (24, 40), color=80)
    res = rollback_service.predict_image(test_img)
    assert res.status == InferenceStatus.MODEL_READY
    assert res.predicted_class in UPSTREAM_CLASSES
    assert 0.0 <= res.confidence <= 1.0


def test_dual_tier_uncertainty_and_taxonomy_notes():
    """Verify dual-tier uncertainty classification and authoritative benchmark taxonomy notes."""
    service = get_inference_service()
    service.reload()

    # Highly ambiguous flat RGB input
    test_img = Image.new("RGB", (300, 200), color=(128, 128, 128))
    res = service.predict_image(test_img)

    assert res.status == InferenceStatus.MODEL_READY
    assert res.uncertainty_tier in ("AMBIGUOUS_HYPOTHESIS", "LOW_CONFIDENCE_HYPOTHESIS", "ACCEPTED_HYPOTHESIS")
    assert res.input_validation is not None
    assert "warnings" in res.input_validation

    if res.predicted_class in ("Cell", "Cell-Multi"):
        assert res.taxonomy_note is not None
        assert "crystalline silicon" in res.taxonomy_note
    elif res.predicted_class in ("Hot-Spot", "Hot-Spot-Multi"):
        assert res.taxonomy_note is not None
        assert "thin-film" in res.taxonomy_note
    elif res.predicted_class == "Cracking":
        assert res.taxonomy_note is not None
        assert "Electroluminescence" in res.taxonomy_note


