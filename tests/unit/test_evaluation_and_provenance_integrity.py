"""Focused regression tests verifying model evaluation integrity, low-confidence uncertainty handling,
inspection lifecycle state machine enforcement, and synthetic data provenance.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.core.formatting import format_power_loss, is_synthetic_asset, is_synthetic_telemetry
from src.database.models.inspection import ALLOWED_INSPECTION_TRANSITIONS, InspectionStatus
from src.diagnosis.engine import DiagnosticEngine, SeverityLevel, ReviewStatus


def test_evaluation_report_integrity_and_axis_ordering():
    """Verify evaluation report metrics match test partition and confusion matrix axes are aligned."""
    eval_path = Path("models/classifier/active/evaluation_report.json")
    meta_path = Path("models/classifier/active/metadata.json")

    assert eval_path.exists(), "evaluation_report.json must exist in active model directory"
    assert meta_path.exists(), "metadata.json must exist in active model directory"

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 1. Version consistency
    assert eval_data["model_version"] == meta_data["model_version"]
    assert eval_data["split_manifest"] == meta_data["split_manifest_source"]

    # 2. Test metrics integrity
    test_metrics = eval_data["test_metrics"]
    assert test_metrics["sample_count"] == meta_data["test_sample_count"]
    assert test_metrics["sample_count"] == 1568
    assert test_metrics["overall_accuracy"] == 0.632
    assert test_metrics["macro_f1"] == 0.5029
    assert test_metrics["weighted_f1"] == 0.6341

    # 3. Class ordering and confusion matrix dimension check
    class_names = test_metrics["class_names"]
    assert len(class_names) == 12
    assert class_names == meta_data["class_names"]

    cm = np.array(test_metrics["confusion_matrix"])
    assert cm.shape == (12, 12), "Confusion matrix must be 12x12"

    # 4. Verify confusion matrix row sums match per-class test support
    per_class = test_metrics["per_class_metrics"]
    for i, cname in enumerate(class_names):
        row_sum = int(np.sum(cm[i, :]))
        support = int(per_class[cname]["support"])
        assert row_sum == support, f"Confusion matrix row {i} sum ({row_sum}) must equal support ({support}) for {cname}"

    # 5. Confirm weak classes are faithfully documented
    assert per_class["Soiling"]["f1_score"] < 0.25
    assert per_class["Hot-Spot"]["f1_score"] < 0.35
    assert per_class["Cell-Multi"]["f1_score"] < 0.40


def test_split_isolation_and_zero_partition_leakage():
    """Verify that train, validation, and test sets are strictly disjoint."""
    manifest_path = Path("data/processed/proposed_splits.json")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # Disjointness
    assert len(train_ids & val_ids) == 0, "Train and Val must have 0 overlap"
    assert len(train_ids & test_ids) == 0, "Train and Test must have 0 overlap"
    assert len(val_ids & test_ids) == 0, "Val and Test must have 0 overlap"

    # Counts
    assert len(train_ids) == 16884
    assert len(val_ids) == 1548
    assert len(test_ids) == 1568
    assert len(train_ids) + len(val_ids) + len(test_ids) == 20000


def test_low_confidence_and_closely_contested_ambiguity():
    """Verify that DiagnosticEngine flags closely contested or low-confidence predictions as INCONCLUSIVE."""
    engine = DiagnosticEngine()

    # Closely contested between Cracking (31.1%) and Cell-Multi (30.5%)
    probabilities = {
        "Cracking": 0.311,
        "Cell-Multi": 0.305,
        "Cell": 0.120,
        "Hot-Spot": 0.080,
        "Diode": 0.040,
        "No-Anomaly": 0.030,
        "Shadowing": 0.030,
        "Offline-Module": 0.020,
        "Vegetation": 0.020,
        "Soiling": 0.020,
        "Hot-Spot-Multi": 0.012,
        "Diode-Multi": 0.012,
    }

    assessment = engine.diagnose(
        inspection_id=10,
        asset_info={"plant_name": "Daisy Solar Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="Cracking",
        model_confidence=0.311,
        model_probabilities=probabilities,
        irradiance_w_m2=750.0,
        ambient_temp_c=25.0,
        delta_temp_c=None,  # No calibrated delta-T
    )

    # Must be INCONCLUSIVE, unquantified loss, PENDING_REVIEW
    assert assessment.severity == SeverityLevel.INCONCLUSIVE
    assert assessment.estimated_power_loss_percent is None
    assert assessment.review_status == ReviewStatus.PENDING_REVIEW
    assert "Uncorroborated AI Hypothesis" in assessment.fault_category_name
    assert "Cracking" in assessment.severity_justification
    assert "Cell-Multi" in assessment.severity_justification
    assert any("Electroluminescence" in m or "EL" in m for m in assessment.required_additional_measurements)


def test_power_loss_formatting_guardrails():
    """Verify that power loss values never render as None% or NaN% and preserve genuine zeros."""
    # Unquantified / missing
    assert format_power_loss(None) == "Not quantified"
    assert format_power_loss("") == "Not quantified"
    assert format_power_loss(float("nan")) == "Not quantified"
    assert format_power_loss(None, fallback="Unquantified (Needs I-V trace)") == "Unquantified (Needs I-V trace)"
    assert "%" not in format_power_loss(None)

    # Genuine zeros preserved
    assert format_power_loss(0.0) == "0.0%"
    assert format_power_loss(0) == "0.0%"
    assert format_power_loss("0.0") == "0.0%"

    # Real calculated values
    assert format_power_loss(33.3) == "33.3%"
    assert format_power_loss(100.0) == "100.0%"


def test_synthetic_provenance_safeguards():
    """Verify that synthetic test imagery and simulated telemetry are reliably detected."""
    # Synthetic image markers
    assert is_synthetic_asset({"notes": "Synthetic thermal test image", "original_filename": "hotspot.png"}) is True
    assert is_synthetic_asset({"id": 7, "notes": "Any note"}) is True
    assert is_synthetic_asset({"original_filename": "Infrared Solar Panel with Hotspot.png"}) is True
    assert is_synthetic_asset({"original_filename": "drone_ir_flight_4.jpg", "notes": "Real field inspection"}) is False

    # Synthetic telemetry markers
    assert is_synthetic_telemetry({"operating_status": "SYNTHETIC_NORMAL"}) is True
    assert is_synthetic_telemetry({"data_source": "SIMULATED_TEST_DATA"}) is True
    assert is_synthetic_telemetry({"operating_status": "NORMAL", "data_source": "SCADA_MODBUS"}) is False


def test_inspection_lifecycle_state_machine_rules():
    """Verify ALLOWED_INSPECTION_TRANSITIONS rules."""
    sched = InspectionStatus.SCHEDULED.value
    in_prog = InspectionStatus.IN_PROGRESS.value
    comp = InspectionStatus.COMPLETED.value
    arch = InspectionStatus.ARCHIVED.value

    # From SCHEDULED
    assert in_prog in ALLOWED_INSPECTION_TRANSITIONS[sched]
    assert arch in ALLOWED_INSPECTION_TRANSITIONS[sched]
    assert comp not in ALLOWED_INSPECTION_TRANSITIONS[sched], "SCHEDULED cannot directly transition to COMPLETED"

    # From IN_PROGRESS
    assert comp in ALLOWED_INSPECTION_TRANSITIONS[in_prog]
    assert arch in ALLOWED_INSPECTION_TRANSITIONS[in_prog]

    # From COMPLETED
    assert in_prog in ALLOWED_INSPECTION_TRANSITIONS[comp], "COMPLETED can be reopened to IN_PROGRESS"
    assert arch in ALLOWED_INSPECTION_TRANSITIONS[comp]
    assert sched not in ALLOWED_INSPECTION_TRANSITIONS[comp], "COMPLETED cannot revert to SCHEDULED"


def test_calibration_report_and_untouched_test_metrics():
    """Verify probability calibrator was fitted on validation partition and test metrics remain untouched."""
    cal_rep_path = Path("models/classifier/active/calibration_report.json")
    eval_rep_path = Path("models/classifier/active/evaluation_report.json")

    assert cal_rep_path.exists(), "calibration_report.json must exist in active model directory"
    assert eval_rep_path.exists(), "evaluation_report.json must exist in active model directory"

    with open(cal_rep_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    with open(eval_rep_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 1. Calibration fitting provenance
    assert cal_data["fitting_split"] == "validation"
    assert cal_data["sample_count"] == 1548
    assert cal_data["method"] == "isotonic"
    val_metrics = cal_data["validation_metrics"]
    assert val_metrics["calibrated_log_loss"] < val_metrics["raw_log_loss"]
    assert val_metrics["log_loss_reduction_pct"] > 10.0  # At least 10% log-loss reduction
    assert val_metrics["calibrated_brier_score"] < val_metrics["raw_brier_score"]

    # 2. Untouched test partition metrics
    test_metrics = eval_data["test_metrics"]
    assert test_metrics["sample_count"] == 1568
    assert test_metrics["overall_accuracy"] == 0.632
    assert test_metrics["macro_f1"] == 0.5029
    assert test_metrics["weighted_f1"] == 0.6341

