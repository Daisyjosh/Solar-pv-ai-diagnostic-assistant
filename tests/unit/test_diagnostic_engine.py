"""Unit tests for Phase 6 Diagnostic Reasoning Engine."""

import pytest

from src.diagnosis.engine import DiagnosticEngine, ReviewStatus, SeverityLevel


def test_diagnostic_engine_normal_module():
    """Verify diagnostic assessment for No-Anomaly input."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=1,
        asset_info={"plant_name": "Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="No-Anomaly",
        model_confidence=0.88,
        irradiance_w_m2=800.0,
        ambient_temp_c=25.0,
    )

    assert assessment.severity == SeverityLevel.NORMAL
    assert assessment.estimated_power_loss_percent == 0.0
    assert assessment.review_status == ReviewStatus.PENDING_REVIEW
    assert "clear sky" in assessment.environmental_validity_note or "meets" in assessment.environmental_validity_note
    assert len(assessment.corrective_actions) >= 1


def test_diagnostic_engine_diode_fault():
    """Verify bypass diode fault triggers expected severity and ~33.3% power loss."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=2,
        asset_info={"plant_name": "Test Plant", "rated_voc_v": 960.0},
        image_modality="THERMAL_IR",
        model_prediction="Diode",
        model_confidence=0.92,
        irradiance_w_m2=750.0,
        operating_voltage_v=640.0,
    )

    assert assessment.severity == SeverityLevel.MAJOR
    assert assessment.estimated_power_loss_percent == 33.3
    assert assessment.canonical_fault_code == "FLT-DIODE-001"
    assert any("bypass diode" in h.lower() for h in assessment.probable_root_causes)
    assert any("dc arc" in sp.lower() or "de-energize" in sp.lower() for sp in assessment.safety_precautions)


def test_diagnostic_engine_offline_module():
    """Verify offline module triggers CRITICAL severity and 100% loss."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=3,
        asset_info={"plant_name": "Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="Offline-Module",
        model_confidence=0.95,
        irradiance_w_m2=850.0,
    )

    assert assessment.severity == SeverityLevel.CRITICAL
    assert assessment.estimated_power_loss_percent == 100.0
    assert any("mc4" in ca.lower() or "connector" in ca.lower() for ca in assessment.corrective_actions)


def test_diagnostic_engine_low_irradiance_warning():
    """Verify irradiance below 600 W/m2 flags IEC 62446-3 validity warning."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=4,
        asset_info={"plant_name": "Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="Cell",
        irradiance_w_m2=450.0,  # Below IEC minimum
    )

    assert "below IEC 62446-3 minimum threshold" in assessment.environmental_validity_note


def test_diagnostic_engine_uncorroborated_cracking_hypothesis():
    """Verify low-confidence or uncorroborated cracking prediction defaults to INCONCLUSIVE and unquantified loss."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=10,
        asset_info={"plant_name": "Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="Cracking",
        model_confidence=0.311,
        model_probabilities={"Cracking": 0.311, "Cell-Multi": 0.305, "Hot-Spot": 0.120},
        irradiance_w_m2=750.0,
        delta_temp_c=None,  # No measured delta T
    )

    assert assessment.severity == SeverityLevel.INCONCLUSIVE
    assert assessment.estimated_power_loss_percent is None
    assert assessment.review_status == ReviewStatus.PENDING_REVIEW
    assert "Uncorroborated AI Hypothesis" in assessment.fault_category_name
    assert "INCONCLUSIVE" in assessment.severity_justification
    assert any("Electroluminescence (EL)" in m for m in assessment.required_additional_measurements)
    assert any("I-V curve" in m for m in assessment.required_additional_measurements)
    assert any("Hypothesis A" in r for r in assessment.probable_root_causes)
    assert any("Hypothesis B" in r for r in assessment.probable_root_causes)


def test_diagnostic_engine_ambiguous_cell_multi_hypothesis():
    """Verify ambiguous Cell-Multi prediction without delta T defaults to INCONCLUSIVE and unquantified loss."""
    engine = DiagnosticEngine()
    assessment = engine.diagnose(
        inspection_id=11,
        asset_info={"plant_name": "Test Plant"},
        image_modality="THERMAL_IR",
        model_prediction="Cell-Multi",
        model_confidence=0.32,
        model_probabilities={"Cell-Multi": 0.32, "Cracking": 0.30},
        irradiance_w_m2=800.0,
        delta_temp_c=None,
    )

    assert assessment.severity == SeverityLevel.INCONCLUSIVE
    assert assessment.estimated_power_loss_percent is None
    assert "Uncorroborated AI Hypothesis" in assessment.fault_category_name

