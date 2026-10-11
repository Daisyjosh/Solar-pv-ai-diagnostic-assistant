"""Unit tests for Phase 9 PDF Report Generator and Batch Inspection Service."""

import io
from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models.plant import Plant
from src.database.session import Base
from src.reports.pdf_generator import InspectionReportPDFGenerator
from src.services.batch_inspection import BatchInspectionService


def test_pdf_report_generation_returns_valid_pdf_stream():
    """Verify that ReportLab compiles a valid PDF stream from inspection data."""
    generator = InspectionReportPDFGenerator()

    insp_data = {
        "code": "INSP-TEST-001",
        "status": "COMPLETED",
        "inspector_name": "Dr. Solar Engineer",
        "inspection_date": "2026-10-10 12:00 UTC",
        "ambient_temp_c": 28.0,
        "irradiance_w_m2": 780.0,
        "asset_info": {
            "plant_name": "Solaria Array",
            "capacity_dc_mw": 50.0,
            "inverter_code": "INV-01",
            "string_code": "STR-01",
            "module_code": "MOD-01",
        },
    }

    diag_data = {
        "fault_category": "Activated Bypass Diode",
        "canonical_fault_code": "FLT-SUB-001",
        "severity": "MAJOR",
        "severity_justification": "Bypass diode conduction detected on central substring.",
        "model_prediction": "Diode",
        "model_confidence": 0.89,
        "power_loss_percent": 33.3,
        "review_status": "CONFIRMED",
        "reviewed_by": "Senior O&M Lead",
        "root_causes": ["Bypass diode forward conduction due to local shading"],
        "corrective_actions": ["Verify diode forward drop with multimeter"],
        "required_measurements": ["Module open circuit voltage Voc"],
    }

    pdf_bytes = generator.generate(insp_data, diag_data, images_data=[], telemetry_data=[])
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_batch_inspection_service_multi_image_processing():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    plant = Plant(code="P-BATCH", name="Batch Site", location="Nevada", capacity_dc_mw=25.0)
    session.add(plant)
    session.commit()

    service = BatchInspectionService(session)
    batch = service.create_batch(plant.id, "Test Drone Flight", expected_count=2)
    assert batch.status == "PENDING"

    # Create 2 dummy images in memory
    buf1 = io.BytesIO()
    Image.new("L", (24, 40), color=50).save(buf1, format="JPEG")
    buf2 = io.BytesIO()
    Image.new("L", (24, 40), color=150).save(buf2, format="JPEG")

    images_payload = [
        {"filename": "crop_1.jpg", "bytes": buf1.getvalue()},
        {"filename": "crop_2.jpg", "bytes": buf2.getvalue()},
    ]

    summary = service.process_image_crops(batch.id, images_payload)
    assert summary["processed_images"] == 2
    assert summary["status"] == "COMPLETED"
    assert len(summary["results"]) == 2
    assert "fault_distribution" in summary


def test_format_power_loss():
    """Verify format_power_loss safeguards: never 'None%' and preserves genuine 0.0%."""
    from src.core.formatting import format_power_loss

    assert format_power_loss(None) == "Not quantified"
    assert format_power_loss("") == "Not quantified"
    assert format_power_loss(float("nan")) == "Not quantified"
    assert format_power_loss("invalid") == "Not quantified"
    assert format_power_loss(None, fallback="N/A") == "N/A"
    assert format_power_loss(0) == "0.0%"
    assert format_power_loss(0.0) == "0.0%"
    assert format_power_loss("0.0") == "0.0%"
    assert format_power_loss(33.33) == "33.3%"
    assert format_power_loss("12.5") == "12.5%"


def test_synthetic_provenance_helpers():
    """Verify detection of synthetic assets and telemetry."""
    from src.core.formatting import is_synthetic_asset, is_synthetic_telemetry

    # Synthetic image
    assert is_synthetic_asset({"notes": "Synthetic thermal test image", "original_filename": "crop.png"}) is True
    assert is_synthetic_asset({"id": 7, "original_filename": "some_crop.png"}) is True
    assert is_synthetic_asset({"notes": "Normal field image", "original_filename": "drone_ir.jpg"}) is False

    # Synthetic telemetry
    assert is_synthetic_telemetry({"operating_status": "SYNTHETIC_NORMAL", "data_source": "SIMULATED_TEST_DATA"}) is True
    assert is_synthetic_telemetry({"operating_status": "NORMAL", "data_source": "SCADA_MODBUS"}) is False


def test_pdf_report_with_synthetic_telemetry_and_blank_review():
    """Verify PDF generates with synthetic warnings, standards, and blank sign-off when pending."""
    import base64
    import zlib
    from src.reports.pdf_generator import InspectionReportPDFGenerator

    gen = InspectionReportPDFGenerator()
    insp_data = {
        "code": "INSP-REG-01",
        "status": "IN_PROGRESS",
        "inspector_name": "Test Inspector",
        "inspection_date": "2026-10-11 10:00 UTC",
        "ambient_temp_c": 25.0,
        "irradiance_w_m2": 700.0,
        "asset_info": {"plant_name": "Test Plant", "capacity_dc_mw": 10.0},
    }
    diag_data = {
        "fault_category": "Suspected Hotspot",
        "canonical_fault_code": "FLT-CELL-001",
        "severity": "INCONCLUSIVE",
        "severity_justification": "Calibrated temperature gradient required under IEC 62446-3.",
        "model_prediction": "Cracking",
        "model_confidence": 0.311,
        "power_loss_percent": None,
        "review_status": "PENDING_REVIEW",
        "reviewed_by": None,
        "root_causes": ["Microcrack or localized resistance"],
        "corrective_actions": ["Conduct dark-field EL test per IEC 61215 MQT 10.9"],
        "required_measurements": ["Electroluminescence imaging", "Calibrated delta-T"],
    }
    images_data = [
        {
            "id": 7,
            "image_type": "THERMAL_IR",
            "original_filename": "Infrared Solar Panel with Hotspot.png",
            "notes": "Synthetic thermal test image",
            "file_path": "",
        }
    ]
    telemetry_data = [
        {
            "timestamp": "2026-10-11 10:00:00",
            "voltage_v": 680.0,
            "current_a": 8.5,
            "power_kw": 5.78,
            "operating_status": "SYNTHETIC_NORMAL",
            "data_source": "SIMULATED_TEST_DATA",
        }
    ]

    pdf_bytes = gen.generate(insp_data, diag_data, images_data, telemetry_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
