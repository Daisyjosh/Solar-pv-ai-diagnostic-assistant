"""Unit tests for SQLAlchemy ORM models, relationships, and constraints."""

from datetime import date, datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models import (
    ImageAsset,
    ImageType,
    Inspection,
    InspectionStatus,
    Inverter,
    Plant,
    PVModule,
    PVString,
)


def test_create_plant(db_session: Session):
    """Test successful creation and retrieval of a Plant entity."""
    plant = Plant(
        code="PLANT-KA-001",
        name="Pavagada Solar Park Block A",
        location="Tumakuru, Karnataka, India",
        capacity_dc_mw=100.0,
        commissioning_date=date(2020, 1, 15),
        is_active=True,
    )
    db_session.add(plant)
    db_session.commit()

    retrieved = db_session.get(Plant, plant.id)
    assert retrieved is not None
    assert retrieved.code == "PLANT-KA-001"
    assert retrieved.capacity_dc_mw == 100.0
    assert retrieved.is_active is True
    assert retrieved.created_at is not None


def test_plant_unique_code_constraint(db_session: Session):
    """Verify unique constraint on Plant code raises IntegrityError on duplication."""
    plant1 = Plant(
        code="PLANT-DUP-001",
        name="Solar Station 1",
        location="Site 1",
        capacity_dc_mw=10.0,
    )
    db_session.add(plant1)
    db_session.commit()

    plant2 = Plant(
        code="PLANT-DUP-001",
        name="Solar Station 2",
        location="Site 2",
        capacity_dc_mw=20.0,
    )
    db_session.add(plant2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_inverter_string_module_hierarchy(db_session: Session):
    """Test asset hierarchy: Plant -> Inverter -> PVString -> PVModule."""
    plant = Plant(
        code="PLANT-HIER-01",
        name="Hierarchy Test Plant",
        location="Site A",
        capacity_dc_mw=50.0,
    )
    db_session.add(plant)
    db_session.flush()

    inverter = Inverter(
        plant_id=plant.id,
        code="INV-01",
        model="SMA Sunny Central 2500-EV",
        rated_power_kw=2500.0,
    )
    db_session.add(inverter)
    db_session.flush()

    pv_string = PVString(
        inverter_id=inverter.id,
        code="STR-01-A",
        modules_count=30,
        rated_voc_v=1450.0,
        rated_isc_a=11.2,
    )
    db_session.add(pv_string)
    db_session.flush()

    pv_module = PVModule(
        string_id=pv_string.id,
        code="MOD-01-A-01",
        position_in_string=1,
        manufacturer="Longi Solar",
        model="LR5-72HBD-540M",
        rated_power_w=540.0,
    )
    db_session.add(pv_module)
    db_session.commit()

    # Verify bidirectional navigation
    loaded_plant = db_session.get(Plant, plant.id)
    assert len(loaded_plant.inverters) == 1
    assert loaded_plant.inverters[0].code == "INV-01"
    assert len(loaded_plant.inverters[0].strings) == 1
    assert loaded_plant.inverters[0].strings[0].code == "STR-01-A"
    assert len(loaded_plant.inverters[0].strings[0].modules) == 1
    assert loaded_plant.inverters[0].strings[0].modules[0].rated_power_w == 540.0


def test_cascade_deletion(db_session: Session):
    """Verify deleting a Plant cascades to inverters, strings, and modules."""
    plant = Plant(code="PLANT-CASC", name="Cascade Plant", location="Site C", capacity_dc_mw=5.0)
    db_session.add(plant)
    db_session.flush()

    inverter = Inverter(plant_id=plant.id, code="INV-CASC", rated_power_kw=100.0)
    db_session.add(inverter)
    db_session.flush()

    pv_string = PVString(inverter_id=inverter.id, code="STR-CASC", modules_count=20, rated_voc_v=800.0, rated_isc_a=9.0)
    db_session.add(pv_string)
    db_session.flush()

    pv_module = PVModule(string_id=pv_string.id, code="MOD-CASC", position_in_string=1, rated_power_w=400.0)
    db_session.add(pv_module)
    db_session.commit()

    module_id = pv_module.id
    string_id = pv_string.id
    inverter_id = inverter.id

    # Delete parent plant
    db_session.delete(plant)
    db_session.commit()

    # Children should be cascade deleted
    assert db_session.get(Inverter, inverter_id) is None
    assert db_session.get(PVString, string_id) is None
    assert db_session.get(PVModule, module_id) is None


def test_foreign_key_enforcement(db_session: Session):
    """Verify creating a child entity with a nonexistent parent raises IntegrityError."""
    inverter = Inverter(plant_id=99999, code="INV-INVALID", rated_power_kw=500.0)
    db_session.add(inverter)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_inspection_and_image_assets(db_session: Session):
    """Test Inspection entity and ImageAsset distinctions between THERMAL_IR and OPTICAL_RGB."""
    plant = Plant(code="PLANT-INSP", name="Insp Plant", location="Site D", capacity_dc_mw=10.0)
    db_session.add(plant)
    db_session.flush()

    inspection = Inspection(
        code="INSP-2026-TEST-01",
        plant_id=plant.id,
        inspector_name="Dr. Alex Rivera",
        inspection_date=datetime.now(timezone.utc),
        status=InspectionStatus.IN_PROGRESS.value,
        ambient_temp_c=34.2,
        irradiance_w_m2=820.0,
        weather_notes="Clear sky, sunny, 2.1 m/s wind speed",
    )
    db_session.add(inspection)
    db_session.flush()

    # 1. Thermal IR image asset with radiometric calibration
    thermal_img = ImageAsset(
        inspection_id=inspection.id,
        image_type=ImageType.THERMAL_IR.value,
        file_path="data/raw/inspections/1/flir_ir_001.jpg",
        original_filename="flir_ir_001.jpg",
        file_size_bytes=2450000,
        mime_type="image/jpeg",
        checksum_sha256="a" * 64,
        camera_make="FLIR",
        camera_model="Vue Pro R 640",
        emissivity=0.90,
        reflected_temp_c=25.0,
        notes="Radiometric capture showing sub-string hotspot",
    )
    db_session.add(thermal_img)

    # 2. Optical RGB image asset
    rgb_img = ImageAsset(
        inspection_id=inspection.id,
        image_type=ImageType.OPTICAL_RGB.value,
        file_path="data/raw/inspections/1/sony_rgb_001.jpg",
        original_filename="sony_rgb_001.jpg",
        file_size_bytes=8900000,
        mime_type="image/jpeg",
        checksum_sha256="b" * 64,
        camera_make="Sony",
        camera_model="Alpha 7R IV",
        emissivity=None,
        reflected_temp_c=None,
        notes="High-resolution optical orthophoto",
    )
    db_session.add(rgb_img)
    db_session.commit()

    loaded_insp = db_session.get(Inspection, inspection.id)
    assert len(loaded_insp.images) == 2

    # Verify media separation
    types = {img.image_type for img in loaded_insp.images}
    assert ImageType.THERMAL_IR.value in types
    assert ImageType.OPTICAL_RGB.value in types

    thermal_record = next(img for img in loaded_insp.images if img.image_type == ImageType.THERMAL_IR.value)
    assert thermal_record.emissivity == 0.90
    assert thermal_record.camera_make == "FLIR"

    rgb_record = next(img for img in loaded_insp.images if img.image_type == ImageType.OPTICAL_RGB.value)
    assert rgb_record.emissivity is None
    assert rgb_record.camera_make == "Sony"

    # Test cascade deletion of images on inspection delete
    insp_id = inspection.id
    db_session.delete(loaded_insp)
    db_session.commit()

    assert db_session.get(Inspection, insp_id) is None
    assert db_session.get(ImageAsset, thermal_img.id) is None
    assert db_session.get(ImageAsset, rgb_img.id) is None
