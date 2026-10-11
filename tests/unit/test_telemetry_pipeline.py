"""Unit tests for Phase 7 Telemetry Importer and Analytics."""

import io
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models.plant import Plant
from src.database.session import Base
from src.telemetry.analytics import TelemetryAnalyticsService
from src.telemetry.importer import TelemetryImporter
from src.telemetry.providers.scada_adapter import SCADAAdapter


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    # Create dummy plant
    plant = Plant(
        code="PLANT-TEST",
        name="Test Plant",
        location="Test Site",
        capacity_dc_mw=10.0,
    )
    session.add(plant)
    session.commit()
    session.refresh(plant)

    yield session, plant
    session.close()


def test_telemetry_csv_preview_and_commit(memory_db):
    session, plant = memory_db
    csv_content = (
        "Timestamp,Voltage_V,Current_A,Power_kW,Irradiance_W_m2,Ambient_Temp_C,Module_Temp_C\n"
        "2026-10-10 12:00:00,650.5,8.9,5.78,850.0,26.5,45.2\n"
        "2026-10-10 12:05:00,652.0,9.1,5.93,860.0,26.8,45.8\n"
    ).encode("utf-8")

    importer = TelemetryImporter(session)
    preview = importer.preview_import(csv_content, "test_scada.csv")

    assert preview.total_rows == 2
    assert preview.valid_rows_count == 2
    assert "timestamp" in preview.column_mapping
    assert "voltage_v" in preview.column_mapping

    # Commit import
    batch = importer.commit_import(plant.id, csv_content, "test_scada.csv")
    assert batch.records_count == 2
    assert batch.status == "COMMITTED"

    # Verify via analytics service
    analytics = TelemetryAnalyticsService(session)
    records = analytics.get_records(plant.id)
    assert len(records) == 2
    assert records[0]["voltage_v"] == 650.5

    trends = analytics.get_trend_summary(plant.id)
    assert trends["has_data"] is True
    assert trends["record_count"] == 2


def test_scada_adapter_status_unconfigured():
    adapter = SCADAAdapter()
    stat = adapter.get_status()
    assert stat["status"] == "UNCONFIGURED"
    assert stat["is_live_connected"] is False
    assert "unconfigured" in stat["message"].lower()
