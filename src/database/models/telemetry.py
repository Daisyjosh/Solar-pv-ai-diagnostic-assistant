"""Database ORM models for electrical telemetry and SCADA monitoring time-series."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inverter import Inverter
    from src.database.models.plant import Plant
    from src.database.models.string import PVString


class TelemetryBatch(Base):
    """Import record for a batch of ingested electrical telemetry (CSV or Excel)."""

    __tablename__ = "telemetry_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    plant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)  # CSV, EXCEL
    records_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="COMMITTED", nullable=False)
    import_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    plant: Mapped["Plant"] = relationship("Plant", backref="telemetry_batches")
    records: Mapped[List["TelemetryRecord"]] = relationship(
        "TelemetryRecord",
        back_populates="batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "plant_id": self.plant_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "records_count": self.records_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "import_notes": self.import_notes,
            "created_at": self.created_at.isoformat(),
        }


class TelemetryRecord(Base):
    """High-resolution electrical measurement point for solar PV plant assets."""

    __tablename__ = "telemetry_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("telemetry_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    plant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    inverter_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("inverters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    string_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("strings.id", ondelete="SET NULL"), nullable=True, index=True
    )

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    voltage_v: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    irradiance_w_m2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ambient_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    module_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    batch: Mapped["TelemetryBatch"] = relationship("TelemetryBatch", back_populates="records")
    plant: Mapped["Plant"] = relationship("Plant")
    inverter: Mapped[Optional["Inverter"]] = relationship("Inverter")
    string: Mapped[Optional["PVString"]] = relationship("PVString")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "plant_id": self.plant_id,
            "inverter_id": self.inverter_id,
            "string_id": self.string_id,
            "timestamp": self.timestamp.isoformat(),
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "power_kw": self.power_kw,
            "irradiance_w_m2": self.irradiance_w_m2,
            "ambient_temp_c": self.ambient_temp_c,
            "module_temp_c": self.module_temp_c,
            "operating_status": self.operating_status,
        }
