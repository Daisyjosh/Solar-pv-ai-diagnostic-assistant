"""Inspection ORM model tracking solar PV inspection sessions."""

import enum
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.image_asset import ImageAsset
    from src.database.models.inverter import Inverter
    from src.database.models.module import PVModule
    from src.database.models.plant import Plant
    from src.database.models.string import PVString


class InspectionStatus(str, enum.Enum):
    """Lifecycle status of a solar plant inspection event."""

    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


ALLOWED_INSPECTION_TRANSITIONS = {
    InspectionStatus.SCHEDULED.value: {
        InspectionStatus.SCHEDULED.value,
        InspectionStatus.IN_PROGRESS.value,
        InspectionStatus.ARCHIVED.value,
    },
    InspectionStatus.IN_PROGRESS.value: {
        InspectionStatus.IN_PROGRESS.value,
        InspectionStatus.COMPLETED.value,
        InspectionStatus.ARCHIVED.value,
    },
    InspectionStatus.COMPLETED.value: {
        InspectionStatus.COMPLETED.value,
        InspectionStatus.IN_PROGRESS.value,
        InspectionStatus.ARCHIVED.value,
    },
    InspectionStatus.ARCHIVED.value: {
        InspectionStatus.ARCHIVED.value,
        InspectionStatus.IN_PROGRESS.value,
    },
}


class Inspection(Base):
    """Inspection event entity recording metadata, ambient parameters, and linked images."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    plant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    inverter_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("inverters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    string_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("strings.id", ondelete="SET NULL"), nullable=True, index=True
    )
    module_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("modules.id", ondelete="SET NULL"), nullable=True, index=True
    )

    inspector_name: Mapped[str] = mapped_column(String(128), nullable=False)
    inspection_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=InspectionStatus.SCHEDULED.value,
        nullable=False,
    )

    # Environmental & Meteorological readings during inspection (IEC 62446-3)
    ambient_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    irradiance_w_m2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weather_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    general_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    plant: Mapped["Plant"] = relationship("Plant", back_populates="inspections")
    inverter: Mapped[Optional["Inverter"]] = relationship("Inverter", back_populates="inspections")
    string: Mapped[Optional["PVString"]] = relationship("PVString", back_populates="inspections")
    module: Mapped[Optional["PVModule"]] = relationship("PVModule", back_populates="inspections")
    images: Mapped[List["ImageAsset"]] = relationship(
        "ImageAsset",
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Inspection(id={self.id}, code='{self.code}', plant_id={self.plant_id}, status='{self.status}')>"
