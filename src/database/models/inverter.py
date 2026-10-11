"""Inverter ORM model representing solar central or string inverters."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection
    from src.database.models.plant import Plant
    from src.database.models.string import PVString


class Inverter(Base):
    """Solar inverter station mapping strings to plant electrical bus."""

    __tablename__ = "inverters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    plant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    rated_power_kw: Mapped[float] = mapped_column(Float, nullable=False)
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
    plant: Mapped["Plant"] = relationship("Plant", back_populates="inverters")
    strings: Mapped[List["PVString"]] = relationship(
        "PVString",
        back_populates="inverter",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inspections: Mapped[List["Inspection"]] = relationship(
        "Inspection",
        back_populates="inverter",
    )

    def __repr__(self) -> str:
        return f"<Inverter(id={self.id}, code='{self.code}', plant_id={self.plant_id}, rated={self.rated_power_kw}kW)>"
