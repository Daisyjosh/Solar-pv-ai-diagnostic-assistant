"""Plant ORM model representing a solar photovoltaic installation."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection
    from src.database.models.inverter import Inverter


class Plant(Base):
    """Solar photovoltaic generation facility entity."""

    __tablename__ = "plants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(255), nullable=False)
    capacity_dc_mw: Mapped[float] = mapped_column(Float, nullable=False)
    commissioning_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
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
    inverters: Mapped[List["Inverter"]] = relationship(
        "Inverter",
        back_populates="plant",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inspections: Mapped[List["Inspection"]] = relationship(
        "Inspection",
        back_populates="plant",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "location": self.location,
            "capacity_dc_mw": self.capacity_dc_mw,
            "commissioning_date": self.commissioning_date.isoformat() if self.commissioning_date else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<Plant(id={self.id}, code='{self.code}', name='{self.name}', capacity={self.capacity_dc_mw}MW)>"
