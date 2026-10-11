"""Module ORM model representing individual solar PV panels."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection
    from src.database.models.string import PVString


class PVModule(Base):
    """Individual solar photovoltaic panel / module entity."""

    __tablename__ = "modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    string_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("strings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    position_in_string: Mapped[int] = mapped_column(Integer, nullable=False)
    manufacturer: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    rated_power_w: Mapped[float] = mapped_column(Float, nullable=False)
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
    string: Mapped["PVString"] = relationship("PVString", back_populates="modules")
    inspections: Mapped[List["Inspection"]] = relationship(
        "Inspection",
        back_populates="module",
    )

    def __repr__(self) -> str:
        return f"<PVModule(id={self.id}, code='{self.code}', string_id={self.string_id}, position={self.position_in_string})>"
