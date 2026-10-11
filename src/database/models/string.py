"""String ORM model representing a series-connected PV module string."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inverter import Inverter
    from src.database.models.inspection import Inspection
    from src.database.models.module import PVModule


class PVString(Base):
    """Photovoltaic series string topology entity."""

    __tablename__ = "strings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    inverter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inverters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    modules_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rated_voc_v: Mapped[float] = mapped_column(Float, nullable=False)
    rated_isc_a: Mapped[float] = mapped_column(Float, nullable=False)
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
    inverter: Mapped["Inverter"] = relationship("Inverter", back_populates="strings")
    modules: Mapped[List["PVModule"]] = relationship(
        "PVModule",
        back_populates="string",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inspections: Mapped[List["Inspection"]] = relationship(
        "Inspection",
        back_populates="string",
    )

    def __repr__(self) -> str:
        return f"<PVString(id={self.id}, code='{self.code}', inverter_id={self.inverter_id}, modules={self.modules_count})>"
