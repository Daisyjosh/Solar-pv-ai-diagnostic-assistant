"""ImageAsset ORM model managing thermal infrared and optical inspection media."""

import enum
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection


class ImageType(str, enum.Enum):
    """Categorization of PV inspection image media."""

    THERMAL_IR = "THERMAL_IR"      # Infrared radiometric or standardized thermal capture
    OPTICAL_RGB = "OPTICAL_RGB"    # Visible spectrum optical RGB inspection image


class ImageAsset(Base):
    """Metadata record for an uploaded inspection image asset."""

    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Hardware & Optical metadata
    camera_make: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    camera_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Radiometric Thermal calibration parameters (IEC 62446-3)
    # Crucial for accurate temperature measurement; null for optical RGB
    emissivity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reflected_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    inspection: Mapped["Inspection"] = relationship("Inspection", back_populates="images")

    def resolve_absolute_path(self) -> Path:
        """Resolve and validate the canonical absolute filesystem path for this image asset."""
        from src.services.storage import StorageService
        return StorageService().get_image_file_path(self.file_path)

    def __repr__(self) -> str:
        return (
            f"<ImageAsset(id={self.id}, inspection_id={self.inspection_id}, "
            f"type='{self.image_type}', file='{self.original_filename}')>"
        )

