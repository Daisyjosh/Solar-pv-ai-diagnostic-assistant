"""Pydantic schemas for ImageAsset entity."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageTypeEnum(str, Enum):
    """Categorization of PV inspection images."""

    THERMAL_IR = "THERMAL_IR"
    OPTICAL_RGB = "OPTICAL_RGB"


class ImageAssetBase(BaseModel):
    """Base fields for an inspection image asset."""

    image_type: ImageTypeEnum = Field(
        ...,
        description="Type of inspection capture: THERMAL_IR for infrared radiometric, OPTICAL_RGB for visible spectrum.",
    )
    file_path: str = Field(..., min_length=1, max_length=512, description="Relative secure filesystem path.")
    original_filename: str = Field(..., min_length=1, max_length=255)
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes.")
    mime_type: str = Field(..., min_length=3, max_length=64)
    checksum_sha256: str = Field(..., min_length=64, max_length=64, description="Hex-encoded SHA-256 checksum.")
    camera_make: Optional[str] = Field(None, max_length=128)
    camera_model: Optional[str] = Field(None, max_length=128)
    emissivity: Optional[float] = Field(
        None,
        ge=0.01,
        le=1.0,
        description="Radiometric emissivity coefficient (typically 0.85-0.95 for PV glass).",
    )
    reflected_temp_c: Optional[float] = Field(
        None,
        description="Reflected ambient apparent temperature in Celsius.",
    )
    notes: Optional[str] = None


class ImageAssetCreate(ImageAssetBase):
    """Payload to register an image asset for an inspection."""

    @field_validator("checksum_sha256")
    @classmethod
    def validate_hex_hash(cls, v: str) -> str:
        if not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("Checksum must be a valid 64-character hexadecimal SHA-256 string")
        return v.lower()


class ImageAssetResponse(ImageAssetBase):
    """Response representation of an ImageAsset."""

    id: int
    inspection_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
