"""Pydantic schemas for Inspection entity."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.image_asset import ImageAssetResponse
from src.database.models.inspection import InspectionStatus


class InspectionBase(BaseModel):
    """Core attributes for an inspection session."""

    code: str = Field(..., min_length=2, max_length=64, description="Unique inspection identifier, e.g., 'INSP-2026-001'.")
    plant_id: int = Field(..., gt=0, description="Foreign key to the inspected solar plant.")
    inverter_id: Optional[int] = Field(None, gt=0, description="Optional target inverter identifier.")
    string_id: Optional[int] = Field(None, gt=0, description="Optional target string identifier.")
    module_id: Optional[int] = Field(None, gt=0, description="Optional target module identifier.")
    inspector_name: str = Field(..., min_length=2, max_length=128, description="Name or ID of inspecting engineer.")
    inspection_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the inspection session.",
    )
    status: InspectionStatus = Field(
        InspectionStatus.SCHEDULED,
        description="Lifecycle status of inspection.",
    )
    ambient_temp_c: Optional[float] = Field(None, description="Ambient air temperature in Celsius.")
    irradiance_w_m2: Optional[float] = Field(
        None,
        ge=0.0,
        description="Plane-of-array solar irradiance in W/m^2 (IEC 62446-3 recommends >= 600 W/m^2).",
    )
    weather_notes: Optional[str] = Field(None, description="Sky conditions, cloud cover, or wind observations.")
    general_notes: Optional[str] = Field(None, description="Inspection purpose, equipment notes, or observations.")


class InspectionCreate(InspectionBase):
    """Payload to register a new inspection event."""

    pass


class InspectionUpdate(BaseModel):
    """Payload to update an inspection status or environmental notes."""

    status: Optional[InspectionStatus] = None
    ambient_temp_c: Optional[float] = None
    irradiance_w_m2: Optional[float] = Field(None, ge=0.0)
    weather_notes: Optional[str] = None
    general_notes: Optional[str] = None


class InspectionResponse(InspectionBase):
    """Response representation of an inspection event."""

    id: int
    created_at: datetime
    updated_at: datetime
    images_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class InspectionDetailResponse(InspectionResponse):
    """Detailed inspection response with associated image assets."""

    images: List[ImageAssetResponse] = []


class InspectionListResponse(BaseModel):
    """Paginated or listed response of inspections."""

    total: int
    items: List[InspectionResponse]
