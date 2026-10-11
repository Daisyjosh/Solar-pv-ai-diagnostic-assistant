"""Pydantic schemas for Plant entity."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PlantBase(BaseModel):
    """Core attributes for a solar PV plant."""

    code: str = Field(..., min_length=2, max_length=64, description="Unique plant identifier code, e.g., 'PLANT-KA-001'.")
    name: str = Field(..., min_length=2, max_length=255, description="Full plant operational name.")
    location: str = Field(..., min_length=2, max_length=255, description="Geographic location or site coordinates.")
    capacity_dc_mw: float = Field(..., gt=0.0, description="Total DC rated capacity in Megawatts.")
    commissioning_date: Optional[date] = Field(None, description="Date of commercial grid synchronization.")
    is_active: bool = Field(True, description="Whether the plant is currently operational.")


class PlantCreate(PlantBase):
    """Payload to register a new solar PV plant."""

    pass


class PlantUpdate(BaseModel):
    """Payload to update an existing solar PV plant."""

    name: Optional[str] = Field(None, min_length=2, max_length=255)
    location: Optional[str] = Field(None, min_length=2, max_length=255)
    capacity_dc_mw: Optional[float] = Field(None, gt=0.0)
    commissioning_date: Optional[date] = None
    is_active: Optional[bool] = None


class PlantResponse(PlantBase):
    """Response representation of a solar PV plant."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PlantListResponse(BaseModel):
    """Paginated or listed response of plants."""

    total: int
    items: List[PlantResponse]
