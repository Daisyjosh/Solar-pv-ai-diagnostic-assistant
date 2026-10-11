"""Pydantic schemas for Inverter, PVString, and PVModule hierarchy entities."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Inverter Schemas ---
class InverterBase(BaseModel):
    """Base fields for solar inverter station."""

    code: str = Field(..., min_length=1, max_length=64, description="Inverter station code, e.g., 'INV-01'.")
    model: Optional[str] = Field(None, max_length=128, description="Manufacturer equipment model.")
    rated_power_kw: float = Field(..., gt=0.0, description="Rated AC/DC power capacity in kW.")


class InverterCreate(InverterBase):
    """Payload to register an inverter under a plant."""

    pass


class InverterResponse(InverterBase):
    """Response representation of an inverter."""

    id: int
    plant_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InverterListResponse(BaseModel):
    """List response for inverters."""

    total: int
    items: List[InverterResponse]


# --- PVString Schemas ---
class StringBase(BaseModel):
    """Base fields for a series-connected PV module string."""

    code: str = Field(..., min_length=1, max_length=64, description="String identifier code, e.g., 'STR-01-A'.")
    modules_count: int = Field(..., gt=0, le=100, description="Number of series-connected modules in this string.")
    rated_voc_v: float = Field(..., gt=0.0, description="Nominal STC open-circuit voltage Voc in Volts.")
    rated_isc_a: float = Field(..., gt=0.0, description="Nominal STC short-circuit current Isc in Amperes.")


class StringCreate(StringBase):
    """Payload to register a series string under an inverter."""

    pass


class StringResponse(StringBase):
    """Response representation of a PV string."""

    id: int
    inverter_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StringListResponse(BaseModel):
    """List response for strings."""

    total: int
    items: List[StringResponse]


# --- PVModule Schemas ---
class ModuleBase(BaseModel):
    """Base fields for an individual solar PV module."""

    code: str = Field(..., min_length=1, max_length=64, description="Module code or serial, e.g., 'MOD-01-A-01'.")
    position_in_string: int = Field(..., ge=1, le=100, description="1-indexed electrical position within the string.")
    manufacturer: Optional[str] = Field(None, max_length=128, description="Module manufacturer brand.")
    model: Optional[str] = Field(None, max_length=128, description="Module model number.")
    rated_power_w: float = Field(..., gt=0.0, description="Nominal STC maximum power rating in Watts.")


class ModuleCreate(ModuleBase):
    """Payload to register a PV module under a string."""

    pass


class ModuleResponse(ModuleBase):
    """Response representation of a PV module."""

    id: int
    string_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModuleListResponse(BaseModel):
    """List response for modules."""

    total: int
    items: List[ModuleResponse]
