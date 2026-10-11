"""FastAPI router for solar PV asset hierarchy (Inverters, Strings, Modules)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.schemas.asset_hierarchy import (
    InverterCreate,
    InverterListResponse,
    InverterResponse,
    ModuleCreate,
    ModuleListResponse,
    ModuleResponse,
    StringCreate,
    StringListResponse,
    StringResponse,
)
from src.database.models.inverter import Inverter
from src.database.models.module import PVModule
from src.database.models.plant import Plant
from src.database.models.string import PVString
from src.database.session import get_db

router = APIRouter(tags=["Asset Hierarchy"])


# ==============================================================================
# Inverter Endpoints
# ==============================================================================

@router.post(
    "/plants/{plant_id}/inverters",
    response_model=InverterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an inverter under a plant",
)
def create_inverter(
    plant_id: int,
    payload: InverterCreate,
    db: Session = Depends(get_db),
) -> Inverter:
    """Create an inverter station mapped to a plant."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with ID {plant_id} not found.",
        )

    # Check for duplicate inverter code within this plant
    existing = db.scalar(
        select(Inverter).where(Inverter.plant_id == plant_id, Inverter.code == payload.code)
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Inverter with code '{payload.code}' already exists in plant {plant_id}.",
        )

    inverter = Inverter(
        plant_id=plant_id,
        code=payload.code,
        model=payload.model,
        rated_power_kw=payload.rated_power_kw,
    )
    db.add(inverter)
    db.commit()
    db.refresh(inverter)
    return inverter


@router.get(
    "/plants/{plant_id}/inverters",
    response_model=InverterListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all inverters in a plant",
)
def list_inverters(
    plant_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> InverterListResponse:
    """Retrieve all inverters associated with a specific plant."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with ID {plant_id} not found.",
        )

    total = db.scalar(
        select(func.count(Inverter.id)).where(Inverter.plant_id == plant_id)
    ) or 0
    inverters = db.scalars(
        select(Inverter)
        .where(Inverter.plant_id == plant_id)
        .offset(skip)
        .limit(limit)
        .order_by(Inverter.code)
    ).all()

    return InverterListResponse(
        total=total,
        items=[InverterResponse.model_validate(inv) for inv in inverters],
    )


# ==============================================================================
# PVString Endpoints
# ==============================================================================

@router.post(
    "/inverters/{inverter_id}/strings",
    response_model=StringResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a series string under an inverter",
)
def create_string(
    inverter_id: int,
    payload: StringCreate,
    db: Session = Depends(get_db),
) -> PVString:
    """Create a series string mapped to an inverter."""
    inverter = db.get(Inverter, inverter_id)
    if not inverter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inverter with ID {inverter_id} not found.",
        )

    # Check for duplicate string code within this inverter
    existing = db.scalar(
        select(PVString).where(PVString.inverter_id == inverter_id, PVString.code == payload.code)
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"String with code '{payload.code}' already exists in inverter {inverter_id}.",
        )

    pv_string = PVString(
        inverter_id=inverter_id,
        code=payload.code,
        modules_count=payload.modules_count,
        rated_voc_v=payload.rated_voc_v,
        rated_isc_a=payload.rated_isc_a,
    )
    db.add(pv_string)
    db.commit()
    db.refresh(pv_string)
    return pv_string


@router.get(
    "/inverters/{inverter_id}/strings",
    response_model=StringListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all strings under an inverter",
)
def list_strings(
    inverter_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> StringListResponse:
    """Retrieve all strings associated with a specific inverter."""
    inverter = db.get(Inverter, inverter_id)
    if not inverter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inverter with ID {inverter_id} not found.",
        )

    total = db.scalar(
        select(func.count(PVString.id)).where(PVString.inverter_id == inverter_id)
    ) or 0
    strings = db.scalars(
        select(PVString)
        .where(PVString.inverter_id == inverter_id)
        .offset(skip)
        .limit(limit)
        .order_by(PVString.code)
    ).all()

    return StringListResponse(
        total=total,
        items=[StringResponse.model_validate(s) for s in strings],
    )


# ==============================================================================
# PVModule Endpoints
# ==============================================================================

@router.post(
    "/strings/{string_id}/modules",
    response_model=ModuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an individual module under a string",
)
def create_module(
    string_id: int,
    payload: ModuleCreate,
    db: Session = Depends(get_db),
) -> PVModule:
    """Create a module record mapped to a string."""
    pv_string = db.get(PVString, string_id)
    if not pv_string:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"String with ID {string_id} not found.",
        )

    # Validate position is within string count
    if payload.position_in_string > pv_string.modules_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Position {payload.position_in_string} exceeds string capacity "
                f"of {pv_string.modules_count} modules."
            ),
        )

    # Check for duplicate module code or position within this string
    existing = db.scalar(
        select(PVModule).where(
            PVModule.string_id == string_id,
            (PVModule.code == payload.code) | (PVModule.position_in_string == payload.position_in_string),
        )
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Module with code '{payload.code}' or position {payload.position_in_string} "
                f"already exists in string {string_id}."
            ),
        )

    pv_module = PVModule(
        string_id=string_id,
        code=payload.code,
        position_in_string=payload.position_in_string,
        manufacturer=payload.manufacturer,
        model=payload.model,
        rated_power_w=payload.rated_power_w,
    )
    db.add(pv_module)
    db.commit()
    db.refresh(pv_module)
    return pv_module


@router.get(
    "/strings/{string_id}/modules",
    response_model=ModuleListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all modules under a string",
)
def list_modules(
    string_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> ModuleListResponse:
    """Retrieve all modules associated with a specific string."""
    pv_string = db.get(PVString, string_id)
    if not pv_string:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"String with ID {string_id} not found.",
        )

    total = db.scalar(
        select(func.count(PVModule.id)).where(PVModule.string_id == string_id)
    ) or 0
    modules = db.scalars(
        select(PVModule)
        .where(PVModule.string_id == string_id)
        .offset(skip)
        .limit(limit)
        .order_by(PVModule.position_in_string)
    ).all()

    return ModuleListResponse(
        total=total,
        items=[ModuleResponse.model_validate(m) for m in modules],
    )
