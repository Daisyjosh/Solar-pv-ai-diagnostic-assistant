"""FastAPI router for Solar PV Plant entity operations."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.schemas.plant import PlantCreate, PlantListResponse, PlantResponse, PlantUpdate
from src.database.models.plant import Plant
from src.database.session import get_db

router = APIRouter(prefix="/plants", tags=["Plants"])


@router.post(
    "/",
    response_model=PlantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new solar PV plant",
)
def create_plant(
    payload: PlantCreate,
    db: Session = Depends(get_db),
) -> Plant:
    """Create a new solar plant record. Enforces unique plant code constraint."""
    # Check for existing plant code
    existing = db.scalar(select(Plant).where(Plant.code == payload.code))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Plant with code '{payload.code}' already exists.",
        )

    plant = Plant(
        code=payload.code,
        name=payload.name,
        location=payload.location,
        capacity_dc_mw=payload.capacity_dc_mw,
        commissioning_date=payload.commissioning_date,
        is_active=payload.is_active,
    )
    db.add(plant)
    db.commit()
    db.refresh(plant)
    return plant


@router.get(
    "/",
    response_model=PlantListResponse,
    status_code=status.HTTP_200_OK,
    summary="List registered solar PV plants",
)
def list_plants(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
) -> PlantListResponse:
    """Retrieve registered solar plants with optional pagination and active-status filtering."""
    query = select(Plant)
    if is_active is not None:
        query = query.where(Plant.is_active == is_active)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    plants = db.scalars(query.offset(skip).limit(limit).order_by(Plant.code)).all()

    return PlantListResponse(
        total=total,
        items=[PlantResponse.model_validate(p) for p in plants],
    )


@router.get(
    "/{plant_id}",
    response_model=PlantResponse,
    status_code=status.HTTP_200_OK,
    summary="Get plant details by ID",
)
def get_plant(
    plant_id: int,
    db: Session = Depends(get_db),
) -> Plant:
    """Retrieve detailed solar plant metadata by primary key ID."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with ID {plant_id} not found.",
        )
    return plant


@router.put(
    "/{plant_id}",
    response_model=PlantResponse,
    status_code=status.HTTP_200_OK,
    summary="Update plant metadata",
)
def update_plant(
    plant_id: int,
    payload: PlantUpdate,
    db: Session = Depends(get_db),
) -> Plant:
    """Update plant fields."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with ID {plant_id} not found.",
        )

    update_data = payload.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(plant, field, val)

    db.commit()
    db.refresh(plant)
    return plant


@router.delete(
    "/{plant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a solar plant",
)
def delete_plant(
    plant_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a solar plant and cascade deletion to associated inverters and inspections."""
    plant = db.get(Plant, plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plant with ID {plant_id} not found.",
        )
    db.delete(plant)
    db.commit()
