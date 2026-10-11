"""FastAPI router for Inspection session and ImageAsset operations."""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from src.api.schemas.image_asset import ImageAssetCreate, ImageAssetResponse, ImageTypeEnum
from src.api.schemas.inspection import (
    InspectionCreate,
    InspectionDetailResponse,
    InspectionListResponse,
    InspectionResponse,
    InspectionUpdate,
)
from src.database.models.image_asset import ImageAsset
from src.database.models.inspection import (
    ALLOWED_INSPECTION_TRANSITIONS,
    Inspection,
    InspectionStatus,
)
from src.database.models.inverter import Inverter
from src.database.models.module import PVModule
from src.database.models.plant import Plant
from src.database.models.string import PVString
from src.database.session import get_db
from src.services.storage import (
    CorruptImageError,
    DirectoryTraversalError,
    FileTooLargeError,
    StorageService,
    UnsupportedFormatError,
)
from src.vision.metadata import extract_image_metadata

router = APIRouter(prefix="/inspections", tags=["Inspections"])
storage_service = StorageService()


@router.post(
    "/",
    response_model=InspectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new solar PV inspection session",
)
def create_inspection(
    payload: InspectionCreate,
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Create a new inspection session. Validates foreign key hierarchy references."""
    # 1. Validate plant existence
    plant = db.get(Plant, payload.plant_id)
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Referenced Plant with ID {payload.plant_id} does not exist.",
        )

    # 2. Validate optional hierarchy references if provided
    if payload.inverter_id is not None:
        inverter = db.get(Inverter, payload.inverter_id)
        if not inverter or inverter.plant_id != plant.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Referenced Inverter with ID {payload.inverter_id} does not exist or does not belong to plant {plant.id}.",
            )

    if payload.string_id is not None:
        string = db.get(PVString, payload.string_id)
        if not string:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Referenced String with ID {payload.string_id} does not exist.",
            )

    if payload.module_id is not None:
        module = db.get(PVModule, payload.module_id)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Referenced Module with ID {payload.module_id} does not exist.",
            )

    # 3. Check for unique inspection code
    existing = db.scalar(select(Inspection).where(Inspection.code == payload.code))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Inspection with code '{payload.code}' already exists.",
        )

    inspection = Inspection(
        code=payload.code,
        plant_id=payload.plant_id,
        inverter_id=payload.inverter_id,
        string_id=payload.string_id,
        module_id=payload.module_id,
        inspector_name=payload.inspector_name,
        inspection_date=payload.inspection_date,
        status=payload.status.value,
        ambient_temp_c=payload.ambient_temp_c,
        irradiance_w_m2=payload.irradiance_w_m2,
        weather_notes=payload.weather_notes,
        general_notes=payload.general_notes,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    resp = InspectionResponse.model_validate(inspection)
    resp.images_count = 0
    return resp


@router.get(
    "/",
    response_model=InspectionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List solar PV inspection sessions",
)
def list_inspections(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    plant_id: Optional[int] = Query(None, description="Filter by plant ID"),
    status_filter: Optional[InspectionStatus] = Query(None, alias="status", description="Filter by inspection status"),
    db: Session = Depends(get_db),
) -> InspectionListResponse:
    """Retrieve inspection sessions with optional plant and status filtering."""
    query = select(Inspection).options(selectinload(Inspection.images))
    if plant_id is not None:
        query = query.where(Inspection.plant_id == plant_id)
    if status_filter is not None:
        query = query.where(Inspection.status == status_filter.value)

    total_query = select(func.count(Inspection.id))
    if plant_id is not None:
        total_query = total_query.where(Inspection.plant_id == plant_id)
    if status_filter is not None:
        total_query = total_query.where(Inspection.status == status_filter.value)

    total = db.scalar(total_query) or 0
    inspections = db.scalars(query.offset(skip).limit(limit).order_by(Inspection.inspection_date.desc())).all()

    items = []
    for insp in inspections:
        resp = InspectionResponse.model_validate(insp)
        resp.images_count = len(insp.images)
        items.append(resp)

    return InspectionListResponse(total=total, items=items)


@router.get(
    "/{inspection_id}",
    response_model=InspectionDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get inspection details including associated images",
)
def get_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionDetailResponse:
    """Retrieve full inspection details and linked thermal/optical image assets."""
    inspection = db.scalar(
        select(Inspection)
        .options(selectinload(Inspection.images))
        .where(Inspection.id == inspection_id)
    )
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    resp = InspectionDetailResponse.model_validate(inspection)
    resp.images_count = len(inspection.images)
    resp.images = [ImageAssetResponse.model_validate(img) for img in inspection.images]
    return resp


@router.put(
    "/{inspection_id}",
    response_model=InspectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update inspection status or notes",
)
def update_inspection(
    inspection_id: int,
    payload: InspectionUpdate,
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Update inspection status or environmental parameters with lifecycle validation."""
    inspection = db.scalar(
        select(Inspection)
        .options(selectinload(Inspection.images))
        .where(Inspection.id == inspection_id)
    )
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    update_dict = payload.model_dump(exclude_unset=True)
    if "status" in update_dict and update_dict["status"] is not None:
        new_status = update_dict["status"].value
        current_status = inspection.status
        allowed = ALLOWED_INSPECTION_TRANSITIONS.get(current_status, {current_status})
        if new_status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid inspection status transition from '{current_status}' to '{new_status}'. "
                    f"Allowed transitions: {sorted(list(allowed))}"
                ),
            )
        update_dict["status"] = new_status

    for k, v in update_dict.items():
        setattr(inspection, k, v)

    db.commit()
    db.refresh(inspection)

    resp = InspectionResponse.model_validate(inspection)
    resp.images_count = len(inspection.images)
    return resp


@router.post(
    "/{inspection_id}/upload",
    response_model=ImageAssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a thermal IR or optical RGB image file",
)
async def upload_inspection_image(
    inspection_id: int,
    file: UploadFile = File(..., description="Binary image file (JPEG, PNG, TIFF)"),
    image_type: ImageTypeEnum = Form(..., description="Image type: THERMAL_IR or OPTICAL_RGB"),
    notes: Optional[str] = Form(None, description="Optional engineering inspection notes"),
    db: Session = Depends(get_db),
) -> ImageAsset:
    """Securely stream, validate, and store an inspection image.

    Validates container signatures, checks for corrupted content, mitigates decompression bombs,
    extracts available EXIF tags, and associates the image with the target inspection.
    """
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    # 1. Stream file into storage with streaming size and magic byte checks
    try:
        rel_path, checksum, file_size, mime_type = storage_service.save_inspection_image(
            inspection_id=inspection_id,
            file_stream=file.file,
            original_filename=file.filename or "upload.jpg",
        )
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileTooLargeError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except CorruptImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DirectoryTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {e}")

    # 2. Check for duplicate upload in this inspection (same SHA-256)
    existing_duplicate = db.scalar(
        select(ImageAsset).where(
            ImageAsset.inspection_id == inspection_id,
            ImageAsset.checksum_sha256 == checksum,
        )
    )
    if existing_duplicate:
        # Clean up newly written file since duplicate exists
        storage_service.delete_image_file(rel_path)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Duplicate image: an identical file has already been uploaded to inspection "
                f"{inspection_id} with checksum {checksum[:16]}..."
            ),
        )

    # 3. Extract EXIF and radiometric metadata from stored file
    abs_path = storage_service.get_image_file_path(rel_path)
    metadata = extract_image_metadata(abs_path)

    # 4. Persist ImageAsset in database
    try:
        image_asset = ImageAsset(
            inspection_id=inspection_id,
            image_type=image_type.value,
            file_path=rel_path,
            original_filename=file.filename or "upload.jpg",
            file_size_bytes=file_size,
            mime_type=mime_type,
            checksum_sha256=checksum,
            camera_make=metadata.camera_make,
            camera_model=metadata.camera_model,
            emissivity=metadata.emissivity,
            reflected_temp_c=metadata.reflected_temp_c,
            notes=notes,
        )
        db.add(image_asset)
        if inspection.status == InspectionStatus.SCHEDULED.value:
            inspection.status = InspectionStatus.IN_PROGRESS.value
            db.add(inspection)
        db.commit()
        db.refresh(image_asset)
        return image_asset
    except Exception as exc:
        db.rollback()
        # Clean up file on disk if DB persistence fails
        storage_service.delete_image_file(rel_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record image asset in database: {exc}",
        )


@router.get(
    "/{inspection_id}/images/{image_id}/file",
    summary="Download or stream an inspection image file",
)
def get_image_file(
    inspection_id: int,
    image_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Retrieve the authorized binary file for an image asset within storage containment."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    image_asset = db.get(ImageAsset, image_id)
    if not image_asset or image_asset.inspection_id != inspection_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset {image_id} not found for inspection {inspection_id}.",
        )

    try:
        abs_path = storage_service.get_image_file_path(image_asset.file_path)
    except (DirectoryTraversalError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file cannot be retrieved: {exc}",
        )

    return FileResponse(
        path=str(abs_path),
        media_type=image_asset.mime_type,
        filename=image_asset.original_filename,
    )


@router.delete(
    "/{inspection_id}/images/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an image asset and its stored file",
)
def delete_image_asset(
    inspection_id: int,
    image_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete an image asset record and remove its file from storage."""
    image_asset = db.get(ImageAsset, image_id)
    if not image_asset or image_asset.inspection_id != inspection_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset {image_id} not found for inspection {inspection_id}.",
        )

    file_path = image_asset.file_path
    db.delete(image_asset)
    db.commit()
    # Remove file from disk
    storage_service.delete_image_file(file_path)


@router.post(
    "/{inspection_id}/images",
    response_model=ImageAssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register metadata for an existing image asset",
)
def register_image_asset(
    inspection_id: int,
    payload: ImageAssetCreate,
    db: Session = Depends(get_db),
) -> ImageAsset:
    """Directly register metadata for an image asset."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    image_asset = ImageAsset(
        inspection_id=inspection_id,
        image_type=payload.image_type.value,
        file_path=payload.file_path,
        original_filename=payload.original_filename,
        file_size_bytes=payload.file_size_bytes,
        mime_type=payload.mime_type,
        checksum_sha256=payload.checksum_sha256,
        camera_make=payload.camera_make,
        camera_model=payload.camera_model,
        emissivity=payload.emissivity,
        reflected_temp_c=payload.reflected_temp_c,
        notes=payload.notes,
    )
    db.add(image_asset)
    if inspection.status == InspectionStatus.SCHEDULED.value:
        inspection.status = InspectionStatus.IN_PROGRESS.value
        db.add(inspection)
    db.commit()
    db.refresh(image_asset)
    return image_asset


@router.get(
    "/{inspection_id}/images",
    response_model=List[ImageAssetResponse],
    status_code=status.HTTP_200_OK,
    summary="List all image assets associated with an inspection",
)
def list_inspection_images(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> List[ImageAsset]:
    """Retrieve all thermal and optical image assets registered for an inspection."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID {inspection_id} not found.",
        )

    return db.scalars(
        select(ImageAsset).where(ImageAsset.inspection_id == inspection_id).order_by(ImageAsset.created_at)
    ).all()
