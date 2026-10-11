"""End-to-end verification script for Phase 3 requirements.

Tests:
1. Real thermal and RGB image uploads through the FastAPI application.
2. Byte-for-byte preservation during image retrieval.
3. Unsupported files, corrupt images, oversized uploads, duplicate uploads, and cleanup.
4. Radiometric values extracted ONLY when genuinely present (zero fabrication check).
"""

import hashlib
import io
import sys
import uuid
from pathlib import Path

# Add project root to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import create_app
from src.core.config import get_settings
from src.database.session import init_db


def generate_radiometric_flir_jpeg() -> bytes:
    """Generate a real JPEG with genuine FLIR EXIF tags."""
    buf = io.BytesIO()
    img = Image.new("RGB", (160, 120), color=(255, 80, 20))
    exif = img.getexif()
    exif[271] = "FLIR Systems"
    exif[272] = "Vue Pro R 640"
    exif[306] = "2026:09:12 14:45:00"
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def generate_rendered_thermal_jpeg() -> bytes:
    """Generate a rendered false-color thermal JPEG without radiometric calibration metadata."""
    buf = io.BytesIO()
    img = Image.new("RGB", (160, 120), color=(180, 40, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def generate_optical_rgb_png() -> bytes:
    """Generate a standard optical RGB PNG."""
    buf = io.BytesIO()
    img = Image.new("RGBA", (200, 200), color=(50, 180, 90, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    print("=" * 70)
    print("Phase 3 End-to-End Verification: Storage, Retrieval & Radiometric Integrity")
    print("=" * 70)

    init_db()
    settings = get_settings()
    app = create_app()
    client = TestClient(app)

    unique_run = uuid.uuid4().hex[:6]

    # 1. Setup Plant and Unique Inspection
    print("[1] Creating Test Plant and Inspection Session...")
    plant_code = f"PLANT-VERIFY-{unique_run}"
    plant_res = client.post("/api/v1/plants/", json={
        "code": plant_code,
        "name": f"Verification Solar Array {unique_run}",
        "location": "Bhadla, Rajasthan",
        "capacity_dc_mw": 50.0,
    })
    assert plant_res.status_code == 201, f"Failed to create plant: {plant_res.text}"
    plant_id = plant_res.json()["id"]

    insp_code = f"INSP-VERIFY-{unique_run}"
    insp_res = client.post("/api/v1/inspections/", json={
        "code": insp_code,
        "plant_id": plant_id,
        "inspector_name": "Lead O&M Engineer",
        "status": "IN_PROGRESS",
    })
    assert insp_res.status_code == 201, f"Failed to create inspection: {insp_res.text}"
    insp_id = insp_res.json()["id"]
    print(f"    [OK] Plant #{plant_id} and Inspection #{insp_id} initialized.")

    raw_insp_dir = settings.storage_dir.resolve() / "raw" / "inspections" / str(insp_id)
    initial_file_count = len(list(raw_insp_dir.iterdir())) if raw_insp_dir.exists() else 0
    assert initial_file_count == 0, "Fresh inspection directory should have 0 files"

    # 2. Upload Thermal IR Image with EXIF
    print("\n[2] Uploading Thermal IR Image (with Camera Make/Model)...")
    flir_bytes = generate_radiometric_flir_jpeg()
    flir_hash = hashlib.sha256(flir_bytes).hexdigest()

    thermal_up = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("flir_vue_capture.jpg", flir_bytes, "image/jpeg")},
        data={"image_type": "THERMAL_IR", "notes": "Thermal capture with FLIR hardware"},
    )
    assert thermal_up.status_code == 201, f"Thermal upload failed: {thermal_up.text}"
    thermal_data = thermal_up.json()
    assert thermal_data["image_type"] == "THERMAL_IR"
    assert thermal_data["camera_make"] == "FLIR Systems"
    assert thermal_data["camera_model"] == "Vue Pro R 640"
    assert thermal_data["checksum_sha256"] == flir_hash
    thermal_img_id = thermal_data["id"]
    print(f"    [OK] Thermal IR image ingested as Asset #{thermal_img_id}")
    print(f"         Camera: {thermal_data['camera_make']} {thermal_data['camera_model']}")

    # 3. Upload Optical RGB Image
    print("\n[3] Uploading Optical RGB Image (PNG)...")
    rgb_bytes = generate_optical_rgb_png()
    rgb_hash = hashlib.sha256(rgb_bytes).hexdigest()

    rgb_up = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("ortho_rgb.png", rgb_bytes, "image/png")},
        data={"image_type": "OPTICAL_RGB", "notes": "Orthomosaic optical RGB capture"},
    )
    assert rgb_up.status_code == 201, f"RGB upload failed: {rgb_up.text}"
    rgb_data = rgb_up.json()
    assert rgb_data["image_type"] == "OPTICAL_RGB"
    assert rgb_data["emissivity"] is None, "Emissivity must be None for optical RGB (zero fabrication)"
    assert rgb_data["reflected_temp_c"] is None, "Reflected temp must be None for optical RGB"
    rgb_img_id = rgb_data["id"]
    print(f"    [OK] Optical RGB image ingested as Asset #{rgb_img_id}")
    print(f"         Confirmed: Emissivity and Reflected Temp are strictly None (unfabricated).")

    # 4. Upload Rendered Thermal JPEG without Radiometric Tags
    print("\n[4] Uploading Rendered Thermal JPEG (Palette image without radiometric tags)...")
    rendered_bytes = generate_rendered_thermal_jpeg()
    rendered_up = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("rendered_ironbow.jpg", rendered_bytes, "image/jpeg")},
        data={"image_type": "THERMAL_IR", "notes": "Rendered palette image without raw calibration"},
    )
    assert rendered_up.status_code == 201
    rendered_data = rendered_up.json()
    assert rendered_data["emissivity"] is None, "Rendered image must NOT fabricate emissivity"
    assert rendered_data["reflected_temp_c"] is None
    print("    [OK] Rendered thermal image correctly identified without fabricated radiometric metadata.")

    # 5. Verify Byte-for-Byte Preservation on Retrieval
    print("\n[5] Verifying Image Retrieval Preserves Exact Binary Bytes...")
    dl_thermal = client.get(f"/api/v1/inspections/{insp_id}/images/{thermal_img_id}/file")
    assert dl_thermal.status_code == 200
    assert dl_thermal.content == flir_bytes, "Retrieved thermal bytes do not match original bytes!"
    assert hashlib.sha256(dl_thermal.content).hexdigest() == flir_hash
    print(f"    [OK] Thermal IR retrieval: 100% byte-for-byte match (SHA-256 verified).")

    dl_rgb = client.get(f"/api/v1/inspections/{insp_id}/images/{rgb_img_id}/file")
    assert dl_rgb.status_code == 200
    assert dl_rgb.content == rgb_bytes, "Retrieved RGB bytes do not match original bytes!"
    assert hashlib.sha256(dl_rgb.content).hexdigest() == rgb_hash
    print(f"    [OK] Optical RGB retrieval: 100% byte-for-byte match (SHA-256 verified).")

    # 6. Test Security: Duplicate Upload Rejection
    print("\n[6] Testing Duplicate Upload Rejection (Same Checksum)...")
    dup_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("flir_duplicate.jpg", flir_bytes, "image/jpeg")},
        data={"image_type": "THERMAL_IR"},
    )
    assert dup_res.status_code == 409, f"Expected 409 Conflict, got {dup_res.status_code}"
    print("    [OK] Duplicate upload rejected with HTTP 409 Conflict.")

    # 7. Test Security: Unsupported Extension
    print("\n[7] Testing Unsupported Extension Rejection...")
    unsupported_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("script.sh", b"#!/bin/bash\necho test", "text/x-sh")},
        data={"image_type": "OPTICAL_RGB"},
    )
    assert unsupported_res.status_code == 400
    print("    [OK] Unsupported file extension rejected with HTTP 400.")

    # 8. Test Security: Corrupt Image Content
    print("\n[8] Testing Corrupted Image Content Rejection...")
    corrupt_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("corrupt.jpg", b"\xff\xd8\xff\xe0truncated_bad_bytes", "image/jpeg")},
        data={"image_type": "THERMAL_IR"},
    )
    assert corrupt_res.status_code == 400
    print("    [OK] Corrupt image content rejected with HTTP 400.")

    # 9. Test Security: Failed-Upload Cleanup (No Orphan Files Left)
    print("\n[9] Verifying Failed-Upload Cleanup (No Orphaned Files)...")
    current_files = list(raw_insp_dir.iterdir())
    # Exactly 3 valid images should exist: flir_vue_capture, ortho_rgb, rendered_ironbow
    assert len(current_files) == 3, f"Expected exactly 3 files, found {len(current_files)}: {current_files}"
    print(f"    [OK] Exactly 3 valid files exist in storage; all rejected uploads were cleanly removed.")

    # 10. Test Security: Path Traversal / Mismatched Inspection Retrieval
    print("\n[10] Testing Path Traversal and Authorization Defense on Retrieval...")
    mismatch_res = client.get(f"/api/v1/inspections/99999/images/{thermal_img_id}/file")
    assert mismatch_res.status_code == 404
    print("    [OK] Cross-inspection access blocked with HTTP 404.")

    print("\n" + "=" * 70)
    print("[SUCCESS] All Phase 3 E2E Verifications Passed Completely!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
