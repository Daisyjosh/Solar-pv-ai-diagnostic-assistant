"""Integration tests for FastAPI Plant, Inspection, and ImageAsset endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint_with_database(client: TestClient):
    """Verify health endpoint pings the database and reports connected status."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "timestamp_utc" in data


def test_create_and_get_plant(client: TestClient):
    """Test creating and retrieving a solar PV plant."""
    payload = {
        "code": "PLANT-API-001",
        "name": "Bhadla Solar Park Section IV",
        "location": "Phalodi, Rajasthan, India",
        "capacity_dc_mw": 500.0,
        "commissioning_date": "2020-03-01",
        "is_active": True,
    }
    create_res = client.post("/api/v1/plants/", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["id"] is not None
    assert created_data["code"] == "PLANT-API-001"
    assert created_data["capacity_dc_mw"] == 500.0

    plant_id = created_data["id"]

    # Retrieve by ID
    get_res = client.get(f"/api/v1/plants/{plant_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Bhadla Solar Park Section IV"

    # List plants
    list_res = client.get("/api/v1/plants/")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(p["code"] == "PLANT-API-001" for p in list_data["items"])


def test_create_plant_duplicate_code(client: TestClient):
    """Test creating a plant with duplicate code returns 409 Conflict."""
    payload = {
        "code": "PLANT-UNIQUE-CODE",
        "name": "Original Plant",
        "location": "Location A",
        "capacity_dc_mw": 25.0,
    }
    res1 = client.post("/api/v1/plants/", json=payload)
    assert res1.status_code == 201

    res2 = client.post("/api/v1/plants/", json=payload)
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"]


def test_create_plant_validation_error(client: TestClient):
    """Test validation errors on invalid capacity or missing required fields."""
    invalid_payload = {
        "code": "PLANT-BAD",
        "name": "Bad Plant",
        "location": "Location",
        "capacity_dc_mw": -10.0,  # Invalid: capacity must be > 0
    }
    res = client.post("/api/v1/plants/", json=invalid_payload)
    assert res.status_code == 422


def test_create_inspection_lifecycle(client: TestClient):
    """Test end-to-end inspection lifecycle: creation, retrieval, and updates."""
    # 1. Register a plant first
    plant_payload = {
        "code": "PLANT-INSP-API",
        "name": "Inspection Target Plant",
        "location": "Kurnool, Andhra Pradesh, India",
        "capacity_dc_mw": 150.0,
    }
    plant_res = client.post("/api/v1/plants/", json=plant_payload)
    assert plant_res.status_code == 201
    plant_id = plant_res.json()["id"]

    # 2. Create inspection referencing valid plant
    insp_payload = {
        "code": "INSP-2026-API-01",
        "plant_id": plant_id,
        "inspector_name": "Marcus Vance",
        "status": "SCHEDULED",
        "ambient_temp_c": 31.0,
        "irradiance_w_m2": 780.0,
        "weather_notes": "Clear, negligible cloud cover",
        "general_notes": "IEC 62446-3 thermographic baseline",
    }
    insp_res = client.post("/api/v1/inspections/", json=insp_payload)
    assert insp_res.status_code == 201
    insp_data = insp_res.json()
    assert insp_data["code"] == "INSP-2026-API-01"
    assert insp_data["plant_id"] == plant_id
    assert insp_data["images_count"] == 0

    insp_id = insp_data["id"]

    # 3. Retrieve inspection detail
    get_res = client.get(f"/api/v1/inspections/{insp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["inspector_name"] == "Marcus Vance"
    assert get_res.json()["images"] == []

    # 4. Update inspection status
    update_res = client.put(
        f"/api/v1/inspections/{insp_id}",
        json={"status": "IN_PROGRESS", "irradiance_w_m2": 810.0},
    )
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "IN_PROGRESS"
    assert update_res.json()["irradiance_w_m2"] == 810.0


def test_create_inspection_invalid_plant(client: TestClient):
    """Test creating an inspection referencing a nonexistent plant returns 404."""
    insp_payload = {
        "code": "INSP-ORPHAN",
        "plant_id": 999999,
        "inspector_name": "Test Inspector",
    }
    res = client.post("/api/v1/inspections/", json=insp_payload)
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"]


def test_register_image_assets(client: TestClient):
    """Test registering both THERMAL_IR and OPTICAL_RGB ImageAssets for an inspection."""
    # Create plant & inspection
    p_res = client.post(
        "/api/v1/plants/",
        json={"code": "PLANT-IMG", "name": "Image Plant", "location": "Site I", "capacity_dc_mw": 20.0},
    )
    plant_id = p_res.json()["id"]

    i_res = client.post(
        "/api/v1/inspections/",
        json={"code": "INSP-IMG-01", "plant_id": plant_id, "inspector_name": "Alice Green"},
    )
    insp_id = i_res.json()["id"]

    # 1. Register Thermal IR image
    thermal_payload = {
        "image_type": "THERMAL_IR",
        "file_path": f"data/raw/inspections/{insp_id}/thermal_001.jpg",
        "original_filename": "thermal_001.jpg",
        "file_size_bytes": 1048576,
        "mime_type": "image/jpeg",
        "checksum_sha256": "c" * 64,
        "camera_make": "FLIR",
        "camera_model": "T540",
        "emissivity": 0.92,
        "reflected_temp_c": 22.0,
        "notes": "Cell hotspot detected with delta-T ~14C",
    }
    thermal_res = client.post(f"/api/v1/inspections/{insp_id}/images", json=thermal_payload)
    assert thermal_res.status_code == 201
    thermal_data = thermal_res.json()
    assert thermal_data["image_type"] == "THERMAL_IR"
    assert thermal_data["emissivity"] == 0.92

    # 2. Register Optical RGB image
    rgb_payload = {
        "image_type": "OPTICAL_RGB",
        "file_path": f"data/raw/inspections/{insp_id}/optical_001.jpg",
        "original_filename": "optical_001.jpg",
        "file_size_bytes": 4194304,
        "mime_type": "image/jpeg",
        "checksum_sha256": "d" * 64,
        "camera_make": "Hasselblad",
        "camera_model": "L2D-20c",
        "notes": "Surface RGB showing localized bird soiling",
    }
    rgb_res = client.post(f"/api/v1/inspections/{insp_id}/images", json=rgb_payload)
    assert rgb_res.status_code == 201
    assert rgb_res.json()["image_type"] == "OPTICAL_RGB"

    # 3. Invalid image type validation error
    bad_payload = thermal_payload.copy()
    bad_payload["image_type"] = "ULTRASOUND"
    bad_res = client.post(f"/api/v1/inspections/{insp_id}/images", json=bad_payload)
    assert bad_res.status_code == 422

    # 4. Verify images are listed with inspection
    detail_res = client.get(f"/api/v1/inspections/{insp_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["images_count"] == 2
    assert len(detail_data["images"]) == 2


def test_inspection_lifecycle_transitions(client: TestClient):
    """Verify inspection lifecycle state machine enforcement."""
    # 1. Create plant & inspection (defaults to SCHEDULED)
    p_res = client.post("/api/v1/plants/", json={"code": "P-LIFECYCLE", "name": "Lifecycle Solar", "location": "Test Solar Site", "capacity_dc_mw": 5.0})
    assert p_res.status_code == 201
    pid = p_res.json()["id"]

    i_res = client.post("/api/v1/inspections/", json={"code": "INSP-LIFE-01", "plant_id": pid, "inspector_name": "Lifecycle Tester"})
    assert i_res.status_code == 201
    iid = i_res.json()["id"]
    assert i_res.json()["status"] == "SCHEDULED"

    # 2. Direct jump from SCHEDULED to COMPLETED is rejected (must go through IN_PROGRESS)
    bad_jump = client.put(f"/api/v1/inspections/{iid}", json={"status": "COMPLETED"})
    assert bad_jump.status_code == 400
    assert "Invalid inspection status transition" in bad_jump.json()["detail"]

    # 3. SCHEDULED -> IN_PROGRESS is allowed
    progress_res = client.put(f"/api/v1/inspections/{iid}", json={"status": "IN_PROGRESS"})
    assert progress_res.status_code == 200
    assert progress_res.json()["status"] == "IN_PROGRESS"

    # 4. IN_PROGRESS -> COMPLETED is allowed
    completed_res = client.put(f"/api/v1/inspections/{iid}", json={"status": "COMPLETED"})
    assert completed_res.status_code == 200
    assert completed_res.json()["status"] == "COMPLETED"

    # 5. COMPLETED -> ARCHIVED is allowed
    archived_res = client.put(f"/api/v1/inspections/{iid}", json={"status": "ARCHIVED"})
    assert archived_res.status_code == 200
    assert archived_res.json()["status"] == "ARCHIVED"


def test_inspection_auto_advances_on_media_upload(client: TestClient):
    """Verify that uploading or registering media automatically advances status from SCHEDULED to IN_PROGRESS."""
    p_res = client.post("/api/v1/plants/", json={"code": "P-AUTOADV", "name": "Auto Advance Solar", "location": "Test Solar Site 2", "capacity_dc_mw": 8.0})
    assert p_res.status_code == 201
    pid = p_res.json()["id"]

    i_res = client.post("/api/v1/inspections/", json={"code": "INSP-AUTOADV-01", "plant_id": pid, "inspector_name": "Adv Tester"})
    iid = i_res.json()["id"]
    assert i_res.json()["status"] == "SCHEDULED"

    # Register an image asset
    reg_res = client.post(
        f"/api/v1/inspections/{iid}/images",
        json={
            "image_type": "THERMAL_IR",
            "file_path": f"data/raw/inspections/{iid}/thermal.jpg",
            "original_filename": "thermal.jpg",
            "file_size_bytes": 1024,
            "mime_type": "image/jpeg",
            "checksum_sha256": "f" * 64,
        },
    )
    assert reg_res.status_code == 201

    # Check inspection status has auto-advanced to IN_PROGRESS
    check_res = client.get(f"/api/v1/inspections/{iid}")
    assert check_res.status_code == 200
    assert check_res.json()["status"] == "IN_PROGRESS"
