"""Integration tests for asset hierarchy endpoints and secure multipart image uploads."""

import io
from fastapi.testclient import TestClient
from PIL import Image


def _create_jpeg_bytes() -> bytes:
    """Helper generating clean JPEG test bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (120, 120), color=(255, 60, 20))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_png_bytes() -> bytes:
    """Helper generating clean PNG test bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (80, 80), color=(10, 150, 220))
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_asset_hierarchy_workflow(client: TestClient):
    """Test full asset hierarchy creation and validation: Plant -> Inverter -> String -> Module."""
    # 1. Create Plant
    p_res = client.post(
        "/api/v1/plants/",
        json={"code": "PLANT-HIER-API", "name": "Hierarchy Test Site", "location": "Site H", "capacity_dc_mw": 50.0},
    )
    assert p_res.status_code == 201
    plant_id = p_res.json()["id"]

    # 2. Create Inverter
    inv_payload = {"code": "INV-01", "model": "SMA 2500", "rated_power_kw": 2500.0}
    inv_res = client.post(f"/api/v1/plants/{plant_id}/inverters", json=inv_payload)
    assert inv_res.status_code == 201
    inv_id = inv_res.json()["id"]
    assert inv_res.json()["code"] == "INV-01"

    # Duplicate inverter in same plant rejected
    inv_dup = client.post(f"/api/v1/plants/{plant_id}/inverters", json=inv_payload)
    assert inv_dup.status_code == 409

    # List inverters
    inv_list = client.get(f"/api/v1/plants/{plant_id}/inverters")
    assert inv_list.status_code == 200
    assert inv_list.json()["total"] == 1

    # 3. Create PV String under Inverter
    str_payload = {"code": "STR-01-A", "modules_count": 24, "rated_voc_v": 1100.0, "rated_isc_a": 11.5}
    str_res = client.post(f"/api/v1/inverters/{inv_id}/strings", json=str_payload)
    assert str_res.status_code == 201
    str_id = str_res.json()["id"]

    # Duplicate string rejected
    str_dup = client.post(f"/api/v1/inverters/{inv_id}/strings", json=str_payload)
    assert str_dup.status_code == 409

    # List strings
    str_list = client.get(f"/api/v1/inverters/{inv_id}/strings")
    assert str_list.status_code == 200
    assert str_list.json()["total"] == 1

    # 4. Create PV Module under String
    mod_payload = {
        "code": "MOD-01-A-01",
        "position_in_string": 1,
        "manufacturer": "JinkoSolar",
        "model": "Tiger Pro",
        "rated_power_w": 535.0,
    }
    mod_res = client.post(f"/api/v1/strings/{str_id}/modules", json=mod_payload)
    assert mod_res.status_code == 201
    assert mod_res.json()["position_in_string"] == 1

    # Position exceeding string modules_count rejected (modules_count is 24)
    bad_mod = mod_payload.copy()
    bad_mod["code"] = "MOD-EXCEED"
    bad_mod["position_in_string"] = 25
    bad_res = client.post(f"/api/v1/strings/{str_id}/modules", json=bad_mod)
    assert bad_res.status_code == 422

    # Duplicate position rejected
    dup_mod = mod_payload.copy()
    dup_mod["code"] = "MOD-DUP-POS"
    dup_res = client.post(f"/api/v1/strings/{str_id}/modules", json=dup_mod)
    assert dup_res.status_code == 409

    # List modules
    mod_list = client.get(f"/api/v1/strings/{str_id}/modules")
    assert mod_list.status_code == 200
    assert mod_list.json()["total"] == 1


def test_multipart_image_upload_and_download(client: TestClient):
    """Test streaming image upload, retrieval, duplicate rejection, and deletion."""
    # 1. Setup plant & inspection
    p_res = client.post(
        "/api/v1/plants/",
        json={"code": "PLANT-UPLOAD-TEST", "name": "Upload Test Plant", "location": "Site U", "capacity_dc_mw": 10.0},
    )
    plant_id = p_res.json()["id"]

    insp_res = client.post(
        "/api/v1/inspections/",
        json={"code": "INSP-UPLOAD-01", "plant_id": plant_id, "inspector_name": "Dr. Kim"},
    )
    insp_id = insp_res.json()["id"]

    jpeg_data = _create_jpeg_bytes()
    png_data = _create_png_bytes()

    # 2. Upload Thermal IR image
    thermal_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("flir_thermal_01.jpg", jpeg_data, "image/jpeg")},
        data={"image_type": "THERMAL_IR", "notes": "Thermal hotspot on string 1"},
    )
    assert thermal_res.status_code == 201
    thermal_json = thermal_res.json()
    assert thermal_json["image_type"] == "THERMAL_IR"
    assert thermal_json["original_filename"] == "flir_thermal_01.jpg"
    assert len(thermal_json["checksum_sha256"]) == 64
    thermal_id = thermal_json["id"]

    # 3. Duplicate upload of identical file to same inspection rejected (409 Conflict)
    dup_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("flir_thermal_01_duplicate.jpg", jpeg_data, "image/jpeg")},
        data={"image_type": "THERMAL_IR"},
    )
    assert dup_res.status_code == 409
    assert "Duplicate image" in dup_res.json()["detail"]

    # 4. Upload Optical RGB image
    optical_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("sony_rgb_01.png", png_data, "image/png")},
        data={"image_type": "OPTICAL_RGB", "notes": "Optical surface confirmation"},
    )
    assert optical_res.status_code == 201
    assert optical_res.json()["image_type"] == "OPTICAL_RGB"
    optical_id = optical_res.json()["id"]

    # 5. Download image file
    download_res = client.get(f"/api/v1/inspections/{insp_id}/images/{thermal_id}/file")
    assert download_res.status_code == 200
    assert download_res.content == jpeg_data
    assert download_res.headers["content-type"] == "image/jpeg"

    # Download with wrong inspection ID rejected
    wrong_insp_res = client.get(f"/api/v1/inspections/99999/images/{thermal_id}/file")
    assert wrong_insp_res.status_code == 404

    # 6. Reject invalid non-image file
    bad_upload = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("malicious.txt", b"plain text script", "text/plain")},
        data={"image_type": "OPTICAL_RGB"},
    )
    assert bad_upload.status_code == 400

    # 7. Delete image asset
    del_res = client.delete(f"/api/v1/inspections/{insp_id}/images/{optical_id}")
    assert del_res.status_code == 204

    # Verify deleted
    get_del = client.get(f"/api/v1/inspections/{insp_id}/images/{optical_id}/file")
    assert get_del.status_code == 404
