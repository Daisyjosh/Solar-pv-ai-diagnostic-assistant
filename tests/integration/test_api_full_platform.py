"""End-to-End API integration tests for Phase 5.2 through Phase 11."""

import io
from PIL import Image
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.database.models.plant import Plant
from src.database.session import get_db

client = TestClient(app)


def test_vision_api_model_status_and_metrics():
    """Verify vision lifecycle endpoints return live model data."""
    res = client.get("/api/v1/vision/model-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "MODEL_READY"
    assert data["model_version"] is not None
    assert len(data["class_names"]) == 12

    res_metrics = client.get("/api/v1/vision/evaluation-metrics")
    assert res_metrics.status_code == 200
    m_data = res_metrics.json()
    assert m_data["status"] == "SUCCESS"
    assert "test_metrics" in m_data["metrics"]


def test_dashboard_summary_api():
    """Verify high-level dashboard aggregation endpoint."""
    res = client.get("/api/v1/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "total_plants" in data["kpis"]
    assert "fault_distribution" in data


def test_telemetry_scada_status_api():
    """Verify SCADA provider connection status endpoint."""
    res = client.get("/api/v1/telemetry/scada-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("UNCONFIGURED", "DISCONNECTED", "CONNECTED")
    assert "supported_protocols" in data


def test_chat_assistant_api():
    """Verify contextual assistant dialogue endpoint for general and grounded inspection queries."""
    # 1. General query (no inspection context)
    payload_gen = {
        "query": "What is the minimum irradiance for thermal inspection under IEC 62446-3?",
        "session_id": "test_api_session_01",
    }
    res_gen = client.post("/api/v1/chat/message", json=payload_gen)
    assert res_gen.status_code == 200
    data_gen = res_gen.json()
    assert "response" in data_gen
    assert len(data_gen["citations"]) >= 1
    assert data_gen["has_inspection_context"] is False

    # 2. Grounded query on Inspection 10
    payload_insp = {
        "query": "What is the fault classification and what additional measurements are needed for this inspection?",
        "session_id": "test_api_session_02",
        "inspection_id": 10,
    }
    res_insp = client.post("/api/v1/chat/message", json=payload_insp)
    assert res_insp.status_code == 200
    data_insp = res_insp.json()
    assert data_insp["has_inspection_context"] is True
    assert data_insp["has_diagnosis_context"] is True
    resp_text = data_insp["response"]
    assert "INSP-20261011-1223" in resp_text
    assert "Daisy Solar Test Plant" in resp_text
    assert "INCONCLUSIVE" in resp_text
    assert "Uncorroborated AI Hypothesis" in resp_text
    assert "Unquantified" in resp_text
    assert "Electroluminescence (EL)" in resp_text or "dark EL" in resp_text.lower()

    # 3. Validation error on empty query
    res_err = client.post("/api/v1/chat/message", json={"query": "   "})
    assert res_err.status_code == 400
    assert "Query cannot be empty" in res_err.json()["detail"]



def test_inspection_diagnosis_and_pdf_report_lifecycle():
    """Verify end-to-end lifecycle: Plant -> Inspection -> Diagnosis -> Review -> PDF Report."""
    # 1. Create unique plant
    p_code = f"P-E2E-{Image.new('L', (1, 1)).getpixel((0,0))}"
    import uuid
    p_code = f"P-E2E-{uuid.uuid4().hex[:6]}"
    p_res = client.post("/api/v1/plants/", json={
        "code": p_code,
        "name": "Integration Test Plant",
        "location": "Nevada",
        "capacity_dc_mw": 20.0,
    })
    assert p_res.status_code == 201
    plant_id = p_res.json()["id"]

    # 2. Create Inspection
    insp_code = f"INSP-E2E-{uuid.uuid4().hex[:6]}"
    i_res = client.post("/api/v1/inspections/", json={
        "code": insp_code,
        "plant_id": plant_id,
        "inspector_name": "QA Lead",
        "ambient_temp_c": 28.0,
        "irradiance_w_m2": 760.0,
    })
    assert i_res.status_code == 201
    insp_id = i_res.json()["id"]

    # 3. Trigger Diagnosis
    d_res = client.post(f"/api/v1/inspections/{insp_id}/diagnose")
    assert d_res.status_code == 200
    diag_data = d_res.json()
    assert "fault_category" in diag_data
    assert "severity" in diag_data
    diag_id = diag_data["id"]

    # 4. Review Diagnosis
    rev_res = client.put(f"/api/v1/diagnoses/{diag_id}/review", json={
        "new_status": "CONFIRMED",
        "reviewed_by": "Senior O&M Lead",
        "engineer_notes": "Reviewed and verified.",
    })
    assert rev_res.status_code == 200
    assert rev_res.json()["review_status"] == "CONFIRMED"

    # 5. Download PDF Report
    pdf_res = client.get(f"/api/v1/inspections/{insp_id}/report/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000
    assert pdf_res.content.startswith(b"%PDF")


def test_uploaded_image_retrieval_and_inference_lifecycle():
    """Regression test: verify uploaded image file is correctly stored, resolved, and classified by AI vision inference."""
    import uuid
    # 1. Create plant & inspection
    p_code = f"P-INF-{uuid.uuid4().hex[:6]}"
    p_res = client.post("/api/v1/plants/", json={
        "code": p_code,
        "name": "Inference Regression Plant",
        "location": "Arizona",
        "capacity_dc_mw": 15.0,
    })
    assert p_res.status_code == 201
    plant_id = p_res.json()["id"]

    i_code = f"INSP-INF-{uuid.uuid4().hex[:6]}"
    i_res = client.post("/api/v1/inspections/", json={
        "code": i_code,
        "plant_id": plant_id,
        "inspector_name": "Vision QA Lead",
        "ambient_temp_c": 30.0,
        "irradiance_w_m2": 800.0,
    })
    assert i_res.status_code == 201
    insp_id = i_res.json()["id"]

    # 2. Generate a valid 24x40 test thermal crop
    buf = io.BytesIO()
    test_img = Image.new("L", (24, 40), color=128)
    test_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # 3. Upload image via multipart upload
    up_res = client.post(
        f"/api/v1/inspections/{insp_id}/upload",
        files={"file": ("test_thermal_crop.png", png_bytes, "image/png")},
        data={"image_type": "THERMAL_IR", "notes": "Regression test crop"},
    )
    assert up_res.status_code == 201
    asset_id = up_res.json()["id"]
    stored_path = up_res.json()["file_path"]
    assert stored_path.startswith("raw/inspections/")

    # 4. Trigger AI inference on the uploaded asset
    pred_res = client.post(f"/api/v1/vision/predict/{asset_id}")
    assert pred_res.status_code == 200, f"Inference failed with status {pred_res.status_code}: {pred_res.text}"
    pred_data = pred_res.json()

    assert pred_data["status"] == "MODEL_READY"
    assert "predicted_class" in pred_data
    assert pred_data["predicted_class"] is not None
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert len(pred_data["probabilities"]) == 12
    assert pred_data["algorithm"] == "HistGradientBoostingClassifier"

    # 5. Trigger automated diagnosis to ensure vision predictions corroborate
    diag_res = client.post(f"/api/v1/inspections/{insp_id}/diagnose")
    assert diag_res.status_code == 200
    diag_data = diag_res.json()
    assert diag_data["model_prediction"] == pred_data["predicted_class"]


def test_asset_7_inference_regression():
    """Verify that existing Asset #7 can be retrieved from disk and classified by the active model."""
    from src.database.session import SessionLocal
    from src.database.models.image_asset import ImageAsset

    db = SessionLocal()
    asset_7 = db.get(ImageAsset, 7)
    db.close()

    if asset_7 is not None:
        # File retrieval check
        resolved_path = asset_7.resolve_absolute_path()
        assert resolved_path.is_file(), f"Asset #7 file does not exist at {resolved_path}"
        assert resolved_path.stat().st_size > 0

        # Model inference check
        pred_res = client.post("/api/v1/vision/predict/7")
        assert pred_res.status_code == 200
        data = pred_res.json()
        assert data["status"] == "MODEL_READY"
        assert data["predicted_class"] is not None
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["is_low_confidence"] is True
        assert len(data["top_candidates"]) >= 2
        # Verify top candidates reflect close competition between Cracking and candidate classes
        classes = [c["class_name"] for c in data["top_candidates"][:3]]
        assert "Cracking" in classes and ("Cell-Multi" in classes or "No-Anomaly" in classes)
        assert "requires physical electrical / EL corroboration" in data["message"]

        # Ensure inspection 10 diagnosis is marked INCONCLUSIVE and unquantified loss
        diag_res = client.get("/api/v1/inspections/10/diagnosis")
        if diag_res.status_code == 200:
            diag_data = diag_res.json()
            assert diag_data["severity"] == "INCONCLUSIVE"
            assert diag_data["power_loss_percent"] is None
            assert "Uncorroborated AI Hypothesis" in diag_data["fault_category"]



