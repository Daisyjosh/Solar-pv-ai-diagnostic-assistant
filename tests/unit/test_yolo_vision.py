"""Unit tests for YOLO Vision Object Detection Service and Endpoints."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from frontend.api_client import BackendAPIClient
from src.database.models import ImageAsset, Inspection, Plant
from src.vision.yolo_service import (
    BoundingBox,
    YOLODetection,
    YOLOAnalysisResult,
    YOLOVisionService,
    get_yolo_service,
)


def _create_sample_jpeg_bytes() -> bytes:
    """Generate minimal valid JPEG bytes in memory."""
    buf = io.BytesIO()
    im = Image.new("RGB", (40, 24), color=(200, 100, 50))
    im.save(buf, format="JPEG")
    return buf.getvalue()


class TestYOLOVisionService:
    """Test suite for YOLOVisionService logic, status handling, and inference."""

    def test_status_when_weights_missing(self, tmp_path):
        non_existent_weights = tmp_path / "missing_best.pt"
        service = YOLOVisionService(model_path=str(non_existent_weights))
        status = service.get_status()
        assert status["status"] == "WEIGHTS_UNAVAILABLE"
        assert status["is_configured"] is False
        assert "weights found" in status["message"].lower()
        assert status["model_path"] == str(non_existent_weights)

    def test_detect_image_weights_missing(self, tmp_path):
        non_existent_weights = tmp_path / "missing_best.pt"
        service = YOLOVisionService(model_path=str(non_existent_weights))
        img_bytes = _create_sample_jpeg_bytes()
        res = service.detect_image(img_bytes)
        assert res.status == "WEIGHTS_UNAVAILABLE"
        assert res.is_configured is False
        assert res.detections == []
        assert res.total_detections == 0
        assert "not found" in res.message

    def test_detect_image_empty_bytes(self, tmp_path):
        non_existent_weights = tmp_path / "missing_best.pt"
        service = YOLOVisionService(model_path=str(non_existent_weights))
        res = service.detect_image(b"")
        assert res.status == "IMAGE_ERROR"
        assert "empty image bytes" in res.message.lower()

    def test_detect_asset_file_not_found(self):
        service = YOLOVisionService(model_path="dummy.pt")
        res = service.detect_asset("non_existent_path_xyz.jpg")
        assert res.status == "FILE_NOT_FOUND"
        assert "does not exist" in res.message.lower()

    def test_detect_asset_local_file_success_mock(self, tmp_path):
        sample_file = tmp_path / "test_crop.jpg"
        sample_file.write_bytes(_create_sample_jpeg_bytes())

        service = YOLOVisionService(model_path="dummy.pt")
        with patch.object(service, "detect_image") as mock_detect:
            mock_detect.return_value = YOLOAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_path="dummy.pt",
                detections=[
                    YOLODetection(
                        class_id=0,
                        class_name="Hotspot",
                        confidence=0.88,
                        bbox=BoundingBox(xmin=5.0, ymin=5.0, xmax=20.0, ymax=18.0),
                    )
                ],
                total_detections=1,
            )
            res = service.detect_asset(sample_file)
            assert res.status == "SUCCESS"
            assert res.total_detections == 1
            assert res.detections[0].class_name == "Hotspot"
            mock_detect.assert_called_once()

    def test_dependency_unavailable_handling(self, tmp_path):
        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"dummy_weights")
        service = YOLOVisionService(model_path=str(weights_file))

        with patch.object(service, "_check_dependencies", return_value=False):
            status = service.get_status()
            assert status["status"] == "DEPENDENCY_UNAVAILABLE"
            assert status["is_configured"] is False

            res = service.detect_image(_create_sample_jpeg_bytes())
            assert res.status == "DEPENDENCY_UNAVAILABLE"
            assert res.is_configured is False
            assert "ultralytics" in res.message.lower()

    def test_detect_image_success_mocked_ultralytics(self, tmp_path):
        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"dummy_weights")
        service = YOLOVisionService(model_path=str(weights_file), confidence_threshold=0.3)

        mock_box = MagicMock()
        mock_box.xyxy = [np.array([2.0, 3.0, 15.0, 20.0])]
        mock_box.conf = [np.array(0.92)]
        mock_box.cls = [np.array(1)]

        mock_boxes = MagicMock()
        mock_boxes.__iter__.return_value = [mock_box]
        mock_boxes.__len__.return_value = 1

        mock_pred = MagicMock()
        mock_pred.boxes = mock_boxes
        mock_pred.names = {0: "Cell", 1: "Hotspot", 2: "Cracking"}
        mock_pred.plot.return_value = np.zeros((24, 40, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_pred]

        with patch.object(service, "_check_dependencies", return_value=True), \
             patch.object(service, "_load_model", return_value=mock_model):
            img_bytes = _create_sample_jpeg_bytes()
            res = service.detect_image(img_bytes)

            assert res.status == "SUCCESS"
            assert res.is_configured is True
            assert res.total_detections == 1
            assert len(res.detections) == 1
            det = res.detections[0]
            assert det.class_name == "Hotspot"
            assert pytest.approx(det.confidence, 0.01) == 0.92
            assert pytest.approx(det.bbox.xmin, 0.01) == 2.0
            assert pytest.approx(det.bbox.ymax, 0.01) == 20.0
            assert res.annotated_image_base64 is not None

    def test_detect_image_zero_detections_mock(self, tmp_path):
        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"dummy_weights")
        service = YOLOVisionService(model_path=str(weights_file))

        mock_boxes = MagicMock()
        mock_boxes.__iter__.return_value = []
        mock_boxes.__len__.return_value = 0

        mock_pred = MagicMock()
        mock_pred.boxes = mock_boxes
        mock_pred.names = {}

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_pred]

        with patch.object(service, "_check_dependencies", return_value=True), \
             patch.object(service, "_load_model", return_value=mock_model):
            res = service.detect_image(_create_sample_jpeg_bytes())
            assert res.status == "SUCCESS"
            assert res.total_detections == 0
            assert res.detections == []
            assert "no anomalies detected" in res.message.lower()


class TestYOLOApiEndpoints:
    """Test FastAPI endpoints for YOLO Vision."""

    def test_yolo_status_endpoint(self, client: TestClient):
        response = client.get("/api/v1/vision/yolo/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "is_configured" in data
        assert "model_path" in data
        assert "confidence_threshold" in data
        assert "iou_threshold" in data

    def test_yolo_detect_upload_endpoint(self, client: TestClient):
        img_bytes = _create_sample_jpeg_bytes()

        with patch.object(YOLOVisionService, "detect_image") as mock_detect:
            mock_detect.return_value = YOLOAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_path="models/yolo/active/best.pt",
                detections=[
                    YOLODetection(
                        class_id=0,
                        class_name="Diode",
                        confidence=0.85,
                        bbox=BoundingBox(xmin=10.0, ymin=5.0, xmax=30.0, ymax=22.0),
                    )
                ],
                total_detections=1,
                message="Mock success",
            )

            files = {"file": ("test_thermal.jpg", img_bytes, "image/jpeg")}
            response = client.post("/api/v1/vision/yolo/detect-upload", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["total_detections"] == 1
            assert data["detections"][0]["class_name"] == "Diode"

    def test_yolo_detect_asset_endpoint_not_found(self, client: TestClient):
        response = client.post("/api/v1/vision/yolo/detect/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_yolo_detect_asset_endpoint_success(self, client: TestClient, db_session, test_settings):
        # Create a plant, inspection, and image asset in db
        plant = Plant(
            code="PLANT-YOLO-TEST",
            name="Test Solar Plant YOLO",
            location="Test Location YOLO",
            capacity_dc_mw=2.0,
        )
        db_session.add(plant)
        db_session.commit()
        db_session.refresh(plant)

        insp = Inspection(
            code="INSP-YOLO-TEST",
            plant_id=plant.id,
            inspector_name="YOLO Inspector",
            ambient_temp_c=25.0,
            irradiance_w_m2=850.0,
            general_notes="YOLO test inspection",
        )
        db_session.add(insp)
        db_session.commit()
        db_session.refresh(insp)

        storage_dir = Path(test_settings.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        sample_img = storage_dir / "yolo_asset.jpg"
        sample_img.write_bytes(_create_sample_jpeg_bytes())

        asset = ImageAsset(
            inspection_id=insp.id,
            image_type="THERMAL",
            original_filename="yolo_asset.jpg",
            file_path="yolo_asset.jpg",
            file_size_bytes=len(_create_sample_jpeg_bytes()),
            mime_type="image/jpeg",
            checksum_sha256="b" * 64,
        )
        db_session.add(asset)
        db_session.commit()
        db_session.refresh(asset)

        with patch.object(ImageAsset, "resolve_absolute_path", return_value=sample_img), \
             patch.object(YOLOVisionService, "detect_asset") as mock_detect:
            mock_detect.return_value = YOLOAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_path="models/yolo/active/best.pt",
                detections=[
                    YOLODetection(
                        class_id=1,
                        class_name="Hotspot",
                        confidence=0.91,
                        bbox=BoundingBox(xmin=2.0, ymin=4.0, xmax=18.0, ymax=20.0),
                    )
                ],
                total_detections=1,
                message="Asset detection success",
            )

            response = client.post(f"/api/v1/vision/yolo/detect/{asset.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["total_detections"] == 1
            assert data["detections"][0]["class_name"] == "Hotspot"


class TestBackendAPIClientYOLO:
    """Test frontend BackendAPIClient YOLO methods."""

    @patch("httpx.Client.get")
    def test_get_yolo_status_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "WEIGHTS_UNAVAILABLE", "is_configured": False}
        mock_get.return_value = mock_resp

        client = BackendAPIClient(base_url="http://127.0.0.1:8000")
        res = client.get_yolo_status()
        assert res["status"] == "WEIGHTS_UNAVAILABLE"
        assert res["is_configured"] is False

    @patch("httpx.Client.get")
    def test_get_yolo_status_network_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        client = BackendAPIClient(base_url="http://127.0.0.1:8000")
        res = client.get_yolo_status()
        assert res["status"] == "NETWORK_ERROR"
        assert res["is_configured"] is False

    @patch("httpx.Client.post")
    def test_detect_yolo_asset(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "SUCCESS", "total_detections": 2}
        mock_post.return_value = mock_resp

        client = BackendAPIClient(base_url="http://127.0.0.1:8000")
        res = client.detect_yolo_asset(123)
        assert res["status"] == "SUCCESS"
        assert res["total_detections"] == 2

    @patch("httpx.Client.post")
    def test_detect_yolo_upload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "SUCCESS", "total_detections": 1}
        mock_post.return_value = mock_resp

        client = BackendAPIClient(base_url="http://127.0.0.1:8000")
        res = client.detect_yolo_upload(b"test_bytes", "test.jpg")
        assert res["status"] == "SUCCESS"
        assert res["total_detections"] == 1
