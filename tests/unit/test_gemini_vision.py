"""Unit tests for Gemini Multimodal Vision Analysis Service and Endpoints."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.database.models import ImageAsset, Inspection, Plant
from src.vision.gemini_service import (
    GeminiAnalysisResult,
    GeminiVisionService,
    get_gemini_service,
)


def _create_sample_jpeg_bytes() -> bytes:
    """Generate minimal valid JPEG bytes in memory."""
    buf = io.BytesIO()
    im = Image.new("RGB", (40, 24), color=(200, 100, 50))
    im.save(buf, format="JPEG")
    return buf.getvalue()


class TestGeminiVisionService:
    """Test suite for GeminiVisionService logic and error handling."""

    def test_status_when_api_key_missing(self):
        service = GeminiVisionService(api_key="")
        status = service.get_status()
        assert status["is_configured"] is False
        assert "GEMINI_API_KEY" in status["message"]

    def test_status_when_api_key_configured(self):
        service = GeminiVisionService(api_key="test-api-key-12345", model_name="gemini-3.5-flash")
        status = service.get_status()
        assert status["is_configured"] is True
        assert status["model_name"] == "gemini-3.5-flash"
        assert "configured" in status["message"]

    def test_analyze_image_missing_api_key(self):
        service = GeminiVisionService(api_key="")
        result = service.analyze_image(_create_sample_jpeg_bytes())
        assert result.status == "API_KEY_MISSING"
        assert result.is_configured is False
        assert result.analysis_text is None
        assert "API key is not configured" in result.message

    def test_analyze_image_empty_bytes(self):
        service = GeminiVisionService(api_key="valid-dummy-key")
        result = service.analyze_image(b"")
        assert result.status == "IMAGE_ERROR"
        assert "Cannot analyze empty image bytes" in result.message

    @patch("google.genai.Client")
    def test_analyze_image_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = (
            "### 1. Visible Thermal Patterns & Locations\n"
            "Localized hot spot detected on lower-middle cell.\n\n"
            "### 2. Possible Fault Categories\n"
            "- **Cell**: Single-cell hot spot on c-Si module.\n\n"
            "### 3. Alternative Explanations & Uncertainty\n"
            "Potential localized soiling vs internal cell shunting.\n\n"
            "### 4. Recommended Verification Steps\n"
            "Perform I-V curve tracing and dark EL imaging.\n\n"
            "### 5. Image Suitability & Limitations\n"
            "Uncalibrated 8-bit false-color rendering."
        )
        mock_client.models.generate_content.return_value = mock_response

        service = GeminiVisionService(api_key="valid-dummy-key", model_name="gemini-3.5-flash")
        img_bytes = _create_sample_jpeg_bytes()
        result = service.analyze_image(img_bytes, mime_type="image/jpeg")

        assert result.status == "SUCCESS"
        assert result.is_configured is True
        assert result.model_name == "gemini-3.5-flash"
        assert result.analysis_text is not None
        assert "Visible Thermal Patterns" in result.analysis_text
        assert "Possible Fault Categories" in result.analysis_text
        assert "AI-generated visual hypothesis" in result.disclaimer

    @patch("google.genai.Client")
    def test_analyze_image_rate_limit_429(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")

        service = GeminiVisionService(api_key="valid-dummy-key")
        result = service.analyze_image(_create_sample_jpeg_bytes())

        assert result.status == "RATE_LIMIT_EXCEEDED"
        assert "rate limit" in result.message.lower()

    @patch("google.genai.Client")
    def test_analyze_image_service_busy_503(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")

        service = GeminiVisionService(api_key="valid-dummy-key", model_name="gemini-3.5-flash")
        result = service.analyze_image(_create_sample_jpeg_bytes())

        assert result.status == "SERVICE_BUSY"
        assert "high demand" in result.message

    @patch("google.genai.Client")
    def test_analyze_image_model_not_found_404(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("404 NOT_FOUND: Model gemini-invalid does not exist")

        service = GeminiVisionService(api_key="valid-dummy-key", model_name="gemini-invalid")
        result = service.analyze_image(_create_sample_jpeg_bytes())

        assert result.status == "MODEL_ERROR"
        assert "gemini-invalid" in result.message

    @patch("google.genai.Client")
    def test_analyze_image_network_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Failed to connect to host: Connection timed out")

        service = GeminiVisionService(api_key="valid-dummy-key")
        result = service.analyze_image(_create_sample_jpeg_bytes())

        assert result.status == "NETWORK_ERROR"
        assert "Network error" in result.message

    @patch("google.genai.Client")
    def test_analyze_image_invalid_api_key(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID: API key not valid. Please pass a valid API key.")

        service = GeminiVisionService(api_key="invalid-key")
        result = service.analyze_image(_create_sample_jpeg_bytes())

        assert result.status == "INVALID_API_KEY"
        assert "invalid or expired" in result.message

    def test_analyze_asset_file_not_found(self):
        service = GeminiVisionService(api_key="valid-dummy-key")
        result = service.analyze_asset("non_existent_path_to_image_xyz.jpg")
        assert result.status == "FILE_NOT_FOUND"

    def test_analyze_asset_local_file_success(self, tmp_path):
        sample_file = tmp_path / "test_thermal.jpg"
        sample_file.write_bytes(_create_sample_jpeg_bytes())

        service = GeminiVisionService(api_key="dummy-key")
        with patch.object(service, "analyze_image") as mock_analyze:
            mock_analyze.return_value = GeminiAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_name="gemini-2.0-flash",
                analysis_text="Thermal analysis ok",
            )
            result = service.analyze_asset(sample_file)
            assert result.status == "SUCCESS"
            mock_analyze.assert_called_once()


class TestGeminiApiEndpoints:
    """Test FastAPI endpoints for Gemini Vision."""

    def test_gemini_status_endpoint(self, client: TestClient):
        response = client.get("/api/v1/vision/gemini/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_configured" in data
        assert "model_name" in data
        assert "provider" in data

    def test_gemini_analyze_upload_endpoint(self, client: TestClient):
        img_bytes = _create_sample_jpeg_bytes()

        with patch.object(GeminiVisionService, "analyze_image") as mock_analyze:
            mock_analyze.return_value = GeminiAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_name="gemini-2.0-flash",
                analysis_text="Visual analysis from upload test.",
                message="Completed successfully.",
            )

            files = {"file": ("test_thermal.jpg", img_bytes, "image/jpeg")}
            response = client.post("/api/v1/vision/gemini/analyze-upload", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["analysis_text"] == "Visual analysis from upload test."

    def test_gemini_analyze_asset_endpoint_not_found(self, client: TestClient):
        response = client.post("/api/v1/vision/gemini/analyze/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_gemini_analyze_asset_endpoint_success(self, client: TestClient, db_session, test_settings):
        # Create a plant, inspection, and image asset in db
        plant = Plant(
            code="PLANT-GEMINI-TEST",
            name="Test Solar Plant",
            location="Test Location",
            capacity_dc_mw=1.5,
        )
        db_session.add(plant)
        db_session.commit()
        db_session.refresh(plant)

        insp = Inspection(
            code="INSP-GEMINI-TEST",
            plant_id=plant.id,
            inspector_name="Test Inspector",
            ambient_temp_c=25.0,
            irradiance_w_m2=800.0,
            general_notes="Gemini test inspection",
        )
        db_session.add(insp)
        db_session.commit()
        db_session.refresh(insp)

        storage_dir = Path(test_settings.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        sample_img = storage_dir / "gemini_asset.jpg"
        sample_img.write_bytes(_create_sample_jpeg_bytes())

        asset = ImageAsset(
            inspection_id=insp.id,
            image_type="THERMAL",
            original_filename="gemini_asset.jpg",
            file_path="gemini_asset.jpg",
            file_size_bytes=len(_create_sample_jpeg_bytes()),
            mime_type="image/jpeg",
            checksum_sha256="a" * 64,
        )
        db_session.add(asset)
        db_session.commit()
        db_session.refresh(asset)

        with patch.object(ImageAsset, "resolve_absolute_path", return_value=sample_img), \
             patch.object(GeminiVisionService, "analyze_asset") as mock_analyze:
            mock_analyze.return_value = GeminiAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_name="gemini-2.0-flash",
                analysis_text="Asset analyzed successfully.",
                message="Success.",
            )

            response = client.post(f"/api/v1/vision/gemini/analyze/{asset.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["analysis_text"] == "Asset analyzed successfully."
