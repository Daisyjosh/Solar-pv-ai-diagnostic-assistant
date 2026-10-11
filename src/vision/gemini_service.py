"""Gemini Multimodal Vision Analysis Service for Solar PV Thermography.

Provides domain-focused qualitative vision analysis of thermal PV imagery
using Google Gemini API (free tier).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_SOLAR_PV_PROMPT = """You are an expert Solar Photovoltaic (PV) thermography analysis specialist assisting a certified field engineer.

Examine the provided infrared/thermal capture of a solar photovoltaic module or array.

Analyze the image strictly adhering to IEC 62446-3 (Outdoor infrared thermography of PV plants) and IEA-PVPS Task 13 guidelines. Provide your assessment under the following exact sections:

### 1. Visible Thermal Patterns & Locations
Describe observable visual signatures: localized hotspots, heated substrings/bands, patch anomalies, gradient lines, cold disconnected cells, or uniform heating, noting their approximate location within the module matrix.

### 2. Possible Fault Categories
Reference ONLY the following 12 benchmark classes where visual evidence supports them:
- **Cell**: Single-cell hot spot on crystalline silicon (c-Si) module
- **Cell-Multi**: Multi-cell hot spots on crystalline silicon module
- **Cracking**: Module surface cracking or wafer microcrack thermal dissipation
- **Diode**: Activated bypass diode (characteristic 1/3 module heating)
- **Diode-Multi**: Multiple activated bypass diodes (typically 2/3 module heating)
- **Hot-Spot**: Hot spot on a thin-film module (CdTe / CIGS monolithic scribes)
- **Hot-Spot-Multi**: Multiple hot spots on a thin-film module
- **No-Anomaly**: Nominal, uniform thermal distribution across cell matrix
- **Offline-Module**: Entire module uniformly heated or open-circuit
- **Shadowing**: External shadowing from vegetation, structures, or adjacent rows
- **Soiling**: Localized dust, bird dropping, or surface debris fouling
- **Vegetation**: Direct vegetation obstruction or encroachment
State which category (or categories) best explains the visual thermal signature, or state if no class fits.

### 3. Alternative Explanations & Uncertainty
Explain alternative hypotheses (e.g. single-cell hotspot vs wafer microcrack vs localized bird-dropping reverse bias) and sources of visual ambiguity.

### 4. Recommended Verification Steps
List specific physical or electrical checks required to confirm the root cause (e.g. night-time dark-field Electroluminescence (EL) imaging, module I-V curve tracing, open-circuit voltage Voc check, bypass diode forward/reverse resistance measurement, in-plane irradiance pyranometer verification).

### 5. Image Suitability & Limitations
State whether this image is unsuitable or limited for definitive interpretation due to:
- Lack of calibrated radiometric data (8-bit rendered false-color palette vs native radiometry)
- Low spatial resolution or pixel artifacts
- Uncropped full-scene capture vs isolated single-module crop
- Unconfirmed solar irradiance (< 600 W/m² baseline requirement).

IMPORTANT RULES:
- Clearly state that this is an AI-generated visual hypothesis, NOT a confirmed physical diagnosis.
- Do NOT invent or fabricate quantitative temperature measurements, power loss percentages, severities, or numeric confidence scores without calibrated sensor data.
"""


class GeminiAnalysisResult(BaseModel):
    """Structured response from Gemini vision analysis."""

    status: str = Field(description="Operational status: SUCCESS, API_KEY_MISSING, RATE_LIMIT_EXCEEDED, etc.")
    is_configured: bool = Field(default=False, description="Whether GEMINI_API_KEY is available in backend")
    model_name: Optional[str] = Field(default=None, description="Gemini model used for analysis")
    analysis_text: Optional[str] = Field(default=None, description="Markdown analysis provided by Gemini")
    message: str = Field(default="", description="User-facing status or error message")
    disclaimer: str = Field(
        default="AI-generated visual hypothesis for engineering review — not a certified physical diagnosis.",
        description="Mandatory engineering disclaimer",
    )


class GeminiVisionService:
    """Manages Gemini vision API communication and structured response formatting."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        settings = get_settings()
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = settings.gemini_model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")

    def get_status(self) -> Dict[str, Any]:
        """Return operational configuration status of the Gemini vision service."""
        has_key = bool(self.api_key and self.api_key.strip())
        return {
            "is_configured": has_key,
            "model_name": self.model_name,
            "provider": "Google Gemini Multimodal API (Free-Tier)",
            "message": (
                f"Gemini vision service configured with model '{self.model_name}'."
                if has_key
                else "GEMINI_API_KEY not configured. Set GEMINI_API_KEY in backend environment (.env) to enable Gemini analysis."
            ),
        }

    def analyze_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> GeminiAnalysisResult:
        """Send image bytes and domain prompt to Gemini for multimodal vision analysis."""
        if not self.api_key or not self.api_key.strip():
            logger.info("Gemini analysis requested but GEMINI_API_KEY is not configured.")
            return GeminiAnalysisResult(
                status="API_KEY_MISSING",
                is_configured=False,
                model_name=self.model_name,
                message=(
                    "Gemini API key is not configured. Please add GEMINI_API_KEY to your "
                    "backend .env file or environment variables to enable free-tier Gemini vision analysis."
                ),
            )

        if not image_bytes or len(image_bytes) == 0:
            return GeminiAnalysisResult(
                status="IMAGE_ERROR",
                is_configured=True,
                model_name=self.model_name,
                message="Cannot analyze empty image bytes.",
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            logger.info("Dispatching image analysis request to Gemini model '%s'...", self.model_name)
            response = client.models.generate_content(
                model=self.model_name,
                contents=[image_part, GEMINI_SOLAR_PV_PROMPT],
            )

            text = response.text if hasattr(response, "text") and response.text else ""
            if not text:
                return GeminiAnalysisResult(
                    status="EMPTY_RESPONSE",
                    is_configured=True,
                    model_name=self.model_name,
                    message="Gemini returned an empty response for this image.",
                )

            return GeminiAnalysisResult(
                status="SUCCESS",
                is_configured=True,
                model_name=self.model_name,
                analysis_text=text,
                message=f"Gemini vision analysis completed successfully using {self.model_name}.",
            )

        except Exception as exc:
            err_str = str(exc)
            logger.exception("Gemini API call failed: %s", err_str)

            # Classify errors for clear user-facing messages
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                return GeminiAnalysisResult(
                    status="RATE_LIMIT_EXCEEDED",
                    is_configured=True,
                    model_name=self.model_name,
                    message="Gemini API free-tier rate limit reached (15 RPM). Please wait a moment before trying again.",
                )
            elif "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                return GeminiAnalysisResult(
                    status="SERVICE_BUSY",
                    is_configured=True,
                    model_name=self.model_name,
                    message=f"Gemini model '{self.model_name}' is temporarily experiencing high demand (503). Please retry in a few moments.",
                )
            elif "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
                return GeminiAnalysisResult(
                    status="MODEL_ERROR",
                    is_configured=True,
                    model_name=self.model_name,
                    message=f"Gemini model '{self.model_name}' was not found or is unsupported. Check GEMINI_MODEL_NAME.",
                )
            elif "connect" in err_str.lower() or "timeout" in err_str.lower() or "network" in err_str.lower():
                return GeminiAnalysisResult(
                    status="NETWORK_ERROR",
                    is_configured=True,
                    model_name=self.model_name,
                    message=f"Network error connecting to Gemini API: {err_str}",
                )
            elif "api_key" in err_str.lower() or "invalid" in err_str.lower() or "unauthenticated" in err_str.lower():
                return GeminiAnalysisResult(
                    status="INVALID_API_KEY",
                    is_configured=True,
                    model_name=self.model_name,
                    message="Provided GEMINI_API_KEY is invalid or expired. Check your Google AI Studio API key.",
                )
            else:
                return GeminiAnalysisResult(
                    status="ERROR",
                    is_configured=True,
                    model_name=self.model_name,
                    message=f"Gemini analysis failed: {err_str}",
                )

    def analyze_asset(self, image_path: Union[str, Path]) -> GeminiAnalysisResult:
        """Resolve an image asset on disk and dispatch to Gemini vision analysis."""
        path_obj = Path(image_path)
        if not path_obj.exists():
            try:
                from src.services.storage import StorageService
                resolved = StorageService().get_image_file_path(path_obj)
                if resolved.exists():
                    path_obj = resolved
            except Exception:
                pass

        if not path_obj.exists():
            return GeminiAnalysisResult(
                status="FILE_NOT_FOUND",
                is_configured=bool(self.api_key),
                model_name=self.model_name,
                message=f"Image file does not exist on disk: {image_path}",
            )

        suffix = path_obj.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        try:
            with open(path_obj, "rb") as f:
                img_bytes = f.read()
            return self.analyze_image(img_bytes, mime_type=mime_type)
        except Exception as exc:
            return GeminiAnalysisResult(
                status="FILE_READ_ERROR",
                is_configured=bool(self.api_key),
                model_name=self.model_name,
                message=f"Unable to read image file from disk: {exc}",
            )


# Global singleton instance
_gemini_service: Optional[GeminiVisionService] = None


def get_gemini_service() -> GeminiVisionService:
    """Retrieve global GeminiVisionService singleton."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiVisionService()
    return _gemini_service
