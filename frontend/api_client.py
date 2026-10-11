"""Lightweight HTTP API client connecting Streamlit frontend to FastAPI backend."""

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.core.config import get_settings


class BackendAPIClient:
    """Client for communicating with the Solar PV AI Diagnostic Assistant REST API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        settings = get_settings()
        self.base_url = (base_url or settings.backend_api_url).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Ping the backend health endpoint and compute response latency."""
        health_url = f"{self.base_url}/health"
        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(health_url)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "online": True,
                        "status": data.get("status", "healthy"),
                        "database": data.get("database", "connected"),
                        "version": data.get("version", "0.1.0"),
                        "environment": data.get("environment", "development"),
                        "latency_ms": latency_ms,
                        "url": health_url,
                    }
                elif response.status_code == 503:
                    data = response.json()
                    return {
                        "online": True,
                        "status": "degraded",
                        "database": data.get("database", "disconnected"),
                        "latency_ms": latency_ms,
                        "url": health_url,
                        "warning": "API online but database is unreachable.",
                    }
                else:
                    return {
                        "online": False,
                        "status": f"HTTP {response.status_code}",
                        "database": "unknown",
                        "latency_ms": latency_ms,
                        "url": health_url,
                    }
        except httpx.ConnectError:
            return {
                "online": False,
                "status": "offline",
                "database": "unreachable",
                "latency_ms": None,
                "url": health_url,
                "error": "Connection refused. FastAPI server is not running.",
            }
        except httpx.TimeoutException:
            return {
                "online": False,
                "status": "timeout",
                "database": "unreachable",
                "latency_ms": None,
                "url": health_url,
                "error": "Request timed out.",
            }
        except Exception as exc:
            return {
                "online": False,
                "status": "error",
                "database": "unreachable",
                "latency_ms": None,
                "url": health_url,
                "error": str(exc),
            }

    # --- Plant APIs ---
    def get_plants(self) -> List[Dict[str, Any]]:
        """Fetch list of registered solar PV plants."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/plants/")
                if res.status_code == 200:
                    return res.json().get("items", [])
                return []
        except Exception:
            return []

    def create_plant(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Create a new solar plant record."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/plants/", json=payload)
                if res.status_code == 201:
                    return True, res.json()
                elif res.status_code == 409:
                    return False, res.json().get("detail", "Plant code already exists.")
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    # --- Asset Hierarchy APIs ---
    def get_inverters(self, plant_id: int) -> List[Dict[str, Any]]:
        """Fetch inverters for a plant."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/plants/{plant_id}/inverters")
                if res.status_code == 200:
                    return res.json().get("items", [])
                return []
        except Exception:
            return []

    def create_inverter(self, plant_id: int, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Create an inverter under a plant."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/plants/{plant_id}/inverters", json=payload)
                if res.status_code == 201:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_strings(self, inverter_id: int) -> List[Dict[str, Any]]:
        """Fetch strings for an inverter."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/inverters/{inverter_id}/strings")
                if res.status_code == 200:
                    return res.json().get("items", [])
                return []
        except Exception:
            return []

    def create_string(self, inverter_id: int, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Create a string under an inverter."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/inverters/{inverter_id}/strings", json=payload)
                if res.status_code == 201:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_modules(self, string_id: int) -> List[Dict[str, Any]]:
        """Fetch modules for a string."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/strings/{string_id}/modules")
                if res.status_code == 200:
                    return res.json().get("items", [])
                return []
        except Exception:
            return []

    def create_module(self, string_id: int, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Create a module under a string."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/strings/{string_id}/modules", json=payload)
                if res.status_code == 201:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    # --- Inspection APIs ---
    def get_inspections(self, plant_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch list of inspection sessions."""
        params = {}
        if plant_id:
            params["plant_id"] = plant_id
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/inspections/", params=params)
                if res.status_code == 200:
                    return res.json().get("items", [])
                return []
        except Exception:
            return []

    def get_inspection_detail(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Fetch inspection details including linked images."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/inspections/{inspection_id}")
                if res.status_code == 200:
                    return res.json()
                return None
        except Exception:
            return None

    def create_inspection(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Create a new inspection session record."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/inspections/", json=payload)
                if res.status_code == 201:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def update_inspection(self, inspection_id: int, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Update inspection session record (lifecycle status, environmental notes)."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.put(f"{self.base_url}/inspections/{inspection_id}", json=payload)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    # --- Media Upload & Retrieval ---
    def upload_image(
        self,
        inspection_id: int,
        file_bytes: bytes,
        filename: str,
        image_type: str,
        notes: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Upload and ingest an image asset to an inspection."""
        try:
            files = {"file": (filename, file_bytes)}
            data = {"image_type": image_type}
            if notes:
                data["notes"] = notes

            with httpx.Client(timeout=20.0) as client:
                res = client.post(
                    f"{self.base_url}/inspections/{inspection_id}/upload",
                    files=files,
                    data=data,
                )
                if res.status_code == 201:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_image_file_bytes(self, inspection_id: int, image_id: int) -> Optional[bytes]:
        """Download raw bytes of an image asset."""
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(
                    f"{self.base_url}/inspections/{inspection_id}/images/{image_id}/file"
                )
                if res.status_code == 200:
                    return res.content
                return None
        except Exception:
            return None

    # --- Vision Model APIs ---
    def get_vision_model_status(self) -> Dict[str, Any]:
        """Retrieve vision classifier operational status and metadata."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/vision/model-status")
                if res.status_code == 200:
                    return res.json()
                return {"status": "MODEL_UNAVAILABLE", "message": f"Error {res.status_code}"}
        except Exception as exc:
            return {"status": "MODEL_UNAVAILABLE", "message": str(exc)}

    def get_vision_evaluation_metrics(self) -> Dict[str, Any]:
        """Retrieve measured test and validation evaluation report."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/vision/evaluation-metrics")
                if res.status_code == 200:
                    return res.json()
                return {"status": "NO_EVALUATION_REPORT", "metrics": None}
        except Exception:
            return {"status": "ERROR", "metrics": None}

    def predict_image_asset(self, image_asset_id: int) -> Dict[str, Any]:
        """Run anomaly classification against stored image asset."""
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(f"{self.base_url}/vision/predict/{image_asset_id}")
                if res.status_code == 200:
                    return res.json()
                return {"status": "INFERENCE_FAILED", "message": res.json().get("detail", f"Error {res.status_code}")}
        except Exception as exc:
            return {"status": "INFERENCE_FAILED", "message": str(exc)}

    def predict_uploaded_image(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Run classification on directly uploaded image crop."""
        try:
            files = {"file": (filename, file_bytes)}
            with httpx.Client(timeout=15.0) as client:
                res = client.post(f"{self.base_url}/vision/predict-upload", files=files)
                if res.status_code == 200:
                    return res.json()
                return {"status": "INFERENCE_FAILED", "message": res.json().get("detail", f"Error {res.status_code}")}
        except Exception as exc:
            return {"status": "INFERENCE_FAILED", "message": str(exc)}

    # --- Gemini Multimodal Vision APIs ---
    def get_gemini_status(self) -> Dict[str, Any]:
        """Check if Gemini vision service is configured in backend."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/vision/gemini/status")
                if res.status_code == 200:
                    return res.json()
                return {"is_configured": False, "message": "Failed to check status"}
        except Exception:
            return {"is_configured": False, "message": "Connection error"}

    def analyze_gemini_asset(self, image_asset_id: int) -> Dict[str, Any]:
        """Run Gemini vision analysis on an existing image asset."""
        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(f"{self.base_url}/vision/gemini/analyze/{image_asset_id}")
                if res.status_code == 200:
                    return res.json()
                return {"status": "ERROR", "message": res.json().get("detail", f"Error {res.status_code}")}
        except Exception as exc:
            return {"status": "ERROR", "message": str(exc)}

    def analyze_gemini_upload(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Run Gemini vision analysis on directly uploaded image crop."""
        try:
            files = {"file": (filename, file_bytes)}
            with httpx.Client(timeout=45.0) as client:
                res = client.post(f"{self.base_url}/vision/gemini/analyze-upload", files=files)
                if res.status_code == 200:
                    return res.json()
                return {"status": "ERROR", "message": res.json().get("detail", f"Error {res.status_code}")}
        except Exception as exc:
            return {"status": "ERROR", "message": str(exc)}

    # --- Diagnostic Engine APIs ---
    def trigger_diagnosis(self, inspection_id: int) -> Tuple[bool, Any]:
        """Trigger multi-modal evidence synthesis and diagnosis generation."""
        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post(f"{self.base_url}/inspections/{inspection_id}/diagnose")
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_diagnosis(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve stored diagnosis for an inspection."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/inspections/{inspection_id}/diagnosis")
                if res.status_code == 200:
                    return res.json()
                return None
        except Exception:
            return None

    def update_diagnosis_review(
        self,
        diagnosis_id: int,
        new_status: str,
        new_severity: Optional[str],
        reviewed_by: str,
        notes: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Record engineer sign-off and audit log."""
        payload = {
            "new_status": new_status,
            "new_severity": new_severity,
            "reviewed_by": reviewed_by,
            "engineer_notes": notes,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.put(f"{self.base_url}/diagnoses/{diagnosis_id}/review", json=payload)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    # --- Telemetry APIs ---
    def preview_telemetry(self, plant_id: int, file_bytes: bytes, filename: str) -> Tuple[bool, Any]:
        """Generate validation preview for CSV/Excel upload."""
        try:
            files = {"file": (filename, file_bytes)}
            with httpx.Client(timeout=20.0) as client:
                res = client.post(f"{self.base_url}/plants/{plant_id}/telemetry/preview", files=files)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def commit_telemetry(self, plant_id: int, file_bytes: bytes, filename: str, notes: Optional[str] = None) -> Tuple[bool, Any]:
        """Ingest and commit telemetry records."""
        try:
            files = {"file": (filename, file_bytes)}
            data = {"notes": notes} if notes else {}
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/plants/{plant_id}/telemetry/import", files=files, data=data)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_telemetry_trends(self, plant_id: int, hours: int = 24) -> Dict[str, Any]:
        """Retrieve trend summary and chart curves."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/plants/{plant_id}/telemetry/trends", params={"hours": hours})
                if res.status_code == 200:
                    return res.json()
                return {"has_data": False, "message": "Failed to load telemetry trends"}
        except Exception:
            return {"has_data": False, "message": "Connection error"}

    def get_scada_status(self) -> Dict[str, Any]:
        """Check external SCADA adapter status."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/telemetry/scada-status")
                if res.status_code == 200:
                    return res.json()
                return {"status": "DISCONNECTED", "message": "API error"}
        except Exception:
            return {"status": "DISCONNECTED", "message": "Connection error"}

    # --- Chatbot APIs ---
    def send_chat_message(
        self,
        query: str,
        session_id: Optional[str] = None,
        plant_id: Optional[int] = None,
        inspection_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send question to contextual diagnostic assistant."""
        payload = {
            "query": query,
            "session_id": session_id,
            "plant_id": plant_id,
            "inspection_id": inspection_id,
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(f"{self.base_url}/chat/message", json=payload)
                if res.status_code == 200:
                    return res.json()
                # Safely extract error detail from JSON or plaintext response
                detail_msg = res.text
                try:
                    err_payload = res.json()
                    detail_msg = err_payload.get("detail", res.text)
                except Exception:
                    pass
                return {
                    "response": f"Assistant error ({res.status_code}): {detail_msg}",
                    "citations": [],
                }
        except httpx.TimeoutException:
            return {
                "response": "Request timed out while waiting for assistant response. Please try again.",
                "citations": [],
            }
        except Exception as exc:
            return {
                "response": f"Network error communicating with assistant: {str(exc)}",
                "citations": [],
            }

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/chat/sessions/{session_id}/history")
                if res.status_code == 200:
                    return res.json()
                return []
        except Exception:
            return []

    # --- Batch Inspection APIs ---
    def create_batch_inspection(self, plant_id: int, title: str, expected_count: int = 0) -> Tuple[bool, Any]:
        """Initialize batch inspection."""
        payload = {"title": title, "expected_count": expected_count}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/plants/{plant_id}/batch-inspections", json=payload)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def process_batch_images(self, batch_id: int, files_list: List[Tuple[str, bytes]]) -> Tuple[bool, Any]:
        """Upload and process batch images."""
        files = [("files", (fname, fbytes)) for fname, fbytes in files_list]
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.base_url}/batch-inspections/{batch_id}/process", files=files)
                if res.status_code == 200:
                    return True, res.json()
                return False, res.json().get("detail", f"Error {res.status_code}")
        except Exception as exc:
            return False, str(exc)

    def get_batch_inspections(self, plant_id: int) -> List[Dict[str, Any]]:
        """List batch inspections for a plant."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/plants/{plant_id}/batch-inspections")
                if res.status_code == 200:
                    return res.json()
                return []
        except Exception:
            return []

    # --- Dashboard and Reports APIs ---
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Fetch aggregated platform KPIs and distributions."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/dashboard/summary")
                if res.status_code == 200:
                    return res.json()
                return {"kpis": {}, "fault_distribution": {}, "severity_distribution": {}, "recent_inspections": []}
        except Exception:
            return {"kpis": {}, "fault_distribution": {}, "severity_distribution": {}, "recent_inspections": []}

    def download_pdf_report(self, inspection_id: int) -> Optional[bytes]:
        """Download binary PDF report."""
        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.get(f"{self.base_url}/inspections/{inspection_id}/report/pdf")
                if res.status_code == 200:
                    return res.content
                return None
        except Exception:
            return None
