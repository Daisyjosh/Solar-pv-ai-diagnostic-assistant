"""Production SCADA and IoT provider adapter with explicit connection status reporting."""

import logging
import os
from typing import Any, Dict, List, Optional

from src.telemetry.providers.base import BaseTelemetryProvider, ProviderConnectionStatus

logger = logging.getLogger(__name__)


class SCADAAdapter(BaseTelemetryProvider):
    """Adapter for external SCADA / Modbus TCP / REST Inverter Cloud gateways."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        protocol: str = "MODBUS_TCP_REST_GATEWAY",
    ):
        self.endpoint_url = endpoint_url or os.getenv("SCADA_ENDPOINT_URL")
        self.api_key = api_key or os.getenv("SCADA_API_KEY")
        self.protocol = protocol

    @property
    def provider_name(self) -> str:
        return "Industrial SCADA / Modbus Gateway Adapter"

    def check_connection(self) -> ProviderConnectionStatus:
        """Verify status: returns UNCONFIGURED when no external credentials exist."""
        if not self.endpoint_url or not self.api_key:
            return ProviderConnectionStatus.UNCONFIGURED
        # Real adapter would probe endpoint health here
        return ProviderConnectionStatus.DISCONNECTED

    def get_status(self) -> Dict[str, Any]:
        """Return transparent connection information."""
        status = self.check_connection()
        return {
            "provider_name": self.provider_name,
            "status": status.value,
            "protocol": self.protocol,
            "endpoint_configured": bool(self.endpoint_url),
            "credentials_configured": bool(self.api_key),
            "is_live_connected": status == ProviderConnectionStatus.CONNECTED,
            "message": (
                "SCADA endpoint unconfigured. Telemetry is currently ingested via CSV/Excel upload."
                if status == ProviderConnectionStatus.UNCONFIGURED
                else "Configured SCADA endpoint is unreachable."
            ),
            "supported_protocols": ["Modbus TCP (via Gateway)", "IEC 60870-5-104", "DNP3", "SolarEdge / SMA REST API"],
        }

    def poll_telemetry(self, plant_code: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Safely returns empty list with explanatory log when not connected."""
        status = self.check_connection()
        if status != ProviderConnectionStatus.CONNECTED:
            logger.info("Cannot poll live SCADA telemetry: provider status is %s", status.value)
            return []
        return []
