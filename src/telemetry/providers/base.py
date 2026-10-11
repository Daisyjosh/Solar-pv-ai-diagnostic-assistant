"""Abstract Provider Interface for external SCADA, IoT, and Inverter APIs."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderConnectionStatus(str, Enum):
    """Operational connection status for external telemetry providers."""

    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    UNCONFIGURED = "UNCONFIGURED"
    ERROR = "ERROR"


class BaseTelemetryProvider(ABC):
    """Abstract base class for solar PV telemetry ingestion adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the telemetry source or protocol."""
        pass

    @abstractmethod
    def check_connection(self) -> ProviderConnectionStatus:
        """Verify network connectivity and credentials with external data source."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return provider configuration details, endpoints, and health status."""
        pass

    @abstractmethod
    def poll_telemetry(self, plant_code: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch latest telemetry points from external service."""
        pass
