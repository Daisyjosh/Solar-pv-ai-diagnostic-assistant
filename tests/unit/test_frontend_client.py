"""Unit tests for the frontend API client and graceful error handling."""

from frontend.api_client import BackendAPIClient


def test_frontend_client_offline_handling():
    """Verify frontend client handles unreachable server gracefully without throwing unhandled exceptions."""
    # Point client to a port guaranteed not to be running a service
    offline_client = BackendAPIClient(base_url="http://127.0.0.1:59999/api/v1", timeout=0.2)

    health = offline_client.check_health()
    assert health["online"] is False
    assert health["status"] in ("offline", "timeout")
    assert health["database"] == "unreachable"
    assert "error" in health

    plants = offline_client.get_plants()
    assert plants == []

    inspections = offline_client.get_inspections()
    assert inspections == []

    success, error = offline_client.create_plant({"code": "TEST"})
    assert success is False
    assert error is not None
