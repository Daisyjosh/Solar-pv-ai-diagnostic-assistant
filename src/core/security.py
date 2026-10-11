"""Security, role-based authorization, and path traversal defense utilities."""

from enum import Enum
import hashlib
import os
from pathlib import Path
import re
from typing import Optional

from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class UserRole(str, Enum):
    """User operational roles for Solar PV O&M platform."""

    INSPECTOR = "INSPECTOR"      # Can create inspections and upload media
    ENGINEER = "ENGINEER"        # Can review diagnoses, sign off, and import telemetry
    ADMINISTRATOR = "ADMIN"      # Full operational access, plant deletion


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or shell character injection."""
    base = Path(filename).name
    # Strip any dangerous characters
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{hashlib.md5(filename.encode()).hexdigest()[:8]}{Path(filename).suffix}"
    return cleaned


def verify_storage_boundary(target_path: Path, allowed_root: Path) -> Path:
    """Ensure target path resolves strictly within the allowed root directory."""
    resolved_target = target_path.resolve()
    resolved_root = allowed_root.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError:
        raise PermissionError(f"Path traversal detected: {target_path} is outside {allowed_root}")
    return resolved_target


def get_current_user_role(
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
    x_user_role: Optional[str] = Header("ENGINEER"),
) -> UserRole:
    """Extract or validate caller role."""
    # If production API key configured, validate it
    configured_key = os.getenv("APP_API_KEY")
    if configured_key and x_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )

    role_str = (x_user_role or "ENGINEER").upper()
    try:
        return UserRole(role_str)
    except ValueError:
        return UserRole.ENGINEER
