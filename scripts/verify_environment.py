"""Verification script to inspect and validate workspace setup and environment."""

import sys
from pathlib import Path

# Add project root to path for imports
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def check_python_version() -> bool:
    """Ensure Python runtime meets minimum requirement (>= 3.11)."""
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Python runtime: {sys.version.split()[0]} ({sys.executable})")
    if (major, minor) < (3, 11):
        print(f"[!] Warning: Python >= 3.11 recommended. Current: {major}.{minor}")
        return False
    print("    [OK] Python version meets minimum requirements (>= 3.11)")
    return True


def check_directories() -> bool:
    """Verify presence of all expected project directories."""
    expected_dirs = [
        "frontend",
        "src/api/routes",
        "src/core",
        "src/database/models",
        "src/services",
        "src/vision",
        "src/diagnosis",
        "src/telemetry",
        "src/knowledge",
        "src/reports",
        "data/raw",
        "data/processed",
        "data/external",
        "tests/unit",
        "tests/integration",
        "docs",
        "scripts",
    ]
    missing = []
    for d in expected_dirs:
        p = WORKSPACE_ROOT / d
        if not p.is_dir():
            missing.append(d)

    if missing:
        print(f"[!] Missing directories: {', '.join(missing)}")
        return False
    print(f"    [OK] All {len(expected_dirs)} core directories present")
    return True


def check_core_files() -> bool:
    """Verify presence of core foundational files."""
    expected_files = [
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "docs/architecture.md",
        "docs/roadmap.md",
        "src/core/config.py",
        "src/api/main.py",
        "src/database/session.py",
        "tests/conftest.py",
    ]
    missing = []
    for f in expected_files:
        p = WORKSPACE_ROOT / f
        if not p.is_file():
            missing.append(f)

    if missing:
        print(f"[!] Missing core files: {', '.join(missing)}")
        return False
    print(f"    [OK] All {len(expected_files)} foundational files present")
    return True


def check_configuration_loading() -> bool:
    """Verify that settings can be loaded cleanly."""
    try:
        from src.core.config import get_settings

        settings = get_settings()
        print(f"    [OK] Configuration loaded: app_name='{settings.app_name}', env='{settings.app_env}'")
        return True
    except Exception as exc:
        print(f"[!] Failed to load configuration: {exc}")
        return False


def main() -> int:
    """Run all verification checks and exit with status code."""
    print("=" * 60)
    print("Solar PV AI Diagnostic Assistant - Environment Preflight Check")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_directories(),
        check_core_files(),
        check_configuration_loading(),
    ]

    print("-" * 60)
    if all(checks):
        print("[SUCCESS] Environment and workspace configuration verified.")
        return 0
    else:
        print("[FAILURE] One or more environment checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
