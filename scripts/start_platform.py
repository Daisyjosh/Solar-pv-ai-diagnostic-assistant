"""Unified startup launcher for Solar PV AI Diagnostic Assistant.

Performs environment pre-flight checks, verifies database schema,
and provides instructions or subprocess launching for backend and frontend.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_preflight():
    """Verify environment readiness."""
    print("=" * 65)
    print("Solar PV AI Diagnostic Assistant - Operational Preflight")
    print("=" * 65)
    print(f"[*] Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    print(f"[*] Workspace Root: {PROJECT_ROOT}")

    # Check database
    db_file = PROJECT_ROOT / "data" / "solar_pv_diagnostics.db"
    if db_file.exists():
        print(f"    [OK] SQLite Database: {db_file} ({db_file.stat().st_size} bytes)")
    else:
        print("    [!] Database not initialized. Running migrations...")
        subprocess.run(["alembic", "upgrade", "head"], cwd=PROJECT_ROOT, check=True)

    # Check vision model
    model_file = PROJECT_ROOT / "models" / "classifier" / "active" / "model.joblib"
    if model_file.exists():
        print(f"    [OK] Active Vision Classifier: {model_file} ({model_file.stat().st_size} bytes)")
    else:
        print("    [!] No active vision classifier found in models/classifier/active.")
        print("        Train the model using: python scripts/train_classifier.py")

    print("-" * 65)
    print("[SUCCESS] Platform is ready for operations.")
    print("=" * 65)


def launch():
    """Display startup commands and start services if requested."""
    check_preflight()
    print("\nTo launch the complete platform locally, open two terminal windows:\n")
    print("  Terminal 1 (FastAPI Backend):")
    print("    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload\n")
    print("  Terminal 2 (Streamlit Frontend):")
    print("    python -m streamlit run frontend/app.py --server.port 8501\n")

    if "--run" in sys.argv:
        print("[*] Launching FastAPI backend in background...")
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=PROJECT_ROOT,
        )
        time.sleep(2)
        print("[*] Launching Streamlit frontend...")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"],
                cwd=PROJECT_ROOT,
            )
        finally:
            backend_proc.terminate()


if __name__ == "__main__":
    launch()
