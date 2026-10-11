"""Unit tests for the DatasetAuditor scanning and quality reporting engine."""

from pathlib import Path
from PIL import Image

from src.vision.dataset_audit import DatasetAuditor, ProvenanceType


def test_dataset_audit_scanner(tmp_path: Path):
    """Test auditing directory with valid images, zero-byte file, corrupt file, and exact duplicate."""
    auditor = DatasetAuditor(near_dup_hamming_threshold=4)

    # 1. Valid image 1
    img1_path = tmp_path / "valid1.png"
    Image.new("RGB", (100, 100), color=(10, 20, 30)).save(img1_path)

    # 2. Exact duplicate of image 1
    img2_path = tmp_path / "valid1_dup.png"
    img2_path.write_bytes(img1_path.read_bytes())

    # 3. Valid image 2 (different content)
    img3_path = tmp_path / "valid2.jpg"
    Image.new("RGB", (80, 80), color=(200, 100, 50)).save(img3_path)

    # 4. Zero-byte file
    zero_path = tmp_path / "empty.jpg"
    zero_path.write_bytes(b"")

    # 5. Corrupt image file
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"\xff\xd8\xff\xe0truncated_bytes")

    summary = auditor.audit_directory(tmp_path)

    assert summary.valid_images_count == 3  # valid1, valid1_dup, valid2
    assert summary.zero_byte_files_count == 1
    assert summary.corrupt_files_count == 2  # empty + corrupt
    assert summary.exact_duplicates_count == 1
    assert len(summary.exact_duplicate_groups) == 1
    assert summary.format_distribution["PNG"] == 2
    assert summary.format_distribution["JPEG"] == 1

    # Verify markdown generation
    report_md = auditor.generate_markdown_report(summary)
    assert "# Solar PV AI Diagnostic Assistant — Dataset Inventory & Quality Audit" in report_md
    assert "Exact Duplicates (SHA-256):** 1" in report_md


def test_provenance_distinction(tmp_path: Path):
    """Verify synthetic test fixtures are distinguished from operational captures."""
    auditor = DatasetAuditor()

    # Synthetic fixture name
    syn_path = tmp_path / "flir_vue_capture_test.jpg"
    Image.new("RGB", (50, 50), color=(1, 2, 3)).save(syn_path)

    summary = auditor.audit_directory(tmp_path)
    assert len(summary.records) == 1
    assert summary.records[0].provenance == ProvenanceType.SYNTHETIC_VERIFICATION_FIXTURE
