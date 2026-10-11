"""Unit tests for StorageService: security, validation, traversal, and cleanup."""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.services.storage import (
    CorruptImageError,
    DirectoryTraversalError,
    FileTooLargeError,
    StorageService,
    UnsupportedFormatError,
)


def _generate_valid_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Helper to generate valid JPEG image bytes in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(255, 128, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _generate_valid_png(width: int = 50, height: int = 50) -> bytes:
    """Helper to generate valid PNG image bytes in memory."""
    buf = io.BytesIO()
    img = Image.new("RGBA", (width, height), color=(0, 200, 100, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_storage_save_valid_jpeg(tmp_path: Path):
    """Test saving a valid JPEG file with SHA-256 calculation."""
    service = StorageService(storage_dir=tmp_path)
    jpeg_bytes = _generate_valid_jpeg()

    rel_path, checksum, file_size, mime_type = service.save_inspection_image(
        inspection_id=1,
        file_stream=io.BytesIO(jpeg_bytes),
        original_filename="thermal_flir_001.jpg",
    )

    assert rel_path.startswith("raw/inspections/1/")
    assert file_size == len(jpeg_bytes)
    assert mime_type == "image/jpeg"
    assert len(checksum) == 64

    # Verify file exists on disk
    abs_path = service.get_image_file_path(rel_path)
    assert abs_path.is_file()
    assert abs_path.read_bytes() == jpeg_bytes


def test_storage_save_valid_png(tmp_path: Path):
    """Test saving a valid PNG file."""
    service = StorageService(storage_dir=tmp_path)
    png_bytes = _generate_valid_png()

    rel_path, checksum, file_size, mime_type = service.save_inspection_image(
        inspection_id=2,
        file_stream=io.BytesIO(png_bytes),
        original_filename="optical_rgb_001.png",
    )

    assert rel_path.startswith("raw/inspections/2/")
    assert mime_type == "image/png"
    assert file_size == len(png_bytes)


def test_storage_rejects_unsupported_extension(tmp_path: Path):
    """Test rejection of non-whitelisted extensions (e.g. .exe, .sh, .txt)."""
    service = StorageService(storage_dir=tmp_path)
    payload = b"test malicious script"

    with pytest.raises(UnsupportedFormatError) as exc_info:
        service.save_inspection_image(
            inspection_id=1,
            file_stream=io.BytesIO(payload),
            original_filename="payload.exe",
        )
    assert "not supported" in str(exc_info.value)


def test_storage_rejects_misleading_extension(tmp_path: Path):
    """Test rejection of a file named .jpg that does not have JPEG magic bytes."""
    service = StorageService(storage_dir=tmp_path)
    # Plain text disguised as .jpg
    fake_payload = b"Plain text disguised as JPEG image"

    with pytest.raises(UnsupportedFormatError) as exc_info:
        service.save_inspection_image(
            inspection_id=1,
            file_stream=io.BytesIO(fake_payload),
            original_filename="fake_image.jpg",
        )
    assert "magic bytes" in str(exc_info.value)


def test_storage_rejects_corrupted_image_content(tmp_path: Path):
    """Test rejection when magic bytes exist but image decode is corrupt."""
    service = StorageService(storage_dir=tmp_path)
    # Valid JPEG header followed by garbage
    corrupt_jpeg = b"\xff\xd8\xff\xe0" + b"\x00\x10JFIF\x00" + b"truncated garbage data"

    with pytest.raises(CorruptImageError):
        service.save_inspection_image(
            inspection_id=1,
            file_stream=io.BytesIO(corrupt_jpeg),
            original_filename="corrupt.jpg",
        )

    # Ensure no leftover file exists in inspection directory
    insp_dir = tmp_path / "raw" / "inspections" / "1"
    if insp_dir.exists():
        assert len(list(insp_dir.iterdir())) == 0


def test_storage_streaming_size_limit(tmp_path: Path):
    """Test upload size limit is enforced while streaming."""
    service = StorageService(storage_dir=tmp_path)
    # Set limit to 500 bytes for test
    service.max_size_bytes = 500

    jpeg_bytes = _generate_valid_jpeg(width=300, height=300)
    assert len(jpeg_bytes) > 500

    with pytest.raises(FileTooLargeError):
        service.save_inspection_image(
            inspection_id=1,
            file_stream=io.BytesIO(jpeg_bytes),
            original_filename="large.jpg",
        )

    # Partial file must be deleted
    insp_dir = tmp_path / "raw" / "inspections" / "1"
    if insp_dir.exists():
        assert len(list(insp_dir.iterdir())) == 0


def test_storage_directory_traversal_defense(tmp_path: Path):
    """Test prevention of directory traversal attacks."""
    service = StorageService(storage_dir=tmp_path)

    with pytest.raises(DirectoryTraversalError):
        service.get_image_file_path("../../outside_storage/secret.txt")

    with pytest.raises(DirectoryTraversalError):
        service.get_image_file_path("raw/inspections/../../../windows/win.ini")


def test_storage_path_resolution_variations(tmp_path: Path):
    """Test resolution of relative, prefixed, and absolute paths."""
    service = StorageService(storage_dir=tmp_path)
    jpeg_bytes = _generate_valid_jpeg()

    rel_path, _, _, _ = service.save_inspection_image(
        inspection_id=5,
        file_stream=io.BytesIO(jpeg_bytes),
        original_filename="sample.jpg",
    )

    # 1. Standard relative path
    p1 = service.get_image_file_path(rel_path)
    assert p1.is_file()

    # 2. Path prefixed with storage directory name
    prefixed = f"{tmp_path.name}/{rel_path}"
    p2 = service.get_image_file_path(prefixed)
    assert p2 == p1

    # 3. Absolute path
    p3 = service.get_image_file_path(str(p1))
    assert p3 == p1

    # 4. Alias resolve_image_path
    p4 = service.resolve_image_path(rel_path)
    assert p4 == p1

