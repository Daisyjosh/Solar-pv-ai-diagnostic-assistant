"""Unit tests for image metadata and EXIF extraction."""

from pathlib import Path
from PIL import Image

from src.vision.metadata import extract_image_metadata


def test_metadata_extraction_without_exif(tmp_path: Path):
    """Verify that an image without EXIF returns valid dimensions and None for absent fields."""
    img_path = tmp_path / "plain_optical.png"
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    img.save(img_path, format="PNG")

    meta = extract_image_metadata(img_path)
    assert meta.image_width == 320
    assert meta.image_height == 240
    assert meta.image_format == "PNG"
    assert meta.camera_make is None
    assert meta.camera_model is None
    assert meta.captured_at is None
    assert meta.gps_latitude is None
    assert meta.gps_longitude is None
    assert meta.emissivity is None
    assert meta.reflected_temp_c is None
    assert meta.is_radiometric is False


def test_metadata_extraction_with_exif(tmp_path: Path):
    """Verify that camera make and model are correctly extracted from EXIF."""
    img_path = tmp_path / "camera_test.jpg"
    img = Image.new("RGB", (200, 200), color=(200, 100, 50))

    # Construct EXIF dictionary
    exif = img.getexif()
    exif[271] = "FLIR Systems"      # Make
    exif[272] = "Vue Pro R 640"     # Model
    exif[306] = "2026:08:15 11:30:00" # DateTime
    img.save(img_path, format="JPEG", exif=exif)

    meta = extract_image_metadata(img_path)
    assert meta.camera_make == "FLIR Systems"
    assert meta.camera_model == "Vue Pro R 640"
    assert meta.captured_at is not None
    assert meta.captured_at.year == 2026
    assert meta.captured_at.month == 8
