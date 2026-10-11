"""Unit tests for optical RGB and thermal preprocessing pipelines."""

from pathlib import Path
import numpy as np
from PIL import Image

from src.vision.preprocessing import (
    RadiometricStatus,
    RGBPreprocessor,
    ThermalPreprocessor,
)


def test_rgb_preprocessor_quality_screening(tmp_path: Path):
    """Test sharpness and exposure quality screening on optical images."""
    preprocessor = RGBPreprocessor(blur_threshold=50.0, clipping_threshold=0.05)

    # 1. Create a sharp image with high edge contrast
    sharp_path = tmp_path / "sharp.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:80, 20:80] = 200  # High contrast square
    Image.fromarray(arr).save(sharp_path)

    res_sharp = preprocessor.process_rgb(sharp_path)
    assert res_sharp.quality_report.blur_score > 50.0
    assert res_sharp.quality_report.is_blurry is False

    # 2. Create a flat, blurry image (uniform gray)
    blur_path = tmp_path / "blurry.png"
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(blur_path)

    res_blur = preprocessor.process_rgb(blur_path)
    assert res_blur.quality_report.is_blurry is True
    assert any("sharpness" in issue.lower() for issue in res_blur.quality_report.quality_issues)


def test_rgb_non_destructive_output(tmp_path: Path):
    """Verify original RGB image is untouched and derivatives are saved to output_dir."""
    orig_path = tmp_path / "original.jpg"
    img = Image.new("RGB", (300, 200), color=(10, 80, 150))
    img.save(orig_path)
    orig_mtime = orig_path.stat().st_mtime

    out_dir = tmp_path / "processed"
    preprocessor = RGBPreprocessor()
    res = preprocessor.process_rgb(orig_path, output_dir=out_dir, target_size=(150, 100))

    assert res.processed_path is not None
    assert res.processed_path.exists()
    assert res.processed_path != orig_path
    assert res.processed_dimensions == (150, 100)
    # Original file must remain unchanged
    assert orig_path.stat().st_mtime == orig_mtime
    with Image.open(orig_path) as check_img:
        assert check_img.size == (300, 200)


def test_thermal_modality_differentiation(tmp_path: Path):
    """Verify 8-bit palette is NOT_CONFIRMED and 16-bit without tags is UNKNOWN."""
    preprocessor = ThermalPreprocessor()

    # 1. Standard 8-bit palette false-color image -> NOT_CONFIRMED
    rendered_path = tmp_path / "ironbow.jpg"
    Image.new("RGB", (160, 120), color=(150, 20, 200)).save(rendered_path)

    status_rendered, depth_rendered, _ = preprocessor.identify_modality(rendered_path)
    assert status_rendered == RadiometricStatus.NOT_CONFIRMED
    assert depth_rendered == 8

    # 2. 16-bit raw array without calibration tags -> UNKNOWN (do not assume radiometric!)
    raw16_path = tmp_path / "raw16.tif"
    arr16 = np.zeros((100, 100), dtype=np.uint16)
    arr16[30:70, 30:70] = 30000
    Image.fromarray(arr16, mode="I;16").save(raw16_path)

    status_raw16, depth_raw16, notes = preprocessor.identify_modality(raw16_path)
    assert status_raw16 == RadiometricStatus.UNKNOWN
    assert depth_raw16 == 16
    assert "unconfirmed" in notes.lower()
