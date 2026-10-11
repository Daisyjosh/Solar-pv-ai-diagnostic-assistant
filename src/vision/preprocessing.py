"""Separate non-destructive image preprocessing pipelines for optical RGB and thermal PV imagery.

Engineering Principles:
- Optical RGB and thermal modalities are strictly separated.
- 16-bit TIFF files are NOT automatically assumed to be radiometric.
- Radiometric status is classified as CONFIRMED, NOT_CONFIRMED, or UNKNOWN based on verifiable calibration tags.
- Temperature is NEVER inferred from arbitrary false-color palette values.
- Original source files are preserved unmodified; all derivatives are saved non-destructively.
- Quality thresholds are configurable and report anomalies without silently deleting files.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from src.vision.metadata import extract_image_metadata


class RadiometricStatus(str, Enum):
    """Verifiability of quantitative thermal temperature data."""

    CONFIRMED = "CONFIRMED"         # Genuine radiometric sensor stream with verifiable calibration
    NOT_CONFIRMED = "NOT_CONFIRMED" # Rendered 8-bit false-color palette or standard RGB image
    UNKNOWN = "UNKNOWN"             # 16-bit or TIFF file lacking explicit calibration tags


@dataclass
class ImageQualityReport:
    """Diagnostic screening of optical image quality."""

    is_blurry: bool = False
    blur_score: float = 0.0          # Laplacian variance (higher = sharper)
    is_underexposed: bool = False
    is_overexposed: bool = False
    shadow_clipping_ratio: float = 0.0
    highlight_clipping_ratio: float = 0.0
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class RGBProcessingResult:
    """Result of non-destructive optical RGB preprocessing."""

    source_path: Path
    processed_path: Optional[Path]
    orientation_corrected: bool
    original_dimensions: Tuple[int, int]
    processed_dimensions: Tuple[int, int]
    quality_report: ImageQualityReport


@dataclass
class ThermalProcessingResult:
    """Result of thermal image inspection and preprocessing."""

    source_path: Path
    processed_path: Optional[Path]
    radiometric_status: RadiometricStatus
    temperature_matrix_available: bool
    bit_depth: int
    emissivity: Optional[float]
    reflected_temp_c: Optional[float]
    preservation_notes: str


class RGBPreprocessor:
    """Preprocessing and quality assurance pipeline for optical RGB captures."""

    def __init__(self, blur_threshold: float = 80.0, clipping_threshold: float = 0.08):
        """Configure optical quality screening parameters.

        Args:
            blur_threshold: Minimum Laplacian variance below which image is flagged as blurry.
            clipping_threshold: Fraction of pixels at extrema (0 or 255) indicating exposure issues.
        """
        self.blur_threshold = blur_threshold
        self.clipping_threshold = clipping_threshold

    def screen_quality(self, pil_image: Image.Image) -> ImageQualityReport:
        """Screen optical image for focus, underexposure, and overexposure."""
        report = ImageQualityReport()
        try:
            # Convert PIL image to OpenCV grayscale array
            rgb_arr = np.array(pil_image.convert("RGB"))
            gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

            # 1. Blur detection via Laplacian variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            report.blur_score = round(lap_var, 2)
            if lap_var < self.blur_threshold:
                report.is_blurry = True
                report.quality_issues.append(
                    f"Low sharpness: Laplacian variance {report.blur_score} < {self.blur_threshold}"
                )

            # 2. Exposure analysis via histogram extrema
            total_pixels = gray.size
            shadow_pixels = np.count_nonzero(gray < 5)
            highlight_pixels = np.count_nonzero(gray > 250)

            report.shadow_clipping_ratio = round(shadow_pixels / total_pixels, 4)
            report.highlight_clipping_ratio = round(highlight_pixels / total_pixels, 4)

            if report.shadow_clipping_ratio > self.clipping_threshold:
                report.is_underexposed = True
                report.quality_issues.append(
                    f"Severe shadow clipping: {report.shadow_clipping_ratio * 100:.1f}% pixels < 5"
                )

            if report.highlight_clipping_ratio > self.clipping_threshold:
                report.is_overexposed = True
                report.quality_issues.append(
                    f"Severe highlight clipping (possible solar glare): {report.highlight_clipping_ratio * 100:.1f}% pixels > 250"
                )

        except Exception as exc:
            report.quality_issues.append(f"Quality screening failed: {exc}")

        return report

    def process_rgb(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> RGBProcessingResult:
        """Normalize RGB orientation and screen quality non-destructively."""
        with Image.open(image_path) as img:
            orig_dims = (img.width, img.height)

            # 1. EXIF orientation transposition
            transposed = ImageOps.exif_transpose(img)
            orientation_corrected = (transposed.size != img.size)

            # 2. Quality screening
            quality_report = self.screen_quality(transposed)

            # 3. Optional non-destructive derivative output
            proc_path = None
            final_dims = (transposed.width, transposed.height)

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                proc_img = transposed.copy()
                if target_size:
                    proc_img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    final_dims = (proc_img.width, proc_img.height)

                proc_path = output_dir / f"proc_{image_path.name}"
                proc_img.save(proc_path, quality=95)

            return RGBProcessingResult(
                source_path=image_path,
                processed_path=proc_path,
                orientation_corrected=orientation_corrected,
                original_dimensions=orig_dims,
                processed_dimensions=final_dims,
                quality_report=quality_report,
            )


class ThermalPreprocessor:
    """Thermal image ingestion, format identification, and radiometric verification."""

    def identify_modality(self, image_path: Path) -> Tuple[RadiometricStatus, int, str]:
        """Determine whether thermal capture contains verified radiometric measurements.

        Returns:
            Tuple of (RadiometricStatus, bit_depth, explanation).
        """
        try:
            with Image.open(image_path) as img:
                mode = img.mode
                ext = image_path.suffix.lower()

                # Check 1: 8-bit standard RGB or Palette modes
                if mode in ("RGB", "RGBA", "P"):
                    # Standard false-color image (ironbow, rainbow, white-hot RGB)
                    meta = extract_image_metadata(image_path)
                    if meta.is_radiometric and meta.emissivity is not None:
                        return (
                            RadiometricStatus.CONFIRMED,
                            8,
                            "FLIR radiometric RJPEG with verified calibration tags",
                        )
                    return (
                        RadiometricStatus.NOT_CONFIRMED,
                        8,
                        "Rendered 8-bit false-color palette. Cannot infer quantitative temperatures without calibration lookup.",
                    )

                # Check 2: 16-bit Grayscale / TIFF formats
                if mode in ("I;16", "I;16L", "I;16B", "I"):
                    exif = img.getexif()
                    # Check for explicit radiometric tags in TIFF headers (e.g. FLIR IFD 50341 or custom calibration)
                    has_calibration = False
                    if exif:
                        # FLIR metadata or Planck constants tags in TIFF
                        for tag_id in exif.keys():
                            if tag_id in (50341, 50342) or "radiometric" in str(ExifTags.TAGS.get(tag_id, "")).lower():
                                has_calibration = True
                                break

                    if has_calibration:
                        return (
                            RadiometricStatus.CONFIRMED,
                            16,
                            "Calibrated 16-bit radiometric TIFF with verifiable temperature conversion tags",
                        )
                    else:
                        return (
                            RadiometricStatus.UNKNOWN,
                            16,
                            "Raw 16-bit array without verified calibration tags. Preserved as raw array; temperature equation unconfirmed.",
                        )

                # Default fallback
                return (
                    RadiometricStatus.NOT_CONFIRMED,
                    8,
                    f"Standard image mode '{mode}' without thermal calibration.",
                )

        except Exception as exc:
            return (RadiometricStatus.NOT_CONFIRMED, 0, f"Failed to inspect image modality: {exc}")

    def process_thermal(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
    ) -> ThermalProcessingResult:
        """Process thermal capture without destructive color-temperature quantization."""
        status, bit_depth, notes = self.identify_modality(image_path)
        meta = extract_image_metadata(image_path)

        proc_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Copy or export normalized derivative preserving bit-depth
            with Image.open(image_path) as img:
                proc_path = output_dir / f"proc_{image_path.name}"
                img.save(proc_path)

        return ThermalProcessingResult(
            source_path=image_path,
            processed_path=proc_path,
            radiometric_status=status,
            temperature_matrix_available=(status == RadiometricStatus.CONFIRMED),
            bit_depth=bit_depth,
            emissivity=meta.emissivity if status == RadiometricStatus.CONFIRMED else None,
            reflected_temp_c=meta.reflected_temp_c if status == RadiometricStatus.CONFIRMED else None,
            preservation_notes=notes,
        )
