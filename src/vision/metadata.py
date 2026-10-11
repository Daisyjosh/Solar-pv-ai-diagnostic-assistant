"""Image metadata and radiometric calibration tag extraction.

Strict Domain Boundary:
- Distinguishes between standard optical/rendered images and true radiometric files.
- Extracts camera hardware details, GPS coordinates, and radiometric parameters
  ONLY when explicitly present in file headers.
- Never fabricates or estimates missing emissivity, reflected temperature, or coordinates.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import ExifTags, Image

# Protect against decompression bomb attacks (50 Megapixels max)
Image.MAX_IMAGE_PIXELS = 50_000_000


@dataclass
class ExtractedMetadata:
    """Metadata extracted from image headers and EXIF."""

    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    captured_at: Optional[datetime] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    emissivity: Optional[float] = None
    reflected_temp_c: Optional[float] = None
    is_radiometric: bool = False
    image_width: int = 0
    image_height: int = 0
    image_format: str = ""


def _convert_gps_coordinate(coords: Tuple[Any, Any, Any], ref: str) -> Optional[float]:
    """Convert EXIF GPS coordinates (degrees, minutes, seconds) to decimal degrees."""
    try:
        deg = float(coords[0])
        minute = float(coords[1])
        sec = float(coords[2])
        decimal = deg + (minute / 60.0) + (sec / 3600.0)
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def extract_image_metadata(file_path: Path) -> ExtractedMetadata:
    """Extract standard EXIF and radiometric calibration tags from image file.

    Returns structured ExtractedMetadata with None for all unmeasured/absent fields.
    """
    metadata = ExtractedMetadata()

    try:
        with Image.open(file_path) as img:
            metadata.image_width = img.width
            metadata.image_height = img.height
            metadata.image_format = (img.format or "").upper()

            exif_data = img.getexif()
            if not exif_data:
                return metadata

            # Map numerical tags to names
            named_exif: Dict[str, Any] = {}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                named_exif[tag_name] = value

            # Hardware specs
            if "Make" in named_exif:
                metadata.camera_make = str(named_exif["Make"]).strip()
            if "Model" in named_exif:
                metadata.camera_model = str(named_exif["Model"]).strip()

            # Capture timestamp
            timestamp_str = named_exif.get("DateTimeOriginal") or named_exif.get("DateTime")
            if timestamp_str and isinstance(timestamp_str, str):
                try:
                    metadata.captured_at = datetime.strptime(
                        timestamp_str.strip()[:19], "%Y:%m:%d %H:%M:%S"
                    )
                except ValueError:
                    metadata.captured_at = None

            # GPS Info (IFD tag 34853)
            gps_ifd = exif_data.get_ifd(34853) if hasattr(exif_data, "get_ifd") else None
            if gps_ifd:
                gps_named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat = gps_named.get("GPSLatitude")
                lat_ref = gps_named.get("GPSLatitudeRef")
                lon = gps_named.get("GPSLongitude")
                lon_ref = gps_named.get("GPSLongitudeRef")

                if lat and lat_ref:
                    metadata.gps_latitude = _convert_gps_coordinate(lat, lat_ref)
                if lon and lon_ref:
                    metadata.gps_longitude = _convert_gps_coordinate(lon, lon_ref)

            # Radiometric detection (e.g. FLIR RJPEG or calibrated TIFF)
            # True radiometric files include embedded temperature lookup tables or emissivity tags
            # If standard rendered JPEG (palette ironbow), is_radiometric remains False
            camera_lower = (metadata.camera_make or "").lower()
            if "flir" in camera_lower or "workswell" in camera_lower or "opgal" in camera_lower:
                # Check for radiometric maker notes or custom IFD tags
                # Notice: Emissivity and ReflectedTemp are only populated if explicitly encoded
                if "Emissivity" in named_exif:
                    try:
                        metadata.emissivity = float(named_exif["Emissivity"])
                        metadata.is_radiometric = True
                    except (ValueError, TypeError):
                        pass
                if "ReflectedTemperature" in named_exif:
                    try:
                        metadata.reflected_temp_c = float(named_exif["ReflectedTemperature"])
                        metadata.is_radiometric = True
                    except (ValueError, TypeError):
                        pass

    except Exception:
        # File parsing failure: returns default empty metadata
        pass

    return metadata
