"""Secure filesystem storage service for solar PV inspection media.

Enforces:
- Streaming upload size enforcement (no unlimited buffering in memory).
- Magic byte signature and PIL content validation.
- Decompression-bomb attack mitigation.
- Directory traversal and symlink escape defenses.
- Collision-resistant file naming with SHA-256 integrity calculation.
- Automatic partial file cleanup upon validation failure.
"""

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Set, Tuple, Union

from PIL import Image

from src.core.config import get_settings


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class UnsupportedFormatError(StorageError):
    """Raised when file extension or magic byte signature is unsupported."""
    pass


class FileTooLargeError(StorageError):
    """Raised when file exceeds configured upload size limit."""
    pass


class CorruptImageError(StorageError):
    """Raised when image content is corrupt or fails PIL decompression checks."""
    pass


class DirectoryTraversalError(StorageError):
    """Raised when a path escapes the configured root storage directory."""
    pass


class StorageService:
    """Manages secure file persistence and retrieval for inspection media."""

    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    # Magic byte signatures
    MAGIC_SIGNATURES: Dict[str, bytes] = {
        "jpeg": b"\xff\xd8\xff",
        "png": b"\x89PNG\r\n\x1a\n",
        "tiff_le": b"II*\x00",
        "tiff_be": b"MM\x00*",
    }

    def __init__(self, storage_dir: Optional[Path] = None):
        settings = get_settings()
        self.storage_dir = (storage_dir or settings.storage_dir).resolve()
        self.max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        # Ensure base directories exist
        self.raw_dir = (self.storage_dir / "raw" / "inspections").resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """Strip dangerous path characters and limit filename to safe characters."""
        base_name = os.path.basename(filename)
        # Keep only letters, digits, dots, dashes, underscores
        clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
        if not clean_name or clean_name.startswith("."):
            clean_name = f"image_{clean_name}"
        return clean_name[:128]

    def _validate_magic_bytes(self, sample: bytes, ext: str) -> None:
        """Verify that the initial bytes match expected image container signatures."""
        if ext in (".jpg", ".jpeg"):
            if not sample.startswith(self.MAGIC_SIGNATURES["jpeg"]):
                raise UnsupportedFormatError(
                    f"File with extension '{ext}' does not contain valid JPEG magic bytes."
                )
        elif ext == ".png":
            if not sample.startswith(self.MAGIC_SIGNATURES["png"]):
                raise UnsupportedFormatError(
                    "File with extension '.png' does not contain valid PNG magic bytes."
                )
        elif ext in (".tif", ".tiff"):
            if not (
                sample.startswith(self.MAGIC_SIGNATURES["tiff_le"])
                or sample.startswith(self.MAGIC_SIGNATURES["tiff_be"])
            ):
                raise UnsupportedFormatError(
                    f"File with extension '{ext}' does not contain valid TIFF magic bytes."
                )

    def save_inspection_image(
        self,
        inspection_id: int,
        file_stream: BinaryIO,
        original_filename: str,
    ) -> Tuple[str, str, int, str]:
        """Stream and validate image into inspection storage.

        Args:
            inspection_id: ID of the parent inspection.
            file_stream: Binary stream of the uploaded file.
            original_filename: Original filename provided by client.

        Returns:
            Tuple of (relative_storage_path, checksum_sha256, file_size_bytes, mime_type).

        Raises:
            UnsupportedFormatError, FileTooLargeError, CorruptImageError, DirectoryTraversalError.
        """
        # 1. Validate file extension
        sanitized_original = self._sanitize_filename(original_filename)
        ext = os.path.splitext(sanitized_original)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"File extension '{ext}' is not supported. Allowed: {sorted(self.ALLOWED_EXTENSIONS)}"
            )

        # 2. Prepare target destination directory
        target_dir = (self.raw_dir / str(inspection_id)).resolve()
        # Traversal containment check
        if not target_dir.is_relative_to(self.storage_dir):
            raise DirectoryTraversalError("Target directory escapes root storage directory.")
        target_dir.mkdir(parents=True, exist_ok=True)

        # 3. Generate collision-resistant unique filename
        unique_prefix = uuid.uuid4().hex
        stored_filename = f"{unique_prefix}_{sanitized_original}"
        dest_path = (target_dir / stored_filename).resolve()

        if not dest_path.is_relative_to(self.storage_dir):
            raise DirectoryTraversalError("Resolved file path escapes root storage directory.")

        # 4. Stream file to disk in chunks, enforcing size limit and computing SHA-256
        sha256_hash = hashlib.sha256()
        total_bytes = 0
        chunk_size = 64 * 1024  # 64 KB chunks
        magic_sample = b""

        try:
            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = file_stream.read(chunk_size)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > self.max_size_bytes:
                        raise FileTooLargeError(
                            f"File size exceeds maximum allowed upload limit of {self.max_size_bytes // (1024 * 1024)} MB."
                        )

                    if len(magic_sample) < 16:
                        magic_sample += chunk[: 16 - len(magic_sample)]

                    sha256_hash.update(chunk)
                    out_file.write(chunk)

            if total_bytes == 0:
                raise CorruptImageError("Uploaded file is empty (0 bytes).")

            # 5. Validate initial magic bytes
            self._validate_magic_bytes(magic_sample, ext)

            # 6. Verify image decode integrity & decompression bomb safety via Pillow
            try:
                with Image.open(dest_path) as img:
                    img_format = (img.format or "").upper()
                    img.verify()

                # Determine standard MIME type
                if img_format == "JPEG":
                    mime_type = "image/jpeg"
                elif img_format == "PNG":
                    mime_type = "image/png"
                elif img_format == "TIFF":
                    mime_type = "image/tiff"
                else:
                    mime_type = f"image/{img_format.lower()}"

            except Image.DecompressionBombError as dbe:
                raise CorruptImageError(f"Decompression bomb risk detected: {dbe}")
            except Exception as exc:
                raise CorruptImageError(f"File content is corrupt or not a valid image: {exc}")

        except Exception:
            # Clean up partial or failed file immediately
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            raise

        checksum = sha256_hash.hexdigest()
        # Compute relative storage path from storage_dir
        relative_path = dest_path.relative_to(self.storage_dir).as_posix()

        return relative_path, checksum, total_bytes, mime_type

    def get_image_file_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve and validate that an image path strictly resides inside storage_dir.

        Supports:
        - Relative paths from storage_dir (e.g., 'raw/inspections/1/uuid_name.jpg')
        - Relative paths with storage_dir name prefix (e.g., 'data/raw/inspections/1/uuid_name.jpg')
        - Relative paths from workspace root (e.g., './data/raw/inspections/1/uuid_name.jpg')
        - Absolute paths residing within storage_dir (e.g., 'C:/sem-7-project/data/raw/inspections/1/uuid_name.jpg')
        - Windows and POSIX path separators

        Raises:
            DirectoryTraversalError if path escapes storage directory.
            FileNotFoundError if file does not exist.
        """
        clean_path = Path(file_path)
        if clean_path.is_absolute():
            dest_path = clean_path.resolve()
        else:
            # 1. Primary resolution: relative to storage_dir
            dest_path = (self.storage_dir / clean_path).resolve()
            if not dest_path.is_file():
                # 2. Check if path is relative to storage_dir's parent (e.g., 'data/raw/...')
                dest_alt = (self.storage_dir.parent / clean_path).resolve()
                if dest_alt.is_file() and dest_alt.is_relative_to(self.storage_dir):
                    dest_path = dest_alt
                # 3. Check if path starts with storage directory name (e.g., 'data')
                elif clean_path.parts and clean_path.parts[0] in (self.storage_dir.name, "data"):
                    dest_sub = (self.storage_dir / Path(*clean_path.parts[1:])).resolve()
                    if dest_sub.is_file() and dest_sub.is_relative_to(self.storage_dir):
                        dest_path = dest_sub

        if not dest_path.is_relative_to(self.storage_dir):
            raise DirectoryTraversalError("Requested path attempts directory traversal escape.")

        if not dest_path.is_file():
            raise FileNotFoundError(f"Image file does not exist at {file_path}.")

        return dest_path

    def resolve_image_path(self, file_path: Union[str, Path]) -> Path:
        """Alias for get_image_file_path for explicit semantic clarity."""
        return self.get_image_file_path(file_path)

    def delete_image_file(self, relative_path: str) -> bool:
        """Safely delete an image file from storage."""
        try:
            target = self.get_image_file_path(relative_path)
            target.unlink()
            return True
        except Exception:
            return False


def resolve_image_file_path(file_path: Union[str, Path], storage_dir: Optional[Path] = None) -> Path:
    """Centrally resolve an image asset file path against the configured storage root."""
    return StorageService(storage_dir=storage_dir).get_image_file_path(file_path)

