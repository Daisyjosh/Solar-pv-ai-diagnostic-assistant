"""Computer vision pipeline for thermal and optical solar PV inspection."""

from src.vision.dataset_audit import AuditSummary, DatasetAuditor, ProvenanceType
from src.vision.metadata import ExtractedMetadata, extract_image_metadata
from src.vision.preprocessing import (
    ImageQualityReport,
    RadiometricStatus,
    RGBProcessingResult,
    RGBPreprocessor,
    ThermalProcessingResult,
    ThermalPreprocessor,
)
from src.vision.splitting import DatasetSplitReport, ProvenanceSplitter, SplitPartition

__all__ = [
    "extract_image_metadata",
    "ExtractedMetadata",
    "DatasetAuditor",
    "AuditSummary",
    "ProvenanceType",
    "RGBPreprocessor",
    "ThermalPreprocessor",
    "RadiometricStatus",
    "ImageQualityReport",
    "RGBProcessingResult",
    "ThermalProcessingResult",
    "ProvenanceSplitter",
    "DatasetSplitReport",
    "SplitPartition",
]
