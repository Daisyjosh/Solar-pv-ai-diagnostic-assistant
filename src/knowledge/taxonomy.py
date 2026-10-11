"""Standardized solar PV fault taxonomy, observability boundaries, and dataset label mapping registry.

Engineering Principles:
- Grounds fault classifications in IEC 62446-3, IEC 61215, and IEA-PVPS Task 13 where applicable.
- Explicitly separates visually observable findings from root causes requiring electrical measurement.
- Documents exact rationale for public dataset label mappings and records ambiguous/unsupported classes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FaultCategory(str, Enum):
    """Categorization of failure modes and anomalies."""

    CELL_LEVEL = "CELL_LEVEL"
    SUBSTRING_LEVEL = "SUBSTRING_LEVEL"
    MODULE_LEVEL = "MODULE_LEVEL"
    STRING_INVERTER_LEVEL = "STRING_INVERTER_LEVEL"
    SURFACE_EXTERNAL = "SURFACE_EXTERNAL"


class Observability(str, Enum):
    """Diagnosability boundary based on available sensor modalities."""

    VISUAL_OBSERVABLE = "VISUAL_OBSERVABLE"
    # Thermal/optical anomaly is visible, but electrical readings are mandatory to confirm root cause
    REQUIRES_ELECTRICAL_CORROBORATION = "REQUIRES_ELECTRICAL_CORROBORATION"
    # Cannot be reliably identified from thermal/RGB images alone
    REQUIRES_PHYSICAL_INSPECTION = "REQUIRES_PHYSICAL_INSPECTION"


class MappingConfidence(str, Enum):
    """Defensibility of mapping a public dataset label to the canonical taxonomy."""

    DEFENSIBLE = "DEFENSIBLE"       # Direct, technically justified correspondence
    AMBIGUOUS = "AMBIGUOUS"         # Underspecified label that could represent multiple root causes
    UNSUPPORTED = "UNSUPPORTED"     # Label outside platform scope or non-defect artifact


@dataclass(frozen=True)
class TaxonomyEntry:
    """Canonical fault taxonomy entry."""

    code: str
    name: str
    category: FaultCategory
    observability: Observability
    description: str
    iec_reference: Optional[str] = None
    typical_thermal_pattern: Optional[str] = None
    corroborating_measurements: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabelMapping:
    """Mapping rule connecting public dataset labels to canonical taxonomy."""

    dataset_source: str
    raw_label: str
    canonical_code: Optional[str]
    confidence: MappingConfidence
    rationale: str


class TaxonomyRegistry:
    """Registry maintaining canonical fault taxonomy and public label mappings."""

    def __init__(self):
        self._entries: Dict[str, TaxonomyEntry] = {}
        self._mappings: List[LabelMapping] = []
        self._initialize_canonical_entries()
        self._initialize_dataset_mappings()

    def _initialize_canonical_entries(self) -> None:
        entries = [
            TaxonomyEntry(
                code="FLT-CELL-001",
                name="Single-Cell Hot Spot",
                category=FaultCategory.CELL_LEVEL,
                observability=Observability.REQUIRES_ELECTRICAL_CORROBORATION,
                description=(
                    "Localized thermal dissipation on a single solar cell caused by internal shunt, "
                    "localized micro-crack, or severe point shading forcing the cell into reverse bias."
                ),
                iec_reference="IEC 62446-3: Clause 5.2.2 (Point anomaly)",
                typical_thermal_pattern="Sharp localized point anomaly (delta-T > 10C above adjacent cells)",
                corroborating_measurements=["I-V curve trace (shunt resistance slope)", "String operating current"],
            ),
            TaxonomyEntry(
                code="FLT-CELL-002",
                name="Multi-Cell Hot Spot",
                category=FaultCategory.CELL_LEVEL,
                observability=Observability.REQUIRES_ELECTRICAL_CORROBORATION,
                description="Thermal anomalies spanning multiple adjacent or distributed cells on a single module.",
                iec_reference="IEC 62446-3: Clause 5.2.2",
                typical_thermal_pattern="Multiple elevated temperature clusters across distinct cell coordinates",
                corroborating_measurements=["Module Voc", "String operating current"],
            ),
            TaxonomyEntry(
                code="FLT-CELL-003",
                name="Cell Micro-Crack",
                category=FaultCategory.CELL_LEVEL,
                observability=Observability.REQUIRES_PHYSICAL_INSPECTION,
                description="Mechanical silicon wafer fracture; may or may not manifest thermal dissipation unless isolated.",
                iec_reference="IEC 61215-2: MQT 09",
                typical_thermal_pattern="Incipient crack may show linear thermal gradient; often requires Electroluminescence (EL)",
                corroborating_measurements=["Electroluminescence (EL) imaging", "I-V curve trace"],
            ),
            TaxonomyEntry(
                code="FLT-DIODE-001",
                name="Shorted Bypass Diode / Substring Hot",
                category=FaultCategory.SUBSTRING_LEVEL,
                observability=Observability.VISUAL_OBSERVABLE,
                description=(
                    "Bypass diode permanently short-circuited or continuously conducting, "
                    "causing approximately 1/3 or 2/3 of the module to heat up uniformly."
                ),
                iec_reference="IEC 62446-3: Clause 5.2.3 (Sub-string heating)",
                typical_thermal_pattern="Uniform heating of an entire cell sub-string rectangle",
                corroborating_measurements=["Module Voc reduction by ~1/3 (approx. 12-16V loss)", "Bypass diode forward/reverse bias multimeter test"],
            ),
            TaxonomyEntry(
                code="FLT-MOD-001",
                name="Open-Circuit / Disconnected Module",
                category=FaultCategory.MODULE_LEVEL,
                observability=Observability.VISUAL_OBSERVABLE,
                description=(
                    "Entire module electrically disconnected from string circuit (open ribbon or blown diode), "
                    "dissipating no current; entire module operates at Voc and exhibits elevated uniform temperature."
                ),
                iec_reference="IEC 62446-3: Clause 5.2.4 (Module-level heating)",
                typical_thermal_pattern="Uniformly elevated temperature across all cells of the module compared to adjacent operating modules",
                corroborating_measurements=["String voltage vs adjacent strings", "Module Voc check across MC4 connectors"],
            ),
            TaxonomyEntry(
                code="FLT-MOD-002",
                name="Potential-Induced Degradation (PID)",
                category=FaultCategory.MODULE_LEVEL,
                observability=Observability.REQUIRES_ELECTRICAL_CORROBORATION,
                description="High voltage stress causing sodium ion migration from glass to cell, shunting junctions nearest negative ground.",
                iec_reference="IEC TS 62804-1 (PID test)",
                typical_thermal_pattern="Characteristic checkerboard heating pattern intensifying toward negative pole of string",
                corroborating_measurements=["EL imaging", "Significant string power degradation under low irradiance"],
            ),
            TaxonomyEntry(
                code="FLT-SURF-001",
                name="Localized Bird Dropping / Hard Soiling",
                category=FaultCategory.SURFACE_EXTERNAL,
                observability=Observability.VISUAL_OBSERVABLE,
                description="Opaque localized surface deposit blocking illumination on part of a cell, inducing localized hot spot.",
                iec_reference="IEA-PVPS Task 13: Failure Modes",
                typical_thermal_pattern="Localized hot spot matching visible surface contamination on optical RGB overlay",
                corroborating_measurements=["Optical RGB confirmation", "Thermal re-inspection post-cleaning"],
            ),
            TaxonomyEntry(
                code="FLT-SURF-002",
                name="Vegetation / Structural Shadowing",
                category=FaultCategory.SURFACE_EXTERNAL,
                observability=Observability.VISUAL_OBSERVABLE,
                description="External shadow cast across modules causing localized cooling or edge hot spot depending on diode status.",
                iec_reference="IEC 62446-3: Clause 4.3",
                typical_thermal_pattern="Linear, jagged, or diffuse shadow contour correlated with sun angle and site geometry",
                corroborating_measurements=["Solar azimuth/elevation check", "Site horizon analysis"],
            ),
        ]
        for entry in entries:
            self._entries[entry.code] = entry

    def _initialize_dataset_mappings(self) -> None:
        """Establish traceable mappings for public solar PV benchmark labels."""
        # 1. InfraredSolarModules benchmark (Buerhop-Lange / Deitsch et al.)
        self._mappings.extend([
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Hot-Spot",
                canonical_code="FLT-CELL-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Direct correspondence to localized single-cell thermal anomaly.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Hot-Spot-Multi",
                canonical_code="FLT-CELL-002",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Direct correspondence to multi-cell hot spot clusters.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Diode",
                canonical_code="FLT-DIODE-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Matches characteristic rectangular 1/3 substring bypass diode overheating.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Diode-Multi",
                canonical_code="FLT-DIODE-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Multiple substring heating patterns on the same module.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Offline-Module",
                canonical_code="FLT-MOD-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Matches completely open-circuit / zero-current module.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Soiling",
                canonical_code="FLT-SURF-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Corresponds to localized surface deposits causing differential heating.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Vegetation",
                canonical_code="FLT-SURF-002",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="External shading induced by foliage or ground growth.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Cracking",
                canonical_code="FLT-CELL-003",
                confidence=MappingConfidence.AMBIGUOUS,
                rationale=(
                    "Wafer cracking in IR imagery is ambiguous: cracks without localized shunts "
                    "do not emit heat, while cracked cells with shunts appear identical to point hot spots. "
                    "Requires EL or visual corroboration."
                ),
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="Shadowing",
                canonical_code="FLT-SURF-002",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="External geometrical shadow casting.",
            ),
            LabelMapping(
                dataset_source="InfraredSolarModules",
                raw_label="No-Anomaly",
                canonical_code=None,
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Healthy, uniform thermal baseline module.",
            ),
        ])

        # 2. PV-Hawk / General Public UAV labels
        self._mappings.extend([
            LabelMapping(
                dataset_source="PV-Hawk",
                raw_label="hot_spot",
                canonical_code="FLT-CELL-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Direct correspondence to cell hot spot.",
            ),
            LabelMapping(
                dataset_source="PV-Hawk",
                raw_label="bypass_diode",
                canonical_code="FLT-DIODE-001",
                confidence=MappingConfidence.DEFENSIBLE,
                rationale="Direct correspondence to sub-string heating.",
            ),
            LabelMapping(
                dataset_source="PV-Hawk",
                raw_label="defect",
                canonical_code=None,
                confidence=MappingConfidence.AMBIGUOUS,
                rationale="Underspecified generic label; cannot map to specific electrical fault.",
            ),
            LabelMapping(
                dataset_source="PV-Hawk",
                raw_label="glare",
                canonical_code=None,
                confidence=MappingConfidence.UNSUPPORTED,
                rationale="Optical reflection artifact, not a physical PV hardware fault.",
            ),
        ])

    def get_entry(self, code: str) -> Optional[TaxonomyEntry]:
        """Retrieve canonical taxonomy entry by code."""
        return self._entries.get(code)

    def list_entries(self) -> List[TaxonomyEntry]:
        """Return all canonical taxonomy entries."""
        return list(self._entries.values())

    def list_mappings(self) -> List[LabelMapping]:
        """Return all public dataset label mappings."""
        return list(self._mappings)

    def map_label(self, dataset_source: str, raw_label: str) -> Optional[LabelMapping]:
        """Find matching label mapping rule for a public dataset label."""
        for m in self._mappings:
            if m.dataset_source.lower() == dataset_source.lower() and m.raw_label.lower() == raw_label.lower():
                return m
        return None
