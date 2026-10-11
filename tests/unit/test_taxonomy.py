"""Unit tests for solar PV fault taxonomy and public dataset label mapping registry."""

from src.knowledge.taxonomy import (
    FaultCategory,
    MappingConfidence,
    Observability,
    TaxonomyRegistry,
)


def test_taxonomy_canonical_entries():
    """Verify presence of core canonical fault entries across all categories."""
    registry = TaxonomyRegistry()
    entries = registry.list_entries()
    assert len(entries) >= 8

    # Verify key faults exist with correct domain classifications
    hotspot = registry.get_entry("FLT-CELL-001")
    assert hotspot is not None
    assert hotspot.category == FaultCategory.CELL_LEVEL
    assert hotspot.observability == Observability.REQUIRES_ELECTRICAL_CORROBORATION
    assert "IEC 62446-3" in str(hotspot.iec_reference)

    diode = registry.get_entry("FLT-DIODE-001")
    assert diode is not None
    assert diode.category == FaultCategory.SUBSTRING_LEVEL
    assert diode.observability == Observability.VISUAL_OBSERVABLE

    offline = registry.get_entry("FLT-MOD-001")
    assert offline is not None
    assert offline.category == FaultCategory.MODULE_LEVEL


def test_taxonomy_label_mappings():
    """Verify mapping of public dataset labels to canonical entries."""
    registry = TaxonomyRegistry()

    # 1. Defensible direct mapping
    m_hotspot = registry.map_label("InfraredSolarModules", "Hot-Spot")
    assert m_hotspot is not None
    assert m_hotspot.canonical_code == "FLT-CELL-001"
    assert m_hotspot.confidence == MappingConfidence.DEFENSIBLE

    # 2. Defensible diode mapping
    m_diode = registry.map_label("InfraredSolarModules", "Diode")
    assert m_diode is not None
    assert m_diode.canonical_code == "FLT-DIODE-001"
    assert m_diode.confidence == MappingConfidence.DEFENSIBLE

    # 3. Ambiguous mapping (wafer cracking in IR without localized heating)
    m_crack = registry.map_label("InfraredSolarModules", "Cracking")
    assert m_crack is not None
    assert m_crack.confidence == MappingConfidence.AMBIGUOUS
    assert "ambiguous" in m_crack.rationale.lower()

    # 4. Unsupported non-defect glare artifact
    m_glare = registry.map_label("PV-Hawk", "glare")
    assert m_glare is not None
    assert m_glare.confidence == MappingConfidence.UNSUPPORTED
    assert m_glare.canonical_code is None


def test_taxonomy_unknown_label_handling():
    """Verify querying an unmapped label returns None."""
    registry = TaxonomyRegistry()
    m_unknown = registry.map_label("UnknownDataset", "NonExistentDefect")
    assert m_unknown is None
