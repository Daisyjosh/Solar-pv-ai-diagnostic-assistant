"""Engineering Standards Knowledge Store and RAG Document Retrieval.

Ingests approved solar PV technical standards (IEC 62446-3, IEC 61215,
IEA-PVPS Task 13) and provides TF-IDF semantic chunk retrieval with citations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeChunk:
    """Indexed fragment of engineering documentation."""

    chunk_id: str
    doc_title: str
    standard_reference: str
    section: str
    content: str
    keywords: List[str]


# Authoritative engineering documents built directly into platform
DEFAULT_KNOWLEDGE_DOCUMENTS: List[KnowledgeChunk] = [
    KnowledgeChunk(
        chunk_id="iec-62446-3-s5.1",
        doc_title="IEC 62446-3: Photovoltaic (PV) systems - Outdoor infrared thermography",
        standard_reference="IEC 62446-3 Clause 5.1",
        section="Environmental and Operational Inspection Boundaries",
        content=(
            "Thermographic inspections must be performed under clear sky conditions with in-plane solar "
            "irradiance of at least 600 W/m2 (recommended > 700 W/m2). Wind speed must not exceed 4 m/s "
            "to prevent forced convective surface cooling from masking internal thermal gradients (delta-T). "
            "Thermal camera angle of view must be maintained between 50° and 80° relative to module surface "
            "to avoid specular glass reflections and minimize emissivity variation."
        ),
        keywords=["irradiance", "wind", "delta-t", "angle", "emissivity", "convection"],
    ),
    KnowledgeChunk(
        chunk_id="iec-62446-3-s5.2.2",
        doc_title="IEC 62446-3: Photovoltaic (PV) systems - Outdoor infrared thermography",
        standard_reference="IEC 62446-3 Clause 5.2.2",
        section="Thermal Anomaly Classification & Severity Criteria",
        content=(
            "Point thermal anomalies (single cell delta-T between 10°C and 20°C above adjacent normal cells) "
            "are classified as Class 2 (Major) anomalies requiring scheduling of maintenance and electrical "
            "verification. Point anomalies with delta-T > 20°C or with visible backsheet degradation / browning "
            "are classified as Class 3 (Critical), presenting imminent fire, delamination, and power loss risks."
        ),
        keywords=["hot spot", "point anomaly", "severity", "delta-t", "class 2", "class 3", "temperature"],
    ),
    KnowledgeChunk(
        chunk_id="iec-62446-3-s5.2.3",
        doc_title="IEC 62446-3: Photovoltaic (PV) systems - Outdoor infrared thermography",
        standard_reference="IEC 62446-3 Clause 5.2.3",
        section="Bypass Diode Conduction & Substring Faults",
        content=(
            "When a bypass diode is activated or shorted, an entire longitudinal substring of cells exhibits "
            "a uniform elevated temperature. A standard crystalline silicon module with 3 bypass diodes will "
            "lose approximately one-third (33.3%) of its rated voltage output per activated diode. If all diodes "
            "are conducting or the module is open-circuit, the entire module appears uniformly heated under irradiance."
        ),
        keywords=["bypass diode", "substring", "voltage", "voc", "short circuit", "band", "junction box"],
    ),
    KnowledgeChunk(
        chunk_id="iea-pvps-task13-shading",
        doc_title="IEA-PVPS Task 13: Review on Failures of Photovoltaic Modules",
        standard_reference="IEA-PVPS Task 13 Section 3.2",
        section="Partial Shading, Soiling, and Reverse Bias Breakdown",
        content=(
            "Partial shading caused by vegetation overgrowth, structural obstructions, or localized soiling "
            "forces shaded cells to operate in reverse bias. Shaded cells dissipate power produced by unshaded "
            "cells in the series string, leading to localized heating (hotspotting). Sustained localized heating "
            "accelerates ethyl vinyl acetate (EVA) encapsulant yellowing and backsheet embrittlement."
        ),
        keywords=["shading", "soiling", "vegetation", "reverse bias", "eva", "encapsulant", "hotspotting"],
    ),
    KnowledgeChunk(
        chunk_id="iec-61215-m10.9",
        doc_title="IEC 61215: Terrestrial photovoltaic (PV) modules - Design qualification",
        standard_reference="IEC 61215 MQT 10.9",
        section="Hot-Spot Endurance Testing & Wafer Microcracks",
        content=(
            "Microcracks in silicon wafers can arise during manufacturing, transport, or installation mechanical loading. "
            "Wafer cracks that isolate cell fragments lead to current bottlenecks and localized resistance dissipation. "
            "Thermal thermography can identify thermal dissipation along crack paths, but dark-field electroluminescence (EL) "
            "is mandatory to conclusively map inactive wafer fragments and crack propagation."
        ),
        keywords=["microcrack", "cracking", "electroluminescence", "el", "wafer", "mechanical stress"],
    ),
    KnowledgeChunk(
        chunk_id="pv-safety-dc-arc",
        doc_title="Solar PV O&M Safety Manual: High Voltage DC Disconnection",
        standard_reference="OSHA 1910.303 / IEC 60364-7-712",
        section="DC Arc Flash Prevention and Connector Inspection",
        content=(
            "Never disconnect MC4 connectors under load. DC currents do not have a natural zero-crossing, and "
            "breaking connections under load causes severe high-temperature DC arc flashes capable of fatal injury "
            "and catastrophic array fires. Always open the inverter DC isolator or string disconnect breaker before "
            "physically testing, inspecting, or re-coupling module lead connectors."
        ),
        keywords=["safety", "mc4", "dc arc", "disconnection", "connector", "hazard", "isolator"],
    ),
]


class KnowledgeDocumentStore:
    """Vector document store using TF-IDF cosine similarity for offline RAG retrieval."""

    def __init__(self, chunks: Optional[List[KnowledgeChunk]] = None):
        self.chunks = chunks or DEFAULT_KNOWLEDGE_DOCUMENTS
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._fit_corpus()

    def _fit_corpus(self) -> None:
        """Fit TF-IDF index across corpus."""
        corpus = [f"{c.doc_title} {c.section} {c.content} {' '.join(c.keywords)}" for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.05) -> List[Dict[str, Any]]:
        """Retrieve most relevant technical documentation chunks for a query."""
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= min_similarity:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.doc_title,
                    "standard_reference": chunk.standard_reference,
                    "section": chunk.section,
                    "content": chunk.content,
                    "similarity_score": round(score, 4),
                })
        return results
