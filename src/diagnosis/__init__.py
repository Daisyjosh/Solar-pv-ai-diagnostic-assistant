"""Diagnostic reasoning engine and multi-modal evidence fusion.

Engineering Principles:
- Distinguishes clearly between:
  1. Observed image evidence (hotspots, substring patterns)
  2. Measured telemetry (Voc, Isc, power, irradiance, ambient/module temp)
  3. Engineering inferences (candidate root causes)
  4. Verified/confirmed diagnoses (corroborated by electrical measurements)
- Formulates requests for missing electrical measurements whenever visual evidence
  is insufficient to establish root cause.
- Quantifies diagnostic uncertainty and severity in accordance with IEC 62446-3 guidelines.
"""

__all__ = []
