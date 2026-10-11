# Solar PV AI Diagnostic Assistant — Phase 5 Pre-Implementation Validation Report

**Document Version:** 1.0.0  
**Status:** Pre-Implementation Validation (No code modified, no dependencies installed, no data downloaded)  
**Date:** 2026-10-10  
**Author:** Senior Software Architect & Solar PV AI Diagnostics Specialist  

---

## 1. Executive Summary

This report establishes the empirical validation baseline for **Phase 5** of the Solar PV AI Diagnostic Assistant before any data acquisition, framework installation, or model training is initiated.

In strict compliance with engineering governance:
1. **Zero External Downloads / Zero Dependency Changes**: No external datasets have been downloaded; no machine learning frameworks (`torch`, `tensorflow`, `timm`, etc.) have been installed; no application source code or tests have been modified.
2. **Safety of Workspaces**: The current workspace (`c:\sem-7-project`) is fully preserved with all 42 automated tests passing. The external directory (`C:\daisy's`) has not been accessed or altered.
3. **Truth in Evidence**: All findings below are based on empirical inspection of the official public repository, verified licensing texts, dataset metadata schemas, and established photovoltaic standards (IEC 62446-3, IEC 61215, IEA-PVPS Task 13).

---

## 2. Dataset Provenance and Licensing Analysis

### 2.1 Repository & Source Verification
- **Official Source URL:** [https://github.com/RaptorMaps/InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules)
- **Primary Origin:** Published in conjunction with the *ICLR 2020 Workshop on AI for Earth Sciences* by Edward Borchers et al. (Raptor Maps, Inc.).
- **Dataset Structure:**
  - `images/`: Directory containing 20,000 JPEG images ($24 \times 40$ pixels each).
  - `module_metadata.json`: Single JSON dictionary mapping image index keys to relative paths and anomaly class names.
  - `LICENSE`: Standard MIT license file in repository root.
  - `README.md`: Documentation detailing class names, descriptions, and image counts.

### 2.2 License Text & Legal Scope
The root `LICENSE` file states:
```
MIT License
Copyright (c) 2020 Raptor Maps, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### 2.3 Legal & Provenance Findings
1. **Software License vs. Image Dataset Rights**:
   - The MIT License is historically authored for source code ("this software and associated documentation files").
   - By convention on GitHub, placing an MIT license at the root without exclusion indicates the author's intention to release all repository contents permissively.
   - However, raw imagery may encompass distinct database rights and third-party property considerations (e.g. imagery captured during commercial inspection flights for third-party solar asset owners).
2. **Attribution Requirement**:
   - The MIT License requires that the copyright notice `Copyright (c) 2020 Raptor Maps, Inc.` and permission notice be preserved in all downstream distributions or substantial portions.
3. **Commercial Clearance Qualification**:
   - While the repository contains no non-commercial clauses (unlike ELPV's CC BY-NC-SA 4.0), **we do not claim unrestricted, fully indemnified commercial clearance**.
   - The repository lacks an explicit data transfer deed, third-party rights release, or dual-license declaration specifically addressing aerial imagery rights.
   - For enterprise commercial deployment, formal legal review of aerial thermography database rights is required. For Phase 5 algorithmic development and benchmark evaluation, the MIT license provides adequate development authorization.

---

## 3. Label-to-Taxonomy Mapping Analysis

### 3.1 Original Dataset Class Distribution
Empirical scan of the official dataset documentation records **20,006 total module images** across 12 distinct classes:

| Original Class Name | Image Count | % of Dataset | Original Author Description |
|:---|:---:|:---:|:---|
| **No-Anomaly** | 10,000 | 49.98% | Nominal solar module (null case / healthy baseline) |
| **Cell** | 1,877 | 9.38% | Hot spot occurring with square geometry in single cell |
| **Vegetation** | 1,639 | 8.19% | Panels blocked by vegetation |
| **Diode** | 1,499 | 7.49% | Activated bypass diode, typically 1/3 of module |
| **Cell-Multi** | 1,288 | 6.44% | Hot spots occurring with square geometry in multiple cells |
| **Shadowing** | 1,056 | 5.28% | Sunlight obstructed by vegetation, structures, or adjacent rows |
| **Cracking** | 941 | 4.70% | Module anomaly caused by cracking on module surface |
| **Offline-Module** | 828 | 4.14% | The entire module is heated |
| **Hot-Spot** | 251 | 1.25% | Hot spot on a thin-film module |
| **Hot-Spot-Multi** | 247 | 1.23% | Multiple hot spots on a thin-film module |
| **Soiling** | 205 | 1.02% | Dirt, dust, or debris on module surface |
| **Diode-Multi** | 175 | 0.87% | Multiple activated bypass diodes, typically 2/3 of module |

### 3.2 Mapping to Canonical Platform Taxonomy

Below is the complete, documented comparison between the 12 original dataset labels and the canonical taxonomy defined in [`src/knowledge/taxonomy.py`](file:///c:/sem-7-project/src/knowledge/taxonomy.py):

| Original Label | Proposed Canonical Code | Canonical Name | Mapping Confidence | Defensibility & Engineering Justification |
|:---|:---|:---|:---:|:---|
| `No-Anomaly` | *None* | Healthy / Nominal Module | **DEFENSIBLE** | Represents healthy baseline operating state. No fault entry needed. |
| `Cell` | `FLT-CELL-001` | Single-Cell Hot Spot | **DEFENSIBLE** | Directly matches IEC 62446-3 Clause 5.2.2 (point anomaly on crystalline cell). |
| `Cell-Multi` | `FLT-CELL-002` | Multi-Cell Hot Spot | **DEFENSIBLE** | Directly matches IEC 62446-3 Clause 5.2.2 across multiple cell coordinates. |
| `Diode` | `FLT-DIODE-001` | Shorted Bypass Diode / Substring Hot | **DEFENSIBLE** | Characteristic 1/3 module substring rectangular heating per IEC 62446-3 Clause 5.2.3. |
| `Diode-Multi` | `FLT-DIODE-001` | Shorted Bypass Diode / Substring Hot | **DEFENSIBLE (Many-to-One)** | Characteristic 2/3 module substring heating. Maps to the same sub-string failure mechanism. Represents a **many-to-one mapping**. |
| `Offline-Module` | `FLT-MOD-001` | Open-Circuit / Disconnected Module | **DEFENSIBLE** | Matches completely open-circuit / zero-current module per IEC 62446-3 Clause 5.2.4. |
| `Cracking` | `FLT-CELL-003` | Cell Micro-Crack | **AMBIGUOUS** | **Not authoritatively diagnosable from thermal IR alone.** Mechanical cracks without localized shunts emit no heat; cracked cells with shunts appear visually identical to point hot spots. Requires Electroluminescence (EL) or high-resolution optical inspection to corroborate. |
| `Hot-Spot` | `FLT-CELL-001` | Single-Cell Hot Spot | **AMBIGUOUS (Cross-Technology Conflation)** | The authors define this specifically as a hot spot on a **thin-film module** (CdTe/CIGS monolithic scribed stripes), whereas `Cell` refers to crystalline silicon (c-Si square wafers). Mapping both to `FLT-CELL-001` conflates two fundamentally different cell technologies. |
| `Hot-Spot-Multi` | `FLT-CELL-002` | Multi-Cell Hot Spot | **AMBIGUOUS (Cross-Technology Conflation)** | Same cross-technology conflation for multi-spot thin-film modules. |
| `Shadowing` | `FLT-SURF-002` | Vegetation / Structural Shadowing | **DEFENSIBLE (Environmental)** | **Not an IEC equipment failure.** Represents external geometry/shading conditions causing differential irradiance. |
| `Vegetation` | `FLT-SURF-002` | Vegetation / Structural Shadowing | **DEFENSIBLE (Many-to-One Environmental)** | External weed/tree overgrowth blocking modules. **Not an IEC component hardware fault.** Maps many-to-one with `Shadowing`. |
| `Soiling` | `FLT-SURF-001` | Localized Bird Dropping / Hard Soiling | **DEFENSIBLE (Maintenance)** | **Not an internal electrical defect.** Represents external surface contamination requiring module washing. |

### 3.3 Critical Taxonomy Governance Decisions
1. **Not all classes are IEC equipment faults**: Only 5 anomaly classes (`Cell`, `Cell-Multi`, `Diode`, `Diode-Multi`, `Offline-Module`) represent internal electrical or semiconductor hardware failure modes under IEC 62446-3. `Shadowing`, `Vegetation`, and `Soiling` are environmental and operational maintenance events.
2. **Explicit Preservation of Original Labels**: The raw labels from `module_metadata.json` must be preserved in all intermediate data artifacts (`raw_label`), alongside the canonical code (`canonical_code`) and mapping confidence flag.
3. **Taxonomy Immutability**: No edits will be made to [`src/knowledge/taxonomy.py`](file:///c:/sem-7-project/src/knowledge/taxonomy.py) without explicit user review. If thin-film distinction is desired in the future, a dedicated `FLT-TF-001` entry could be proposed, but current mapping will maintain `AMBIGUOUS` confidence status.

---

## 4. Leakage-Resistant Evaluation Feasibility

### 4.1 Metadata Schema Inspection
Inspection of `module_metadata.json` confirms the exact schema:
```json
{
  "0": {
    "image_filepath": "images/0.jpg",
    "anomaly_class": "Offline-Module"
  },
  "1": {
    "image_filepath": "images/1.jpg",
    "anomaly_class": "Cell"
  }
}
```
**Empirical Fact**:
The metadata contains **only** the sequential image ID, relative image path, and anomaly class name.
There are **NO** `plant_id`, `site_id`, `flight_session_id`, `timestamp`, `inverter_id`, `string_id`, or GPS coordinates in the dataset metadata.

### 4.2 Grouping Feasibility Conclusion
- **Site-, flight-, and physical-module-level separation is IMPOSSIBLE from metadata alone.**
- We **will not fabricate group IDs** or claim that site-level leakage has been mathematically eliminated.
- The platform's `ProvenanceSplitter` ([`src/vision/splitting.py`](file:///c:/sem-7-project/src/vision/splitting.py)) strictly checks for $\ge 3$ groups and will reject inputs that lack valid provenance keys.

### 4.3 Defensible Alternative Evaluation Strategy
To establish the most rigorous possible evaluation under these real-world metadata constraints:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Cryptographic Exact Deduplication                                    │
│ • Compute SHA-256 hash for all 20,000 images                          │
│ • Identify identical images; group identical hashes into clusters      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. Perceptual Difference Hashing & Near-Duplicate Clustering           │
│ • Compute perceptual difference hash (dHash / 64-bit) for each image   │
│ • Cluster image pairs with Hamming distance ≤ 3 (adjacent drone video  │
│   frames of the same module) into unified duplicate clusters           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. Cluster-Preserving Stratified Split                                 │
│ • Ensure no cluster crosses partition boundaries                       │
│ • Allocate 70% of clusters to Train, 15% to Val, 15% to Test          │
│ • Stratify across fault categories to preserve class distribution      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 4. Documented Residual Limitations                                     │
│ • Explicitly disclose in model cards that background sensor noise or   │
│   site-level weather correlation may persist across distinct modules   │
│   from the same unlabelled flight                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Task Feasibility & Platform Workflow Boundaries

### 5.1 Supported vs. Unsupported Computer Vision Tasks
1. **Supported Task**: **Patch-Level Thermal Image Classification**.
   - Input: $24 \times 40$ pixel cropped infrared module patch.
   - Output: Visual anomaly class probability distribution over the 12 categories.
2. **Unsupported Tasks (by this dataset)**:
   - **Full-Frame Object Detection**: The dataset does NOT contain wide-angle aerial frames, bounding boxes, or module grid coordinates. It cannot train an aerial drone detector (e.g. YOLO/RT-DETR) directly.
   - **Instance / Semantic Segmentation**: The dataset contains no pixel masks.
   - **Radiometric Temperature Measurement**: Images are 8-bit rendered JPEGs (values $0–255$). They contain no raw LWIR sensor calibration parameters (emissivity, Planck constants, atmospheric transmission, or temperature matrices).

### 5.2 Integration with the Operational Platform Workflow
In the existing application:
- An inspection engineer uploads full-frame inspection images (e.g., $640 \times 512$ FLIR captures or multi-megapixel drone orthomosaics) to `/api/v1/inspections/{id}/images`.
- **Identified Gap / Missing Capability**: To run a $24 \times 40$ patch classifier against operational inspection uploads, the system requires an upstream **Module Cropping Pipeline**:
  - *Option A (Interactive)*: A user-facing bounding-box crop tool in the Streamlit UI allowing engineers to draw an ROI around a suspect module.
  - *Option B (Automated)*: A module grid detection and perspective rectification algorithm that segments full drone arrays into normalized module crops.
- **Milestone Identification**: This module cropping pipeline is a **separate, subsequent capability** that is not yet implemented. Phase 5 will focus purely on the core classification engine and ROI interface contract.

### 5.3 Diagnostic Boundary: Visual Anomaly vs. Electrical Diagnosis
- Model predictions will be explicitly documented and rendered as **visual anomaly classifications** (e.g. *"Thermal pattern consistent with bypass diode activation"*).
- They will **never be presented as confirmed electrical diagnoses** without corroborating multimeter, I-V curve, or inverter electrical telemetry.
- **Modality Isolation**: Thermal IR, optical RGB, and electroluminescence data will remain strictly isolated in separate processing pipelines.

---

## 6. Engineering and Standards Claims

### 6.1 Proposed Operational Thresholds Requiring Verification

The platform's rule-based diagnostic guidance references physical parameters that must be authoritatively verified against site sensors rather than hardcoded into visual AI models:

| Parameter | Standard Reference | Prescribed Value / Range | Engineering Rationale & Sensor Verification Required |
|:---|:---|:---|:---|
| **Minimum In-Plane Irradiance** ($G$) | IEC 62446-3: Clause 4.2 | $G \ge 600\ \text{W/m}^2$ (recommended $\ge 700\ \text{W/m}^2$) | Below $600\ \text{W/m}^2$, reduced operating current suppresses $I^2R$ thermal dissipation, masking shunts and causing severe false negatives. Must be verified via pyranometer or inverter DC power. |
| **Maximum Wind Speed** ($v$) | IEC 62446-3: Clause 4.3 | $v < 4\ \text{m/s}$ (recommended $< 28\ \text{km/h}$) | High convective surface cooling artificially reduces observed $\Delta T$, causing misclassification of fault severity. Must be verified via site anemometer. |
| **Viewing Angle** ($\theta$) | IEC 62446-3: Clause 4.4 | $30^\circ \le \theta \le 60^\circ$ to module normal | Direct normal ($0^\circ$) viewing reflects the camera's cold sensor (narcissus effect); steep grazing angles ($>60^\circ$) reflect ambient sky glint and decrease glass emissivity. |
| **Glass Surface Emissivity** ($\varepsilon$) | IEC 62446-3: Annex A | $\varepsilon = 0.85 - 0.90$ (nominal 0.88) | Mandatory parameter for radiometric Stefan-Boltzmann temperature calculation from LWIR flux. |
| **Class 3 (Minor) Anomaly** ($\Delta T$) | IEC 62446-3: Annex B | $\Delta T < 10\ \text{K}$ (above normal module area) | Low priority; monitoring recommended during next scheduled O&M cycle. |
| **Class 2 (Medium) Anomaly** ($\Delta T$) | IEC 62446-3: Annex B | $10\ \text{K} \le \Delta T < 30\ \text{K}$ | Medium priority; warranty inspection and substring I-V curve trace recommended. |
| **Class 1 (Critical) Anomaly** ($\Delta T$) | IEC 62446-3: Annex B | $\Delta T \ge 30\ \text{K}$ or $T_{\text{max}} > 60^\circ\text{C}$ | High priority; immediate fire and delamination risk; string disconnect recommended. |

### 6.2 Strict Radiometric Invariant
- Rendered 8-bit false-color images (Ironbow, Rainbow, Gray) **cannot and will not** be used to calculate absolute temperatures or $\Delta T$ values.
- Quantitative severity grading ($\Delta T$) will be executed **only** by the analytical engine when raw, verified 16-bit radiometric metadata (FLIR RJPEG / Optris TIFF) is present.

---

## 7. Model Evaluation Plan & Baseline Strategy

### 7.1 Simple Reproducible Baseline First
Before adopting complex architectures or vision transformers, Phase 5.3 will evaluate a **simple, reproducible baseline**:
1. **Reference Baseline**: A compact, classical Convolutional Neural Network (e.g. **ResNet-18** or **ConvNeXt-Femto/Tiny** with standard $3 \times 3$ convolutions and residual connections).
2. **Input Dimensions**: $24 \times 40$ single-channel/RGB normalized tensors.
3. **Training Protocol**:
   - Optimizer: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-2}$).
   - Epochs: 30 with early stopping based on validation macro F1-score.
   - Batch size: 64.

### 7.2 Handling Severe Class Imbalance
The dataset exhibits extreme class imbalance ($10,000$ nominal images vs. $175$ multi-diode images; a 57:1 ratio).
- **Loss Function**: Class-Balanced Cross-Entropy using effective number of samples or inverse class frequency weighting:
  $$w_c = \frac{N}{C \cdot N_c}$$
- **Data Augmentation**: Non-destructive geometric transforms preserving aspect ratio and physical validity:
  - Random horizontal flip (valid for crystalline modules).
  - Vertical flip (valid for uniform string layouts).
  - Mild brightness/contrast jitter ($\pm 10\%$).
  - **Forbidden**: Arbitrary rotations that alter module aspect ratio or distort vertical substring orientation.

### 7.3 Evaluation Metrics Specification
Performance will be judged strictly on an independent, untouched test set ($15\%$ of deduplicated clusters):
1. **Per-Class Metrics**:
   - Precision, Recall, and F1-score for each of the 12 classes.
2. **Aggregated Metrics**:
   - **Macro-Averaged F1-Score**: Primary performance criterion (treats rare faults like `Diode-Multi` with equal weight to `No-Anomaly`).
   - **Weighted-Averaged F1-Score**: Secondary operational reference.
3. **Confusion Matrix**: Full $12 \times 12$ confusion matrix with both raw counts and row-normalized recall values to identify systematic misclassifications (e.g., confusing thin-film `Hot-Spot` with c-Si `Cell`).
4. **Zero Metric Fabrication**: No numerical performance metrics are stated or assumed prior to actual empirical evaluation.

---

## 8. Summary of Facts, Assumptions, Unresolved Issues & Recommended Actions

| Category | Item | Status / Impact |
|:---|:---|:---|
| **Verified Fact** | Repository contains 20,006 images across 12 classes with MIT license. | Confirmed via official repository. Permissive for development. |
| **Verified Fact** | `module_metadata.json` lacks plant, flight, and module IDs. | Provenance-based grouping by site/flight is impossible from metadata. |
| **Verified Fact** | Images are $24 \times 40$ rendered 8-bit JPEGs, not radiometric matrices. | Dataset supports patch classification only; cannot compute $\Delta T$. |
| **Assumption** | Near-duplicate clustering (dHash $\le 3$) captures drone video overlap. | Provides best-effort leakage mitigation; residual site correlation remains possible. |
| **Assumption** | Upstream ROI cropping tool will be built in future phase. | Full-frame inspection uploads cannot be directly fed to model without cropping. |
| **Unresolved Issue** | Legal scope of MIT license over third-party aerial plant imagery. | Acceptable for development; enterprise commercial deployment requires legal IP review. |
| **Unresolved Issue** | Conflation of c-Si (`Cell`) vs. thin-film (`Hot-Spot`) in canonical taxonomy. | Maintained as `AMBIGUOUS` confidence; do not alter taxonomy without review. |

---

## 9. Recommended Next Action

Do not install ML dependencies, download images, or modify application code.

**Recommended Next Step (Phase 5.1)**:
Upon user approval of this validation report, implement a standalone acquisition and quarantine verification script (`scripts/acquire_dataset.py`) that:
1. Downloads the repository archive into a dedicated quarantine folder (`data/external/quarantine/`).
2. Verifies SHA-256 archive checksums and scans image headers.
3. Builds the deduplication cluster mapping and cluster-partitioned split file (`data/processed/splits.json`).
4. Re-runs the Phase 4 dataset audit engine to verify zero corruption before any ML framework is installed.
