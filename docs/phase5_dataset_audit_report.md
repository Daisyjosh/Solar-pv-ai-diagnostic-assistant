# Solar PV AI Diagnostic Assistant — Phase 5.1 Dataset Audit & Quarantine Report

**Audit Timestamp (UTC):** `2026-10-10T11:19:48.776353+00:00`  
**Dataset Name:** `InfraredSolarModules`  
**Quarantine Location:** `data/external/quarantine/`  
**Evaluation Status:** Pre-Training Quarantine Review (Zero Models Trained)  

---

## 1. Provenance and Licensing Tracking

* **Official Source URL:** [https://github.com/RaptorMaps/InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules)
* **Archive Filename:** `2020-02-14_InfraredSolarModules.zip`
* **Computed Archive SHA-256:** `b82c706bc719b045ac4f8930570d81767a8a170d0998ca3e09283b585db05b5e`
* **Upstream Published Checksum:** `None Provided`
* **Integrity Note:** No trusted published checksum (SHA-256 or MD5) was provided upstream by Raptor Maps in repository documentation or releases. The recorded SHA-256 is an empirical baseline computed from the initial official repository download.
* **Repository License:** MIT License (Copyright (c) 2020 Raptor Maps, Inc.)
* **Attribution Notice:** Borchers, Edward et al. 'InfraredSolarModules', ICLR 2020 Workshop on AI for Earth Sciences. Copyright (c) 2020 Raptor Maps, Inc.

### Licensing Scope & Commercial Clearance Status
> [!WARNING]
> **Commercial Clearance Qualification**:
> The repository root specifies standard MIT text covering 'software and associated documentation'. Because aerial drone thermography may encompass distinct database rights and third-party asset owner considerations, this dataset is authorized for Phase 5 technical development and benchmarking, but unrestricted commercial clearance is NOT claimed without dedicated legal IP review.

#### Unresolved Legal Questions
- Does the standard MIT license text unambiguously grant database and image reproduction rights for commercial redistribution?
- Are thermal images captured during commercial client flights subject to third-party facility confidentiality or utility asset NDAs?
- Does the lack of an explicit Creative Commons or Open Data Commons deed restrict embedding raw images in commercial software distributions?

---

## 2. Ingestion & Archive Structure Integrity

* **Total Images in Metadata (`module_metadata.json`):** 20000
* **Total Image Files on Disk:** 20000
* **Valid Images Ingested:** 20000
* **Corrupt Image Files:** 0
* **Missing Image Files:** 0
* **Unexpected Image Files:** 2
* **Unexpected Artifact Details:**
  - `InfraredSolarModules/.DS_Store`
  - `__MACOSX/InfraredSolarModules/._.DS_Store`
* **Image Format & Dimensions:** Verified 100% standard JPEG format, $24 \times 40$ pixels, 3 channels (RGB).

### Discrepancy Analysis vs. Official Documentation
| Class Name | Documented in README | Observed in Metadata | Discrepancy |
|:---|:---:|:---:|:---:|
| `Soiling` | 205 | 204 | -1 |
| `Offline-Module` | 828 | 827 | -1 |
| `Hot-Spot-Multi` | 247 | 246 | -1 |
| `Cracking` | 941 | 940 | -1 |
| `Hot-Spot` | 251 | 249 | -2 |

---

## 3. Observed Class Distribution

| Class Name | Observed Images | % of Dataset | Documented in README |
|:---|:---:|:---:|:---:|
| `No-Anomaly` | 10000 | 50.00% | 10000 |
| `Cell` | 1877 | 9.38% | 1877 |
| `Vegetation` | 1639 | 8.20% | 1639 |
| `Diode` | 1499 | 7.50% | 1499 |
| `Cell-Multi` | 1288 | 6.44% | 1288 |
| `Shadowing` | 1056 | 5.28% | 1056 |
| `Cracking` | 940 | 4.70% | 941 |
| `Offline-Module` | 827 | 4.13% | 828 |
| `Hot-Spot` | 249 | 1.24% | 251 |
| `Hot-Spot-Multi` | 246 | 1.23% | 247 |
| `Soiling` | 204 | 1.02% | 205 |
| `Diode-Multi` | 175 | 0.88% | 175 |

---

## 4. Deduplication & Candidate Cluster Analysis

* **Verified Exact Byte Duplicate Groups (Identical SHA-256):** 22 groups (22 pairs, 44 images).
  - Exact Pairs with Conflicting Upstream Labels: 6 pairs (12 images) due to upstream annotation noise.
* **Perceptual dHash Distance 0 Pairs:**
  - Identical-Byte dHash 0 Pairs: 22 pairs (coincides with exact SHA-256 duplicates).
  - Different-Byte dHash 0 Pairs: 46,025 pairs (24,516 pairs carry conflicting upstream labels).
* **Candidate Near-Duplicate Pairs (dHash Hamming 1 to 3):** 916,068 pairwise connections.
* **Connected Components at Hamming $\le 3$:** 10,362 components.
  - Largest Connected Component: 9,356 images (46.78% of dataset).
  - Component Nature: Spans all 12 classes; internal pairwise Hamming distances reach 51 bits (mean 16.78 bits). Extreme pair (IDs 6104 and 4983) has Hamming distance 34 bits.
  - Scientific Finding: The 9,356 component is an **artifact of transitive chaining** across a continuous low-contrast gradient manifold, **not** a cluster of physical module duplicates.

> [!NOTE]
> **Deduplication Methodology & Constraints**:
> - **Exact Byte Duplicates**: Inviolable atomic units. Must never cross partition boundaries.
> - **Perceptual Hashing (dHash)**: An approximate visual gradient heuristic. Low Hamming distance does **NOT** establish physical-module identity.
> - **Different-Byte dHash 0 Pairs**: Must **not** automatically be treated as physical duplicates.
> - **Label Preservation**: Original upstream labels must be preserved; conflicting-label pairs must remain documented.
> - See full documentation in [`docs/dataset_split_constraints.md`](dataset_split_constraints.md).

---

## 5. Label Ambiguities and Domain Scope

| Label / Category | Technical Ambiguity & Diagnostic Boundary |
|:---|:---|
| `Cracking` | Mechanical wafer cracks cannot be confirmed in thermal IR alone without EL imaging. |
| `Hot-Spot / Hot-Spot-Multi` | Defined for thin-film modules; conflating with crystalline cells blurs distinct PV technologies. |
| `Shadowing / Vegetation` | External site and environmental conditions, NOT internal IEC component equipment defects. |
| `Soiling` | Surface soiling is a cleaning/maintenance event, not an internal electrical defect. |

---

## 6. Known Methodological Limitations

- Site-, plant-, and flight-level metadata are completely absent in module_metadata.json.
- Perceptual hashing identifies visual pattern similarity, NOT ground-truth physical module identity.
- Cluster-aware partitioning eliminates visual duplicate contamination but CANNOT establish site-independent generalization.
- 8-bit rendered JPEGs do not support radiometric temperature or delta-T calculation.

> [!IMPORTANT]
> **Zero Site-Level Leakage Prevention Guarantee**:
> Because `plant_id`, `site_id`, and `flight_id` are completely absent from `module_metadata.json`, site-level data leakage **cannot be mathematically eliminated**. The proposed cluster-aware split partitions data by visual duplicate clusters, but residual background correlation from unlabelled flights remains possible.

---

## 7. Operational Baseline & Next Actions

1. **Approved Baseline**: Option A (`data/processed/proposed_splits.json`) is retained as the provisional Phase 5.1 baseline.
2. **Rejected Option B**: Option B (`data/processed/proposed_splits_option_b.json`) was rejected due to severe validation/test class starvation.
3. **Formal Constraints Recorded**: All 6 core split constraints are formally codified in [`docs/dataset_split_constraints.md`](dataset_split_constraints.md).
4. **Hold State**: Do not install ML frameworks, train a model, or begin Phase 5.2 without explicit user authorization.
