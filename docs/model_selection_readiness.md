# Solar PV AI Diagnostic Assistant — Model Selection & Architecture Readiness

## 1. Executive Summary

This document evaluates candidate computer vision architectures and formalizes the data-readiness strategy for the **Solar PV AI Diagnostic Assistant**.

In strict accordance with Phase 4 policy:
- No model weights have been downloaded.
- No model has been trained.
- No predictive accuracy or detection confidence is claimed.
- YOLO is not assumed to be the default or only viable architecture.

---

## 2. Problem Formulation & Task Requirements

Solar PV visual inspection encompasses two distinct imaging modalities with fundamentally different feature distributions:

```
[ Modality 1: Optical RGB Inspection ]
  ├─ Sensor: High-resolution visible spectrum (4K / 20MP RGB)
  ├─ Physics: Surface reflectance, absorption, and refraction
  ├─ Targets: Glass cracks, snail trails, bird soiling, dust/soiling accumulation, burn marks
  └─ Challenges: Sun glint, cloud reflections, angle-dependent glare

[ Modality 2: Thermal Infrared (IR) Thermography ]
  ├─ Sensor: Long-Wave Infrared (LWIR 7.5–14 µm, 640x512 radiometric or rendered)
  ├─ Physics: Radiated thermal flux proportional to localized I²R power dissipation
  ├─ Targets: Single-cell shunts, shorted bypass diodes, open modules, PID gradients
  └─ Challenges: Ambient temperature drift, convective wind cooling, false-color quantization
```

### Candidate Task Formulations

| Formulation | Description | Pros | Cons |
|:---|:---|:---|:---|
| **Patch-Level Classification** | Predicts defect class for pre-cropped individual solar modules. | Low annotation cost; works on standard image classification backbones. | Requires an external module-cropping pipeline; cannot localize cell-level coordinates or multi-fault modules. |
| **Bounding-Box Object Detection** | Predicts bounding boxes `[x, y, w, h]` and class labels across full plant frames. | Direct end-to-end aerial analysis; rich public benchmark support (InfraredSolarModules, PV-Hawk); fast inference. | Approximate rectangular boxes; does not segment irregular diagonal cell cracks or diffuse PID patterns. |
| **Instance Segmentation** | Predicts pixel-exact polygonal masks for defective cells and sub-strings. | Precise area measurement of hot spots; calculates exact cell aspect ratios and sub-string coverage. | 4x to 6x higher annotation cost; slower inference latency. |

**Architectural Recommendation for Phase 5**:
Adopt a two-tier strategy:
1. **Tier 1 (Detection & Localization)**: Bounding-box object detection for wide-area aerial flight frames to locate anomalous modules, hot spots, and substring heating patterns.
2. **Tier 2 (Evidence & Radiometric Extraction)**: Cropping localized module regions of interest (ROIs) to calculate thermal contrast ($\Delta T$) and verify IEC 62446-3 severity criteria.

---

## 3. Candidate Model Architecture Evaluation

| Architecture | Paradigm | Strengths for Solar PV | Weaknesses / Risks | Recommended Use Case |
|:---|:---|:---|:---|:---|
| **RT-DETR** (Real-Time Detection Transformer) | Transformer (Deformable Attention) | Anchor-free architecture eliminates hand-crafted anchor tuning; superior global receptive field across repetitive solar module arrays; robust against aspect-ratio distortion in oblique drone shots. | Requires larger dataset volume (>5,000 annotated images) to train effectively from scratch or requires COCO transfer learning. | **Primary candidate for utility-scale aerial inspections.** |
| **Faster R-CNN / Mask R-CNN** (ResNet-50-FPN) | Two-Stage Detector | Gold standard localization accuracy on small objects (single-cell point hot spots); highly sample-efficient on modest datasets (<2,000 images). | Slower inference (~50–80ms on GPU); heavyweight memory footprint; less suitable for low-power edge devices. | **Benchmark reference and high-precision ground-truth verification.** |
| **YOLOv8 / YOLOv11** | Single-Stage CNN | Ultra-fast inference (<10ms on GPU/NPU); low memory footprint; readily exportable to ONNX/TensorRT for drone companion computers. | Single-stage anchor/grid sensitivity can yield false positives on repetitive array reflection artifacts. | **Candidate for real-time edge streaming on aerial inspection drones.** |

---

## 4. Current Dataset Gap & Empirical Findings

### Current Repository State (Phase 4 Scan)
- `data/external/`: **Empty** (`.gitkeep` only). No external benchmark datasets currently exist in the repository.
- `data/raw/inspections/`: Contains **8 test verification images** generated during Phase 3 automated testing.
- **Empirical Conclusion**: There are currently **zero operational field training datasets** in the workspace. Training a deep learning vision model at this stage would be methodologically fraudulent.

---

## 5. Lawful Data Acquisition & Annotation Roadmap for Phase 5

To achieve model-selection readiness before Phase 5 training, the following lawful data-acquisition pipeline is planned:

### 1. Ingestion of Open-Access Benchmarks
- **InfraredSolarModules Dataset**:
  - *Source*: FAU Erlangen-Nürnberg & Helmholtz Institute.
  - *Volume*: 20,000 thermal IR module patches across 11 defect categories.
  - *License*: Creative Commons Attribution 4.0 International (CC-BY 4.0).
  - *Applicability*: Patch-level classification and localized thermal anomaly evaluation.
- **PV-Hawk / UAV Aerial Thermography Datasets**:
  - *Source*: Open-access utility-scale drone orthomosaic flights.
  - *License*: Open-access academic/commercial evaluation.
  - *Applicability*: Full-frame multi-module object detection.

### 2. Annotation & Harmonization Protocol
- Ingest annotations and harmonize labels using [`src/knowledge/taxonomy.py`](file:///c:/sem-7-project/src/knowledge/taxonomy.py).
- Verify container magic bytes and check for corruptions using [`src/vision/dataset_audit.py`](file:///c:/sem-7-project/src/vision/dataset_audit.py).
- Partition datasets by site and flight transect using [`src/vision/splitting.py`](file:///c:/sem-7-project/src/vision/splitting.py) to prevent spatial data leakage.

### 3. Model Benchmark Evaluation in Phase 5
- Train candidate architectures (RT-DETR vs. Faster R-CNN vs. YOLOv8) under identical group-partitioned splits.
- Evaluate models using standard engineering metrics:
  - $\text{mAP}_{50}$ and $\text{mAP}_{50-95}$ across all standardized IEC fault classes.
  - Precision-Recall curves.
  - Confusion matrix highlighting ambiguous class overlap (e.g., cell crack vs. point hotspot).
