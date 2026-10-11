# Solar PV AI Diagnostic Assistant — Architecture & Domain Specification

## 1. Executive Summary

**Solar PV AI Diagnostic Assistant** is an enterprise operations, inspection, and predictive maintenance platform designed for utility-scale solar photovoltaic (PV) power plants, commercial & industrial (C&I) rooftop installations, and independent inspection service providers.

The platform combines multi-spectral computer vision (thermal infrared and high-resolution RGB), electrical string telemetry (I-V curves, open-circuit voltage $V_{oc}$, short-circuit current $I_{sc}$, operating point $V_{mp}/I_{mp}$), environmental meteorological data (plane-of-array irradiance $G$, ambient temperature $T_{amb}$, module temperature $T_{mod}$), and physics-based engineering reasoning into a coherent diagnostic workflow.

---

## 2. Core Architectural Principles

### 2.1 Rigorous Evidence Separation
In solar PV operations and maintenance, an infrared thermal gradient is **not** synonymous with an electrical fault diagnosis. The architecture enforces strict ontological boundaries across five distinct data and reasoning tiers:

```
[ Tier 1: Raw Evidence ]
  ├─ Thermal IR radiometric matrix / RJPEG / FLIR sensor stream
  ├─ Optical RGB imagery (orthomosaic, handheld, or UAV close-up)
  └─ In-situ meteorological data (irradiance G, ambient Tamb, wind speed)
          │
          ▼
[ Tier 2: Feature & Anomaly Extraction ]
  ├─ Thermal delta-T calculation (ΔT_cell, ΔT_substring, ΔT_module)
  ├─ Geometric segmentation (module boundary, cell coordinate grid)
  └─ Computer Vision detection (hot spot, multi-cell, bypass pattern, PID gradient)
          │
          ▼
[ Tier 3: Electrical Telemetry Corroboration ]
  ├─ Operating string current & voltage (I_string, V_string)
  ├─ Benchmark parameters (normalized STC Voc, Isc vs nominal datasheet)
  └─ SCADA inverter-level yield & clipping telemetry
          │
          ▼
[ Tier 4: Diagnostic Reasoning & Inference Engine ]
  ├─ Multi-modal evidence fusion & hypothesis ranking
  ├─ Uncertainty quantification (identifying incomplete evidence)
  └─ Dynamic measurement requests (prompting engineer for Voc, Isc, diode tests)
          │
          ▼
[ Tier 5: Decision Support & Corrective O&M Action ]
  ├─ IEC 62446-3 severity classification (Class 1, Class 2, Class 3)
  ├─ Power loss / yield degradation estimation
  └─ Engineering report generation (formal PDF audit export)
```

1. **Observed Image Evidence**: Spatial temperature distributions, hot-spot patterns, thermal gradients ($\Delta T$), and optical surface markings (soiling, cracks, discoloration, glass breakage).
2. **Computer Vision Prediction**: Model-generated bounding boxes, masks, and class probabilities with explicit uncertainty bounds.
3. **Measured Telemetry**: Empirically measured electrical and environmental values ($V_{oc}, I_{sc}, G, T_{mod}$) with sensor calibration metadata.
4. **Engineering Inference**: Algorithmic reasoning correlating observed heat patterns with electrical circuits and environmental constraints.
5. **Verified/Confirmed Diagnosis**: Diagnoses validated through corroborating electrical measurements or physical site inspection.
6. **Recommended Maintenance Action**: Actionable O&M directives prioritized by safety risk, energy yield impact, and equipment warranty terms.

---

## 3. Photovoltaic Engineering & Standards Foundation

### 3.1 IEC 62446-3 Thermographic Standards
The diagnostic engine adheres strictly to **IEC 62446-3** (*Photovoltaic (PV) systems – Requirements for testing, documentation and maintenance – Part 3: Photovoltaic modules and plants – Outdoor infrared thermography*):

- **Environmental Test Criteria**:
  - Plane-of-array irradiance $G \ge 600\text{ W/m}^2$ (recommended $> 700\text{ W/m}^2$) under clear sky conditions.
  - Wind speed $< 4\text{ m/s}$ (high wind causes convective cooling, suppressing $\Delta T$).
  - Camera viewing angle between $50^\circ$ and $80^\circ$ to the module surface normal (minimizing glass reflection while avoiding emissivity degradation).
- **Thermal Delta ($\Delta T$) Categorization**:
  - $\Delta T_{cell}$: Temperature difference between the hottest point on a cell and the average temperature of fault-free neighboring cells on the same module under steady irradiance.
  - $\Delta T_{module}$: Temperature difference between an anomalous module and neighboring fault-free modules in the same string.
- **Severity Classification**:
  - **Class 1 (Minor Anomaly)**: $\Delta T < 10^\circ\text{C}$. Localized thermal stress or non-critical soiling. Monitored during next regular inspection cycle.
  - **Class 2 (Medium Anomaly)**: $10^\circ\text{C} \le \Delta T < 20^\circ\text{C}$ or substring heating. Likely bypass diode conduction, localized cell fracture, or heavy shadowing. Remediation scheduled during next planned maintenance.
  - **Class 3 (Critical Anomaly)**: $\Delta T \ge 20^\circ\text{C}$, junction box overheating, or open-circuit module. Risk of backsheet burn-through, delamination, fire hazard, or significant string output loss. Immediate intervention required.

### 3.2 Failure Modes Covered by Diagnostic Knowledge Base
- **Cell-Level Faults**: Single-cell hot spots (shunts, micro-cracks, localized resistive heating), multi-cell hot spots, snail trails with localized heating.
- **Substring & Diode Faults**: 1/3 or 2/3 module overheating caused by shorted or continuously forward-biased bypass diodes.
- **Module-Level Faults**: Completely open-circuit module (disconnected ribbon, damaged junction box), severe Potential-Induced Degradation (PID) characterized by exponential heating gradient along strings nearest the negative pole.
- **String & Inverter Faults**: Blown string fuses, disconnected strings, inverter clipping or MPPT tracking collapse.
- **External & Environmental Factors**: Soiling (uniform dust vs localized bird droppings), vegetation shading, and optical reflection artifacts.

---

## 4. System Architecture & Module Boundaries

```
c:\sem-7-project\
├── frontend/             # Streamlit Operations Portal
├── src/
│   ├── api/              # FastAPI REST endpoints (/health, /inspections, /diagnosis)
│   ├── core/             # Pydantic Settings, application security, logging
│   ├── database/         # SQLAlchemy ORM, SQLite/PostgreSQL engine, sessions
│   ├── services/         # Business logic & workflow orchestrators
│   ├── vision/           # Thermal radiometric parsing, optical processing, CV inference contracts
│   ├── diagnosis/        # Evidence fusion, uncertainty evaluation, dynamic measurement queries
│   ├── telemetry/        # CSV/Excel/SCADA electrical data parsing and normalization
│   ├── knowledge/        # IEC standards taxonomy, physics equations, rules engine
│   └── reports/          # ReportLab PDF engineering audit generator
├── data/                 # Raw/Processed storage (git-ignored)
├── tests/                # Unit and integration test suites
├── docs/                 # Architecture, domain guides, and phased roadmaps
└── scripts/              # Verification & maintenance utilities
```

### Module Responsibilities:
- **`src.api`**: Clean RESTful interface exposing inspection management, file uploads, diagnostic execution, and report downloads. Does not contain raw business or vision logic.
- **`src.core`**: Central configuration loaded via environment variables and `.env` using Pydantic Settings.
- **`src.database`**: Independent ORM layer managing persistent entities with an initial SQLite target and seamless PostgreSQL migration path.
- **`src.vision`**: Image I/O, radiometric temperature calibration, geometric homography/rectification, and model-agnostic inference wrappers.
- **`src.diagnosis`**: The reasoning engine that takes visual findings, compares them with electrical measurements, computes Bayesian/heuristic uncertainty, and generates questions for missing data.
- **`src.telemetry`**: File parsers (CSV, Excel) validating string current, voltage, and meteorological sensors against physical sanity limits.
- **`src.knowledge`**: Formalized rules and threshold matrices directly mapped from IEC standards and physics principles.
- **`src.reports`**: High-fidelity PDF generation creating timestamped, traceable engineering inspection reports with audit-grade provenance.

---

## 5. Security & Operational Integrity

1. **Zero Secret Leakage**: No API keys, credentials, or production tokens exist in code or repository history.
2. **Data Isolation**: Raw plant imagery and client telemetry are stored in git-ignored data directories with strict path validation to prevent directory traversal.
3. **No Fabricated Data**: If an electrical parameter ($V_{oc}$, $I_{sc}$) has not been measured, the platform reports `Status: Unmeasured` and highlights diagnostic uncertainty rather than generating synthetic confidence scores.
