# Solar PV AI Diagnostic Assistant

**Enterprise AI-powered solar photovoltaic inspection, multi-modal diagnosis, and O&M decision support platform.**

Designed for utility-scale solar plant operators, operations & maintenance (O&M) engineers, aerial inspection teams, and renewable energy asset managers.

---

## 1. Executive Summary & Capabilities

The **Solar PV AI Diagnostic Assistant** is a full-stack, industrial-grade operations and maintenance (O&M) platform that bridges thermal thermography, optical RGB imagery, electrical SCADA/telemetry, and international engineering standards into an integrated decision-support system.

### Key Capabilities
- **Trained AI Vision Pipeline**: 2,950-dimensional domain feature extraction (raw intensity, spatial gradients, row/column thermal statistics, higher-order contrast moments) feeding an empirical `HistGradientBoostingClassifier` trained on 16,884 samples across 12 module defect categories. Real measured test accuracy: **63.20%**, test macro-F1: **0.5029**, test weighted-F1: **0.6341**.
- **Evidence-Based Diagnostic Engine**: Implements strict **IEC 62446-3** thermographic criteria ($\\Delta T$ thresholds, solar irradiance minimums $\\ge 600\\,\\text{W/m}^2$, ambient normalization), synthesizing visual findings and electrical telemetry into justifiable fault hypotheses, power loss estimates, safety directives, and corrective actions.
- **Electrical Telemetry Ingestion**: Transaction-safe CSV/Excel batch ingestion with fuzzy column matching, electrical range checks ($V$, $I$, $P$, irradiance, module temperature), and time-series query analytics.
- **Engineering Knowledge Assistant (RAG)**: Ingests IEC 62446-3, IEC 61215, IEA-PVPS Task 13, and OSHA safety standards into a TF-IDF vector retrieval engine; grounds conversational answers in live inspection telemetry, imagery, and diagnoses.
- **Audit-Ready PDF Dossiers**: ReportLab generator compiling multi-page engineering reports complete with asset tree context, thermal image previews, class probability distributions, telemetry trends, and formal engineer sign-off blocks.
- **15-View Streamlit Operations Portal**: Comprehensive operator interface covering fleet KPIs, plant management, asset hierarchy, image uploads, AI inference, diagnosis review, telemetry analytics, interactive RAG chatbot, inspection comparison, batch triage, PDF exports, and SCADA connectivity status.

---

## 2. System Architecture

```
                                  [ Streamlit Portal ]
                               (15 Operational Views)
                                         │  HTTP / JSON
                                         ▼
                                  [ FastAPI Backend ]
                      (46 Endpoints across 10 API Routers)
                                         │
    ┌────────────────┬───────────────────┼───────────────────┬────────────────┐
    ▼                ▼                   ▼                   ▼                ▼
[ Vision Service ] [ Diagnostic Engine ] [ Telemetry Engine ] [ RAG Assistant ] [ PDF Generator ]
 HistGradientBoost   IEC 62446-3 Rules    CSV/Excel Parser    TF-IDF Vectorizer  ReportLab Engine
 2,950 Features      ΔT Severity          Time-Series Stats   Standards Store    Audit Dossiers
    │                │                   │                   │                │
    └────────────────┴───────────────────┼───────────────────┴────────────────┘
                                         ▼
                             [ SQLAlchemy ORM / SQLite ]
                              (Alembic Migration Tracked)
```

### Directory Structure
```
c:\sem-7-project\
├── alembic/                  # Database migration versions (0001 initial, 0002 full platform)
├── data/
│   ├── external/quarantine/  # Quarantined InfraredSolarModules dataset (20,000 images)
│   ├── processed/            # Leakage-resistant split manifest (proposed_splits.json)
│   ├── raw/inspections/      # Inspection image store (thermal & RGB)
│   └── solar_pv_diagnostics.db # Active SQLite database
├── frontend/
│   ├── app.py                # 15-view Streamlit portal
│   └── api_client.py         # Type-safe API client wrapper
├── models/
│   └── classifier/active/    # Active model.joblib, metadata.json, evaluation_report.json
├── src/
│   ├── api/                  # FastAPI routers (plants, inspections, assets, vision, etc.)
│   ├── core/                 # Pydantic settings, logging, path security sanitizers
│   ├── database/models/      # SQLAlchemy ORM entity models (11 tables)
│   ├── diagnosis/            # DiagnosticEngine and hypothesis synthesis
│   ├── knowledge/            # Standards document store and contextual assistant
│   ├── reports/              # ReportLab PDF dossier compiler
│   ├── services/             # BatchInspectionService and coordination
│   ├── telemetry/            # Ingestion parser, analytics, and SCADA adapter
│   └── vision/               # Feature extractor, SolarPVClassifier, inference service
├── tests/
│   ├── integration/          # Full API workflow and lifecycle tests (12 tests)
│   └── unit/                 # Domain logic, safety, and component tests (58 tests)
└── scripts/
    ├── start_platform.py     # Operational preflight and unified launcher
    ├── train_classifier.py   # Vision model training and evaluation script
    └── verify_environment.py # Environment integrity verification
```

---

## 3. Machine Learning Vision Pipeline

The computer vision subsystem classifies thermal crops into 12 canonical PV condition classes without synthetic data or hallucinated accuracy claims.

### Real Measured Test Performance
Trained on 16,884 samples (`data/processed/proposed_splits.json`, Option A partition):
- **Overall Accuracy**: **63.20%**
- **Macro Average F1**: **0.5029**
- **Weighted Average F1**: **0.6341**
- **Validation Accuracy**: **62.66%**

### Per-Class Test Metrics
| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **No-Anomaly** | 0.771 | 0.852 | **0.809** | 1,000 |
| **Cell** | 0.540 | 0.635 | **0.584** | 188 |
| **Cell-Multi** | 0.407 | 0.380 | **0.393** | 129 |
| **Cracking** | 0.600 | 0.250 | **0.353** | 24 |
| **Hot-Spot** | 0.370 | 0.278 | **0.317** | 25 |
| **Hot-Spot-Multi** | 0.250 | 0.222 | **0.235** | 9 |
| **Shadowing** | 0.575 | 0.426 | **0.489** | 108 |
| **Diode** | 0.793 | 0.767 | **0.780** | 180 |
| **Diode-Multi** | 0.417 | 0.476 | **0.444** | 21 |
| **Soiling** | 0.375 | 0.286 | **0.324** | 21 |
| **Offline-Module** | 0.718 | 0.627 | **0.669** | 83 |
| **Vegetation** | 0.600 | 0.439 | **0.507** | 41 |

*Note: All inference predictions clearly output the active model version, test accuracy disclaimer, and advise physical electrical verification.*

---

## 4. API Endpoints Catalog (FastAPI)

The backend exposes 46 REST endpoints across 10 functional modules at `http://127.0.0.1:8000`:

| Router | Method | Path | Description |
| :--- | :---: | :--- | :--- |
| **Health** | `GET` | `/health` | Liveness check |
| | `GET` | `/ready` | Readiness check with database ping |
| **Plants** | `GET, POST` | `/api/v1/plants/` | List and create solar power plants |
| | `GET, PUT` | `/api/v1/plants/{id}` | Retrieve and update plant details |
| **Assets** | `POST` | `/api/v1/plants/{id}/inverters` | Register inverter under plant |
| | `GET` | `/api/v1/plants/{id}/inverters` | List plant inverters |
| | `POST` | `/api/v1/inverters/{id}/strings` | Register string under inverter |
| | `GET` | `/api/v1/inverters/{id}/strings` | List inverter strings |
| | `POST` | `/api/v1/strings/{id}/modules` | Register module under string |
| | `GET` | `/api/v1/strings/{id}/modules` | List string modules |
| | `GET` | `/api/v1/plants/{id}/hierarchy` | Complete recursive asset hierarchy tree |
| **Inspections**| `GET, POST` | `/api/v1/inspections/` | List and create field inspections |
| | `GET, PUT` | `/api/v1/inspections/{id}` | Inspection details and environmental conditions |
| | `GET` | `/api/v1/inspections/{id}/compare/{other_id}` | Side-by-side inspection comparison |
| **Images** | `POST` | `/api/v1/inspections/{id}/images/upload` | Multipart thermal/optical image upload |
| | `GET` | `/api/v1/inspections/{id}/images` | List images registered to inspection |
| | `GET` | `/api/v1/images/{id}/download` | Stream image file content |
| **Vision** | `GET` | `/api/v1/vision/model-status` | Model training metadata and health |
| | `GET` | `/api/v1/vision/evaluation-metrics` | Test split precision, recall, and F1 |
| | `POST` | `/api/v1/vision/predict/{image_id}` | Run inference on registered image |
| | `POST` | `/api/v1/vision/predict-upload` | Direct ad-hoc image file inference |
| **Diagnosis** | `POST` | `/api/v1/inspections/{id}/diagnose` | Run IEC 62446-3 diagnostic assessment |
| | `GET` | `/api/v1/inspections/{id}/diagnosis` | Fetch current diagnosis and hypotheses |
| | `PUT` | `/api/v1/diagnoses/{id}/review` | Human engineer review sign-off & override |
| **Telemetry** | `POST` | `/api/v1/plants/{id}/telemetry/preview` | Upload and preview CSV/Excel records |
| | `POST` | `/api/v1/plants/{id}/telemetry/import` | Commit validated telemetry batch |
| | `GET` | `/api/v1/plants/{id}/telemetry/batches` | List uploaded telemetry batches |
| | `GET` | `/api/v1/plants/{id}/telemetry/records` | Query telemetry time-series |
| | `GET` | `/api/v1/plants/{id}/telemetry/trends` | Time-series statistics (V, I, P, G, T) |
| | `GET` | `/api/v1/telemetry/scada-status` | Check live SCADA gateway connection |
| **Chatbot** | `POST` | `/api/v1/chat/message` | Multi-turn RAG query with IEC citations |
| | `GET` | `/api/v1/chat/sessions/{id}/history` | Fetch chat conversation history |
| **Reports** | `GET` | `/api/v1/inspections/{id}/report/pdf` | Stream compiled PDF engineering report |
| | `GET` | `/api/v1/dashboard/summary` | Fleet KPIs, fault distribution, recent runs |
| **Batch** | `POST` | `/api/v1/plants/{id}/batch-inspections` | Triage zip/archive of drone image crops |
| | `GET` | `/api/v1/batch-inspections/{id}` | Session status, anomaly breakdown, log |

---

## 5. Operations Portal (15 Streamlit Views)

1. **Dashboard Overview**: Fleet overview, total inspections, open issues, IEC severity breakdown, fault category bar chart, and recent inspection table.
2. **Plant Management**: Register new utility solar assets (code, capacity DC MW, location) and view active plants.
3. **Asset Hierarchy Tree**: Inverter -> String -> Module tree visualizer with capacity and count rollups.
4. **New Field Inspection**: Schedule inspections, assign engineers, and record IEC 62446-3 baseline conditions (irradiance W/m2, ambient temperature).
5. **Thermal Image Gallery**: Upload and view high-resolution radiometric thermal captures with emissivity and reflected temperature metadata.
6. **Optical RGB Gallery**: Upload and view optical RGB surface imagery to cross-examine physical soiling, discoloration, or glass breakage.
7. **AI Image Analysis**: Trigger vision classifier inference on any captured image; displays predicted class, confidence, probabilities, and evaluation disclaimers.
8. **Diagnostic Assessment**: Fuse imagery and electrical data into root-cause hypotheses, estimated yield loss, safety directives, and submit official engineer sign-offs.
9. **Telemetry Import & Trends**: Ingest SCADA/logger CSV or Excel files, preview columns, and visualize I-V, power, and temperature trends.
10. **Engineering Chat Assistant**: Ask technical questions grounded in IEC standards and current inspection data with RAG context.
11. **Inspection Comparison**: Select two historical inspections on the same asset and compare environmental drift, severity changes, and electrical loss side-by-side.
12. **Batch Drone Inspection**: Process archives of drone module crops, auto-generate batch triage summaries, and flag critical modules.
13. **PDF Engineering Reports**: Generate and download audit-ready ReportLab engineering reports formatted for plant asset managers.
14. **Model Performance & Metrics**: Inspect model parameters, training sample sizes, validation/test splits, confusion matrices, and per-class metrics.
15. **System Config & SCADA Status**: Review database health, path security sanitizers, active directory sizes, and live SCADA gateway status.

---

## 6. Quickstart & Execution

### 6.1 Prerequisites
- Python 3.11+ (tested on Python 3.13.1 on Windows).
- Dependencies installed via `pip install -r requirements.txt`.

### 6.2 Preflight Environment Check
Run the preflight launcher to verify database migrations, directories, and trained model artifacts:
```powershell
python scripts/start_platform.py
```

### 6.3 Launching the Platform

**Terminal 1 — FastAPI Backend**:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

**Terminal 2 — Streamlit Frontend Portal**:
```powershell
python -m streamlit run frontend/app.py --server.port 8501
```
Open your browser to `http://localhost:8501`.

---

## 7. Verification & Automated Testing

The complete platform test suite consists of **70 automated tests** covering unit logic, integration workflows, dataset integrity, and end-to-end lifecycles.

Execute the full suite:
```powershell
pytest -v
```
**Results**: `70 passed, 0 failed, 100% pass rate`.

Test breakdown:
- `tests/integration/`: 12 tests validating plant creation, asset hierarchies, image upload, vision status, dashboard summary, SCADA status, chatbot Q&A, and full inspection -> diagnosis -> review -> PDF export lifecycle.
- `tests/unit/`: 58 tests validating the 2,950-dimensional feature extractor, `HistGradientBoostingClassifier`, `DiagnosticEngine` IEC rules, telemetry CSV parsing, SCADA unconfigured state, RAG document indexing, safe extraction path traversal defense, and duplicate clustering.

---

## 8. Engineering Boundaries & Operational Disclaimers

1. **Dataset Provenance & Generalization**: The vision classifier is trained on the curated 20,000-sample *InfraredSolarModules* dataset. Because public drone datasets do not provide verified flight or site IDs, cross-site generalization cannot be guaranteed. The model is an assistive triage tool, not an autonomous replacement for a certified thermographer.
2. **Radiometric vs Rendered Images**: Public datasets contain 8-bit rendered grayscale/colormap module crops. True surface temperature calculation (Delta-T) requires calibrated radiometric EXIF metadata or direct FLIR/RJPEG radiometry.
3. **Telemetry Integration**: The SCADA adapter defaults to `UNCONFIGURED` until live plant Modbus/OPC-UA endpoints and credentials are provided. CSV and Excel file imports are fully active.
4. **Engineer Oversight**: All AI diagnostic assessments are marked `PENDING_REVIEW` by default. Under IEC 62446-3, final severity classification and maintenance dispatch require qualified engineering sign-off.
