# Solar PV AI Diagnostic Assistant — Phased Engineering Roadmap

This document outlines the phased development roadmap for the **Solar PV AI Diagnostic Assistant**. Each phase represents an isolated, reviewable unit of work with explicit deliverables and acceptance criteria.

---

## Phase Summary Matrix

| Phase | Title | Focus Area | Status |
|:---:|:---|:---|:---:|
| **1** | **Workspace & Foundation** | Architecture, configuration, folder structure, docs, minimal tests | **Current** |
| **2** | **Core Backend & Frontend** | FastAPI skeleton, Streamlit baseline, database models, health checks | Scheduled |
| **3** | **Inspection & Media Ingestion** | Inspection management, panel/string metadata, secure file upload | Scheduled |
| **4** | **Dataset Audit & Normalization** | Thermal vs RGB audit, taxonomy harmonization, data hygiene | Scheduled |
| **5** | **Computer Vision Integration** | Model evaluation, training/fine-tuning, benchmark evaluation | Scheduled |
| **6** | **Evidence-Based Diagnostic Engine**| Multi-modal fusion, dynamic measurement requests, uncertainty | Scheduled |
| **7** | **Telemetry Ingestion & SCADA** | CSV/Excel parser, string/inverter mapping, electrical validation | Scheduled |
| **8** | **Engineering Assistant & Knowledge**| Conversational assistant, IEC standard RAG, domain reasoning | Scheduled |
| **9** | **PDF Reporting & Batch Workflow** | Professional PDF reports, batch inspection runs, export history | Scheduled |
| **10**| **Plant Monitoring & Prioritization**| Plant-level dashboard, fault ranking, trend & degradation tracking| Scheduled |
| **11**| **Hardening & Real-World Validation**| Security audit, end-to-end integration tests, field validation | Scheduled |

---

## Phase 1: Workspace, Architecture, Configuration, and Documentation (Current Phase)
- **Objective**: Establish the modular Python architecture, configuration templates, version control rules, and foundational verification test harness.
- **Key Deliverables**:
  - Full modular directory structure (`frontend/`, `src/`, `data/`, `tests/`, `docs/`, `scripts/`).
  - `.gitignore` and `.env.example` with zero credential exposure.
  - Pinned foundational dependencies (`requirements.txt`, `pyproject.toml`).
  - Core settings module (`src/core/config.py`) and FastAPI entrypoint (`src/api/main.py`).
  - Architecture specification (`docs/architecture.md`) and development roadmap (`docs/roadmap.md`).
  - Minimal automated test suite (`tests/unit/test_environment.py`) and verification utility (`scripts/verify_environment.py`).
- **Acceptance Criteria**:
  - Directory hierarchy created and internally consistent.
  - No secret keys, dummy mock models, or synthetic detections committed.
  - `python scripts/verify_environment.py` and `pytest tests/unit/test_environment.py` execute with 100% pass rate.

---

## Phase 2: Minimal Backend, Frontend, Database, Health Checks, and Automated Tests
- **Objective**: Stand up active backend API services, SQLite database persistence with SQLAlchemy ORM, Streamlit health status, and end-to-end service communication.
- **Key Deliverables**:
  - SQLAlchemy ORM schemas for `Plant`, `Inverter`, `String`, `Module`, `Inspection`, `ImageAsset`.
  - Database initialization scripts and Alembic migration harness.
  - REST endpoints for database health, system info, and configuration inspection.
  - Streamlit dashboard communicating with the FastAPI backend via HTTP.
  - Automated integration tests verifying API and database CRUD operations.
- **Acceptance Criteria**:
  - Backend launches cleanly via `uvicorn src.api.main:app`.
  - Frontend launches via `streamlit run frontend/app.py` and renders live backend status.
  - Database tables created automatically in local SQLite file without errors.

---

## Phase 3: Inspection Creation, Panel/String Metadata, Image Upload, and Secure File Handling
- **Objective**: Allow engineers to create new inspection sessions, define plant asset hierarchy (Plant -> Inverter -> String -> Module), and securely upload thermal and RGB imagery.
- **Key Deliverables**:
  - Asset hierarchy data models and API endpoints.
  - Multi-file image uploader supporting thermal (RJPEG, TIFF) and optical (JPEG, PNG).
  - Secure storage handler with SHA-256 deduplication, file extension validation, and virus/traversal protection.
  - Exif and thermal metadata extractor (camera model, timestamp, GPS, emissivity, reflected temperature).
- **Acceptance Criteria**:
  - Operator can create an inspection session with plant/string metadata.
  - Images upload securely, undergo validation, and store with recorded metadata.
  - Uploaded files are verified against file size and extension constraints.

---

## Phase 4: Dataset Inventory, Normalization, Fault Taxonomy, and Dataset Quality Checks
- **Objective**: Perform a comprehensive audit of external training and validation datasets before any model selection or training.
- **Key Deliverables**:
  - Ingestion and cataloging scripts for thermal and RGB datasets.
  - Dataset quality audit: resolution check, aspect ratio distribution, corrupted file detection, duplicate identification.
  - Fault taxonomy harmonization (mapping disparate dataset labels into standardized IEC 62446-3 categories).
  - Leakage-resistant dataset splitting strategy (splitting by plant or flight transect, NOT random image shuffling).
- **Acceptance Criteria**:
  - Catalog report detailing image counts, classes, resolutions, and quality flags.
  - Formal mapping table reconciling dataset labels to unified fault taxonomy.
  - Train/Validation/Test split created with zero plant-level or string-level data leakage.

---

## Phase 5: Model Selection, Training/Fine-Tuning, Evaluation, and Model Integration
- **Objective**: Select appropriate computer vision model architectures (object detection / instance segmentation), train/fine-tune models, and benchmark performance.
- **Key Deliverables**:
  - Architecture evaluation (e.g., comparing YOLO variants, Faster R-CNN, Mask R-CNN, RT-DETR for thermal imagery).
  - Training scripts with reproducible random seeds, data augmentations (thermal-specific flips, radiometric scaling).
  - Model evaluation reporting mAP@50, mAP@50-95, precision, recall, and confusion matrix across fault classes.
  - Model inference service with confidence thresholding and non-maximum suppression (NMS).
- **Acceptance Criteria**:
  - Model achieves verified benchmark criteria on test dataset.
  - Inference runner integrated into `src/vision` with clean input/output contracts.
  - Detections return bounding boxes, labels, and calibrated thermal delta-T values.

---

## Phase 6: Evidence-Based Diagnostic Workflow & Dynamic Measurement Requests
- **Objective**: Build the core diagnostic engine that fuses visual observations with electrical rules and dynamically requests missing telemetry.
- **Key Deliverables**:
  - Diagnostic reasoning engine implementing IEC 62446-3 fault classification logic.
  - Evidence matrix: linking thermal patterns ($\Delta T_{cell}$, $\Delta T_{substring}$) to candidate electrical causes.
  - Interactive questionnaire engine: generating targeted queries for missing measurements ($V_{oc}$, $I_{sc}$, string voltage, bypass diode check).
  - Uncertainty scoring algorithm quantifying confidence based on available vs required evidence.
- **Acceptance Criteria**:
  - For ambiguous thermal anomalies (e.g., hot cell), the engine explicitly prompts for string current and $V_{oc}$.
  - When measurements are supplied, the diagnosis updates with revised confidence, root cause, and severity.
  - System clearly reports "Insufficient Evidence" when required measurements are absent.

---

## Phase 7: SCADA/IoT CSV & Excel Ingestion, Panel/String Association, and Data Validation
- **Objective**: Enable bulk ingestion of electrical telemetry from plant data loggers, SCADA exports, and I-V curve tracers.
- **Key Deliverables**:
  - Parsers for CSV and Excel files containing electrical strings and inverter channels.
  - Unit normalizer and timestamp aligner.
  - Data validation engine: physical boundary checks ($I_{sc} > I_{mp}$, $V_{oc} > V_{mp}$, irradiance sanity).
  - Automated string-to-module mapping associating telemetry rows to inspection entities.
- **Acceptance Criteria**:
  - Ingestion of real-world sample CSV/Excel files with validation feedback.
  - Corrupted or impossible values (e.g., negative voltage under illumination) flagged with descriptive warnings.
  - Telemetry seamlessly links to the inspection diagnostic engine.

---

## Phase 8: Conversational Diagnostic Assistant & Engineering Knowledge Retrieval
- **Objective**: Provide a natural language engineering assistant that allows operators to query inspection findings, understand probable causes, and review IEC standards.
- **Key Deliverables**:
  - Provider-agnostic LLM interface (supporting configurable providers: Anthropic, OpenAI, or local models).
  - Engineering Retrieval-Augmented Generation (RAG) over IEC 62446-3, IEC 61215, and inverter maintenance manuals.
  - Structured prompt contracts ensuring the assistant only references verified inspection data and engineering literature.
  - Conversational interface integrated into the Streamlit portal.
- **Acceptance Criteria**:
  - Assistant accurately explains diagnostic rationale and cites relevant IEC clauses.
  - Hallucination guardrails prevent inventing unmeasured electrical metrics.
  - Assistant operates in offline/disabled mode when no LLM API key is configured.

---

## Phase 9: PDF Reports, Inspection History, and Batch Inspection
- **Objective**: Implement audit-ready PDF engineering inspection report generation and batch inspection processing.
- **Key Deliverables**:
  - Professional ReportLab PDF template with solar plant header, asset IDs, thermal/RGB visual callouts, electrical tables, and signature blocks.
  - Downloadable PDF generation endpoint in FastAPI and export button in Streamlit.
  - Inspection history log allowing operators to revisit past inspections and compare findings over time.
  - Batch image processing pipeline for flight missions with dozens or hundreds of module captures.
- **Acceptance Criteria**:
  - Generated PDF adheres to engineering reporting standards with crisp graphics and clear typography.
  - Unmeasured metrics are explicitly marked as "Not Measured" in generated reports.
  - Inspection records persist and load accurately from history.

---

## Phase 10: Plant Dashboard, Fault Prioritization, and Trend Analysis
- **Objective**: Aggregate inspections at the solar plant level to provide O&M managers with fleet-wide health monitoring and work-order prioritization.
- **Key Deliverables**:
  - Plant health dashboard displaying overall defect count, severity breakdown (Class 1, 2, 3), and estimated power degradation.
  - Asset health map visualizing defective strings and modules across array layouts.
  - Defect prioritization queue ranked by severity, safety hazard, and revenue impact.
  - Historical degradation trend charts tracking defect progression over successive inspection dates.
- **Acceptance Criteria**:
  - Dashboard updates dynamically based on completed inspections.
  - Plant managers can filter anomalies by severity, array, string, and defect category.

---

## Phase 11: Integration Testing, Security Hardening, Usability Improvements, and Field Validation
- **Objective**: Conduct rigorous end-to-end integration testing, security penetration checks, UX polish, and pilot field validation.
- **Key Deliverables**:
  - Comprehensive end-to-end automated test suite covering full inspection lifecycle.
  - Security hardening: input sanitization, rate limiting, secure header configuration.
  - User experience refinements based on O&M field technician feedback.
  - Field validation protocol comparing platform diagnoses against verified physical teardowns.
- **Acceptance Criteria**:
  - All test suites pass with > 90% branch coverage on core business logic.
  - Zero critical or high-severity vulnerabilities identified in dependency or static analysis.
  - Platform ready for live field pilot deployment.
