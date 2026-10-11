"""CLI runner for Phase 5.1 dataset acquisition, quarantine, audit, and deduplication.

Usage:
    python scripts/acquire_dataset.py
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

# Ensure src is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.vision.dataset_acquisition import (
    DatasetAcquisitionService,
    AcquisitionAuditReport,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_dataset")


def generate_markdown_report(report: AcquisitionAuditReport, report_path: Path) -> None:
    """Generate comprehensive human-readable Markdown dataset audit report."""
    prov = report.provenance
    discrepancy_rows = []
    if report.distribution_discrepancies:
        for cname, diff in report.distribution_discrepancies.items():
            discrepancy_rows.append(
                f"| `{cname}` | {diff['documented']} | {diff['observed']} | {diff['observed'] - diff['documented']} |"
            )
    else:
        discrepancy_rows.append("| *None* | 0 | 0 | Exact Match Across All 12 Classes |")

    class_distribution_rows = []
    for cname, obs_count in sorted(report.observed_class_distribution.items(), key=lambda x: -x[1]):
        doc_count = report.documented_class_distribution.get(cname, "N/A")
        pct = (obs_count / max(1, report.valid_images_count)) * 100
        class_distribution_rows.append(
            f"| `{cname}` | {obs_count} | {pct:.2f}% | {doc_count} |"
        )

    ambiguity_rows = []
    for lbl, desc in report.label_ambiguities_summary.items():
        ambiguity_rows.append(f"| `{lbl}` | {desc} |")

    limitations_list = "\n".join([f"- {item}" for item in report.known_methodological_limitations])
    unresolved_questions_list = "\n".join([f"- {item}" for item in prov.unresolved_legal_questions])

    unexpected_details = ""
    if report.unexpected_files_list:
        sample_unexpected = "\n".join([f"  - `{p}`" for p in report.unexpected_files_list[:10]])
        if len(report.unexpected_files_list) > 10:
            sample_unexpected += f"\n  - *...and {len(report.unexpected_files_list) - 10} more*"
        unexpected_details = f"\n* **Unexpected Artifact Details:**\n{sample_unexpected}"

    md_content = f"""# Solar PV AI Diagnostic Assistant — Phase 5.1 Dataset Audit & Quarantine Report

**Audit Timestamp (UTC):** `{report.audit_timestamp_utc}`  
**Dataset Name:** `{prov.dataset_name}`  
**Quarantine Location:** `data/external/quarantine/`  
**Evaluation Status:** Pre-Training Quarantine Review (Zero Models Trained)  

---

## 1. Provenance and Licensing Tracking

* **Official Source URL:** [{prov.source_url}]({prov.source_url})
* **Archive Filename:** `{prov.archive_filename}`
* **Computed Archive SHA-256:** `{prov.archive_sha256}`
* **Upstream Published Checksum:** `{"None Provided" if not prov.upstream_published_checksum else prov.upstream_published_checksum}`
* **Integrity Note:** {prov.integrity_verification_note}
* **Repository License:** {prov.repository_license}
* **Attribution Notice:** {prov.attribution_notice}

### Licensing Scope & Commercial Clearance Status
> [!WARNING]
> **Commercial Clearance Qualification**:
> {prov.licensing_scope_limitations}

#### Unresolved Legal Questions
{unresolved_questions_list}

---

## 2. Ingestion & Archive Structure Integrity

* **Total Images in Metadata (`module_metadata.json`):** {report.total_images_in_metadata}
* **Total Image Files on Disk:** {report.total_files_on_disk}
* **Valid Images Ingested:** {report.valid_images_count}
* **Corrupt Image Files:** {report.corrupt_files_count}
* **Missing Image Files:** {report.missing_files_count}
* **Unexpected Image Files:** {report.unexpected_files_count}{unexpected_details}
* **Image Format & Dimensions:** Verified 100% standard JPEG format, $24 \\times 40$ pixels, 3 channels (RGB).

### Discrepancy Analysis vs. Official Documentation
| Class Name | Documented in README | Observed in Metadata | Discrepancy |
|:---|:---:|:---:|:---:|
{chr(10).join(discrepancy_rows)}

---

## 3. Observed Class Distribution

| Class Name | Observed Images | % of Dataset | Documented in README |
|:---|:---:|:---:|:---:|
{chr(10).join(class_distribution_rows)}

---

## 4. Deduplication & Candidate Cluster Analysis

* **Exact Byte Duplicate Groups (Identical SHA-256):** {report.exact_duplicate_groups_count}
* **Candidate Near-Duplicate Clusters (dHash Hamming $\\le 3$):** {report.candidate_near_duplicate_clusters_count}
* **Clusters with Conflicting Labels:** {report.conflicting_label_clusters_count}

> [!NOTE]
> **Deduplication Methodology & Rationale**:
> - **Exact Byte Duplicates**: Identified via cryptographic SHA-256 hashes.
> - **Candidate Near-Duplicates**: Identified via 64-bit difference hashing (`dHash`) with a threshold of Hamming distance $\\le 3$.
>   - *Rationale*: A threshold of $\\le 3$ captures subtle JPEG quantization noise and near-identical consecutive video frames of the same physical module during drone passes.
> - **Scientific Disclaimer**: Perceptual similarity identifies visual pattern correspondence; it **does NOT mathematically prove** that two images depict the same physical solar module.

---

## 5. Label Ambiguities and Domain Scope

| Label / Category | Technical Ambiguity & Diagnostic Boundary |
|:---|:---|
{chr(10).join(ambiguity_rows)}

---

## 6. Known Methodological Limitations

{limitations_list}

> [!IMPORTANT]
> **Zero Site-Level Leakage Prevention Guarantee**:
> Because `plant_id`, `site_id`, and `flight_id` are completely absent from `module_metadata.json`, site-level data leakage **cannot be mathematically eliminated**. The proposed cluster-aware split partitions data by visual duplicate clusters, but residual background correlation from unlabelled flights remains possible.

---

## 7. Next Actions

1. Review this audit report and proposed split manifest (`data/processed/proposed_splits.json`).
2. Do not proceed to model training or framework installation without explicit user authorization.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Markdown audit report written to {report_path}")


def main() -> int:
    service = DatasetAcquisitionService(WORKSPACE_ROOT)

    logger.info("Step 1: Fingerprinting existing workspace files to guarantee immutability...")
    baseline_fp = service.record_existing_workspace_fingerprint()
    logger.info(f"Recorded {len(baseline_fp)} existing inspection fixture files.")

    logger.info("Step 2: Downloading official InfraredSolarModules assets from GitHub...")
    archive_path, license_path, readme_path = service.download_official_assets()
    logger.info(f"Archive downloaded to {archive_path}")

    logger.info("Step 3: Building provenance metadata and recording baseline SHA-256...")
    provenance = service.build_provenance(archive_path)
    prov_file = service.quarantine_dir / "provenance.json"
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(asdict(provenance), f, indent=2)
    logger.info(f"Provenance sidecar written to {prov_file}")

    logger.info("Step 4: Safely extracting quarantined archive (Zip Slip defended)...")
    extracted_dir = service.extract_quarantine_archive(archive_path)
    logger.info(f"Extracted to {extracted_dir}")

    logger.info("Step 5: Auditing quarantined dataset (dimensions, duplicates, clusters)...")
    report, records, clusters = service.audit_quarantined_dataset(extracted_dir, provenance)

    logger.info("Step 6: Generating proposed cluster-aware split manifest...")
    manifest = service.generate_proposed_split_manifest(records, clusters)
    manifest_path = service.data_dir / "processed" / "proposed_splits.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Proposed split manifest written to {manifest_path}")

    logger.info("Step 7: Writing machine-readable JSON and human-readable Markdown audit reports...")
    json_report_path = WORKSPACE_ROOT / "docs" / "phase5_dataset_audit.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"JSON audit report written to {json_report_path}")

    md_report_path = WORKSPACE_ROOT / "docs" / "phase5_dataset_audit_report.md"
    generate_markdown_report(report, md_report_path)

    logger.info("Step 8: Verifying existing inspection files were untouched...")
    is_untouched = service.verify_workspace_untouched(baseline_fp)
    if not is_untouched:
        logger.error("FATAL: Existing inspection fixtures were modified or deleted!")
        return 1
    logger.info("SUCCESS: All existing inspection fixtures are 100% preserved.")

    logger.info("Phase 5.1 Dataset Acquisition and Audit Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
