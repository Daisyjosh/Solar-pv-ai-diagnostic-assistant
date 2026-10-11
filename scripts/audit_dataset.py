"""CLI utility executing dataset audit and generating reproducible reports."""

import json
import sys
from pathlib import Path

# Add project root to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.config import get_settings
from src.vision.dataset_audit import DatasetAuditor


def main() -> int:
    print("=" * 60)
    print("Solar PV AI Diagnostic Assistant — Dataset Inventory & Quality Audit")
    print("=" * 60)

    settings = get_settings()
    target_data_dir = settings.storage_dir.resolve()
    print(f"[*] Scanning configured storage directory: {target_data_dir}")

    auditor = DatasetAuditor(near_dup_hamming_threshold=4)
    summary = auditor.audit_directory(target_data_dir)

    # 1. Output Markdown Report
    docs_dir = WORKSPACE_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = docs_dir / "dataset_inventory_report.md"
    markdown_content = auditor.generate_markdown_report(summary)
    report_md_path.write_text(markdown_content, encoding="utf-8")
    print(f"[+] Markdown report written to: {report_md_path}")

    # 2. Output JSON Report
    report_json_path = docs_dir / "dataset_inventory.json"
    json_data = {
        "scan_timestamp_utc": summary.scan_timestamp_utc,
        "scanned_directory": summary.scanned_directory,
        "total_files_scanned": summary.total_files_scanned,
        "valid_images_count": summary.valid_images_count,
        "corrupt_files_count": summary.corrupt_files_count,
        "zero_byte_files_count": summary.zero_byte_files_count,
        "format_distribution": summary.format_distribution,
        "provenance_distribution": summary.provenance_distribution,
        "radiometric_distribution": summary.radiometric_distribution,
        "exact_duplicates_count": summary.exact_duplicates_count,
        "exact_duplicate_groups": summary.exact_duplicate_groups,
        "near_duplicate_candidate_pairs": summary.near_duplicate_candidate_pairs,
        "license_documentation": summary.license_documentation,
        "records": [
            {
                "relative_path": r.relative_path,
                "file_size_bytes": r.file_size_bytes,
                "format": r.format,
                "width": r.width,
                "height": r.height,
                "channels": r.channels,
                "provenance": r.provenance.value,
                "sha256": r.sha256,
                "radiometric_status": r.radiometric_status,
                "is_corrupt": r.is_corrupt,
            }
            for r in summary.records
        ],
    }
    report_json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[+] Machine-readable JSON report written to: {report_json_path}")

    print("\n--- Inventory Summary ---")
    print(f"Total Files Scanned:       {summary.total_files_scanned}")
    print(f"Valid Images:              {summary.valid_images_count}")
    print(f"Corrupt / Zero-Byte:       {summary.corrupt_files_count}")
    print(f"Exact Duplicate Groups:    {len(summary.exact_duplicate_groups)}")
    print(f"Candidate Near-Duplicates: {len(summary.near_duplicate_candidate_pairs)}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
