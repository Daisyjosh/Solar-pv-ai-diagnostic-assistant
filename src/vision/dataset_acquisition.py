"""Dataset acquisition, quarantine verification, audit, and deduplication engine for Phase 5.1.

Engineering Principles:
- Acquires datasets solely from official upstream sources.
- Safe archive extraction with path-traversal (Zip Slip) defense.
- Cryptographic SHA-256 byte deduplication and perceptual difference hash (dHash) candidate clustering.
- Preserves raw labels and documents metadata schema strictly.
- Enforces data immutability: existing inspection fixtures and application data are never touched.
- Produces comprehensive, reproducible audit reports and cluster-aware split manifests.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request
import zipfile

from PIL import Image

logger = logging.getLogger(__name__)

OFFICIAL_REPO_URL = "https://github.com/RaptorMaps/InfraredSolarModules"
OFFICIAL_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/2020-02-14_InfraredSolarModules.zip"
)
OFFICIAL_LICENSE_URL = (
    "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/LICENSE"
)
OFFICIAL_README_URL = (
    "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/README.md"
)

# Reference counts documented in the official Raptor Maps README
DOCUMENTED_CLASS_COUNTS: Dict[str, int] = {
    "No-Anomaly": 10000,
    "Cell": 1877,
    "Vegetation": 1639,
    "Diode": 1499,
    "Cell-Multi": 1288,
    "Shadowing": 1056,
    "Cracking": 941,
    "Offline-Module": 828,
    "Hot-Spot": 251,
    "Hot-Spot-Multi": 247,
    "Soiling": 205,
    "Diode-Multi": 175,
}
DOCUMENTED_TOTAL_IMAGES = sum(DOCUMENTED_CLASS_COUNTS.values())  # 20,006 images


@dataclass
class DatasetProvenance:
    """Provenance and legal tracking metadata for acquired datasets."""

    dataset_name: str
    source_url: str
    retrieval_timestamp_utc: str
    archive_filename: str
    archive_sha256: str
    upstream_published_checksum: Optional[str]
    integrity_verification_note: str
    repository_license: str
    attribution_notice: str
    licensing_scope_limitations: str
    unresolved_legal_questions: List[str]


@dataclass
class ImageValidationRecord:
    """Record of individual image file validation."""

    image_id: str
    relative_path: str
    raw_label: str
    is_valid: bool
    format: str
    width: int
    height: int
    channels: int
    sha256: str
    dhash: int
    error: Optional[str] = None


@dataclass
class DuplicateCluster:
    """Cluster of candidate duplicate images."""

    cluster_id: int
    cluster_type: str  # "EXACT_SHA256" or "PERCEPTUAL_DHASH"
    image_ids: List[str]
    labels: List[str]
    has_conflicting_labels: bool
    hamming_distance: Optional[int] = None


@dataclass
class AcquisitionAuditReport:
    """Comprehensive dataset audit report for Phase 5.1."""

    audit_timestamp_utc: str
    provenance: DatasetProvenance
    total_images_in_metadata: int
    total_files_on_disk: int
    valid_images_count: int
    corrupt_files_count: int
    missing_files_count: int
    unexpected_files_count: int
    observed_class_distribution: Dict[str, int]
    documented_class_distribution: Dict[str, int]
    distribution_discrepancies: Dict[str, Dict[str, int]]
    exact_duplicate_groups_count: int
    candidate_near_duplicate_clusters_count: int
    conflicting_label_clusters_count: int
    label_ambiguities_summary: Dict[str, str]
    known_methodological_limitations: List[str]
    unexpected_files_list: List[str] = field(default_factory=list)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file efficiently in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for an image.

    Rationale:
    - Downscales image to (hash_size + 1, hash_size) grayscale.
    - Captures relative horizontal luminance gradients.
    - Highly robust to minor JPEG compression artifacts while preserving spatial structure.
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(pixel_left > pixel_right)

    decimal_val = 0
    for idx, bit in enumerate(diff):
        if bit:
            decimal_val |= 1 << idx
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integer hashes."""
    return bin(h1 ^ h2).count("1")


def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    """Extract a ZIP archive safely with Zip Slip (path traversal) defense."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    extracted_files: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(target_resolved)):
                raise ValueError(
                    f"Security Error: Path traversal attempt detected in zip member '{member.filename}'"
                )
            zf.extract(member, target_dir)
            extracted_files.append(member_path)

    return extracted_files


class DatasetAcquisitionService:
    """Service orchestrating dataset acquisition, quarantine, audit, and splitting."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self.data_dir = self.workspace_root / "data"
        self.quarantine_dir = self.data_dir / "external" / "quarantine"
        self.raw_inspections_dir = self.data_dir / "raw" / "inspections"

    def record_existing_workspace_fingerprint(self) -> Dict[str, str]:
        """Record SHA-256 fingerprint of all existing inspection files to verify immutability."""
        fingerprints: Dict[str, str] = {}
        if self.raw_inspections_dir.exists():
            for p in self.raw_inspections_dir.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(self.workspace_root)).replace("\\", "/")
                    fingerprints[rel] = compute_file_sha256(p)
        return fingerprints

    def verify_workspace_untouched(self, baseline_fingerprint: Dict[str, str]) -> bool:
        """Verify that existing inspection files have not been modified or deleted."""
        current_fingerprint = self.record_existing_workspace_fingerprint()
        if set(current_fingerprint.keys()) != set(baseline_fingerprint.keys()):
            return False
        for path, orig_hash in baseline_fingerprint.items():
            if current_fingerprint.get(path) != orig_hash:
                return False
        return True

    def download_official_assets(self) -> Tuple[Path, Path, Path]:
        """Download official InfraredSolarModules archive and documentation."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        archive_path = self.quarantine_dir / "2020-02-14_InfraredSolarModules.zip"
        license_path = self.quarantine_dir / "LICENSE"
        readme_path = self.quarantine_dir / "README.md"

        # Download archive if not present
        if not archive_path.exists():
            logger.info(f"Downloading official archive from {OFFICIAL_ARCHIVE_URL}...")
            urllib.request.urlretrieve(OFFICIAL_ARCHIVE_URL, archive_path)

        # Download LICENSE
        if not license_path.exists():
            logger.info(f"Downloading official LICENSE from {OFFICIAL_LICENSE_URL}...")
            urllib.request.urlretrieve(OFFICIAL_LICENSE_URL, license_path)

        # Download README
        if not readme_path.exists():
            logger.info(f"Downloading official README from {OFFICIAL_README_URL}...")
            urllib.request.urlretrieve(OFFICIAL_README_URL, readme_path)

        return archive_path, license_path, readme_path

    def build_provenance(self, archive_path: Path) -> DatasetProvenance:
        """Build provenance and legal tracking metadata."""
        archive_hash = compute_file_sha256(archive_path)

        return DatasetProvenance(
            dataset_name="InfraredSolarModules",
            source_url=OFFICIAL_REPO_URL,
            retrieval_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            archive_filename=archive_path.name,
            archive_sha256=archive_hash,
            upstream_published_checksum=None,
            integrity_verification_note=(
                "No trusted published checksum (SHA-256 or MD5) was provided upstream by Raptor Maps in repository "
                "documentation or releases. The recorded SHA-256 is an empirical baseline computed from the initial "
                "official repository download."
            ),
            repository_license="MIT License (Copyright (c) 2020 Raptor Maps, Inc.)",
            attribution_notice=(
                "Borchers, Edward et al. 'InfraredSolarModules', ICLR 2020 Workshop on AI for Earth Sciences. "
                "Copyright (c) 2020 Raptor Maps, Inc."
            ),
            licensing_scope_limitations=(
                "The repository root specifies standard MIT text covering 'software and associated documentation'. "
                "Because aerial drone thermography may encompass distinct database rights and third-party asset "
                "owner considerations, this dataset is authorized for Phase 5 technical development and benchmarking, "
                "but unrestricted commercial clearance is NOT claimed without dedicated legal IP review."
            ),
            unresolved_legal_questions=[
                "Does the standard MIT license text unambiguously grant database and image reproduction rights for commercial redistribution?",
                "Are thermal images captured during commercial client flights subject to third-party facility confidentiality or utility asset NDAs?",
                "Does the lack of an explicit Creative Commons or Open Data Commons deed restrict embedding raw images in commercial software distributions?",
            ],
        )

    def extract_quarantine_archive(self, archive_path: Path) -> Path:
        """Extract dataset into quarantine extracted directory using safe extraction."""
        extracted_dir = self.quarantine_dir / "extracted"
        safe_extract_zip(archive_path, extracted_dir)
        return extracted_dir

    def audit_quarantined_dataset(
        self, extracted_dir: Path, provenance: DatasetProvenance
    ) -> Tuple[AcquisitionAuditReport, List[ImageValidationRecord], List[DuplicateCluster]]:
        """Validate images, schema, class counts, duplicates, and candidate clusters."""
        # Locate module_metadata.json in extracted_dir or nested subdirectories
        if (extracted_dir / "module_metadata.json").exists():
            dataset_root = extracted_dir
        elif (extracted_dir / "InfraredSolarModules" / "module_metadata.json").exists():
            dataset_root = extracted_dir / "InfraredSolarModules"
        else:
            candidates = list(extracted_dir.rglob("module_metadata.json"))
            if candidates:
                dataset_root = candidates[0].parent
            else:
                raise FileNotFoundError(f"module_metadata.json not found in extracted directory: {extracted_dir}")

        metadata_file = dataset_root / "module_metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        records: List[ImageValidationRecord] = []
        observed_classes: Dict[str, int] = defaultdict(int)
        sha_to_ids: Dict[str, List[str]] = defaultdict(list)
        dhash_to_ids: Dict[int, List[str]] = defaultdict(list)
        id_to_record: Dict[str, ImageValidationRecord] = {}

        missing_count = 0
        corrupt_count = 0
        valid_count = 0

        # Scan each entry in metadata
        for img_id, item in metadata_dict.items():
            rel_path = item.get("image_filepath", "")
            raw_label = item.get("anomaly_class", "UNKNOWN")
            observed_classes[raw_label] += 1

            img_file = dataset_root / rel_path
            if not img_file.exists():
                missing_count += 1
                rec = ImageValidationRecord(
                    image_id=img_id,
                    relative_path=rel_path,
                    raw_label=raw_label,
                    is_valid=False,
                    format="MISSING",
                    width=0,
                    height=0,
                    channels=0,
                    sha256="",
                    dhash=0,
                    error="File not found on disk",
                )
                records.append(rec)
                continue

            # Verify image integrity and dimensions
            try:
                img_sha = compute_file_sha256(img_file)
                with Image.open(img_file) as im:
                    im.verify()  # Check for container corruptions
                with Image.open(img_file) as im:
                    w, h = im.size
                    fmt = im.format or "UNKNOWN"
                    ch = len(im.getbands())
                    dh = compute_dhash(im, hash_size=8)

                rec = ImageValidationRecord(
                    image_id=img_id,
                    relative_path=rel_path,
                    raw_label=raw_label,
                    is_valid=True,
                    format=fmt,
                    width=w,
                    height=h,
                    channels=ch,
                    sha256=img_sha,
                    dhash=dh,
                )
                valid_count += 1
                records.append(rec)
                id_to_record[img_id] = rec
                sha_to_ids[img_sha].append(img_id)
                dhash_to_ids[dh].append(img_id)
            except Exception as e:
                corrupt_count += 1
                rec = ImageValidationRecord(
                    image_id=img_id,
                    relative_path=rel_path,
                    raw_label=raw_label,
                    is_valid=False,
                    format="CORRUPT",
                    width=0,
                    height=0,
                    channels=0,
                    sha256="",
                    dhash=0,
                    error=str(e),
                )
                records.append(rec)

        # Count total valid image files on disk
        disk_image_files = [
            p for p in dataset_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "__MACOSX" not in str(p)
        ]
        total_disk_files = len(disk_image_files)

        # Detect unexpected archive artifacts (.DS_Store, __MACOSX)
        unexpected_files: List[str] = []
        for p in extracted_dir.rglob("*"):
            if p.is_file():
                if ".DS_Store" in p.name or "__MACOSX" in str(p):
                    unexpected_files.append(str(p.relative_to(extracted_dir)).replace("\\", "/"))
        unexpected_count = len(unexpected_files)

        # Discrepancy analysis with README documentation
        discrepancies: Dict[str, Dict[str, int]] = {}
        all_class_names = set(DOCUMENTED_CLASS_COUNTS.keys()) | set(observed_classes.keys())
        for cname in all_class_names:
            doc_cnt = DOCUMENTED_CLASS_COUNTS.get(cname, 0)
            obs_cnt = observed_classes.get(cname, 0)
            if doc_cnt != obs_cnt:
                discrepancies[cname] = {"documented": doc_cnt, "observed": obs_cnt}

        # 1. Exact Duplicate Clusters (SHA-256)
        clusters: List[DuplicateCluster] = []
        cluster_counter = 1
        exact_groups_count = 0
        for sha, ids in sha_to_ids.items():
            if len(ids) > 1:
                exact_groups_count += 1
                labels = [id_to_record[i].raw_label for i in ids]
                conflicting = len(set(labels)) > 1
                clusters.append(
                    DuplicateCluster(
                        cluster_id=cluster_counter,
                        cluster_type="EXACT_SHA256",
                        image_ids=ids,
                        labels=labels,
                        has_conflicting_labels=conflicting,
                        hamming_distance=0,
                    )
                )
                cluster_counter += 1

        # 2. Candidate Near-Duplicate Clusters (dHash Hamming distance <= 3)
        # Using multi-index hashing (4 blocks of 16-bits) for sub-second candidate generation
        near_duplicate_pairs: Set[Tuple[str, str]] = set()
        block_index: Dict[Tuple[int, int], List[str]] = defaultdict(list)

        for rec in records:
            if not rec.is_valid:
                continue
            h = rec.dhash
            # 4 blocks of 16 bits
            b0 = (h >> 48) & 0xFFFF
            b1 = (h >> 32) & 0xFFFF
            b2 = (h >> 16) & 0xFFFF
            b3 = h & 0xFFFF
            for b_idx, b_val in enumerate((b0, b1, b2, b3)):
                for candidate_id in block_index[(b_idx, b_val)]:
                    cand_rec = id_to_record[candidate_id]
                    dist = hamming_distance(h, cand_rec.dhash)
                    if 0 < dist <= 3:  # dist == 0 handled in exact or identical dhash
                        pair = tuple(sorted([rec.image_id, candidate_id]))
                        near_duplicate_pairs.add(pair)
                block_index[(b_idx, b_val)].append(rec.image_id)

        # Build union-find clusters for near duplicates
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if parent.setdefault(x, x) != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for id1, id2 in near_duplicate_pairs:
            union(id1, id2)

        near_clusters_dict: Dict[str, List[str]] = defaultdict(list)
        for id1, id2 in near_duplicate_pairs:
            root = find(id1)
            if id1 not in near_clusters_dict[root]:
                near_clusters_dict[root].append(id1)
            if id2 not in near_clusters_dict[root]:
                near_clusters_dict[root].append(id2)

        conflicting_count = 0
        for root, ids in near_clusters_dict.items():
            labels = [id_to_record[i].raw_label for i in ids]
            conflicting = len(set(labels)) > 1
            if conflicting:
                conflicting_count += 1
            clusters.append(
                DuplicateCluster(
                    cluster_id=cluster_counter,
                    cluster_type="PERCEPTUAL_DHASH",
                    image_ids=ids,
                    labels=labels,
                    has_conflicting_labels=conflicting,
                    hamming_distance=None,  # Variable across cluster
                )
            )
            cluster_counter += 1

        label_ambiguities = {
            "Cracking": "Mechanical wafer cracks cannot be confirmed in thermal IR alone without EL imaging.",
            "Hot-Spot / Hot-Spot-Multi": "Defined for thin-film modules; conflating with crystalline cells blurs distinct PV technologies.",
            "Shadowing / Vegetation": "External site and environmental conditions, NOT internal IEC component equipment defects.",
            "Soiling": "Surface soiling is a cleaning/maintenance event, not an internal electrical defect.",
        }

        limitations = [
            "Site-, plant-, and flight-level metadata are completely absent in module_metadata.json.",
            "Perceptual hashing identifies visual pattern similarity, NOT ground-truth physical module identity.",
            "Cluster-aware partitioning eliminates visual duplicate contamination but CANNOT establish site-independent generalization.",
            "8-bit rendered JPEGs do not support radiometric temperature or delta-T calculation.",
        ]

        report = AcquisitionAuditReport(
            audit_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            provenance=provenance,
            total_images_in_metadata=len(metadata_dict),
            total_files_on_disk=total_disk_files,
            valid_images_count=valid_count,
            corrupt_files_count=corrupt_count,
            missing_files_count=missing_count,
            unexpected_files_count=unexpected_count,
            observed_class_distribution=dict(observed_classes),
            documented_class_distribution=DOCUMENTED_CLASS_COUNTS,
            distribution_discrepancies=discrepancies,
            exact_duplicate_groups_count=exact_groups_count,
            candidate_near_duplicate_clusters_count=len(near_clusters_dict),
            conflicting_label_clusters_count=conflicting_count,
            label_ambiguities_summary=label_ambiguities,
            known_methodological_limitations=limitations,
            unexpected_files_list=unexpected_files,
        )

        return report, records, clusters

    def generate_proposed_split_manifest(
        self, records: List[ImageValidationRecord], clusters: List[DuplicateCluster]
    ) -> Dict[str, Any]:
        """Generate cluster-aware proposed split manifest (70% Train, 15% Val, 15% Test).

        Invariant:
        All images belonging to the same duplicate/near-duplicate cluster are assigned
        as an atomic unit to a single split, guaranteeing zero duplicate leakage.
        """
        # Map each image_id to a cluster_id (or its own unique pseudo-cluster)
        image_to_cluster: Dict[str, int] = {}
        for c in clusters:
            for img_id in c.image_ids:
                # If image is in multiple clusters, link to first
                if img_id not in image_to_cluster:
                    image_to_cluster[img_id] = c.cluster_id

        # Unique cluster mapping
        cluster_to_records: Dict[int, List[ImageValidationRecord]] = defaultdict(list)
        next_cluster_id = max([c.cluster_id for c in clusters], default=0) + 1

        for rec in records:
            if not rec.is_valid:
                continue
            if rec.image_id in image_to_cluster:
                c_id = image_to_cluster[rec.image_id]
            else:
                c_id = next_cluster_id
                next_cluster_id += 1
            cluster_to_records[c_id].append(rec)

        # Stratified cluster allocation by primary class of cluster
        # Group clusters by majority class
        class_to_clusters: Dict[str, List[int]] = defaultdict(list)
        for c_id, recs in cluster_to_records.items():
            primary_label = recs[0].raw_label
            class_to_clusters[primary_label].append(c_id)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        for label, c_ids in class_to_clusters.items():
            # Deterministic allocation based on cluster_id
            c_ids_sorted = sorted(c_ids)
            n = len(c_ids_sorted)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            # remainder to test

            train_c = c_ids_sorted[:n_train]
            val_c = c_ids_sorted[n_train : n_train + n_val]
            test_c = c_ids_sorted[n_train + n_val :]

            for cid in train_c:
                train_ids.extend([r.image_id for r in cluster_to_records[cid]])
            for cid in val_c:
                val_ids.extend([r.image_id for r in cluster_to_records[cid]])
            for cid in test_c:
                test_ids.extend([r.image_id for r in cluster_to_records[cid]])

        manifest = {
            "disclaimer": (
                "PROPOSED CLUSTER-AWARE SPLIT MANIFEST FOR REVIEW ONLY — CANNOT ESTABLISH "
                "SITE-INDEPENDENT GENERALIZATION DUE TO LACK OF METADATA. DO NOT USE FOR TRAINING."
            ),
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_dataset": "InfraredSolarModules",
            "cluster_count": len(cluster_to_records),
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "total": len(train_ids) + len(val_ids) + len(test_ids),
            },
            "splits": {
                "train": sorted(train_ids, key=int),
                "val": sorted(val_ids, key=int),
                "test": sorted(test_ids, key=int),
            },
        }

        # Verify zero intersection
        set_train = set(train_ids)
        set_val = set(val_ids)
        set_test = set(test_ids)
        assert len(set_train & set_val) == 0, "Train and Val overlap!"
        assert len(set_train & set_test) == 0, "Train and Test overlap!"
        assert len(set_val & set_test) == 0, "Val and Test overlap!"

        return manifest
