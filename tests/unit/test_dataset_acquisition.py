"""Unit tests for Phase 5.1 dataset acquisition, quarantine, audit, and deduplication."""

import io
import json
from pathlib import Path
import tempfile
import zipfile

from PIL import Image
import pytest

from src.vision.dataset_acquisition import (
    DatasetAcquisitionService,
    DuplicateCluster,
    ImageValidationRecord,
    compute_dhash,
    compute_file_sha256,
    hamming_distance,
    safe_extract_zip,
)


def test_provenance_schema_and_integrity_note(tmp_path: Path):
    """Test that provenance object records source, lack of published checksum, and legal questions."""
    dummy_zip = tmp_path / "test_archive.zip"
    dummy_zip.write_bytes(b"dummy zip content for sha test")

    service = DatasetAcquisitionService(tmp_path)
    provenance = service.build_provenance(dummy_zip)

    assert provenance.dataset_name == "InfraredSolarModules"
    assert provenance.source_url == "https://github.com/RaptorMaps/InfraredSolarModules"
    assert provenance.archive_sha256 == compute_file_sha256(dummy_zip)
    assert provenance.upstream_published_checksum is None
    assert "No trusted published checksum" in provenance.integrity_verification_note
    assert "MIT License" in provenance.repository_license
    assert "Raptor Maps, Inc." in provenance.attribution_notice
    assert len(provenance.unresolved_legal_questions) >= 3


def test_safe_extract_zip_prevents_path_traversal(tmp_path: Path):
    """Test that safe_extract_zip detects and blocks Zip Slip path traversal."""
    malicious_zip = tmp_path / "malicious.zip"
    target_extract = tmp_path / "extracted"

    # Create zip containing path traversal member
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../../escape.txt", "malicious payload")

    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        safe_extract_zip(malicious_zip, target_extract)

    # Ensure file was not extracted outside
    assert not (tmp_path.parent / "escape.txt").exists()


def test_safe_extract_zip_extracts_legitimate_files(tmp_path: Path):
    """Test that safe_extract_zip extracts legitimate directory structures cleanly."""
    valid_zip = tmp_path / "valid.zip"
    target_extract = tmp_path / "extracted"

    with zipfile.ZipFile(valid_zip, "w") as zf:
        zf.writestr("module_metadata.json", json.dumps({"0": {"image_filepath": "images/0.jpg"}}))
        zf.writestr("images/0.jpg", b"fake jpeg bytes")

    extracted = safe_extract_zip(valid_zip, target_extract)
    assert len(extracted) == 2
    assert (target_extract / "module_metadata.json").exists()
    assert (target_extract / "images" / "0.jpg").exists()


def test_dhash_computation_and_hamming_distance():
    """Test perceptual difference hash (dHash) properties and Hamming distance."""
    # Image 1: horizontal gradient from black to white
    im1 = Image.new("L", (24, 40), color=0)
    for x in range(24):
        for y in range(40):
            im1.putpixel((x, y), int(x * 10))

    # Image 2: identical gradient
    h1 = compute_dhash(im1)
    h2 = compute_dhash(im1)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Image 3: slight modification (1 pixel shifted)
    im3 = im1.copy()
    im3.putpixel((0, 0), 255)
    h3 = compute_dhash(im3)
    dist = hamming_distance(h1, h3)
    assert dist <= 3  # Near duplicate

    # Image 4: completely reversed gradient
    im4 = Image.new("L", (24, 40), color=0)
    for x in range(24):
        for y in range(40):
            im4.putpixel((x, y), int((23 - x) * 10))
    h4 = compute_dhash(im4)
    assert hamming_distance(h1, h4) > 10  # Very different


def test_duplicate_clustering_and_conflicting_labels(tmp_path: Path):
    """Test duplicate and near-duplicate cluster detection including label conflict flagging."""
    service = DatasetAcquisitionService(tmp_path)
    extracted_dir = tmp_path / "extracted"
    img_dir = extracted_dir / "images"
    img_dir.mkdir(parents=True)

    # Create 3 images:
    # img 0 and img 1 are exact duplicates but have DIFFERENT labels
    im = Image.new("RGB", (24, 40), color=(100, 100, 100))
    im.save(img_dir / "0.jpg", format="JPEG")
    im.save(img_dir / "1.jpg", format="JPEG")

    # img 2 is distinct
    im2 = Image.new("RGB", (24, 40), color=(200, 20, 20))
    im2.save(img_dir / "2.jpg", format="JPEG")

    meta = {
        "0": {"image_filepath": "images/0.jpg", "anomaly_class": "Cell"},
        "1": {"image_filepath": "images/1.jpg", "anomaly_class": "Diode"},  # Conflicting label
        "2": {"image_filepath": "images/2.jpg", "anomaly_class": "No-Anomaly"},
    }
    with open(extracted_dir / "module_metadata.json", "w") as f:
        json.dump(meta, f)

    dummy_zip = tmp_path / "dummy.zip"
    dummy_zip.write_bytes(b"dummy zip content")
    prov = service.build_provenance(dummy_zip)
    report, records, clusters = service.audit_quarantined_dataset(extracted_dir, prov)

    assert report.total_images_in_metadata == 3
    assert report.valid_images_count == 3
    assert report.exact_duplicate_groups_count == 1

    # Check exact cluster
    exact_cluster = [c for c in clusters if c.cluster_type == "EXACT_SHA256"][0]
    assert set(exact_cluster.image_ids) == {"0", "1"}
    assert exact_cluster.has_conflicting_labels is True
    assert set(exact_cluster.labels) == {"Cell", "Diode"}


def test_proposed_split_manifest_zero_leakage(tmp_path: Path):
    """Test that cluster-aware split manifest strictly enforces zero partition leakage."""
    service = DatasetAcquisitionService(tmp_path)

    records = [
        ImageValidationRecord("0", "images/0.jpg", "Cell", True, "JPEG", 24, 40, 3, "sha0", 100),
        ImageValidationRecord("1", "images/1.jpg", "Cell", True, "JPEG", 24, 40, 3, "sha0", 100),
        ImageValidationRecord("2", "images/2.jpg", "Diode", True, "JPEG", 24, 40, 3, "sha2", 200),
        ImageValidationRecord("3", "images/3.jpg", "No-Anomaly", True, "JPEG", 24, 40, 3, "sha3", 300),
        ImageValidationRecord("4", "images/4.jpg", "No-Anomaly", True, "JPEG", 24, 40, 3, "sha4", 400),
    ]

    # Cluster: 0 and 1 are duplicate cluster
    clusters = [
        DuplicateCluster(
            cluster_id=1,
            cluster_type="EXACT_SHA256",
            image_ids=["0", "1"],
            labels=["Cell", "Cell"],
            has_conflicting_labels=False,
            hamming_distance=0,
        )
    ]

    manifest = service.generate_proposed_split_manifest(records, clusters)

    train = set(manifest["splits"]["train"])
    val = set(manifest["splits"]["val"])
    test = set(manifest["splits"]["test"])

    # Zero overlap check
    assert len(train & val) == 0
    assert len(train & test) == 0
    assert len(val & test) == 0

    # Ensure 0 and 1 were not separated into different splits
    both_in_train = ("0" in train) and ("1" in train)
    both_in_val = ("0" in val) and ("1" in val)
    both_in_test = ("0" in test) and ("1" in test)
    assert both_in_train or both_in_val or both_in_test

    assert "PROPOSED CLUSTER-AWARE SPLIT MANIFEST FOR REVIEW ONLY" in manifest["disclaimer"]


def test_existing_workspace_fingerprint_preservation(tmp_path: Path):
    """Test that existing workspace files are fingerprinted and modifications detected."""
    workspace = tmp_path / "workspace"
    raw_inspections = workspace / "data" / "raw" / "inspections" / "1"
    raw_inspections.mkdir(parents=True)

    test_file = raw_inspections / "sample.jpg"
    test_file.write_bytes(b"initial image bytes")

    service = DatasetAcquisitionService(workspace)
    fp = service.record_existing_workspace_fingerprint()

    rel_key = "data/raw/inspections/1/sample.jpg"
    assert rel_key in fp

    # Verify untouched returns True initially
    assert service.verify_workspace_untouched(fp) is True

    # Modify file and verify untouched returns False
    test_file.write_bytes(b"tampered content")
    assert service.verify_workspace_untouched(fp) is False


def test_zero_distance_dhash_different_bytes():
    """Verify that images with different bytes/pixels can yield identical dHash (distance 0)."""
    im1 = Image.new("L", (24, 40), color=0)
    for x in range(24):
        for y in range(40):
            im1.putpixel((x, y), int(x * 5))

    # Add uniform brightness offset: gradient relative order preserved but different bytes
    im2 = Image.new("L", (24, 40), color=0)
    for x in range(24):
        for y in range(40):
            im2.putpixel((x, y), int(x * 5) + 50)

    buf1 = io.BytesIO()
    im1.save(buf1, format="PNG")
    bytes1 = buf1.getvalue()

    buf2 = io.BytesIO()
    im2.save(buf2, format="PNG")
    bytes2 = buf2.getvalue()

    assert bytes1 != bytes2
    h1 = compute_dhash(im1)
    h2 = compute_dhash(im2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0


def test_transitive_similarity_chaining():
    """Verify that single-linkage clustering can connect items across a similarity chain despite large endpoint distance."""
    hashes = {
        "A": 0b000000,
        "B": 0b000011,  # dist 2 from A
        "C": 0b001111,  # dist 2 from B
        "D": 0b111111,  # dist 2 from C, dist 6 from A
    }
    threshold = 3

    adj = {k: set() for k in hashes}
    for k1 in hashes:
        for k2 in hashes:
            if k1 != k2 and hamming_distance(hashes[k1], hashes[k2]) <= threshold:
                adj[k1].add(k2)

    visited = set()
    component = []
    queue = ["A"]
    visited.add("A")
    while queue:
        curr = queue.pop(0)
        component.append(curr)
        for nbr in adj[curr]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    assert set(component) == {"A", "B", "C", "D"}
    assert hamming_distance(hashes["A"], hashes["D"]) == 6
    assert hamming_distance(hashes["A"], hashes["D"]) > threshold


def test_manifest_partition_integrity_and_exact_duplicate_isolation():
    """Verify partition integrity and exact duplicate isolation in both proposed split manifests."""
    workspace = Path(__file__).resolve().parent.parent.parent
    manifest_a_path = workspace / "data" / "processed" / "proposed_splits.json"
    manifest_b_path = workspace / "data" / "processed" / "proposed_splits_option_b.json"

    assert manifest_a_path.exists(), "proposed_splits.json must exist"
    assert manifest_b_path.exists(), "proposed_splits_option_b.json must exist"

    meta_path = workspace / "data" / "external" / "quarantine" / "extracted" / "InfraredSolarModules" / "module_metadata.json"
    assert meta_path.exists(), "module_metadata.json must exist"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    all_ids = set(str(k) for k in meta.keys())
    assert len(all_ids) == 20000

    # 22 verified exact byte duplicate pairs (SHA-256)
    exact_pairs = [
        ["1585", "1144"], ["2448", "1566"], ["2584", "3087"], ["2746", "2825"],
        ["2755", "3146"], ["2858", "2973"], ["2889", "2700"], ["3063", "2790"],
        ["3376", "3179"], ["3431", "3071"], ["3449", "3330"], ["3479", "2860"],
        ["3539", "2908"], ["3915", "7027"], ["450", "7484"],  ["4518", "161"],
        ["543", "5171"],  ["556", "188"],   ["640", "505"],   ["6888", "483"],
        ["7387", "1124"], ["9227", "9571"],
    ]

    for manifest_path, name in [(manifest_a_path, "Option A"), (manifest_b_path, "Option B")]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train = set(str(i) for i in manifest["splits"]["train"])
        val = set(str(i) for i in manifest["splits"]["val"])
        test = set(str(i) for i in manifest["splits"]["test"])

        # 1. Zero overlaps between partitions
        assert len(train & val) == 0, f"{name}: Train and Val overlap"
        assert len(train & test) == 0, f"{name}: Train and Test overlap"
        assert len(val & test) == 0, f"{name}: Val and Test overlap"

        # 2. Complete 20,000 accounting
        assigned = train | val | test
        assert len(assigned) == 20000, f"{name}: Total unique assigned count is not 20000"
        assert assigned == all_ids, f"{name}: Assigned IDs do not match metadata IDs"

        # 3. No duplicates within splits
        assert len(manifest["splits"]["train"]) == len(train), f"{name}: Duplicate IDs in train"
        assert len(manifest["splits"]["val"]) == len(val), f"{name}: Duplicate IDs in val"
        assert len(manifest["splits"]["test"]) == len(test), f"{name}: Duplicate IDs in test"

        # 4. Strict exact duplicate isolation across all 22 verified pairs
        for pair in exact_pairs:
            pair_set = set(pair)
            splits_with_pair = sum([
                bool(pair_set & train),
                bool(pair_set & val),
                bool(pair_set & test),
            ])
            assert splits_with_pair == 1, f"{name}: Exact duplicate pair {pair} split across partitions"
