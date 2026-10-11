"""Unit tests for provenance-based, leakage-resistant dataset partitioning."""

import pytest

from src.vision.splitting import ProvenanceSplitter


def test_splitting_insufficient_groups():
    """Verify splitter refuses to partition when fewer than 3 independent groups exist."""
    splitter = ProvenanceSplitter(random_seed=42)

    # 10 images from only 2 flights
    items = [
        {"image_id": f"img_{i}", "flight_session_id": "FLIGHT-A" if i < 5 else "FLIGHT-B"}
        for i in range(10)
    ]

    report = splitter.split_by_provenance(items, group_key="flight_session_id")
    assert report.is_defensible is False
    assert report.unique_groups_count == 2
    assert "insufficient" in report.methodology_notes.lower()
    assert report.train_count == 0


def test_splitting_with_sufficient_groups():
    """Verify group-based partitioning ensures zero group overlap across Train, Val, and Test."""
    splitter = ProvenanceSplitter(
        train_target_ratio=0.60,
        val_target_ratio=0.20,
        test_target_ratio=0.20,
        random_seed=42,
    )

    # 20 images across 5 distinct solar plant sites
    items = []
    for site_idx in range(5):
        site_id = f"SITE-0{site_idx + 1}"
        for img_idx in range(4):
            items.append({"image_id": f"{site_id}_img_{img_idx}", "plant_id": site_id})

    report = splitter.split_by_provenance(items, group_key="plant_id")
    assert report.is_defensible is True
    assert report.unique_groups_count == 5
    assert report.total_items == 20
    assert report.train_count + report.val_count + report.test_count == 20

    # Verify zero group leakage: intersection of group sets must be empty
    train_groups = set(report.train_groups)
    val_groups = set(report.val_groups)
    test_groups = set(report.test_groups)

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0

    # Verify images in train belong strictly to train_groups
    for item in report.train_items:
        assert item["plant_id"] in train_groups
    for item in report.val_items:
        assert item["plant_id"] in val_groups
    for item in report.test_items:
        assert item["plant_id"] in test_groups


def test_splitting_invalid_ratios():
    """Verify exception on ratios that do not sum to 1.0."""
    with pytest.raises(ValueError):
        ProvenanceSplitter(train_target_ratio=0.8, val_target_ratio=0.3, test_target_ratio=0.1)
