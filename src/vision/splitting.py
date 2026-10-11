"""Provenance-based, leakage-resistant dataset partitioning utilities.

Engineering Principles:
- Partitions datasets strictly by independent physical entities: plant, flight session, string, or site.
- Keeps all captures from a single flight session or physical array block in the same partition.
- Explicitly refuses to partition when provenance groups are insufficient (< 3 groups),
  rather than falling back to random image shuffling that causes spatial data leakage.
- Never falsely claims zero leakage solely because a partition executed.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SplitPartition(str, Enum):
    """Dataset partition assignment."""

    TRAIN = "TRAIN"
    VAL = "VAL"
    TEST = "TEST"


@dataclass
class DatasetSplitReport:
    """Outcome and integrity report for a dataset split."""

    is_defensible: bool
    group_key: str
    total_items: int
    unique_groups_count: int
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_groups: List[str] = field(default_factory=list)
    val_groups: List[str] = field(default_factory=list)
    test_groups: List[str] = field(default_factory=list)
    train_items: List[Any] = field(default_factory=list)
    val_items: List[Any] = field(default_factory=list)
    test_items: List[Any] = field(default_factory=list)
    methodology_notes: str = ""
    leakage_risks: List[str] = field(default_factory=list)


class ProvenanceSplitter:
    """Partitions dataset samples by provenance grouping to eliminate spatial correlation leakage."""

    def __init__(
        self,
        train_target_ratio: float = 0.70,
        val_target_ratio: float = 0.15,
        test_target_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """Configure target partition ratios and reproducible seed."""
        total = train_target_ratio + val_target_ratio + test_target_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Partition ratios must sum to 1.0 (got {total})")

        self.train_ratio = train_target_ratio
        self.val_ratio = val_target_ratio
        self.test_ratio = test_target_ratio
        self.seed = random_seed

    def split_by_provenance(
        self,
        items: List[Dict[str, Any]],
        group_key: str,
    ) -> DatasetSplitReport:
        """Partition items into Train, Val, and Test by grouping on group_key.

        Args:
            items: List of dictionaries representing dataset images.
            group_key: Key identifying physical independence (e.g. 'plant_id', 'flight_id', 'inspection_id').

        Returns:
            DatasetSplitReport with partition lists or explicit invalidity explanation.
        """
        report = DatasetSplitReport(
            is_defensible=False,
            group_key=group_key,
            total_items=len(items),
            unique_groups_count=0,
        )

        if not items:
            report.methodology_notes = "Cannot split empty dataset."
            return report

        # 1. Group items by provenance identifier
        groups: Dict[str, List[Dict[str, Any]]] = {}
        missing_group_count = 0

        for item in items:
            grp_val = item.get(group_key)
            if grp_val is None or str(grp_val).strip() == "":
                missing_group_count += 1
                grp_id = "__MISSING_PROVENANCE__"
            else:
                grp_id = str(grp_val).strip()
            groups.setdefault(grp_id, []).append(item)

        report.unique_groups_count = len([g for g in groups.keys() if g != "__MISSING_PROVENANCE__"])

        if missing_group_count > 0:
            report.leakage_risks.append(
                f"{missing_group_count} items lack provenance key '{group_key}'."
            )

        # 2. Defensibility check: At least 3 independent groups required
        if report.unique_groups_count < 3:
            report.is_defensible = False
            report.methodology_notes = (
                f"Insufficient independent provenance groups ({report.unique_groups_count} found, 3 minimum required). "
                f"Randomly partitioning images from only {report.unique_groups_count} site/flight sessions "
                f"would cause severe spatial data leakage between training and evaluation."
            )
            return report

        # 3. Deterministic group allocation using seeded hash sorting
        sorted_groups = sorted([g for g in groups.keys() if g != "__MISSING_PROVENANCE__"])

        # Sort groups by stable seeded hash to avoid bias
        def group_hash(g_id: str) -> str:
            return hashlib.sha256(f"{self.seed}_{g_id}".encode()).hexdigest()

        shuffled_groups = sorted(sorted_groups, key=group_hash)

        num_groups = len(shuffled_groups)
        num_train = max(1, int(round(num_groups * self.train_ratio)))
        num_val = max(1, int(round(num_groups * self.val_ratio)))

        # Ensure test gets at least 1 group if total >= 3
        if num_train + num_val >= num_groups:
            num_train = max(1, num_groups - 2)
            num_val = 1

        train_grps = set(shuffled_groups[:num_train])
        val_grps = set(shuffled_groups[num_train : num_train + num_val])
        test_grps = set(shuffled_groups[num_train + num_val :])

        report.train_groups = sorted(list(train_grps))
        report.val_groups = sorted(list(val_grps))
        report.test_groups = sorted(list(test_grps))

        for grp_id, grp_items in groups.items():
            if grp_id in train_grps:
                report.train_items.extend(grp_items)
            elif grp_id in val_grps:
                report.val_items.extend(grp_items)
            elif grp_id in test_grps:
                report.test_items.extend(grp_items)
            else:
                # Missing provenance items isolated into train to avoid test contamination
                report.train_items.extend(grp_items)

        report.train_count = len(report.train_items)
        report.val_count = len(report.val_items)
        report.test_count = len(report.test_items)
        report.is_defensible = True

        report.methodology_notes = (
            f"Successfully partitioned across {report.unique_groups_count} independent '{group_key}' groups. "
            f"Allocated {len(train_grps)} groups to TRAIN, {len(val_grps)} to VAL, and {len(test_grps)} to TEST. "
            f"Zero images from the same {group_key} appear across multiple partitions."
        )

        report.leakage_risks.append(
            f"Group-level partitioning isolates spatial site correlation; residual sensor or environmental correlation may still exist."
        )

        return report
