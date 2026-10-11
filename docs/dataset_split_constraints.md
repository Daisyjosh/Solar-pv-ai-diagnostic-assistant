# Solar PV AI Diagnostic Assistant — Dataset Split Constraints & Evaluation Guidelines

This document formally records the engineering and methodological constraints governing dataset partitioning, duplicate isolation, and model evaluation for the *InfraredSolarModules* dataset prior to initiating any model training.

---

## 1. Core Evaluation Constraints

### Constraint 1: Absence of Site and Flight Identifiers (Site-Independent Generalization Limitation)
The upstream dataset metadata (`module_metadata.json`) provides only `image_filepath` and `anomaly_class`. It completely lacks `plant_id`, `site_id`, `flight_id`, `camera_model`, and geospatial coordinates.
- **Methodological Impact**: Partitioning samples across Train, Validation, and Test cannot prevent images from the same flight or utility site from appearing in both training and evaluation sets.
- **Mandatory Reporting Rule**: No evaluation results produced on this dataset may claim or imply *site-independent domain generalization* or *cross-plant transferability*. Performance metrics reflect in-domain held-out module classification only.

### Constraint 2: Inviolable Isolation of Exact SHA-256 Duplicates
Exactly 22 verified duplicate groups (22 pairs, 44 images) share identical file bytes (matching SHA-256 digests).
- **Rule**: Exact SHA-256 duplicate groups must be treated as **indivisible atomic units**. All members of an exact duplicate group must be assigned to the same partition (Train, Validation, or Test). They must never cross partition boundaries.

### Constraint 3: Perceptual Similarity (dHash) Is a Heuristic, Not Physical Identity
Difference hashing (`dHash`) measures relative gradient directions across downsampled $8 \times 9$ pixel projections.
- **Methodological Impact**: Perceptual hash proximity reflects similarity in visual gradient patterns, **not** physical proof that two crops originate from the same photovoltaic module.
- **Mandatory Reporting Rule**: Low Hamming distance must be documented strictly as a perceptual similarity heuristic. It must never be described as ground-truth physical module identity or flight tracking.

### Constraint 4: Transitive Chaining in Connected Components (The 9,356-Image Component)
Single-linkage graph clustering at a distance threshold of $\le 3$ joins nodes $(A, B)$ and $(B, C)$ even when $A$ and $C$ are visually dissimilar.
- **Observed Manifestation**: A giant connected component of 9,356 images (46.78% of the dataset) forms across all 12 classes, with internal pairwise Hamming distances ranging up to 51 bits (mean 16.78 bits) and an extreme pair (IDs 6104 and 4983) having distance 34 bits (greater than expected distance between random bitstrings).
- **Mandatory Reporting Rule**: The 9,356-image component is an **artifact of transitive chaining** across a continuous manifold of low-contrast thermal imagery. Connected components must **not** be characterized as "verified duplicate groups".

### Constraint 5: Different-Byte dHash Distance 0 Pairs Are Not Physical Duplicates
There are 46,025 image pairs that yield a dHash Hamming distance of 0 despite having different file bytes (different SHA-256 digests).
- **Observed Manifestation**: Over 53% of these pairs (24,516 pairs) carry conflicting ground-truth labels (e.g., ID 5983 `Cell` vs. ID 876 `Diode-Multi`; ID 17652 `No-Anomaly` vs. ID 5983 `Cell`).
- **Methodological Impact**: Flat, subtle, or uniform background gradients frequently collapse to identical bit patterns when evaluating simple sign differences.
- **Mandatory Reporting Rule**: Different-byte dHash-zero pairs must **not** automatically be treated as physical duplicates, nor should they be forced into indivisible clusters that distort dataset stratification.

### Constraint 6: Label Preservation and Conflicting-Label Documentation
The original upstream labels must be preserved exactly as published.
- **Observed Manifestation**: Exactly 6 pairs of images with identical SHA-256 bytes possess conflicting upstream labels:
  - `['3915', '7027']`: `['Cell-Multi', 'Cracking']`
  - `['450', '7484']`: `['Offline-Module', 'Cracking']`
  - `['4518', '161']`: `['Cell-Multi', 'Offline-Module']`
  - `['6888', '483']`: `['Hot-Spot', 'Offline-Module']`
  - `['7387', '1124']`: `['Cracking', 'Diode']`
  - `['543', '5171']`: `['Offline-Module', 'Cell']`
- **Mandatory Handling Rule**: Do not arbitrarily overwrite, relabel, or delete these images during partitioning. Document conflicting labels as known upstream annotation noise and retain them within the training partition.

---

## 2. Operational Status of Manifests

1. **Option A (`data/processed/proposed_splits.json`)**:
   - **Status**: Approved as provisional Phase 5.1 baseline.
   - **Properties**: 16,884 Train / 1,548 Val / 1,568 Test. Isolates all connected components indivisibly while maintaining positive, balanced representation across all 12 classes in Validation and Test.
2. **Option B (`data/processed/proposed_splits_option_b.json`)**:
   - **Status**: Rejected.
   - **Rationale**: While attempting to achieve exact 70/15/15 partition counts under rigid connected-component indivisibility, it caused severe class starvation (0 samples for `Soiling` and `Shadowing` in Test).
