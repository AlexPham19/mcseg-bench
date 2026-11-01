# Split definitions

## CSAW-S

The development pool contains **237 images from 115 supplied patient groups**. It combines eligible records from earlier development and assessment pools after excluding **15 images from 8 expert-linked development groups**. Consequently, the historical 88-image assessment pool is **not** an independent test set for this five-fold design.

Split seed: **20261024**. Membership is copied from existing registries, not regenerated from the seed. A model's training seed does not change these partitions. All views of one supplied patient group remain together within each fold. Each development image and group appears in validation exactly once across the five folds.

| Fold | Train images | Validation images | Train groups | Validation groups |
| --- | ---: | ---: | ---: | ---: |
| 1 | 193 | 44 | 92 | 23 |
| 2 | 185 | 52 | 92 | 23 |
| 3 | 194 | 43 | 92 | 23 |
| 4 | 189 | 48 | 92 | 23 |
| 5 | 187 | 50 | 92 | 23 |

Files:

- `development.csv`: complete eligible development membership.
- `fold_1/` through `fold_5/`: exact `train.csv` and `validation.csv` lists.
- `folds.json`: validation group assignments; train groups are the development complement.
- `expert_evaluation.csv`: **26 images / 23 supplied expert groups**, evaluated against annotators A and B separately on the same image set. This is not 52 independent images.
- `excluded_expert_linked.csv`: 15 excluded development records; these do not enter any fold.

The first fold matches the current comparison registry: **193 train / 44 validation**. These counts must not be applied to every fold. Provider group identifiers and the registered conservative cross-release exclusion policy are preserved; this does not certify independently verified clinical patient linkage. Prior expert-cohort inspection and adaptive development must be disclosed when interpreting subsequent evaluations.

The `image_filename` and `mask_filename` columns refer to the project's prepared CSAW-S image/mask files. Paths are basenames, not server-specific paths. Image IDs such as `009_0` and group IDs such as `009` are strings; preserve leading zeros when reading CSVs. The expert list preserves its original namespaced group identifiers.

## INbreast

INbreast is **evaluation only** for these models. It has no training, validation, calibration or threshold-selection role, and these files do not create a new INbreast train/test split.

| File | Images | Reference definition |
| --- | ---: | --- |
| `contour_27.csv` | 27 | Existing contour-reference cohort |
| `assumed_negative_67.csv` | 67 | Source BI-RADS 1 images without XML, scored with assumed-empty MC references |
| `evaluation_94.csv` | 94 | Exact union of the preceding two cohorts |

**INbreast27 is a subset of INbreast94**, not a separate independent test population. Report contour segmentation and the assumed-negative unmatched-prediction burden separately. The 67 assumed-empty references are **not independently verified pixelwise MC-free labels**. Their source `source_dense_negative_valid=false` status is retained. A missing XML or reference filename is intentionally blank; it is not permission to treat all unannotated INbreast images as negative.

`group_proxy_id` is the existing anonymized filename/metadata grouping token: **44 proxies in INbreast94**. True patient linkage is not independently verified. Do not label this as 44 verified patients. Image, XML and reference filenames identify the expected source/prepared files without including any file contents. For the assumed-negative subset, the empty reference is an evaluation assumption, not a distributed mask file.

## Validation and interpretation

Run `python scripts/validate_splits.py` from the repository root. Validation reads only the shipped metadata. It does not decode images, access external cohorts, fit models or verify annotation correctness. A successful result establishes export consistency, not clinical validity or untouched-test status.

`metadata.json` preserves the source manifest SHA-256 fingerprints and project-relative provenance. `checksums.sha256` protects every release file outside Git's internal state. No server home paths, raw pixels, learned weights, scores or predictions are included.

These are project-specific split definitions, not an official dataset-provider split. Source datasets: [CSAW-S paper](https://proceedings.mlr.press/v119/matsoukas20a.html) and [INbreast paper](https://doi.org/10.1016/j.acra.2011.09.014).
