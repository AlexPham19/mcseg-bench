# Baseline evaluation protocol

## Data and partitions

- Official fitting, development, calibration, and test inputs are **real medical images only**. Generated lesions/images and synthetic medical patches are excluded. Test arrays are software fixtures, never benchmark training examples or results.
- Preserve [registered split membership](../splits/README.md). Pilot: fold 1,193 training/44 development images, model seed 42, split seed 20261024. Extend using the five registered folds and preregistered model seeds42/43/44.
- Keep related patient/group views together. Audit linkage, actual mask classes, completeness, orientation, spacing, and preprocessing. Fit dataset statistics and mine negatives from training data only.
- CSAW44 has informed optimization: report it as development, not an independent test. Score expert A/B separately on the same 26 images, not 52 independent images. Disclose prior test inspection/adaptation.
- INbreast is evaluation-only. Its27 contour-reference images are nested within94; the67 assumed-empty images are not verified MC-free negatives. Its44 grouping proxies are not independently verified patients. Declare any checkpoint/data overlap.

## Private data manifest

UTF-8 CSV; preserve IDs as strings and use paths relative to the data root:

```text
case_id,group_id,split,image,mask,label_status,image_sha256,mask_sha256
```

`split` is `train`, `dev`, or `test`. Reference status:

| Status | Treatment |
|---|---|
| `dense_verified` | Audited complete reference; eligible for primary evaluation and required by training/calibration |
| `contour_positive` | Positive contours with completeness limitations; supplementary evaluation |
| `assumed_empty` | Explicit empty mask under a declared assumption; supplementary evaluation |
| `unknown` | Excluded from scoring, never silently negative |

Use consistent binary0/1 or0/255 masks in original image coordinates. Audit ignore/multiclass conversion. Include test rows before freezing the manifest/calibrating. Use separate A/B manifests. The organizer controls hidden test labels; file/hash checks are not access control or annotation certification.

## Plain U-Net baseline

The only model is the plain U-Net trained from scratch. The complete recipe is defined in [BASELINE_SETTINGS.md](BASELINE_SETTINGS.md), backed by executable defaults in [settings.py](../mcbench/settings.py).

Training batch 4 and inference batch 8 are different settings. Percentile input normalization is separate from network GroupNorm. The 50 x128 budget means 6,400 optimizer updates, not 50 dataset passes. Checkpoint selection at 0.5 precedes final development threshold calibration.

## Operating-point lock

For the selected checkpoint, search thresholds 0.05..0.95 in steps of 0.05 on development data:

- `segmentation`: maximize positive-image macro IoU; ties favor higher component recall, lower FP/image, then higher threshold.
- `cad_fp0.5`, `cad_fp1`, `cad_fp2`, `cad_fp4`, `cad_fp8`: maximize component recall within the corresponding FP/image budget; ties favor lower FP/image, higher IoU, then higher threshold. No qualifying threshold means infeasible.

The lock binds manifest hash, model hash, and pipeline ID, and records development fingerprints/curve. Apply unchanged to test. Report actual test FP even if it exceeds the development budget. These engineering budgets are not clinically accepted alert limits. Pixel-threshold sweeps can merge/split objects and are not automatically alert-level FROC.

## Metrics

```text
Foreground Dice = 2TP / (2TP + FP + FN)
Foreground IoU  = TP / (TP + FP + FN)
```

Primary segmentation endpoint: **foreground macro IoU on positive `dense_verified` images**. Also report positive-image Dice, pooled Dice/IoU, pooled pixel precision/recall, and FP/FN pixels. All-image Dice with empty-empty=1 is supplementary. Contour-positive/assumed-empty results are separate from primary metrics.

Label 8-connected components on the original full image. Match with maximum-cardinality one-to-one matching using at least one overlapping pixel:

```text
Matched = K            Component precision = K / Npred
FN = Ngt - K           Component recall    = K / Ngt
FP regions = Npred - K
```

One merged prediction can cover several GT objects but match only one. Report no-overlap and covered-but-unmatched GT, zero-overlap and overlapping-but-unmatched predictions, and FP pixels inside matched components. Supplement matching with GT-coverage/pair-IoU thresholds 0.1/0.5. Recompute components after geometry changes.

Subgroups: tiny area<11 native pixels versus regular≥11; isolated if the nearest other GT-centroid distance>128 native pixels, otherwise clustered. These are technical reference-mask groups, not clinical cluster definitions. Report nominal/minimum/maximum matches when optimal matchings have ambiguous identities; disclose small denominators and resolution differences.

Report FP/image overall and separately for positive/negative images, plus clean-negative rate. Zero-denominator ratios are undefined/null. A GT-empty image has no component-recall denominator. Reference components are not necessarily biological specks; MC segmentation is not cancer diagnosis.

## Uncertainty and outputs

Evaluation provides 2000-replicate supplied-group percentile bootstrap intervals, keeping related views together. Report seed variability separately. Seeds/folds/components/repeated annotations are not additional independent patients; proxy groups do not establish verified linkage. Multi-seed hierarchical analyses and reader studies are outside this baseline.

Retain native probabilities, checkpoint/configuration, source/environment versions, manifest/weight hashes, selected epoch, threshold lock, private per-image metrics, aggregate results, and runtime/hardware. Declare any reused checkpoint exposure; this baseline trains from scratch. Report all registered seeds/folds, failures, and exclusions.
