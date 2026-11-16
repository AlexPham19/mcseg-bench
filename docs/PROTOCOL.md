# Shared benchmark protocol

This is a proposed protocol for the handoff edition dated November 15, 2026. Repository/model availability statements refer to that recorded audit, not an ongoing availability check. The [pipeline guide](PIPELINE.md) explains the execution sequence.

## 1. Scope and reproducibility levels

The audited [mcseg-bench repository](https://github.com/AlexPham19/mcseg-bench) provides splits, checksums, a validator, and cohort notes. It does not yet provide the complete training/inference/evaluation workflow. This package is an independent starter for that missing workflow and does not alter published split membership.

Recommended task: **fully automatic microcalcification-mask segmentation against an explicitly defined reference**, with component localization evaluated separately. Results describe algorithms on a particular cohort/reference. MC masks, clinically suspicious clusters, and cancer diagnosis are different targets.

Freeze the protocol version, source commit, data hashes, and reference policy before execution. Distinguish:

- **R1:** rerun inference using released weights.
- **R2:** retrain a public implementation under this benchmark's data/protocol.
- **R3:** reproduce the original paper's results, requiring its data/splits/recipe/weights and access conditions. R1/R2 do not establish R3, and public code alone does not guarantee100% reproduction.

## Data policy: real medical images only

**All official benchmark training, development, calibration, and test inputs must be real medical images from the organizer's audited manifests.** Synthetic lesions, rendered mammograms, procedural objects, and generated synthetic training patches are excluded; there is no synthetic-data benchmark track in this project.

Cropping, flips, and declared intensity normalization of real images are ordinary transformations of real data. They do not introduce generated lesions. Foundation-model pretraining and its data exposure must be declared separately; medical fine-tuning and benchmark evaluation still use only the eligible real images.

The optional procedural demo and unit-test fixtures are software checks only. Their arrays, checkpoints, and scores are excluded from official benchmark fitting, calibration, test cohorts, and result tables.

## 2. Development and confirmatory phases

| Phase | Data and seeds | Purpose | Permitted interpretation |
|---|---|---|---|
| Pilot v0 | CSAW fold 1:193 training /44 development images; model seed 42 | Debug adapters, runtime, losses, and metrics | Exploratory comparison under a shared protocol |
| Confirmatory v1 | Five development folds plus an independently locked test; registered model seeds 42/43/44 | Estimate stability and final performance | Held-out conclusions within the test population/reference scope |

The upstream **split seed is20261024**. Model seed 42 controls initialization/training, not partition membership. Other folds have185/52,194/43,189/48, and 187/50 training/development images. Preserve membership and hashes rather than regenerating partitions from the seed.

The 237-image development pool contains 115 supplied groups. Keep all related views/exams from the same patient together when linkage permits. A supplied group/proxy is not independently verified clinical patient linkage. Prevent patches or images from crossing roles. Fit dataset statistics and mine hard negatives using eligible real training data only. Declared deterministic per-image normalization is an inference operation, distinct from fitting statistics on the test cohort.

**CSAW44 has repeatedly informed optimization.** It is development data, not a held-out leaderboard. The expert set contains 26 images/23 supplied groups with annotators A/B on the same images. Score the references separately; they are not52 independent images. Audit prior expert-data inspection. If it informed development, disclose that and seek new data for confirmatory testing. Renaming or reshuffling inspected data does not restore an untouched test.

INbreast94 remains an evaluation supplement:27 contour-reference images plus67 assumed-empty images. Do not tune thresholds on it. The 67 images lack independent dense-negative verification; clean-reference results are not clinical specificity. The 44 grouping proxies are not44 verified patients. INbreast27 is nested within94, not an independent second cohort. HDoGReg weights with uncertain INbreast training/validation exposure cannot be treated as a clean external zero-shot comparator there.

Report partition level, reference standards, and external testing explicitly, consistent with [CLAIM2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/).

## 3. Organizer-owned data contract

Use a UTF-8 CSV manifest and preserve IDs as strings, including leading zeros:

```text
case_id,group_id,split,image,mask,label_status,image_sha256,mask_sha256
```

- `split`: `train`, `dev`, or `test`. Paths are relative to the data root and may not escape it.
- `dense_verified`: the reference is audited as sufficiently complete for pixel/component evaluation and background interpretation over the image. This is an organizer audit requirement, not a default inherited from a dataset name.
- `contour_positive`: positive contours exist, but completeness/background interpretation is limited; report supplemental reference-relative metrics.
- `assumed_empty`: an explicit empty mask based on a declared assumption; supplemental reporting only, excluded from primary specificity/precision calculations.
- `unknown`: exclude from supervised fitting/scoring rather than treating as negative. Ignore-region/multiclass annotations need an audited conversion; the starter rejects masks not represented consistently as binary.

Masks must use 0/1 or0/255 and the original image orientation/shape. Record source classes, annotators, native spacing where available, valid-region policy, exclusions, and reasons. Version DICOM polarity/VOI/LUT handling, image transforms, contour rasterization, and annotation-completeness assumptions.

For regional/cluster annotations, connected components are **reference components**, not necessarily individual biological specks. The [CSAW-S paper](https://proceedings.mlr.press/v119/matsoukas20a.html) concerns tumor segmentation; the project's MC class mapping must be traced from actual masks/code.

The starter does not establish annotation correctness, clinical linkage, data permissions, or untouched-test status. Checksums establish file integrity only.

## 4. Comparison tracks

**Track A — common recipe.** Compare plain U-Net and backbone/decoder adaptations under shared eligible data, patch sampling, optimizer/tuning budget, native tiled inference, and postprocessing. Declare pretrained versus scratch initialization, trainable/total parameters, pretraining exposure, VRAM, and time.

**Track B — full system.** Preserve registered system-specific recipes for nnU-Net auto-configuration, complete HDoGReg, DeepMiCa, and DINO+HDoG+classifier systems. Use the same splits, references, native-coordinate outputs, and evaluator. Report preprocessing/compute differences. This compares systems; it does not isolate architecture alone.

**Separate prompt-assisted track.** Fix prompt source/count and interaction cost for MedSAM. GT-derived boxes belong to an oracle track. Automatic prompting must come from a detector trained on eligible training data, with its missed proposals included in end-to-end evaluation.

Synthetic medical training data are outside this benchmark and must not be introduced as another track. Any additional real training cohort requires organizer approval and an explicit declaration that preserves the registered comparison conditions. Declare foundation-model pretraining separately. A foundation encoder alone is not an MC segmentor without a trained prediction head.

## 5. Training, checkpoint selection, and threshold calibration

The pilot uses fold 1/model seed 42. A fuller study should register five folds and three seeds before execution, with a practical compute budget. Fifteen runs do not create fifteen independent cohorts.

The proposed common-recipe starting point is 256×256 patches, 50 epochs ×128 optimizer steps, batch size 4, AdamW at 1e-3, and equally weighted BCE + soft Dice. After selecting an image, positive-centered sampling is used with probability 0.5 when that image contains GT; otherwise sampling is uniform, with flips. This is an initial recipe, not a claimed optimum. Assess convergence on development data and allocate fair budgets; full-system comparators may retain their declared author recipes.

The bundled U-Net has two downsampling levels, GroupNorm, and base width 16. It is independently implemented, not the historical4.37M-parameter PlainUNet or DinoUNet graph. Select its checkpoint by development positive-image foreground macro IoU at threshold 0.5; choose the earliest epoch on ties.

For the selected fixed checkpoint, calibrate once on development data using thresholds0.05..0.95 in steps of 0.05. Register any component-verifier score grid and trial budget separately rather than giving one method an unreported larger search.

Lock two families of operating points:

1. **Segmentation:** maximize development positive-image macro foreground IoU. Break ties by higher component recall, lower FP/image, then higher threshold.
2. **CAD engineering:** maximize development component recall subject to FP/image≤B, for B∈{0.5,1,2,4,8}. Break ties by lower FP/image, higher IoU, then higher threshold. Report infeasible if the grid has no qualifying point. These are research budgets, not clinically accepted alert limits.

Apply the locked model/configuration/threshold to test data. Report actual test FP even if it exceeds the development budget. Never retune on test labels to force the budget. The lock records the development curve/fingerprints, model hash, pipeline ID, and manifest hash; model/pipeline/manifest changes are rejected. This is a reproducibility safeguard, not an access-controlled competition server. A hidden test requires separate label access and centralized evaluation.

## 6. Shared metrics

### Pixel metrics

Evaluate foreground rather than background-inclusive mIoU:

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

The segmentation primary is **macro foreground IoU over positive images with eligible reliable references**. Also report positive-image macro Dice, pooled Dice/IoU, pooled pixel precision/recall, and FP/FN pixels. All-image averages using empty-empty=1 are explicitly labeled supplements; clean negatives must not obscure contour performance. Never replace a legitimate zero with 1 using `metric or 1.0`.

### Component metrics

Label 8-connected components on the original full image. The legacy primary graph has an edge for at least one overlapping pixel; use **maximum-cardinality one-to-one matching**:

```text
TPc = K                component precision = K / Npred
FNc = Ngt - K          component recall    = K / Ngt
FPc = Npred - K
```

If a stage changes geometry, relabel and recompute scores/matching rather than reusing the previous component identities.

One-pixel overlap is lenient. Supplement it with maximum matching requiring GT coverage≥0.1/0.5 and pair IoU≥0.1/0.5. These registered engineering sensitivity analyses do not certify clinical localization. Do not invent a clinical millimeter tolerance without an appropriate annotation/reference definition.

Required error decomposition:

- `GT FN = no-overlap GT + covered-but-unmatched GT`, exposing merges/conflicts.
- `FP regions = zero-overlap predictions + overlapping-but-unmatched predictions`, exposing splits/duplicates.
- FP pixels inside a matched component still degrade contour quality; component precision alone misses this oversegmentation.
- Report FP/image overall and separately on positive/negative images, clean-negative rate, and images with predictions. Ratios with zero denominator are undefined/null; negative-image component recall has no defined denominator.

### Subgroups and ambiguity

Use original-reference geometry: tiny area<11 native pixels, regular area≥11; isolated if the nearest other GT-centroid distance>128 native pixels, clustered otherwise. These inherited technical definitions are not clinical cluster criteria. Do not resize GT before grouping. Where valid spacing exists, add preregistered physical-unit strata for cross-scanner comparisons. A subgroup with only four objects cannot support broad superiority claims.

GT match identities can differ between equally optimal maximum matchings. Report nominal and minimum/maximum matched counts for each subgroup. `covered_gt` is a different metric: one merged prediction can touch several GT components but match only one. Do not equate coverage with one-to-one recall.

Choose image/object/pixel metrics according to the actual task, as discussed in [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).

## 7. Detection FROC and reader workflow

Pixel-threshold sweeps can merge/split objects, making matching nonmonotonic. The starter returns an operating-point curve, **not automatically a conventional fixed-candidate scored FROC**.

For alert-level FROC, define fixed candidate geometry, alert grouping, and one score per alert using training/development data. Sweep that score, match under a locked rule, and report lesion/reference sensitivity against false marks/image or exam. State the denominator and observed curve range. Do not use GT to group alerts at inference. Alert-level FROC/grouping is not implemented in this starter.

A small FP region is not necessarily a reader-visible mark. Clinical studies additionally need marks/exam, reading time, lesion/patient-level sensitivity, appropriate diagnostic specificity, and multireader/multicase design. Plan washout/crossover and sample size with clinical/statistical collaborators. A global negative score alone does not justify removing an image from reading.

## 8. Statistics and reporting

Bootstrap patient/supplied groups rather than independent components; keep related views/exams together. Seed/fold variability and patient-sampling uncertainty answer different questions and must be reported separately. The starter provides2000-replicate group-percentile intervals and paired B−A comparisons for one run pair; it does not implement multi-seed hierarchical CI, MRMC, or multiple-comparison correction.

Compare models on the same groups/test using paired effects and intervals, with endpoints/comparisons chosen beforehand. A/B annotations on the same26 images are reference sensitivity analyses, not doubled sample size. If linkage is only a proxy, call the procedure group bootstrap rather than verified-patient bootstrap. Registration after observing results is not preregistration.

Submit source commit/dirty-diff hash, environment lock, weight hashes, data-manifest hash, pretraining exposure, checkpoint criterion, threshold lock, native probabilities, prediction metadata, private per-image metrics, public aggregates, runtime/VRAM/GPU/cost, and failure analysis. Match inference hardware, precision, thread/batch/tiling/postprocessing settings or disclose differences.

## 9. Integration with the existing repository

Preserve the audited split snapshot/validator. Review additions of `mcbench`, `docs`, `tests`, and `scripts`; do not commit medical data or local runs. The upstream validator currently requires checksums to cover all release files. Adding code requires an explicit versioned checksum update or a clearly documented split-snapshot-only checksum policy; do not silently weaken it to obtain PASS.

Students do not need a 3GB standalone project or 76 weights to begin. The organizer prepares data once; students supply adapters and predictions. Select a release license for new code and preserve upstream code/weight terms. Every adapter must pass the acceptance criteria in [MODEL_READINESS.md](MODEL_READINESS.md).
