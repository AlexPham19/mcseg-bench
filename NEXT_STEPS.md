# Organizer checklist and next steps

**Recommendation:** establish a reproducible benchmark with two separate objectives: contour quality and component recall under an FP budget. For the method-development project, prioritize a verifier conditioned on the particular component being scored, using detailed local and wider-context crops. Add mask refinement when FP pixels remain attached to a true-positive region.

CSAW44 has already been used repeatedly for threshold selection. Further improvements on it are development results, not independent test confirmation.

## Organizer setup: do this once

**Real-image-only scope:** all official fitting/development/calibration/test inputs must be real medical images. Exclude generated lesions, rendered/procedural mammograms, and synthetic medical training patches. Keep software-demo fixtures and their scores outside the study.

1. **P0 — establish label meaning.** Record the source-class mapping, whether annotations describe individual specks or larger regions, which areas are exhaustively annotated, and which negatives have been checked. The CSAW-S paper concerns mammographic tumor segmentation; the dataset name alone does not certify that the project's masks delineate individual MC specks. Trace the actual masks being used. Do not assign `dense_verified` merely to make the code run.
2. **P0 — freeze data and access conditions.** Obtain data from the provider; version preprocessing and SHA256 hashes; inspect orientation, spacing, patient linkage, and sharing permissions. Keep the INbreast67 subset as `assumed_empty`, not verified MC-free or cancer-free images.
3. **P0 — preserve published split membership.** Start with fold 1: 193 training /44 development images and model seed 42. The split seed remains20261024. Evaluate expert annotators A/B separately. Audit prior inspection of expert data; if it informed development, disclose that and obtain a new test set for confirmatory claims.
4. **P1 — prepare one shared private manifest.** Include relative paths, group IDs, split roles, reference status, and image/mask hashes. Run the upstream validator and this package's converter. Freeze the manifest. Where practical, the organizer retains test labels and performs centralized evaluation.
5. **P1 — assign three initial baselines.** Plain U-Net, nnU-Net2D, and DINO-U-Net should share split membership, data eligibility, declared pretraining conditions, and evaluator. Each model submits native-resolution probabilities and run metadata. Add HDoGReg/DeepMiCa after adapter qualification. The legacy `author_recipe_benchmark_20261022` has useful adapter source, but uses CSAW130/34 and has a different artifact state; consult model readiness before reuse. Do not launch its controller by default.
6. **P1 — lock the protocol before real runs.** Use positive-image foreground IoU as the segmentation primary metric. Also report positive-image Dice, pooled pixel P/R, component recall/precision, FP/image, clean-negative rate, subgroups, and stricter-overlap analyses. Fix the threshold grid and checkpoint-selection rule before test evaluation.
7. **P2 — extend the study.** Move from fold 1/seed 42 to five folds and three registered seeds. Use external testing only when checkpoint training exposure is sufficiently understood. Report paired patient/group analyses, uncertainty, and inference cost. Seeds/folds are not additional independent patients.
8. **P2 — prepare a release.** Select a license for the new code and establish the terms for dependencies, weights, and data. Review integrations into the repository. Adding code requires an explicit versioned update to the upstream checksum policy rather than a silent validator change.

## Work assigned to each model runner

- Own one adapter. Do not independently change split membership, metric definitions, negative-reference policy, the threshold grid, or registered postprocessing.
- Record source commit, complete configuration, preprocessing, weight hashes, loss, sampler, pretraining datasets, logs, random seed, hardware, and elapsed time.
- Fit on training data; select checkpoints and thresholds on development data. Submit test predictions before receiving test results. Report every registered seed, including failed and poor runs.
- Submit a model card, `run.json`, native probability maps, threshold lock, private per-image metric JSON, public aggregate results, and three failure cases selected by a common rule. The organizer controls private per-image artifacts.
- State whether the method improves contour quality, recall at the same FP budget, runtime, or a particular subgroup. A single metric does not establish an overall winner.

## Next experiments for the method-development project

1. **Freeze comparison systems:** DINO20+GRID, historical best-Dice+MEAN, AND/TRIM+classifier, and HR-AND+CG. Score them with the same evaluator. Historical weights are system-level comparators, not architecture-only controls when recipes differ.
2. **Train a component verifier on out-of-fold training candidates:** use a native-resolution detail crop, component mask/center, and context crop. Include hard negatives near TPs; ignore uncertain annotations. Compare local-only, local+context, then local+context+global inputs.
3. **Refine or split masks:** merged GT objects and FP pixels attached to TPs require a mask head or explicit splitting. A scalar keep/drop classifier cannot alter a component's internal geometry.
4. **Measure every stage:** candidate coverage/recall, matching after geometry, after verification, and after refinement. Track lost TPs and reductions in FP regions/pixels. Test a soft YOLO prior or rescue branch rather than assuming a hard gate is beneficial.
5. **Predeclare go/no-go criteria:** improved recall/tiny recall at the same development FP budget, fewer FPs at the same development recall, and/or better contour IoU. Repeat across seeds/folds and confirm on a new test set. Use paired uncertainty estimates; do not choose a meaningful-effect cutoff after observing test results.

## Completed in this package

- Analysis, proposed protocol, public-repository audit, shared evaluator, reference trainer, demo, and tests.
- Software execution checks using synthetic data. These checks do not establish medical performance.
- A self-contained source/documentation handoff that does not require the legacy server to read or run the demo.
- English documentation and a pipeline guide showing data roles, artifacts, and implemented versus proposed stages.

## Still required

No real-data model benchmark yet satisfies the entire proposed protocol. Annotation and patient-linkage audits remain necessary. Clean external testing is not established for every checkpoint. Reader studies and clinical-deployment evidence are absent. Existing projects provide starting points, not a complete clinically validated benchmark.
