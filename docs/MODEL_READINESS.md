# Model and project readiness

Audit date: November 15, 2026. [REPOSITORY_AUDIT.json](../reports/REPOSITORY_AUDIT.json) records commits, file counts, and GitHub-detected licenses. Available training code is not equivalent to a qualified run under this protocol. No external model was newly trained on medical data for this handoff.

| Project/model | Available evidence | Appropriate readiness level | Required before the main benchmark |
|---|---|---|---|
| **Current mcseg-bench snapshot** | Splits, checksums, validator;24 files | Useful metadata foundation | Train/infer/evaluator, adapters, data audit, test governance, license |
| **Existing DINO project** | Graph, weights, inference, and replayed ablations | Internal result reproduction; legacy/full-system track | Package training recipe/provenance; audit fairness/exposure; retrain for confirmation |
| **Existing Plain U-Net** | Code and 5-epoch/two-seed pilot | Internal control | Appropriate training budget and shared protocol; the pilot does not establish architectural inferiority |
| **Bundled reference U-Net** | Independent trainer/inference/evaluator; CPU synthetic smoke passed | Software baseline ready for assignment | Run on audited real data; not the historical PlainUNet graph |
| **nnU-Netv2 2D** | Official training/inference framework; Apache-2.0 | **Recommended second baseline**, R2 candidate | Adapter, tiny-target resampling audit, registered seeds/budget |
| **DinoUNet / DINOv3** | Official integration, backbone and training code; related variant used internally | Foundation R2 candidate; existing internal checkpoint R1 | Avoid counting the same graph twice; weight terms, CUDA ops, shared recipe |
| **Complete HDoGReg** | Official train/infer and released FPN; internal checkpoint inference performed | **Internal R1 inference completed**; conditional R2 | Retrain on CSAW for clean INbreast transfer; pin compatibility; audit exposure and redistribution terms |
| **DeepMiCa** | U-Net train/test and preprocessing/patch scripts; MIT | Feasible MC-specific R2 candidate | Shared adapter/split, rebuilt environment and qualification; retired bundle is not ready to run |
| **MedSAM** | Official code/checkpoint; repository Apache-2.0 | Separate prompt-assisted track | Prompt source/cost; proposal misses for automatic use; GT boxes are oracle prompts |
| **DINOv2/v3 with another decoder** | Public foundation encoders | Encoder-ablation R2 candidate | Train a head, control decoder/recipe; not an already replicated MC segmentation paper |
| **MC-GenRef** | Commit `ddb32fa...`; one README, no implementation entrypoints | **Not runnable from the audited release** | Wait for code, weights, configuration, and recipe; a guessed reimplementation is not official replication |
| **GMIC** | Global/local classification concept | Verifier inspiration or classification control | Different from pixel MC segmentation; needs a task-appropriate head and training |

Official sources: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [DinoUNet](https://github.com/yifangao112/DinoUNet), [DINOv3](https://github.com/facebookresearch/dinov3), [HDoGReg](https://github.com/cmarasinou/HDoGReg), [DeepMiCa](https://github.com/ales-git/DeepMiCa), [MedSAM](https://github.com/bowang-lab/MedSAM), [MC-GenRef](https://github.com/hwcho-research/MC-GenRef).

## Legacy internal benchmark: what can be reused?

Remote source remains for `mcseg/experiments/author_recipe_benchmark_20261022`, including adapters, a training controller, qualification, scoring, and freeze-before-evaluate logic for DeepMiCa/HDoGReg/nnU-Net. It is the closest task-specific implementation reference found during the audit.

Its protocol uses **CSAW130 fitting /34 validation**, seeds 20261009/10/11, and the earlier INbreast27 setup, not the current193/44 split and seed 42. Historical notes describe storage/runtime interruptions. The expected current report directory was not found at the README's path during inspection. An old handoff does not establish that all nine runs finished or that required payloads remain.

The related `deepmica_image_only_preprocessing_20261023` includes `RETIRED_20261103.md`: source/recipes/results/logs were retained, while unnecessary weights, generated data, caches, and dedicated environments were removed. Do not present it as a runnable model bundle.

Recommendation: use that source as an implementation reference for full-system adapters, then repin dependencies and qualify them against the new manifest. Do not launch its controller by default or substitute results from its different split into the new leaderboard. This package does not copy vendor source or depend on the old server.

## Real-image eligibility

Official benchmark fitting, development, calibration, and test inputs must be real medical images from the audited manifests. Public code or a historical score does not establish eligibility. If a checkpoint's medical training data origin is unknown or includes synthetic medical examples, audit its provenance or retrain on the frozen real-image split before treating it as a compliant benchmark entry. Foundation pretraining is declared separately.

## Suggested assignments

**Round 1:** plain U-Net, nnU-Net2D, and DINO-U-Net: scratch control, strong self-configuring baseline, and foundation-feature baseline.

**Round 2:** DeepMiCa, retrained HDoGReg, and DINO+geometry+verifier/refiner. Add MedSAM only with a defined prompting track. MC-GenRef has no runnable implementation in the audited snapshot; paper scores cannot substitute for a run under this protocol.

Distinguish R1 released-weight inference, R2 training public code under the new protocol, and R3 original-paper reproduction. No external project is guaranteed to reproduce100% of a paper. Data/splits/recipe/weights/environment must agree; bitwise equality also depends on hardware/kernels.

## Licenses and redistribution

GitHub API returned `license=null` for mcseg-bench/HDoGReg/DinoUNet and a custom/unclassified license (`NOASSERTION`) for DINOv3. This does not mean public domain. Check repository/submodule/weight terms before redistribution or commercial use. This package supplies links, audit metadata, and independently written code, not vendor source or weights. SPDX detection is not a complete legal review.

## Adapter acceptance criteria

1. Training does not use test labels; disclose group partitions, training exposure, and licenses.
2. Forward/inference passes on procedural samples; small test objects retain correct coordinates through transformations.
3. Native probabilities satisfy the interface; hashes/configuration/environment/logs are present; geometry changes trigger score/matching recomputation.
4. Development threshold locking and shared evaluation pass. No GT prompts in the main automatic track; unknown references are not silently negative.
5. Registered seed/fold artifacts support rerunning. Report executed results, failures, and limitations rather than filling the leaderboard with published paper numbers.
