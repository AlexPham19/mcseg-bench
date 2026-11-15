# Sources and evidence scope

Checked 2026-11-15. Repository commits and file inventories are recorded separately in REPOSITORY_AUDIT.json. No new medical-cohort inference was performed for this handoff.

## Current project evidence

- Existing numeric replay and precision/recall tables: `outputs/csaw44-model-comparison-20261115/COMPARISON_VI.md` in the working project. The handoff contains only aggregate metrics and newly plotted aggregate figures, not the medical illustration payload.
- Historical best-Dice comparison and HDoGReg replication notes: `analysis/CSAW44_BEST_DICE_COMPARISON_20261114.md`, `analysis/HDOGREG_INBREAST_20261114.md` in the working project.
- Remote legacy source audit: `mcseg/experiments/author_recipe_benchmark_20261022/{README.md,HANDOFF.md,EXECUTION_NOTES.md}`. Different CSAW130/34 protocol; source existence is verified, current nine-run completion is not.
- Retired legacy DeepMiCa record: `deepmica_image_only_preprocessing_20261023/RETIRED_20261103.md`. Source/recipe retained; do not advertise as runnable model bundle.

## Benchmark definition and model implementations

- [Benchmark repository README](https://github.com/AlexPham19/mcseg-bench/blob/d7ac04e89e6fbe85165a0c33836f979b9b7d59b9/README.md).
- [Split policy and cohort limitations](https://github.com/AlexPham19/mcseg-bench/blob/d7ac04e89e6fbe85165a0c33836f979b9b7d59b9/splits/README.md).
- [Split metadata](https://github.com/AlexPham19/mcseg-bench/blob/d7ac04e89e6fbe85165a0c33836f979b9b7d59b9/splits/metadata.json).
- [Existing split validator](https://github.com/AlexPham19/mcseg-bench/blob/d7ac04e89e6fbe85165a0c33836f979b9b7d59b9/scripts/validate_splits.py).
- [CSAW-S original paper](https://proceedings.mlr.press/v119/matsoukas20a.html).
- [HDoGReg official source](https://github.com/cmarasinou/HDoGReg).
- [DeepMiCa official source](https://github.com/ales-git/DeepMiCa).
- [DinoUNet official source](https://github.com/yifangao112/DinoUNet).
- [DINOv3 official source and model terms](https://github.com/facebookresearch/dinov3).
- [nnU-Net official source](https://github.com/MIC-DKFZ/nnUNet).
- [MedSAM official source](https://github.com/bowang-lab/MedSAM).
- [MC-GenRef official repository](https://github.com/hwcho-research/MC-GenRef): API audit found only one README and no implementation entrypoints.

## Metrics, reporting and clinical evaluation

- [Metrics Reloaded, Nature Methods](https://www.nature.com/articles/s41592-023-02151-z): task-aware image/object/pixel metric selection and aggregation.
- [CLAIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/): reporting data partitions, references and testing.
- [DECIDE-AI, Nature Medicine](https://www.nature.com/articles/s41591-022-01772-9): reporting early live clinical evaluation of AI decision support.
- [FDA Clinical Decision Support guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software).
- [FDA CDS decision step](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support): medical-image analysis context. Regulatory pathway depends on intended use; no particular clearance class is asserted here.
- [Medical Image Analysis scope](https://shop.elsevier.com/journals/medical-image-analysis/1361-8415).
- [IEEE TMI information for authors](https://ieeexplore.ieee.org/ielx7/42/9393623/09393624.pdf).
- [Radiology: Artificial Intelligence author instructions](https://pubs.rsna.org/page/ai/author-instructions).

The suggested project direction, priority order, loss/recipe defaults, FP budgets and publication fit are recommendations for this project, not results or acceptance criteria supplied by these sources.
