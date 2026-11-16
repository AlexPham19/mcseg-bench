# Model runner quickstart

## A. Optional software check, separate from the real-image benchmark

Use Python 3.10+ and run commands from the package root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m mcbench demo --out runs/demo
```

Official benchmark training/development/test data must be real medical images. Start the real-data setup in section B; the demo below is an optional software check and supplies no benchmark training examples or eligible results.

The demo creates procedural bright-speck arrays, `manifest.csv`, `threshold_lock.json`, and `test_results.json`. No mammograms are included. `primary` contains metrics on `dense_verified` references, `per_image` supports auditing, and `ci` bootstraps supplied groups.

For a visual explanation, read the [pipeline guide](PIPELINE.md).

## B. The organizer prepares real data once

Clone the split repository at the frozen commit and run its validator from that repository. Obtain authorized data, prepare images/masks, and audit labels and group linkage according to the shared protocol.

Example train/development conversion, using placeholder local paths:

```bash
python scripts/prepare_manifest.py --train-csv ../mcseg-bench/splits/csaw_s/fold_1/train.csv --dev-csv ../mcseg-bench/splits/csaw_s/fold_1/validation.csv --data-root data/csaw --images-dir images --masks-dir masks --label-status dense_verified --out data/fold1.csv
python -m mcbench validate --manifest data/fold1.csv --data-root data/csaw
```

Use `dense_verified` only after the reference audit. The converter defaults to `unknown` to avoid silently treating uncertain labels as complete GT. For different source columns, supply `--case-column`, `--group-column`, `--image-column`, and `--mask-column`. The converter does not generate new splits or verify annotation/patient identity.

If test evaluation is planned, include test rows in **the same manifest before threshold calibration**. Use `--test-csv` only when that table has the correct selected reference-mask column. Expert annotators A/B require separate manifests, with the same images and their corresponding reference masks. Do not duplicate case IDs or use one reference for both annotators.

INbreast contour-positive/assumed-empty rows need the correct per-row status and explicit masks consistent with the reference policy. Assumed negatives are excluded from primary metrics. For a hidden test, the organizer performs evaluation separately with access to test references. The starter's manifest checks inspect file existence/hashes; it is not an access-control boundary.

## C. Train the bundled reference U-Net

Install PyTorch for your hardware/CUDA first, or use `python -m pip install -r requirements-train.txt` if the default build is suitable. In the commands below, `data/fold1.csv` is the organizer's frozen manifest, already containing test rows if test evaluation is planned.

```bash
python -m mcbench train --manifest data/fold1.csv --data-root data/csaw --out runs/unet42 --seed 42 --epochs 50 --steps 128 --batch 4 --patch 256 --device cuda:0
python -m mcbench predict --manifest data/fold1.csv --data-root data/csaw --checkpoint runs/unet42/model.pt --split dev --out runs/unet42-dev --device cuda:0
python -m mcbench lock --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-dev --out runs/unet42-lock.json
```

Checkpoint selection maximizes development foreground IoU at 0.5. Threshold calibration then searches the registered development grid for that fixed checkpoint. Model fitting uses training references, selection uses development references, and prediction uses images only. This new reference implementation does not reproduce the historical PlainUNet/DinoUNet graph.

When the same manifest contains the test role:

```bash
python -m mcbench predict --manifest data/fold1.csv --data-root data/csaw --checkpoint runs/unet42/model.pt --split test --out runs/unet42-test --device cuda:0
python -m mcbench evaluate --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-test --split test --lock runs/unet42-lock.json --op segmentation --out runs/unet42-test-metrics.json
python -m mcbench evaluate --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-test --split test --lock runs/unet42-lock.json --op cad_fp2 --out runs/unet42-test-cad2.json
```

With training/development data only, use `--split dev` and label the result as development. If no threshold satisfies a CAD budget on development data, that operating point is infeasible. Do not search test labels for a replacement threshold. Changes to the manifest/model/pipeline after calibration are rejected. Use new output folders to avoid mixing artifacts.

## D. Plug in another model

Your model's train/inference code should export:

```text
pred_dev/<case_id>.npy
pred_dev/run.json
pred_test/<case_id>.npy
pred_test/run.json
```

Each `.npy` must be a floating foreground-probability array in [0,1], with the original H×W coordinates. Do not submit logits, RGB overlays, or thumbnails. Map resized/tiled predictions back to native coordinates; never resize GT to fit the output. Fixed binary masks belong to a declared fixed-system setting and do not provide a useful probability-threshold curve.

`run.json` requires a real 64-character lowercase hexadecimal `model_sha256`, a versioned `pipeline_id` covering preprocessing/model/postprocessing, and `split` set to `dev` or `test`. Keep the same model hash and pipeline ID between development and test predictions. For a multi-weight system, hash a canonical JSON containing every weight/config hash and submit that JSON as well. Do not invent a hash.

Then use the same `lock` and `evaluate` commands. Complete the [model card](MODEL_CARD_TEMPLATE.md); minimal run metadata does not replace provenance or weight-access terms.

## E. Paired comparison and submission

```bash
python -m mcbench compare --a runs/model-a-test.json --b runs/model-b-test.json --out runs/paired-b-minus-a.json --bootstrap 2000
```

Both results must use the same manifest/split and primary cases/groups. Differences are B−A, with paired group bootstrap intervals for one pair of runs. This is not a multi-seed hierarchical analysis or MRMC study. Register primary endpoints and comparison pairs before viewing test results.

Submit source commit/diff, configuration, logs/environment, checkpoint hashes, probability maps, threshold lock, private per-image metrics, public aggregates, and runtime/VRAM. Report all registered seeds/folds and failures. State the particular objective improved rather than declaring a universal winner from Dice alone.

## F. CPU training smoke test

After running the demo in section A:

```bash
python -m mcbench train --manifest runs/demo/manifest.csv --data-root runs/demo --out runs/unet-smoke --epochs 1 --steps 2 --batch 2 --patch 32 --width 8 --device cpu
python -m mcbench predict --manifest runs/demo/manifest.csv --data-root runs/demo --checkpoint runs/unet-smoke/model.pt --split dev --out runs/smoke-dev
python -m mcbench lock --manifest runs/demo/manifest.csv --data-root runs/demo --pred runs/smoke-dev --out runs/smoke-lock.json
```

This verifies software execution only. Do not include smoke-test IoU in a medical benchmark table.
