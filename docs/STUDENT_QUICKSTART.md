# Run the plain U-Net baseline

Run from the repository root with Python 3.10+ and new output directories.

## 1. Install

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
python scripts/validate_splits.py
```

Install a hardware-appropriate PyTorch build separately if needed. Obtain real data from its providers; none are downloaded/bundled. Run repository-integrity checks on the clean release before creating local data/environments/results; see [DEVELOPMENT.md](DEVELOPMENT.md).

## 2. Prepare the real-image manifest

The organizer audits labels, completeness, grouping, and geometry. Example with placeholder local data paths:

```bash
python scripts/prepare_manifest.py --train-csv splits/csaw_s/fold_1/train.csv --dev-csv splits/csaw_s/fold_1/validation.csv --data-root data/csaw --images-dir images --masks-dir masks --label-status dense_verified --out data/fold1.csv
python -m mcbench validate --manifest data/fold1.csv --data-root data/csaw
```

Use `dense_verified` only after audit; the default is `unknown`. The converter does not resplit or certify annotations. Preserve original class definitions and grouping limitations.

For testing, include test rows in the same frozen manifest **before training/calibration**. Use `--test-csv` only with the correct reference-mask column. Expert A/B require separate manifests; the converter cannot infer the correct annotator mask. INbreast contour/assumed-empty rows need their respective statuses and explicit masks, not blanket `dense_verified` labels.

## 3. Train and select a checkpoint

```bash
python -m mcbench train --manifest data/fold1.csv --data-root data/csaw --out runs/unet42 --seed 42 --epochs 50 --steps 128 --batch 4 --patch 256 --device cuda:0
```

Use `--device cpu` for CPU execution. `model.pt` is selected by development foreground IoU at 0.5; `training.json` records configuration, history, and fingerprints. This plain U-Net trains from scratch. Test references do not select weights.

## 4. Lock thresholds on development data

```bash
python -m mcbench predict --manifest data/fold1.csv --data-root data/csaw --checkpoint runs/unet42/model.pt --split dev --out runs/unet42-dev --device cuda:0
python -m mcbench lock --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-dev --out runs/unet42-lock.json
```

The lock saves the best-IoU threshold and feasible FP-budget operating points for the fixed checkpoint. Do not replace it after viewing test results.

## 5. Predict and evaluate test images

Requires test rows already included in the frozen manifest:

```bash
python -m mcbench predict --manifest data/fold1.csv --data-root data/csaw --checkpoint runs/unet42/model.pt --split test --out runs/unet42-test --device cuda:0
python -m mcbench evaluate --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-test --split test --lock runs/unet42-lock.json --op segmentation --out runs/unet42-test-metrics.json
python -m mcbench evaluate --manifest data/fold1.csv --data-root data/csaw --pred runs/unet42-test --split test --lock runs/unet42-lock.json --op cad_fp2 --out runs/unet42-test-cad2.json
```

An infeasible development budget cannot be replaced by a test-selected threshold. If only train/dev exist, evaluate with `--split dev` and call it development. The organizer performs hidden-test evaluation with separate reference access.

Predictions are native H×W floating probabilities in [0,1], saved as `<case_id>.npy` plus `run.json`. Model hash/pipeline ID must agree across development and test. Prediction uses images only; reference usage follows split roles.

## 6. Save the baseline result

Keep weights, training log/configuration, native probabilities/run metadata, frozen manifest, threshold lock, result JSON, source/environment, and runtime/hardware. Report every registered seed/fold and failure. See [PROTOCOL.md](PROTOCOL.md) for metric/reference definitions.
