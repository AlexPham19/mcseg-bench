# Final plain U-Net settings

These consolidate the existing baseline recipe, not a proven optimum. The executable source of truth is [mcbench/settings.py](../mcbench/settings.py). CLI overrides are different declared runs; their actual values are saved in `training.json`.

## Input normalization

For each **full grayscale image, before cropping or tiling**:

```text
x = float32(input image)
lo = percentile(x[x > 0], 1)
hi = percentile(x[x > 0], 99.5)
normalized = clip((x - lo) / (hi - lo), 0, 1)
```

Use NumPy's default linear percentile interpolation. Return zeros if no positive intensity exists or `hi <= lo`. `x > 0` selects input intensities, not GT-positive pixels or an anatomical breast mask. Use the same function for train/dev/test. There is no per-patch normalization, cohort-statistics fitting, z-score, ImageNet normalization, maximum-value division, resizing, CLAHE, or histogram equalization in this baseline. The organizer must supply consistently prepared grayscale images with correct orientation/polarity and version DICOM preprocessing separately.

## Model and fitting

| Setting | Final default |
|---|---|
| Benchmark inputs | Real medical images and audited references only |
| Pilot | CSAW-S fold1: 193 train /44 dev; expert26 images scored against A/B separately |
| Model seed / split seed | **42 /20261024**; model seeds do not change partitions |
| Architecture | Plain U-Net from scratch; grayscale input, foreground-logit output; two downsampling levels; base width 16; two 3×3 convolutions/block; bilinear upsampling and skip concatenation |
| Feature normalization | **GroupNorm, 4 groups**, then ReLU; no BatchNorm |
| Training patch | **256×256 native pixels** |
| Training batch | **4 patches per optimizer update**; accumulation 1 |
| Budget | **50 epochs ×128 updates = 6,400 optimizer updates**; an epoch is not a full dataset pass |
| Optimizer | **AdamW: lr 0.001, weight decay 0.01, betas (0.9,0.999), epsilon 1e-8**, AMSGrad off |
| LR schedule | Constant; no warmup, scheduler, or early stopping |
| Loss | Mean BCE-with-logits + mean per-patch soft Dice loss, coefficients **1:1** |
| Soft Dice | `(2*sum(p*y)+1)/(sum(p)+sum(y)+1)` per patch; smoothing **1** |
| Class weights | None; no `pos_weight`, focal loss, or component weights |
| Sampling | Uniform image choice; conditional on foreground existing, GT-pixel-centered crop with probability 0.5; otherwise uniform crop |
| Center jitter | Independent inclusive integer x/y offsets `[-patch//4,+patch//4]`; clamp crop origin to image |
| Augmentation | Independent horizontal/vertical flips, each probability 0.5; no rotations or intensity augmentation |
| Small-image padding | Edge-pad image, zero-pad GT |
| Precision | FP32, no AMP |
| Execution | In-process patch sampling, no DataLoader workers; at most 4 PyTorch CPU threads |
| Reproducibility | Seed Python/NumPy/PyTorch; deterministic algorithms requested; cross-hardware bitwise identity not guaranteed |
| Checkpoint | Maximum dev positive-image foreground macro IoU at **0.5**; earliest epoch on ties |

The positive-center rule does not guarantee half of every batch is positive. It samples GT pixels, not components uniformly.

## Inference and operating points

| Setting | Final default |
|---|---|
| Tile size | **256×256**; follows the checkpoint's training patch size |
| Stride | **128**, generally `patch//2` |
| Inference batch | **8 tiles per forward pass**, including dev checkpoint selection |
| Stitching | Sigmoid per tile, Hann-weighted probability averaging, window floor **0.001** |
| Coordinates | Remove padding to restore native H×W; never resize GT |
| Threshold grid | **0.05,0.10,...,0.95**, development only |
| Segmentation point | Maximum dev positive-image macro IoU, with protocol tie breaks |
| Detection points | Maximum dev component recall under FP/image budgets **0.5,1,2,4,8**; infeasible if no point qualifies |
| Test | Freeze model, normalization, and operating thresholds; no test tuning |
| Confidence intervals | **2,000** supplied-group bootstrap replicates by default |

**Training batch 4 and inference batch 8 serve different purposes.** Inference batching affects memory/throughput and performs no optimizer updates. Input percentile normalization and the network's GroupNorm are also separate operations. The checkpoint-selection threshold 0.5 is fixed for comparing epochs; the final prediction threshold is selected separately using dev data.

`training.json` includes actual `resolved_settings`, optimizer/schedule, precision, stride, loss, and training history/fingerprints. `run.json` records inference batch/stride, normalization, and stitching. Use these execution records and the current baseline branch, not an old downloaded handoff, to interpret a run. Changing normalization/loss/data eligibility requires a new declared recipe and recalibration, not reuse of an old threshold lock.

See [PROTOCOL.md](PROTOCOL.md) for matching/subgroup/reference definitions. These changes make previously implicit defaults explicit; they do not introduce a new tuned recipe or claim real-data performance.
