# Microcalcification segmentation benchmark

## Benchmark process

```text
Real images + reference masks
    -> fixed patient/group splits
    -> train the plain U-Net baseline
    -> select a checkpoint on development data
    -> lock thresholds on development data
    -> predict test images with the frozen model
    -> evaluate with the locked thresholds
    -> report segmentation, component detection, and false positives
```

1. **Prepare real data.** Use real medical images and audited reference masks only. Crops, flips, and intensity normalization are allowed; generated lesions/images are excluded. Keep related views/exams in the same patient/group partition.
2. **Use the registered splits.** Start with CSAW-S fold 1: **193 training /44 development images**, model seed **42**. Preserve split membership and split seed **20261024**. The expert set has **26 images**, scored against annotators A and B separately. See [split definitions](splits/README.md).
3. **Train the baseline.** Fit the plain U-Net on training images/masks. Choose its checkpoint by foreground macro IoU on positive development images at threshold 0.5. Test labels must not select weights.
4. **Lock operating points.** For the selected checkpoint, search thresholds 0.05–0.95 in steps of 0.05 on development data. Save the best-IoU point and component-recall points under FP/image budgets 0.5,1,2,4,8.
5. **Evaluate once.** Predict test images at their original resolution and apply the locked model/thresholds. Do not tune on test results. INbreast is evaluation-only; report its 27 contour-reference and 67 assumed-empty images separately.
6. **Report results.** Include positive-image macro IoU/Dice, pooled pixel precision/recall, component precision/recall, matched/FN/FP counts, FP/image, clean-negative rate, and tiny/regular × isolated/cluster results. Save the manifest, model hash, configuration, threshold lock, and per-image metrics.

Commands: [Run the baseline](docs/STUDENT_QUICKSTART.md). Definitions and reference limitations: [Evaluation protocol](docs/PROTOCOL.md).
