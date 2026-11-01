# MC segmentation benchmark splits

Note: Use one fixed seed - let's do "42", and only use Fold 1 for train/test/validation first

**Why we only have 94 images in InBreast: I have filtered all ineligible images, so we only have 94 (67 negative + 27 positive images). (Images must be either negative, or "pixel-level contour only", cluster approximation/only "point" is not accepted). Let me know if there's any problem!**

What we do: Train, validation in CSAW-S, test on CSAW-S (expert set)

Exact split membership and evaluation cohorts used by the microcalcification segmentation project, exported from registered metadata on **31 October 2026**. This initial repository release contains split files and their validator; it does not contain the training implementation, model weights, images, masks or patient clinical records.

- [Split definitions and usage](splits/README.md)
- [CSAW-S development list](splits/csaw_s/development.csv)
- [CSAW-S five folds](splits/csaw_s/folds.json)
- [INbreast94 evaluation list](splits/inbreast/evaluation_94.csv)
- [Counts and source-manifest fingerprints](splits/metadata.json)

## Quick verification

Requires Python 3.9+ and the standard library only:

```bash
python scripts/validate_splits.py
```

The validator checks exact file checksums, image/group separation, once-only validation membership across five folds, expert-linked exclusions, and the nested INbreast27/67/94 cohort definitions.

Obtain source data separately through the original dataset providers. These metadata files do not redistribute the underlying datasets or change their access conditions.
