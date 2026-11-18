# Baseline software checks

The runtime contains one model: the plain U-Net in `mcbench/baseline.py`. Other modules load data, select development thresholds, or evaluate predictions.

```text
mcbench/baseline.py       U-Net, patch sampling, training, tiled inference
mcbench/settings.py       Canonical defaults, shared by CLI and runtime
mcbench/io.py             manifest/reference/probability validation
mcbench/metrics.py        pixel/component metrics, group bootstrap
mcbench/__main__.py       validate, train, predict, lock, evaluate
scripts/prepare_manifest.py
scripts/validate_splits.py
scripts/verify_integrity.py
tests/                   software fixtures and metric/protocol checks
```

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/validate_splits.py
python scripts/verify_integrity.py
python -m unittest discover -s tests -v
python -m mcbench --help
```

Tests use small arrays in temporary directories to check matching, empty masks, group leakage, threshold locks, and probability validation. They are software fixtures only. The public demo/comparison commands have been removed; there is no generated-data benchmark entry point.

`checksums.sha256` describes the clean release tree, excluding itself, Git internals, and Python bytecode caches. The split validator requires that exact membership. Run release checks before creating ignored local data/results or a virtual environment inside the repository. Model-level `validate` checks the private data manifest separately.

For release edits, preserve registered split files, normalize text to LF per `.gitattributes`, and regenerate checksums from intended release files. Exclude local data, predictions, weights, environments, and caches. Check staged Git bytes against the manifest before publishing.
