"""Tiny generated arrays for software tests only; never benchmark training data."""
import csv
from pathlib import Path

import numpy as np

from mcbench.io import sha, write_json


def make_fixture(root):
    root = Path(root)
    (root/"images").mkdir(parents=True)
    (root/"masks").mkdir()
    rng = np.random.default_rng(42)
    rows = []
    for i in range(9):
        case = f"test_fixture_{i}"
        role = "train" if i < 3 else "dev" if i < 6 else "test"
        gt = np.zeros((24, 28), dtype=np.uint8)
        if i % 3 != 2:
            gt[5:7, 6:8] = 1
            gt[15:18, 20:23] = 1
        image = rng.uniform(.05, .2, gt.shape).astype(np.float32) + gt * .6
        image[10:12, 12:14] = .7
        np.save(root/"images"/(case+".npy"), image)
        np.save(root/"masks"/(case+".npy"), gt)
        rows.append(dict(case_id=case, group_id=f"fixture_group_{i}", split=role,
                         image=f"images/{case}.npy", mask=f"masks/{case}.npy", label_status="dense_verified",
                         image_sha256=sha(root/"images"/(case+".npy")), mask_sha256=sha(root/"masks"/(case+".npy"))))
        if role != "train":
            pred = root/("pred_"+role)
            pred.mkdir(exist_ok=True)
            np.save(pred/(case+".npy"), image.astype(np.float32))
    with (root/"manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for role in ("dev", "test"):
        write_json(root/("pred_"+role)/"run.json", dict(model_sha256="f"*64, pipeline_id="software_test_fixture", split=role))
    return root/"manifest.csv"
