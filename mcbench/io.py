import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROLES = {"train", "dev", "test"}
LABELS = {"dense_verified", "contour_positive", "assumed_empty", "unknown"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            a = np.asarray(im).copy()
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"Expected finite 2D grayscale array: {path.name}")
    return a


def relative(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest file paths must stay within --data-root")
    return path


def manifest(path, data_root=None, check_files=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"case_id", "group_id", "split", "image", "mask", "label_status", "image_sha256", "mask_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires nonempty rows and columns {sorted(required)}")
    ids, groups, images, image_hash_roles = set(), {}, {}, {}
    for row in rows:
        cid, gid, role = row["case_id"], row["group_id"], row["split"]
        if not cid or Path(cid).name != cid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cid):
            raise ValueError("case_id must be a safe basename using letters, digits, underscore or hyphen")
        if cid in ids or not gid or role not in ROLES or row["label_status"] not in LABELS:
            raise ValueError("Duplicate case ID, invalid group, role or label status")
        if gid in groups and groups[gid] != role:
            raise ValueError("Group leakage across splits")
        if row["image"] in images:
            raise ValueError("Duplicate image path; score annotators in separate manifests")
        ids.add(cid)
        groups[gid] = role
        images[row["image"]] = role
        for key in ("image", "mask"):
            if row[key]:
                digest = row[key+"_sha256"]
                if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Manifest requires lowercase SHA256 for every provided file")
        if row.get("image_sha256"):
            digest = row["image_sha256"]
            if digest in image_hash_roles and image_hash_roles[digest] != role:
                raise ValueError("Identical image hash across splits")
            image_hash_roles[digest] = role
        if row["label_status"] != "unknown" and not row["mask"]:
            raise ValueError("Explicit reference required, including assumed-empty mask")
        if check_files:
            if data_root is None:
                raise ValueError("--data-root required for file validation")
            for key in ("image", "mask"):
                if row[key]:
                    p = relative(data_root, row[key])
                    if not p.is_file():
                        raise ValueError(f"Missing {key} file for {cid}")
                    if row.get(key + "_sha256") and sha(p) != row[key + "_sha256"]:
                        raise ValueError(f"Changed {key} for {cid}")
    return rows


def reference(row, root):
    if row["label_status"] == "unknown":
        raise ValueError("Unknown reference cannot be scored or used for supervised training")
    a = array(relative(root, row["mask"]))
    values = set(np.unique(a).tolist())
    if not (values.issubset({0,1}) or values.issubset({0,255})):
        raise ValueError("Expected binary mask (0/1 or 0/255); convert ignore/multiclass labels explicitly")
    a = a > 0
    if row["label_status"] == "assumed_empty" and a.any():
        raise ValueError("assumed_empty mask contains foreground")
    if row["label_status"] == "contour_positive" and not a.any():
        raise ValueError("contour_positive mask is empty")
    return a


def probability(pred_dir, cid, shape):
    a = array(Path(pred_dir) / (cid + ".npy"))
    if a.shape != shape or not np.issubdtype(a.dtype, np.floating) or a.min() < 0 or a.max() > 1:
        raise ValueError(f"Expected native-shape floating probabilities in [0,1]: {cid}")
    return a
