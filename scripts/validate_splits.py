"""Verify shipped split membership and checksums without accessing source images."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows):
    ids = [row["image_id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate image IDs")
    return {row["image_id"]: row for row in rows}


def groups(rows):
    return {row["group_id"] for row in rows}


def main():
    metadata = json.loads((ROOT / "splits/metadata.json").read_text(encoding="utf-8"))
    development = keyed(read_csv("splits/csaw_s/development.csv"))
    require(len(development) == 237, "CSAW-S development image count differs")
    require(len(groups(development.values())) == 115, "CSAW-S group count differs")
    excluded = keyed(read_csv("splits/csaw_s/excluded_expert_linked.csv"))
    expert = keyed(read_csv("splits/csaw_s/expert_evaluation.csv"))
    require(len(excluded) == 15, "Excluded record count differs")
    require(len(expert) == 26 and len(groups(expert.values())) == 23,
            "Expert evaluation count differs")
    require(not groups(excluded.values()) & groups(development.values()),
            "An excluded patient group entered development")
    expert_suffixes = {row["group_id"].rsplit("/", 1)[-1] for row in expert.values()}
    require(not expert_suffixes & groups(development.values()),
            "Registered conservative expert suffix exclusion was not retained")
    assignments = json.loads((ROOT / "splits/csaw_s/folds.json").read_text(encoding="utf-8"))
    require(assignments["split_seed"] == metadata["split_seed"] == 20261024,
            "Split seed differs")
    validation_images, validation_groups = Counter(), Counter()
    summaries = []
    for expected, assignment in zip(metadata["csaw_s"]["folds"], assignments["folds"]):
        fold = expected["fold"]
        require(fold == assignment["fold"], "Fold ordering differs")
        train = keyed(read_csv(f"splits/csaw_s/fold_{fold}/train.csv"))
        val = keyed(read_csv(f"splits/csaw_s/fold_{fold}/validation.csv"))
        require(not train.keys() & val.keys(), f"Fold {fold}: image overlap")
        require(not groups(train.values()) & groups(val.values()),
                f"Fold {fold}: group overlap")
        require(train.keys() | val.keys() == development.keys(),
                f"Fold {fold}: development membership differs")
        for image_id, row in (list(train.items()) + list(val.items())):
            require(row == development[image_id], f"Fold {fold}: changed record {image_id}")
        require(len(train) == expected["train_images"] and len(val) == expected["validation_images"],
                f"Fold {fold}: image counts differ")
        require(len(groups(train.values())) == expected["train_groups"] and
                len(groups(val.values())) == expected["validation_groups"],
                f"Fold {fold}: group counts differ")
        require(groups(val.values()) == set(assignment["validation_group_ids"]),
                f"Fold {fold}: JSON/CSV assignments differ")
        validation_images.update(val.keys())
        validation_groups.update(groups(val.values()))
        summaries.append(f"Fold {fold}: {len(train)} train / {len(val)} validation images")
    require(len(summaries) == 5, "Expected exactly five folds")
    require(set(validation_images) == set(development) and set(validation_images.values()) == {1},
            "Every development image must validate once")
    require(set(validation_groups) == groups(development.values()) and
            set(validation_groups.values()) == {1}, "Every development group must validate once")

    cohort = keyed(read_csv("splits/inbreast/evaluation_94.csv"))
    contours = keyed(read_csv("splits/inbreast/contour_27.csv"))
    negatives = keyed(read_csv("splits/inbreast/assumed_negative_67.csv"))
    require((len(cohort), len(contours), len(negatives)) == (94, 27, 67), "INbreast counts differ")
    require(not contours.keys() & negatives.keys(), "INbreast strata overlap")
    require(contours.keys() | negatives.keys() == cohort.keys(), "INbreast94 union differs")
    for image_id, row in (list(contours.items()) + list(negatives.items())):
        require(cohort[image_id] == row, f"INbreast subset record differs: {image_id}")
    require(all(row["reference_empty_assumed"] == "false" and row["reference_filename"]
                for row in contours.values()), "Contour-reference policy differs")
    require(all(row["reference_empty_assumed"] == "true" and
                row["source_dense_negative_valid"] == "false" and
                not row["xml_filename"] and not row["reference_filename"]
                for row in negatives.values()), "Assumed-negative validity metadata differs")
    require(len({row["group_proxy_id"] for row in cohort.values()}) == 44,
            "INbreast grouping proxy count differs")
    require(metadata["inbreast"]["evaluation_only"] is True and
            metadata["inbreast"]["group_proxy_linkage_verified"] is False,
            "INbreast role/linkage limitation changed")

    checksums = ROOT / "checksums.sha256"
    require(checksums.exists(), "Missing checksum manifest")
    listed = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected_hash, name = line.split("  ", 1)
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe checksum path")
        require(sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash,
                f"Checksum mismatch: {name}")
        listed.add(relative.as_posix())
    release_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                     if p.is_file() and not any(part in {".git", "__pycache__"}
                                              for part in p.relative_to(ROOT).parts)
                     and p.name != "checksums.sha256"}
    require(listed == release_files, "Checksum manifest does not cover exactly the release files")
    print("PASS: metadata checksums and split membership")
    print("\n".join(summaries))
    print("CSAW-S: 237 development images / 115 groups; 26 expert images; 15 exclusions")
    print("INbreast: 94 evaluation-only images = 27 contour + 67 assumed-empty; 44 unverified proxies")


if __name__ == "__main__":
    main()
