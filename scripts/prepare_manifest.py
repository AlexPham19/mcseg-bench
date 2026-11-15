"""Convert LOCAL official-repo split CSVs into a private path/hash manifest.

Does not download data, infer patient linkage, certify annotations or re-split.
Run from the handoff root: python scripts/prepare_manifest.py --help
"""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mcbench.io import sha, manifest, relative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",required=True,type=Path)
    ap.add_argument("--dev-csv",required=True,type=Path)
    ap.add_argument("--test-csv",type=Path)
    ap.add_argument("--data-root",required=True,type=Path)
    ap.add_argument("--images-dir",default="images")
    ap.add_argument("--masks-dir",default="masks")
    ap.add_argument("--case-column",default="image_id")
    ap.add_argument("--group-column",default="group_id")
    ap.add_argument("--image-column",default="image_filename")
    ap.add_argument("--mask-column",default="mask_filename")
    ap.add_argument("--label-status",choices=("dense_verified","contour_positive","unknown"),default="unknown",
                    help="Default unknown deliberately blocks scoring/training until reference audit")
    ap.add_argument("--out",required=True,type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite manifest")
    rows = []
    for role,path in (("train",args.train_csv),("dev",args.dev_csv),("test",args.test_csv)):
        if path is None:
            continue
        with path.open(encoding="utf-8-sig",newline="") as f:
            source = list(csv.DictReader(f))
        for r in source:
            im = (Path(args.images_dir)/r[args.image_column]).as_posix()
            mask = (Path(args.masks_dir)/r[args.mask_column]).as_posix()
            rows.append(dict(case_id=r[args.case_column],group_id=r[args.group_column],split=role,image=im,mask=mask,
                             label_status=args.label_status,image_sha256=sha(relative(args.data_root,im)),
                             mask_sha256=sha(relative(args.data_root,mask))))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest(args.out,args.data_root,True)
    print("PASS private manifest built; retain original source split checksums separately")


if __name__ == "__main__":
    main()
