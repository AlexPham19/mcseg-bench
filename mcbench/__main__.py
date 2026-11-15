import argparse
from pathlib import Path
import platform
import sys

import numpy as np

from .io import manifest, probability, read_json, reference, relative, sha, write_json
from .metrics import measure, aggregate, bootstrap, paired_comparison


def run_metadata(pred_dir, role):
    meta = read_json(Path(pred_dir)/"run.json")
    digest = meta.get("model_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not meta.get("pipeline_id") or meta.get("split") != role:
        raise ValueError("run.json requires model_sha256, pipeline_id and matching split")
    return meta


def score(rows,root,pred_dir,role,threshold):
    selected = [r for r in rows if r["split"] == role]
    if not selected:
        raise ValueError("Requested split is empty")
    result = []
    for row in selected:
        if row["label_status"] == "unknown":
            continue
        gt = reference(row,root)
        p = probability(pred_dir,row["case_id"],gt.shape)
        result.append(dict(case_id=row["case_id"],group_id=row["group_id"],label_status=row["label_status"],
                           **measure(gt,p >= threshold)))
    return result


def calibration(rows,root,pred_dir,manifest_path):
    meta = run_metadata(pred_dir,"dev")
    dev = [r for r in rows if r["split"] == "dev"]
    if not dev or any(r["label_status"] != "dense_verified" for r in dev):
        raise ValueError("Threshold calibration requires audited dense_verified dev references")
    points = []
    for t in np.round(np.arange(.05,.951,.05),2):
        points.append(dict(threshold=float(t),**aggregate(score(rows,root,pred_dir,"dev",t))))
    valid = [p for p in points if p["positive_macro_iou"] is not None]
    if not valid:
        raise ValueError("Need positive dev images")
    segmentation = max(valid,key=lambda p:(p["positive_macro_iou"],p["component_recall"],-p["fp_per_image"],p["threshold"]))
    operating = {"segmentation": dict(status="feasible",threshold=segmentation["threshold"],dev=segmentation)}
    for budget in (.5,1,2,4,8):
        eligible = [p for p in valid if p["fp_per_image"] <= budget]
        best = max(eligible,key=lambda p:(p["component_recall"],-p["fp_per_image"],p["positive_macro_iou"],p["threshold"])) if eligible else None
        operating[f"cad_fp{budget:g}"] = dict(status="feasible" if best else "infeasible",threshold=best["threshold"] if best else None,dev=best,budget_fp_per_image=budget)
    fingerprints = [{"case_id":r["case_id"],"image_sha256":sha(relative(root,r["image"])),
                     "mask_sha256":sha(relative(root,r["mask"])),"prediction_sha256":sha(Path(pred_dir)/(r["case_id"]+".npy"))} for r in dev]
    return dict(schema_version=1,protocol="mcbench-v0.1",manifest_sha256=sha(manifest_path),model_sha256=meta["model_sha256"],
                pipeline_id=meta["pipeline_id"],calibration_split="dev",threshold_grid=[p["threshold"] for p in points],
                operating_points=operating,dev_curve=points,dev_fingerprints=fingerprints,
                note="Engineering operating points, not validated clinical burden limits. Pixel-threshold sweep can change topology non-monotonically; not a conventional fixed-candidate FROC.")


def evaluate(rows,root,pred_dir,role,lock,path,op,replicates):
    if role == "train":
        raise ValueError("Use dev/test for evaluation")
    meta = run_metadata(pred_dir,role)
    if lock["manifest_sha256"] != sha(path) or any(lock[k] != meta[k] for k in ("model_sha256","pipeline_id")):
        raise ValueError("Manifest, model or pipeline differs from locked calibration")
    point = lock["operating_points"][op]
    if point["status"] != "feasible":
        raise ValueError("No dev-feasible threshold for this budget; do not invent a test threshold")
    measured = score(rows,root,pred_dir,role,point["threshold"])
    primary = [r for r in measured if r["label_status"] == "dense_verified"]
    return dict(protocol="mcbench-v0.1",split=role,operating_point=op,threshold=point["threshold"],
                model_sha256=meta["model_sha256"],pipeline_id=meta["pipeline_id"],manifest_sha256=sha(path),
                primary=aggregate(primary),ci=bootstrap(primary,replicates) if primary and replicates else None,
                contour_reference_supplement=aggregate([r for r in measured if r["label_status"] == "contour_positive"]),
                assumed_empty_supplement=aggregate([r for r in measured if r["label_status"] == "assumed_empty"]),
                unknown_references_excluded=sum(r["split"]==role and r["label_status"]=="unknown" for r in rows),
                input_fingerprints=[dict(case_id=r["case_id"],mask_sha256=sha(relative(root,r["mask"])),
                                         prediction_sha256=sha(Path(pred_dir)/(r["case_id"]+".npy"))) for r in rows if r["split"]==role and r["label_status"]!="unknown"],
                per_image=measured,environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))


def demo(out):
    import csv
    import hashlib
    from scipy import ndimage
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Demo output must be new or empty")
    (out/"images").mkdir(parents=True)
    (out/"masks").mkdir()
    rng = np.random.default_rng(42)
    rows = []
    # Procedural, nonmedical bright-speck arrays. Test label access here only creates a fixture.
    for i in range(12):
        cid = f"synthetic_{i:02}"
        role = "train" if i<6 else "dev" if i<9 else "test"
        gt = np.zeros((96,112),np.uint8)
        if i%3 != 2:
            gt[20:22,25:27] = 1
            gt[28:31,31:34] = 1
            gt[65:69,80:84] = 1
        im = rng.uniform(.05,.25,gt.shape).astype(np.float32)
        im += gt*.6
        im[47:49,52:54] = .72  # bright non-target control
        np.save(out/"images"/(cid+".npy"),im)
        np.save(out/"masks"/(cid+".npy"),gt)
        rows.append(dict(case_id=cid,group_id=f"procedural_group_{i//1:02}",split=role,
                         image="images/"+cid+".npy",mask="masks/"+cid+".npy",label_status="dense_verified",
                         image_sha256=sha(out/"images"/(cid+".npy")),mask_sha256=sha(out/"masks"/(cid+".npy"))))
        if role != "train":
            pd = out/("pred_"+role)
            pd.mkdir(exist_ok=True)
            p = ndimage.gaussian_filter(im,.5).clip(0,1).astype(np.float32)
            np.save(pd/(cid+".npy"),p)
    with (out/"manifest.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model_sha = hashlib.sha256(b"synthetic image-only blur predictor v1").hexdigest()
    for role in ("dev","test"):
        write_json(out/("pred_"+role)/"run.json",dict(model_sha256=model_sha,pipeline_id="synthetic_blur_image_only",split=role))
    lock = calibration(rows,out,out/"pred_dev",out/"manifest.csv")
    write_json(out/"threshold_lock.json",lock)
    result = evaluate(rows,out,out/"pred_test","test",lock,out/"manifest.csv","segmentation",100)
    write_json(out/"test_results.json",result)
    print("PASS synthetic demo: manifest -> dev threshold lock -> test metrics+group CI")


def main():
    ap = argparse.ArgumentParser(description="MC segmentation benchmark starter; no data download or clinical use")
    sub = ap.add_subparsers(dest="command",required=True)
    p = sub.add_parser("demo"); p.add_argument("--out",required=True,type=Path)
    p = sub.add_parser("compare")
    p.add_argument("--a",required=True,type=Path); p.add_argument("--b",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path); p.add_argument("--bootstrap",type=int,default=2000)
    for command in ("validate","lock","evaluate","train","predict"):
        p = sub.add_parser(command)
        p.add_argument("--manifest",required=True,type=Path)
        p.add_argument("--data-root",required=True,type=Path)
        if command in ("lock","evaluate"):
            p.add_argument("--pred",required=True,type=Path)
        if command != "validate":
            p.add_argument("--out",required=True,type=Path)
        if command in ("evaluate","predict"):
            p.add_argument("--split",choices=("dev","test"),default="test")
        if command == "evaluate":
            p.add_argument("--lock",required=True,type=Path)
            p.add_argument("--op",default="segmentation")
            p.add_argument("--bootstrap",type=int,default=2000)
        if command in ("train","predict"):
            p.add_argument("--device",default="cpu")
        if command == "predict":
            p.add_argument("--checkpoint",required=True,type=Path)
        if command == "train":
            for key,default in (("seed",42),("epochs",50),("steps",128),("batch",4),("patch",256),("width",16)):
                p.add_argument("--"+key,type=int,default=default)
            p.add_argument("--lr",type=float,default=.001)
    args = ap.parse_args()
    if args.command == "demo":
        demo(args.out)
        return
    if args.command == "compare":
        if args.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")
        write_json(args.out,paired_comparison(read_json(args.a),read_json(args.b),args.bootstrap))
        print("Wrote paired group-bootstrap differences B minus A")
        return
    rows = manifest(args.manifest,args.data_root,check_files=True)
    if args.command == "validate":
        print("PASS manifest IDs/groups/roles/paths/checksums; annotation quality/linkage still require audit")
    elif args.command == "lock":
        if args.out.exists():
            raise ValueError("Refusing to overwrite an existing threshold lock")
        write_json(args.out,calibration(rows,args.data_root,args.pred,args.manifest))
        print("Locked dev-only segmentation and CAD budget thresholds")
    elif args.command == "evaluate":
        if args.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")
        write_json(args.out,evaluate(rows,args.data_root,args.pred,args.split,read_json(args.lock),args.manifest,args.op,args.bootstrap))
        print("Wrote locked evaluation; no threshold selection on evaluated split")
    elif args.command == "train":
        from .baseline import train
        train(rows,args.data_root,args.out,args.seed,args.epochs,args.steps,args.batch,args.patch,args.width,args.lr,args.device)
        info = read_json(args.out/"training.json")
        info["manifest_sha256"] = sha(args.manifest)
        info["training_inputs"] = [dict(case_id=r["case_id"],image_sha256=sha(relative(args.data_root,r["image"])),mask_sha256=sha(relative(args.data_root,r["mask"]))) for r in rows if r["split"] in ("train","dev")]
        write_json(args.out/"training.json",info)
    else:
        from .baseline import predict
        predict(rows,args.data_root,args.checkpoint,args.out,args.split,args.device)


if __name__ == "__main__":
    main()
