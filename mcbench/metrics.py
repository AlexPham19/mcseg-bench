"""Native-pixel foreground metrics and one-to-one maximum-cardinality matching."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

STRUCTURE = np.ones((3, 3), dtype=np.uint8)
GROUPS = ("tiny_isolated", "tiny_cluster", "regular_isolated", "regular_cluster")


def div(a, b):
    return float(a / b) if b else None


def match(graph):
    if not graph.shape[0] or not graph.shape[1]:
        return np.full(graph.shape[0], -1, dtype=int)
    return maximum_bipartite_matching(graph.tocsr(), perm_type="column")


def count_matches(graph):
    return int((match(graph) >= 0).sum())


def measure(gt, pred, tiny_area=11, isolated_distance=128):
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError("GT and prediction must have identical 2D shape")
    gl, ng = ndimage.label(gt, STRUCTURE)
    pl, npred = ndimage.label(pred, STRUCTURE)
    ga = np.bincount(gl.ravel(), minlength=ng + 1)[1:]
    pa = np.bincount(pl.ravel(), minlength=npred + 1)[1:]
    both = gt & pred
    tp = int(both.sum())
    fp, fn = int((pred & ~gt).sum()), int((gt & ~pred).sum())
    if tp:
        pairs, intersections = np.unique(np.column_stack((gl[both] - 1, pl[both] - 1)), axis=0, return_counts=True)
    else:
        pairs, intersections = np.empty((0, 2), int), np.array([], int)
    def graph(select):
        e = pairs[select]
        return coo_matrix((np.ones(len(e), dtype=np.uint8), (e[:, 0], e[:, 1])), shape=(ng, npred)).tocsr()
    g = graph(np.ones(len(pairs), bool))
    assignment = match(g)
    k = int((assignment >= 0).sum())
    covered = len(np.unique(pairs[:, 0]))
    ptouched = len(np.unique(pairs[:, 1]))
    strict = {}
    for cut in (0.1, 0.5):
        if len(pairs):
            coverage = intersections / ga[pairs[:, 0]]
            iou = intersections / (ga[pairs[:, 0]] + pa[pairs[:, 1]] - intersections)
        else:
            coverage = iou = np.array([])
        strict[f"gt_coverage_{cut}"] = count_matches(graph(coverage >= cut))
        strict[f"pair_iou_{cut}"] = count_matches(graph(iou >= cut))
    strata = {}
    if ng:
        centers = np.asarray(ndimage.center_of_mass(gt, gl, np.arange(1, ng + 1)))
        nearest = cKDTree(centers).query(centers, k=2)[0][:, 1] if ng > 1 else np.array([np.inf])
        tiny, isolated = ga < tiny_area, nearest > isolated_distance
        for name, mask in zip(GROUPS, (tiny & isolated, tiny & ~isolated, ~tiny & isolated, ~tiny & ~isolated)):
            # A transversal matroid basis has min S = rank(all)-rank(complement S), max S = rank(S).
            # These bounds avoid treating an arbitrary maximum-matching identity as certain.
            strata[name] = dict(gt=int(mask.sum()), matched=int(((assignment >= 0) & mask).sum()),
                                matched_min=k - count_matches(g[~mask]), matched_max=count_matches(g[mask]))
    else:
        strata = {name: dict(gt=0, matched=0, matched_min=0, matched_max=0) for name in GROUPS}
    return dict(gt=ng, predicted=npred, matched=k, fn_components=ng-k, fp_components=npred-k,
                gt_no_overlap=ng-covered, gt_covered_unmatched=covered-k,
                pred_no_overlap=npred-ptouched, pred_overlap_unmatched=ptouched-k,
                covered_gt=covered, pixel_tp=tp, pixel_fp=fp, pixel_fn=fn,
                component_precision=div(k, npred), component_recall=div(k, ng),
                pixel_precision=div(tp, tp+fp), pixel_recall=div(tp, tp+fn),
                dice_positive=div(2*tp, 2*tp+fp+fn) if ng else None,
                iou_positive=div(tp, tp+fp+fn) if ng else None,
                dice_all_empty_one=div(2*tp, 2*tp+fp+fn) if tp+fp+fn else 1.0,
                clean_negative=int(ng == 0 and npred == 0), strict_matched=strict, groups=strata)


def aggregate(rows):
    if not rows:
        return None
    sums = {key: sum(r[key] for r in rows) for key in (
        "gt", "predicted", "matched", "fn_components", "fp_components", "gt_no_overlap", "gt_covered_unmatched",
        "pred_no_overlap", "pred_overlap_unmatched", "covered_gt", "pixel_tp", "pixel_fp", "pixel_fn", "clean_negative")}
    positives, negatives = [r for r in rows if r["gt"]], [r for r in rows if not r["gt"]]
    tp, fp, fn, k = (sums[x] for x in ("pixel_tp", "pixel_fp", "pixel_fn", "matched"))
    mean = lambda values: float(np.mean(values)) if values else None
    return dict(**sums, images=len(rows), positive_images=len(positives), negative_images=len(negatives),
                positive_macro_dice=mean([r["dice_positive"] for r in positives]),
                positive_macro_iou=mean([r["iou_positive"] for r in positives]),
                all_macro_dice_empty_one=mean([r["dice_all_empty_one"] for r in rows]),
                pooled_dice=div(2*tp, 2*tp+fp+fn), pooled_iou=div(tp, tp+fp+fn),
                component_precision=div(k, sums["predicted"]), component_recall=div(k, sums["gt"]),
                pixel_precision=div(tp, tp+fp), pixel_recall=div(tp, tp+fn),
                fp_per_image=div(sums["fp_components"], len(rows)),
                fp_per_negative=div(sum(r["fp_components"] for r in negatives), len(negatives)),
                fp_per_positive=div(sum(r["fp_components"] for r in positives), len(positives)),
                clean_negative_rate=div(sums["clean_negative"], len(negatives)),
                strict_matched={key: sum(r["strict_matched"][key] for r in rows) for key in rows[0]["strict_matched"]},
                groups={name: {key: sum(r["groups"][name][key] for r in rows) for key in ("gt", "matched", "matched_min", "matched_max")} for name in GROUPS})


def bootstrap(rows, n=2000, seed=42):
    """Resample entire supplied patient groups. Not a fix for unknown/incorrect linkage."""
    groups = sorted({r["group_id"] for r in rows})
    bygroup = {g: [r for r in rows if r["group_id"] == g] for g in groups}
    keys = ("positive_macro_iou", "positive_macro_dice", "component_recall", "component_precision", "fp_per_image", "clean_negative_rate")
    values = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    if len(groups) < 2:
        return {"status": "insufficient_groups", "groups": len(groups)}
    for _ in range(n):
        sampled = [r for g in rng.choice(groups, len(groups), replace=True) for r in bygroup[g]]
        stats = aggregate(sampled)
        for key in keys:
            if stats[key] is not None:
                values[key].append(stats[key])
    return dict(method="percentile group bootstrap", groups=len(groups), replicates=n, seed=seed,
                intervals={key: dict(low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975)), valid_replicates=len(v)) if v else None for key, v in values.items()})


def paired_comparison(a, b, n=2000, seed=42):
    """Paired group bootstrap, differences B minus A, one run/seed pair only."""
    if a["manifest_sha256"] != b["manifest_sha256"] or a["split"] != b["split"]:
        raise ValueError("Paired comparison requires the same manifest and split")
    ra = {r["case_id"]:r for r in a["per_image"] if r["label_status"] == "dense_verified"}
    rb = {r["case_id"]:r for r in b["per_image"] if r["label_status"] == "dense_verified"}
    if not ra or ra.keys() != rb.keys() or any(ra[k]["group_id"] != rb[k]["group_id"] for k in ra):
        raise ValueError("Paired primary cases/groups must agree exactly")
    groups = sorted({r["group_id"] for r in ra.values()})
    bygroup = {g:sorted(k for k in ra if ra[k]["group_id"]==g) for g in groups}
    keys = ("positive_macro_iou","positive_macro_dice","component_recall","component_precision","fp_per_image")
    def differences(ids):
        sa,sb = aggregate([ra[k] for k in ids]),aggregate([rb[k] for k in ids])
        return {k:sb[k]-sa[k] if sa[k] is not None and sb[k] is not None else None for k in keys}
    point = differences(sorted(ra))
    samples = {k:[] for k in keys}
    rng = np.random.default_rng(seed)
    if len(groups)>1:
        for _ in range(n):
            delta = differences([cid for g in rng.choice(groups,len(groups),replace=True) for cid in bygroup[g]])
            for k in keys:
                if delta[k] is not None:
                    samples[k].append(delta[k])
    return dict(direction="B minus A",groups=len(groups),images=len(ra),seed=seed,replicates=n,
                difference=point,ci={k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),valid_replicates=len(v)) if v else None for k,v in samples.items()},
                note="Paired supplied-group percentile bootstrap for one run pair; no multiplicity correction or multi-seed hierarchy.")
