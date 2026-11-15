import csv
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from mcbench.io import manifest, probability, write_json, reference
from mcbench.metrics import measure, aggregate, bootstrap, match, count_matches, paired_comparison
from mcbench.__main__ import demo, evaluate


class ProtocolTests(unittest.TestCase):
    def test_merge_is_not_two_matches(self):
        gt = np.zeros((8,12),bool)
        gt[3,2]=gt[3,8]=True
        pred = np.zeros_like(gt); pred[3,2:9]=True
        m=measure(gt,pred)
        self.assertEqual((m["covered_gt"],m["matched"],m["gt_covered_unmatched"],m["fp_components"]),(2,1,1,0))
        self.assertEqual(m["strict_matched"]["pair_iou_0.5"],0)

    def test_duplicate_predictions_overlap_one_gt(self):
        gt=np.zeros((9,9),bool); gt[2:7,2:7]=True
        pred=np.zeros_like(gt); pred[2,2]=pred[6,6]=True
        m=measure(gt,pred)
        self.assertEqual((m["matched"],m["fp_components"],m["pred_overlap_unmatched"]),(1,1,1))
        self.assertEqual(m["strict_matched"]["gt_coverage_0.1"],0)

    def test_empty_and_total_miss(self):
        gt=np.zeros((4,4),bool)
        empty=measure(gt,gt)
        self.assertIsNone(empty["component_precision"])
        self.assertIsNone(empty["component_recall"])
        self.assertEqual(empty["clean_negative"],1)
        pred=gt.copy(); pred[0,0]=1
        fp=measure(gt,pred)
        self.assertEqual(fp["component_precision"],0)
        self.assertIsNone(fp["dice_positive"])
        miss=measure(pred,gt)
        self.assertEqual(miss["dice_positive"],0)  # Never coerce zero to1.
        stats=aggregate([empty,miss])
        self.assertEqual(stats["positive_macro_dice"],0)
        self.assertEqual(stats["all_macro_dice_empty_one"],.5)

    def test_diagonal_8_connectivity(self):
        x=np.eye(4,dtype=bool)
        self.assertEqual(measure(x,x)["gt"],1)

    def test_group_matching_bounds_against_enumeration(self):
        rng=np.random.default_rng(3)
        for _ in range(60):
            edges=rng.random((4,4))<.45
            graph=csr_matrix(edges)
            k=count_matches(graph)
            subgroup=np.array([1,0,1,0],bool)
            achieved=[]
            for a in product(range(-1,4),repeat=4):
                selected=[v for v in a if v>=0]
                if len(selected)!=k or len(set(selected))!=k:
                    continue
                if all(v<0 or edges[i,v] for i,v in enumerate(a)):
                    achieved.append(sum(v>=0 and subgroup[i] for i,v in enumerate(a)))
            self.assertEqual(min(achieved),k-count_matches(graph[~subgroup]))
            self.assertEqual(max(achieved),count_matches(graph[subgroup]))

    def test_patient_leakage_and_renamed_copy(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"manifest.csv"
            rows=[dict(case_id="a",group_id="p",split="train",image="a.npy",mask="a.npy",label_status="dense_verified",image_sha256="a"*64,mask_sha256="a"*64),
                  dict(case_id="b",group_id="p",split="dev",image="b.npy",mask="b.npy",label_status="dense_verified",image_sha256="b"*64,mask_sha256="b"*64)]
            def write():
                with path.open("w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
            write()
            with self.assertRaisesRegex(ValueError,"Group leakage"):
                manifest(path)
            rows[1]["group_id"]="other"; rows[1]["image_sha256"]="a"*64; write()
            with self.assertRaisesRegex(ValueError,"Identical image hash"):
                manifest(path)

    def test_probability_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"case.npy"
            for a in (np.zeros((3,3),np.uint8), np.full((3,3),np.nan), np.full((3,3),1.01), np.zeros((2,3),float)):
                np.save(path,a)
                with self.assertRaises(ValueError):
                    probability(d,"case",(3,3))

    def test_unknown_or_mixed_encoding_reference_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            row=dict(mask="mask.npy",label_status="unknown")
            with self.assertRaisesRegex(ValueError,"Unknown"):
                reference(row,d)
            row["label_status"]="dense_verified"
            np.save(Path(d)/"mask.npy",np.array([[0,1,255]],np.uint8))
            with self.assertRaisesRegex(ValueError,"binary mask"):
                reference(row,d)

    def test_lock_and_model_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"demo"
            demo(root)
            import json
            lock=json.loads((root/"threshold_lock.json").read_text())
            rows=manifest(root/"manifest.csv",root,True)
            results=evaluate(rows,root,root/"pred_test","test",lock,root/"manifest.csv","segmentation",10)
            self.assertEqual(results["primary"]["images"],3)
            write_json(root/"pred_test"/"run.json",dict(model_sha256="a"*64,pipeline_id=lock["pipeline_id"],split="test"))
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluate(rows,root,root/"pred_test","test",lock,root/"manifest.csv","segmentation",0)
            lock["operating_points"]["cad_fp0.5"]["status"]="infeasible"
            write_json(root/"pred_test"/"run.json",dict(model_sha256=lock["model_sha256"],pipeline_id=lock["pipeline_id"],split="test"))
            with self.assertRaisesRegex(ValueError,"dev-feasible"):
                evaluate(rows,root,root/"pred_test","test",lock,root/"manifest.csv","cad_fp0.5",0)

    def test_group_bootstrap_does_not_count_views_as_people(self):
        x=np.eye(3,dtype=bool)
        a=dict(group_id="patient1",**measure(x,x))
        b=dict(group_id="patient2",**measure(x,np.zeros_like(x)))
        ci=bootstrap([a,a,b],50)
        self.assertEqual(ci["groups"],2)
        self.assertEqual(ci["intervals"]["positive_macro_iou"]["valid_replicates"],50)

    def test_paired_comparison_and_mismatched_cohort(self):
        x=np.eye(3,dtype=bool)
        rows_a=[dict(case_id=f"case{i}",group_id=f"p{i}",label_status="dense_verified",**measure(x,np.zeros_like(x))) for i in range(3)]
        rows_b=[dict(case_id=f"case{i}",group_id=f"p{i}",label_status="dense_verified",**measure(x,x)) for i in range(3)]
        a=dict(manifest_sha256="x",split="test",per_image=rows_a)
        b=dict(manifest_sha256="x",split="test",per_image=rows_b)
        result=paired_comparison(a,b,20)
        self.assertEqual(result["difference"]["positive_macro_iou"],1.)
        self.assertEqual(result["ci"]["component_recall"]["low"],1.)
        b["per_image"]=rows_b[:-1]
        with self.assertRaises(ValueError):
            paired_comparison(a,b,20)


if __name__ == "__main__":
    unittest.main()
