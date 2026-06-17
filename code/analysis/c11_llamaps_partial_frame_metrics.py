from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CATEGORY_CSV = DATA_DIR / "llamaps_utt_categories.csv"
OUT_CSV = DATA_DIR / "llamaps_partial_frame_metrics.csv"

XPS = REPO_ROOT.parent / "xps_forensic"
RAW = DATA_DIR / "raw_e5_cross_dataset"
IOU_OP_DIR = DATA_DIR

DETECTORS = ["bam", "cfprf", "mrm"]
NATIVE_MS = {"bam": 160.0, "cfprf": 20.0, "mrm": 20.0}
LABEL_MS = 10.0
FRAME_THR = {"bam": 0.9999083280563354, "cfprf": 0.3542732000350952, "mrm": 0.5038838982582092}

GUARD_SEG = {"bam": 32.07, "cfprf": 46.77, "mrm": 48.54}
GUARD_OP = {"bam": 0.47, "cfprf": 0.37, "mrm": 0.51}
GUARD_LOC = {"bam": 0.44, "cfprf": 0.60, "mrm": 0.61}

sys.path.insert(0, str(REPO_ROOT / "lib"))
from xps_forensic.utils.metrics import (
    compute_eer, compute_tIoU,
    _pool_scores_to_windows, _pool_labels_to_windows,
    upsample_binary_predictions_to_label_grid,
)

def load_ternary():
    return {r["utterance_id"]: int(r["ternary_label"])
            for r in csv.DictReader(open(CATEGORY_CSV))}

def seg_eer(fs_all, fl_all, mask, res, shift):
    ps, pl = [], []
    for i in range(len(fs_all)):
        if not mask[i]:
            continue
        a = np.asarray(fs_all[i]).ravel()
        b = np.asarray(fl_all[i]).ravel()
        if a.size == 0 or b.size == 0:
            continue
        sw = _pool_scores_to_windows(a, shift, res, agg="min")
        lw = _pool_labels_to_windows(b, LABEL_MS, res, rule="any")
        m = min(len(sw), len(lw))
        ps.extend(sw[:m].tolist()); pl.extend(lw[:m].tolist())
    if len(set(pl)) < 2:
        return float("nan")
    return 100 * compute_eer(np.array(ps), np.array(pl))[0]

def loc_iou(fs_all, fl_all, spoof_idx, keep, thr, shift):
    vals = []
    for idx in spoof_idx:
        pred = (fs_all[idx].astype(np.float32) >= thr).astype(int)
        up = upsample_binary_predictions_to_label_grid(
            pred, pred_frame_shift_ms=float(shift), label_frame_shift_ms=LABEL_MS)
        fl = fl_all[idx].astype(np.int32)
        m = min(len(up), len(fl))
        vals.append(compute_tIoU(up[:m], fl[:m]))
    vals = np.array(vals)
    return vals.mean(), vals[keep].mean()

def run():
    tmap = load_ternary()
    rows = []
    for d in DETECTORS:
        ids = np.load(RAW / f"{d}_llamapartialspoof_utt_ids.npy", allow_pickle=True)
        ul = np.load(RAW / f"{d}_llamapartialspoof_utt_labels.npy")
        tern = np.array([tmap[str(i)] for i in ids])
        keepP = (tern == 0) | (tern == 1)

        fs = np.load(RAW / f"{d}_llamapartialspoof_frame_scores.npy", allow_pickle=True)
        fl = np.load(RAW / f"{d}_llamapartialspoof_frame_labels.npy", allow_pickle=True)
        res = NATIVE_MS[d]
        seg_full = seg_eer(fs, fl, np.ones(len(fs), bool), res, res)
        seg_part = seg_eer(fs, fl, keepP, res, res)

        spoof_idx = np.where(ul == 1)[0]
        keep_sp = tern[spoof_idx] == 1
        loc_full, loc_part = loc_iou(fs, fl, spoof_idx, keep_sp, FRAME_THR[d], res)
        del fs, fl; gc.collect()

        iou_op = np.load(IOU_OP_DIR / f"iou_dist_{d}_llamapartialspoof.npy", allow_pickle=True)
        op_full, op_part = float(iou_op.mean()), float(iou_op[keepP].mean())

        print(f"{d.upper():6s} Seg-EER {seg_full:6.2f}->{seg_part:6.2f}  "
              f"IoU-op {op_full:.2f}->{op_part:.2f}  IoU-loc {loc_full:.2f}->{loc_part:.2f}")
        for metric, full, part in [("seg_eer", seg_full, seg_part),
                                   ("iou_operational", op_full, op_part),
                                   ("iou_loc_conditional", loc_full, loc_part)]:
            rows.append({"metric": metric, "detector": d.upper(),
                         "full": f"{full:.2f}", "partial_only": f"{part:.2f}"})

        assert abs(seg_full - GUARD_SEG[d]) < 0.05, (d, seg_full)
        assert abs(op_full - GUARD_OP[d]) < 0.01, (d, op_full)
        assert abs(loc_full - GUARD_LOC[d]) < 0.01, (d, loc_full)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "detector", "full", "partial_only"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {OUT_CSV} ({len(rows)} rows)")
    print("Reproduction guards passed: full-set Seg-EER / IoU match published values.")

if __name__ == "__main__":
    run()
