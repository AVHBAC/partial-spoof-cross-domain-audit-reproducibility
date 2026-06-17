from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_E1 = DATA_DIR / "raw_e1_baseline"
RAW_E5 = DATA_DIR / "raw_e5_cross_dataset"
OUT_CSV = DATA_DIR / "recalibration_decomposition.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
FAR_TARGET = 0.01
_E5_KEY = {"LlamaPS": "llamapartialspoof", "HQ-MPSD": "hqmpsd"}

def load(ds: str, det: str):
    if ds == "PartialSpoof":
        base = RAW_E1 / f"{det}_utt"
    else:
        base = RAW_E5 / f"{det}_{_E5_KEY[ds]}_utt"
    return np.load(f"{base}_scores.npy"), np.load(f"{base}_labels.npy")

def compute_eer_threshold(scores, labels) -> float:
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(thr[i])

def metrics_at(scores, labels, thresh):
    pred = (scores >= thresh).astype(int)
    ng, ns = int((labels == 0).sum()), int((labels == 1).sum())
    bpcer = ((pred == 1) & (labels == 0)).sum() / ng if ng else float("nan")
    apcer = ((pred == 0) & (labels == 1)).sum() / ns if ns else float("nan")
    return 100 * bpcer, 100 * apcer, 100 * (bpcer + apcer) / 2

def run():
    e1 = json.load(open(RAW_E1 / "results.json"))
    eer_thr_indom = {d: float(e1[d]["utt_eer_threshold"]) for d in DETECTORS}
    far_thr_indom = {}
    for d in DETECTORS:
        s, l = load("PartialSpoof", d)
        far_thr_indom[d] = float(np.percentile(s[l == 0], 100 * (1 - FAR_TARGET)))

    rows = []

    for ds in ["PartialSpoof", "LlamaPS", "HQ-MPSD"]:
        for d in DETECTORS:
            s, l = load(ds, d)
            t_fixed = far_thr_indom[d]
            t_recal = float(np.percentile(s[l == 0], 100 * (1 - FAR_TARGET)))
            bf, af, cf = metrics_at(s, l, t_fixed)
            br, ar, cr = metrics_at(s, l, t_recal)
            rows.append({"level": "detector", "operating_point": "FAR1", "dataset": ds,
                         "unit": d.upper(),
                         "fixed_BPCER": f"{bf:.2f}", "fixed_APCER": f"{af:.2f}", "fixed_ACER": f"{cf:.2f}",
                         "recal_BPCER": f"{br:.2f}", "recal_APCER": f"{ar:.2f}", "recal_ACER": f"{cr:.2f}"})

    for ds in ["PartialSpoof", "LlamaPS", "HQ-MPSD"]:
        arrs = {d: load(ds, d) for d in DETECTORS}
        n = min(len(arrs[d][0]) for d in DETECTORS)
        labels = arrs["bam"][1][:n]

        def ensemble_metrics(thr_map):
            votes = np.zeros(n, dtype=int)
            for d in DETECTORS:
                votes += (arrs[d][0][:n] >= thr_map[d]).astype(int)
            pred = (votes >= 2).astype(int)
            ng, ns = int((labels == 0).sum()), int((labels == 1).sum())
            bp = ((pred == 1) & (labels == 0)).sum() / ng if ng else float("nan")
            ap = ((pred == 0) & (labels == 1)).sum() / ns if ns else float("nan")
            return 100 * bp, 100 * ap, 100 * (bp + ap) / 2

        eer_thr_target = {d: compute_eer_threshold(arrs[d][0][:n], labels) for d in DETECTORS}
        bf, af, cf = ensemble_metrics(eer_thr_indom)
        br, ar, cr = ensemble_metrics(eer_thr_target)
        rows.append({"level": "ensemble", "operating_point": "EER", "dataset": ds,
                     "unit": "MAJORITY",
                     "fixed_BPCER": f"{bf:.2f}", "fixed_APCER": f"{af:.2f}", "fixed_ACER": f"{cf:.2f}",
                     "recal_BPCER": f"{br:.2f}", "recal_APCER": f"{ar:.2f}", "recal_ACER": f"{cr:.2f}"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'level':9s}{'ds':13s}{'unit':10s}| fixed B/A/ACER        | recal B/A/ACER")
    for r in rows:
        print(f"{r['level']:9s}{r['dataset']:13s}{r['unit']:10s}| "
              f"{r['fixed_BPCER']:>6s}/{r['fixed_APCER']:>6s}/{r['fixed_ACER']:>6s} | "
              f"{r['recal_BPCER']:>6s}/{r['recal_APCER']:>6s}/{r['recal_ACER']:>6s}")
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

    def cell(level, ds, unit, key):
        return float(next(r[key] for r in rows if r["level"] == level and r["dataset"] == ds and r["unit"] == unit))
    assert abs(cell("detector", "LlamaPS", "MRM", "fixed_BPCER") - 88.95) < 0.1, "MRM FAR1 BPCER"
    assert abs(cell("detector", "LlamaPS", "BAM", "fixed_BPCER") - 28.71) < 0.1, "BAM FAR1 BPCER"
    assert abs(cell("ensemble", "LlamaPS", "MAJORITY", "fixed_BPCER") - 68.7) < 0.2, "ensemble BPCER"
    assert abs(cell("ensemble", "LlamaPS", "MAJORITY", "fixed_ACER") - 35.6) < 0.2, "ensemble ACER"
    assert abs(cell("ensemble", "PartialSpoof", "MAJORITY", "fixed_ACER") - 0.6) < 0.1, "in-domain ensemble ACER"
    print("Reproduction guards passed: fixed-threshold numbers match published values.")

if __name__ == "__main__":
    run()
