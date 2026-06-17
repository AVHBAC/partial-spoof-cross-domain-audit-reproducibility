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
OUT_CSV = DATA_DIR / "fusion_rules.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
_E5_KEY = {"LlamaPS": "llamapartialspoof", "HQ-MPSD": "hqmpsd"}

def load(ds, det):
    base = RAW_E1 / f"{det}_utt" if ds == "PartialSpoof" else RAW_E5 / f"{det}_{_E5_KEY[ds]}_utt"
    return np.load(f"{base}_scores.npy"), np.load(f"{base}_labels.npy")

def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        return 100 * float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
    except ValueError:
        i = int(np.argmin(np.abs(fpr - fnr)))
        return 100 * float((fpr[i] + fnr[i]) / 2.0)

def rates(pred, labels):
    ng, ns = int((labels == 0).sum()), int((labels == 1).sum())
    bp = ((pred == 1) & (labels == 0)).sum() / ng if ng else float("nan")
    ap = ((pred == 0) & (labels == 1)).sum() / ns if ns else float("nan")
    return 100 * bp, 100 * ap, 100 * (bp + ap) / 2

def run():
    e1 = json.load(open(RAW_E1 / "results.json"))
    thr = {d: float(e1[d]["utt_eer_threshold"]) for d in DETECTORS}
    zstat = {}
    for d in DETECTORS:
        s, l = load("PartialSpoof", d)
        g = s[l == 0]
        zstat[d] = (float(g.mean()), float(g.std()) + 1e-9)

    rows = []
    for ds in ["PartialSpoof", "LlamaPS", "HQ-MPSD"]:
        arrs = {d: load(ds, d) for d in DETECTORS}
        n = min(len(arrs[d][0]) for d in DETECTORS)
        lab = arrs["bam"][1][:n]
        S = {d: arrs[d][0][:n] for d in DETECTORS}

        best_single = min(eer(S[d], lab) for d in DETECTORS)
        z = np.mean([(S[d] - zstat[d][0]) / zstat[d][1] for d in DETECTORS], axis=0)
        soft_eer = eer(z, lab)

        binp = {d: (S[d] >= thr[d]).astype(int) for d in DETECTORS}
        votes = sum(binp[d] for d in DETECTORS)
        maj_b, maj_a, maj_c = rates((votes >= 2).astype(int), lab)
        or_b, or_a, or_c = rates((votes >= 1).astype(int), lab)
        and_b, and_a, and_c = rates((votes >= 3).astype(int), lab)

        rows.append({
            "dataset": ds, "best_single_EER": f"{best_single:.2f}",
            "softvote_EER": f"{soft_eer:.2f}",
            "majority_BPCER": f"{maj_b:.1f}", "majority_APCER": f"{maj_a:.1f}", "majority_ACER": f"{maj_c:.1f}",
            "OR_BPCER": f"{or_b:.1f}", "OR_APCER": f"{or_a:.1f}", "OR_ACER": f"{or_c:.1f}",
            "AND_BPCER": f"{and_b:.1f}", "AND_APCER": f"{and_a:.1f}", "AND_ACER": f"{and_c:.1f}",
        })
        print(f"{ds:13s} best-single EER={best_single:5.2f}  soft-vote EER={soft_eer:5.2f}  "
              f"MAJ ACER={maj_c:4.1f}  OR ACER={or_c:4.1f}  AND ACER={and_c:4.1f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

    llama = next(r for r in rows if r["dataset"] == "LlamaPS")
    indom = next(r for r in rows if r["dataset"] == "PartialSpoof")
    assert abs(float(llama["majority_BPCER"]) - 68.7) < 0.2, llama
    assert abs(float(llama["majority_ACER"]) - 35.6) < 0.2, llama
    assert abs(float(indom["majority_ACER"]) - 0.6) < 0.1, indom
    print("Reproduction guard passed: majority-vote numbers match published values.")

if __name__ == "__main__":
    run()
