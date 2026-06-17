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
OUT_CSV = DATA_DIR / "confidence_intervals.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
_E5_KEY = {"LlamaPS": "llamapartialspoof", "HQ-MPSD": "hqmpsd"}
Z = 1.96

def load(ds, det):
    base = RAW_E1 / f"{det}_utt" if ds == "PartialSpoof" else RAW_E5 / f"{det}_{_E5_KEY[ds]}_utt"
    return np.load(f"{base}_scores.npy"), np.load(f"{base}_labels.npy")

def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        return float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
    except ValueError:
        i = int(np.argmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2.0)

def normal_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return p - Z * se, p + Z * se

def wilson_ci(p, n):
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return centre - half, centre + half

def emit(rows, scope, dataset, metric, unit, p, n, method):
    lo, hi = (normal_ci if method == "normal" else wilson_ci)(p, n)
    rows.append({
        "scope": scope, "dataset": dataset, "metric": metric, "unit": unit,
        "point_pct": f"{100*p:.2f}", "ci_low_pct": f"{100*lo:.2f}",
        "ci_high_pct": f"{100*hi:.2f}", "half_width_pp": f"{100*(hi-lo)/2:.2f}",
        "n": n, "method": method,
    })

def run():
    e1 = json.load(open(RAW_E1 / "results.json"))
    thr = {d: float(e1[d]["utt_eer_threshold"]) for d in DETECTORS}
    rows = []

    for ds in ["PartialSpoof", "LlamaPS", "HQ-MPSD"]:
        for d in DETECTORS:
            s, l = load(ds, d)
            ng, ns = int((l == 0).sum()), int((l == 1).sum())
            e = eer(s, l)
            emit(rows, "detector", ds, "Utt-EER", d.upper(), e, min(ng, ns), "normal")
        arrs = {d: load(ds, d) for d in DETECTORS}
        n = min(len(arrs[d][0]) for d in DETECTORS)
        lab = arrs["bam"][1][:n]
        votes = sum((arrs[d][0][:n] >= thr[d]).astype(int) for d in DETECTORS)
        pred = (votes >= 2).astype(int)
        ng, ns = int((lab == 0).sum()), int((lab == 1).sum())
        bpcer = ((pred == 1) & (lab == 0)).sum() / ng
        apcer = ((pred == 0) & (lab == 1)).sum() / ns
        emit(rows, "ensemble", ds, "BPCER", "MAJORITY", bpcer, ng, "wilson")
        emit(rows, "ensemble", ds, "APCER", "MAJORITY", apcer, ns, "wilson")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    maxhw = max(float(r["half_width_pp"]) for r in rows)
    for r in rows:
        print(f"{r['dataset']:13s} {r['metric']:8s} {r['unit']:9s} "
              f"{r['point_pct']:>6s}% [{r['ci_low_pct']:>6s}, {r['ci_high_pct']:>6s}]  "
              f"+/-{r['half_width_pp']:>4s} pp  (n={r['n']}, {r['method']})")
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")
    print(f"Max half-width across all reported rates: {maxhw:.2f} pp "
          f"-> CIs add no decision value; not printed in the paper.")

if __name__ == "__main__":
    run()
