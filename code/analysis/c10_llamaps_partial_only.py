from __future__ import annotations

import argparse
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
CATEGORY_CSV = DATA_DIR / "llamaps_utt_categories.csv"
OUT_CSV = DATA_DIR / "llamaps_partial_only_results.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
PAIRS = [("bam", "cfprf"), ("bam", "mrm"), ("cfprf", "mrm")]
_LABEL_FILES = ["label_R01TTS.0.a.txt", "label_R01TTS.0.b.txt"]

def _ternary(line: str):
    p = line.split()
    if len(p) < 4:
        return None, None
    uid, raw = p[0], p[2]
    has_spoof = has_bona = False
    for seg in p[3:]:
        a = seg.rsplit("-", 1)
        if len(a) == 2:
            if a[1] == "spoof":
                has_spoof = True
            else:
                has_bona = True
    if raw == "bonafide":
        t = 0
    elif has_spoof and has_bona:
        t = 1
    elif has_spoof:
        t = 2
    else:
        t = 0
    return uid, t

def build_category_csv(root: Path) -> None:
    cat_name = {0: "bonafide", 1: "partial", 2: "fully_fake"}
    rows = []
    for lf in _LABEL_FILES:
        path = root / lf
        if not path.exists():
            raise SystemExit(f"label file not found: {path}")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            uid, t = _ternary(line)
            if uid is None:
                continue
            rows.append({"utterance_id": uid, "category": cat_name[t],
                         "ternary_label": t, "binary_label": min(t, 1),
                         "is_partial_spoof": int(t == 1)})
    CATEGORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORY_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in cat_name.values()}
    print(f"Wrote {CATEGORY_CSV} ({len(rows)} utterances); counts={counts}")

def load_category_map() -> dict:
    if not CATEGORY_CSV.exists():
        raise SystemExit(f"{CATEGORY_CSV} not found; run --build-categories first.")
    m = {}
    for r in csv.DictReader(open(CATEGORY_CSV)):
        m[r["utterance_id"]] = int(r["ternary_label"])
    return m

def eer(scores, labels) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        return 100 * float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
    except ValueError:
        i = int(np.argmin(np.abs(fpr - fnr)))
        return 100 * float((fpr[i] + fnr[i]) / 2.0)

def eer_threshold(scores, labels) -> float:
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i])

def main() -> None:
    tmap = load_category_map()
    e1 = json.load(open(RAW_E1 / "results.json"))
    thr = {d: float(e1[d]["utt_eer_threshold"]) for d in DETECTORS}

    raw = {}
    for d in DETECTORS:
        ids = np.load(RAW_E5 / f"{d}_llamapartialspoof_utt_ids.npy", allow_pickle=True)
        raw[d] = {
            "id": np.array([str(x) for x in ids]),
            "s": np.load(RAW_E5 / f"{d}_llamapartialspoof_utt_scores.npy"),
            "l": np.load(RAW_E5 / f"{d}_llamapartialspoof_utt_labels.npy"),
        }

    rows = []
    print("Utt-EER (%):  det    FULL  PARTIAL  FULLY")
    for d in DETECTORS:
        tern = np.array([tmap[i] for i in raw[d]["id"]])
        s, l = raw[d]["s"], raw[d]["l"]
        e_full = eer(s, l)
        mP = (tern == 0) | (tern == 1)
        mF = (tern == 0) | (tern == 2)
        e_part, e_fully = eer(s[mP], l[mP]), eer(s[mF], l[mF])
        print(f"             {d.upper():5s} {e_full:6.2f} {e_part:7.2f} {e_fully:6.2f}")
        for nm, v in [("full", e_full), ("partial_only", e_part), ("fully_only", e_fully)]:
            rows.append({"metric": "utt_eer", "subset": nm, "unit": d.upper(),
                         "pair": "", "value": f"{v:.2f}"})

    common = set(raw["bam"]["id"]) & set(raw["cfprf"]["id"]) & set(raw["mrm"]["id"])
    order = [i for i in raw["bam"]["id"] if i in common]
    pos = {d: {u: k for k, u in enumerate(raw[d]["id"])} for d in DETECTORS}
    S = {d: np.array([raw[d]["s"][pos[d][i]] for i in order]) for d in DETECTORS}
    tern = np.array([tmap[i] for i in order])
    lab = (tern >= 1).astype(int)
    maskP = (tern == 0) | (tern == 1)
    n = len(order)
    print(f"\naligned N={n}  bonafide={int((tern==0).sum())} "
          f"partial={int((tern==1).sum())} fully={int((tern==2).sum())} "
          f"(fully-fake = {100*(tern==2).sum()/(tern>=1).sum():.1f}% of spoof)")

    binp = {d: (S[d] >= thr[d]).astype(int) for d in DETECTORS}

    def rates(pred, mask):
        l = lab[mask]; p = pred[mask]
        ng, ns = int((l == 0).sum()), int((l == 1).sum())
        bp = ((p == 1) & (l == 0)).sum() / ng
        ap = ((p == 0) & (l == 1)).sum() / ns
        return 100 * bp, 100 * ap, 100 * (bp + ap) / 2

    votes = sum(binp[d] for d in DETECTORS)
    maj_fixed = (votes >= 2).astype(int)
    thr_tgt = {d: eer_threshold(S[d], lab) for d in DETECTORS}
    votes_r = sum((S[d] >= thr_tgt[d]).astype(int) for d in DETECTORS)
    maj_recal = (votes_r >= 2).astype(int)

    print("\nensemble majority (BPCER/APCER/ACER):")
    for nm, mask in [("full", np.ones(n, bool)), ("partial_only", maskP)]:
        bf = rates(maj_fixed, mask); br = rates(maj_recal, mask)
        print(f"  {nm:12s} fixed {bf[0]:.1f}/{bf[1]:.1f}/{bf[2]:.1f}   "
              f"recal {br[0]:.1f}/{br[1]:.1f}/{br[2]:.1f}")
        for lbl, trip in [("fixed", bf), ("recal", br)]:
            for mk, v in zip(["BPCER", "APCER", "ACER"], trip):
                rows.append({"metric": f"ensemble_{lbl}_{mk}", "subset": nm,
                             "unit": "MAJORITY", "pair": "", "value": f"{v:.1f}"})

    zs = {}
    for d in DETECTORS:
        s1 = np.load(RAW_E1 / f"{d}_utt_scores.npy"); l1 = np.load(RAW_E1 / f"{d}_utt_labels.npy")
        g = s1[l1 == 0]; zs[d] = (float(g.mean()), float(g.std()) + 1e-9)
    z = np.mean([(S[d] - zs[d][0]) / zs[d][1] for d in DETECTORS], axis=0)
    print("\nfusion (LlamaPS):")
    for nm, mask in [("full", np.ones(n, bool)), ("partial_only", maskP)]:
        best = min(eer(S[d][mask], lab[mask]) for d in DETECTORS)
        sv = eer(z[mask], lab[mask])
        orr = rates((votes >= 1).astype(int), mask)
        andd = rates((votes >= 3).astype(int), mask)
        print(f"  {nm:12s} best-single={best:.2f} soft-vote={sv:.2f} "
              f"OR-ACER={orr[2]:.1f} AND-ACER={andd[2]:.1f}")
        rows += [{"metric": "best_single_EER", "subset": nm, "unit": "", "pair": "", "value": f"{best:.2f}"},
                 {"metric": "softvote_EER", "subset": nm, "unit": "", "pair": "", "value": f"{sv:.2f}"},
                 {"metric": "OR_ACER", "subset": nm, "unit": "", "pair": "", "value": f"{orr[2]:.1f}"},
                 {"metric": "AND_ACER", "subset": nm, "unit": "", "pair": "", "value": f"{andd[2]:.1f}"}]

    err = {d: (binp[d] != lab).astype(int) for d in DETECTORS}
    print("\nJaccard error overlap (id-aligned):")
    for a, b in PAIRS:
        for nm, mask in [("full", np.ones(n, bool)), ("partial_only", maskP)]:
            ea, eb = err[a][mask], err[b][mask]
            both = int((ea & eb).sum()); u = int(ea.sum() + eb.sum() - both)
            j = both / u if u > 0 else float("nan")
            rows.append({"metric": "jaccard_err", "subset": nm, "unit": "",
                         "pair": f"{a.upper()}/{b.upper()}", "value": f"{j:.4f}"})
        full_j = float(next(r["value"] for r in rows if r["metric"] == "jaccard_err"
                            and r["subset"] == "full" and r["pair"] == f"{a.upper()}/{b.upper()}"))
        part_j = float(next(r["value"] for r in rows if r["metric"] == "jaccard_err"
                            and r["subset"] == "partial_only" and r["pair"] == f"{a.upper()}/{b.upper()}"))
        print(f"  {a.upper()}/{b.upper():6s} full={full_j:.4f} partial={part_j:.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "subset", "unit", "pair", "value"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {OUT_CSV} ({len(rows)} rows)")

    def cell(metric, subset, unit="", pair=""):
        return float(next(r["value"] for r in rows if r["metric"] == metric
                          and r["subset"] == subset and r["unit"] == unit and r["pair"] == pair))
    assert abs(cell("utt_eer", "full", "BAM") - 15.81) < 0.05
    assert abs(cell("utt_eer", "full", "MRM") - 28.31) < 0.05
    assert abs(cell("ensemble_fixed_BPCER", "full", "MAJORITY") - 68.7) < 0.3
    assert abs(cell("ensemble_fixed_ACER", "full", "MAJORITY") - 35.6) < 0.3
    print("Reproduction guards passed (per-detector EER + ensemble BPCER/ACER match published).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-categories", metavar="LLAMAPS_ROOT")
    args = ap.parse_args()
    if args.build_categories:
        build_category_csv(Path(args.build_categories))
    else:
        main()
