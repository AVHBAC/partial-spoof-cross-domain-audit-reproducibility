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
RAW_E5 = DATA_DIR / "raw_e5_cross_dataset"
RAW_E1 = DATA_DIR / "raw_e1_baseline"
CATEGORY_CSV = DATA_DIR / "hqmpsd_utt_categories.csv"
OUT_CSV = DATA_DIR / "hqmpsd_composition_results.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
PAIRS = [("bam", "cfprf"), ("bam", "mrm"), ("cfprf", "mrm")]

DIR_TO_CATEGORY = {
    "Bonafide": ("bonafide", 0),
    "Fully_Fake": ("fully_fake", 2),
    "Partial_Fake_Clean": ("partial_clean", 1),
    "Partial_Fake_Noisy": ("partial_noisy", 1),
}
SPOOF_PARTIAL = {"partial_clean", "partial_noisy"}

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, _, _ = roc_curve(labels, scores, pos_label=1)
    _, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        return float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
    except ValueError:
        diff = np.abs(fpr - fnr)
        idx = int(np.nanargmin(diff))
        return float((fpr[idx] + fnr[idx]) / 2.0)

def build_category_csv(dataset_root: Path) -> None:
    english = None
    for cand in (dataset_root / "17929533" / "English", dataset_root / "English", dataset_root):
        if (cand / "Bonafide").exists():
            english = cand
            break
    if english is None:
        raise SystemExit(f"Could not find HQ-MPSD English/ tree under {dataset_root}")

    rows = []
    for dir_name, (cat, raw_label) in DIR_TO_CATEGORY.items():
        d = english / dir_name
        if not d.exists():
            print(f"  [warn] absent: {d}")
            continue
        for f in sorted(d.glob("*.flac")):
            rows.append(
                {
                    "utterance_id": f.stem,
                    "category": cat,
                    "raw_utt_label": raw_label,
                    "binary_label": min(raw_label, 1),
                    "is_partial_spoof": int(cat in SPOOF_PARTIAL),
                }
            )
    CATEGORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORY_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"Wrote {CATEGORY_CSV}  ({len(rows)} utterances)")
    print(f"  category counts: {counts}")

def load_category_map() -> dict[str, str]:
    if not CATEGORY_CSV.exists():
        raise SystemExit(
            f"{CATEGORY_CSV} not found. Regenerate with:\n"
            f"  python {Path(__file__).name} --build-categories /path/to/HQ-MPSD-EN"
        )
    cat: dict[str, str] = {}
    with open(CATEGORY_CSV) as fh:
        for r in csv.DictReader(fh):
            cat[r["utterance_id"]] = r["category"]
    return cat

def jaccard_pairs(scores, labels, thresholds, mask):
    preds = {d: (scores[d][mask] >= thresholds[d]).astype(int) for d in DETECTORS}
    lbm = labels[mask]
    errs = {d: (preds[d] != lbm).astype(int) for d in DETECTORS}
    out = {}
    for a, b in PAIRS:
        ea, eb = errs[a], errs[b]
        both = int((ea & eb).sum())
        union = int(ea.sum() + eb.sum() - both)
        out[(a, b)] = {
            "jaccard": (both / union) if union > 0 else float("nan"),
            "n_err_a": int(ea.sum()),
            "n_err_b": int(eb.sum()),
            "n_both": both,
        }
    return out

def main() -> None:
    cat_map = load_category_map()

    thr_json = json.load(open(RAW_E1 / "results.json"))
    thresholds = {d: float(thr_json[d]["utt_eer_threshold"]) for d in DETECTORS}

    scores, labels, ids_ref = {}, None, None
    for d in DETECTORS:
        ids = np.load(RAW_E5 / f"{d}_hqmpsd_utt_ids.npy", allow_pickle=True)
        sc = np.load(RAW_E5 / f"{d}_hqmpsd_utt_scores.npy")
        lb = np.load(RAW_E5 / f"{d}_hqmpsd_utt_labels.npy")
        if ids_ref is None:
            ids_ref, labels = ids, lb
        elif not np.array_equal(ids, ids_ref):
            raise SystemExit(f"id ordering for {d} differs from reference")
        scores[d] = sc
    n = len(ids_ref)

    cats = np.array([cat_map[i] for i in ids_ref])
    is_bona = cats == "bonafide"
    is_fully = cats == "fully_fake"
    is_partial = np.isin(cats, list(SPOOF_PARTIAL))

    assert is_bona.sum() == 6101, is_bona.sum()
    assert is_partial.sum() == 6103 + 3281, is_partial.sum()
    assert is_fully.sum() == 6103, is_fully.sum()
    assert (labels[is_partial] == 1).all() and (labels[is_fully] == 1).all()
    assert (labels[is_bona] == 0).all()

    subsets = {
        "full": np.ones(n, dtype=bool),
        "partial_only": is_bona | is_partial,
        "fully_only": is_bona | is_fully,
    }

    rows = []
    print(f"\nHQ-MPSD composition: N={n}  "
          f"bonafide={int(is_bona.sum())}  "
          f"partial={int(is_partial.sum())}  fully_fake={int(is_fully.sum())}  "
          f"(fully-fake = {100*is_fully.sum()/(is_partial.sum()+is_fully.sum()):.1f}% of spoof)\n")

    print("Utt-EER (%):")
    print(f"  {'det':6s} {'full':>8s} {'partial':>8s} {'fully':>8s}")
    for d in DETECTORS:
        vals = {}
        for name, mask in subsets.items():
            vals[name] = compute_eer(scores[d][mask], labels[mask])
        print(f"  {d:6s} {100*vals['full']:8.2f} {100*vals['partial_only']:8.2f} {100*vals['fully_only']:8.2f}")
        for name in subsets:
            rows.append({"metric": "utt_eer", "subset": name, "detector": d.upper(),
                         "pair": "", "value": f"{vals[name]:.6f}",
                         "value_pct": f"{100*vals[name]:.2f}"})

    print("\nJaccard error overlap (in-domain thresholds):")
    print(f"  {'pair':12s} {'full':>8s} {'partial':>8s} {'fully':>8s}")
    jac = {name: jaccard_pairs(scores, labels, thresholds, mask) for name, mask in subsets.items()}
    for a, b in PAIRS:
        pair = f"{a.upper()}/{b.upper()}"
        print(f"  {pair:12s} {jac['full'][(a,b)]['jaccard']:8.4f} "
              f"{jac['partial_only'][(a,b)]['jaccard']:8.4f} "
              f"{jac['fully_only'][(a,b)]['jaccard']:8.4f}")
        for name in subsets:
            rows.append({"metric": "jaccard_err", "subset": name, "detector": "",
                         "pair": pair, "value": f"{jac[name][(a,b)]['jaccard']:.6f}",
                         "value_pct": f"{100*jac[name][(a,b)]['jaccard']:.2f}"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "subset", "detector", "pair", "value", "value_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

    full_eer = {r["detector"]: float(r["value_pct"]) for r in rows
                if r["metric"] == "utt_eer" and r["subset"] == "full"}
    assert abs(full_eer["BAM"] - 8.74) < 0.02, full_eer
    assert abs(full_eer["MRM"] - 11.77) < 0.02, full_eer
    full_jac = {r["pair"]: float(r["value"]) for r in rows
                if r["metric"] == "jaccard_err" and r["subset"] == "full"}
    assert abs(full_jac["CFPRF/MRM"] - 0.4504) < 0.001, full_jac
    print("Reproduction guard passed: full-set numbers match the published values.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-categories", metavar="HQ_MPSD_ROOT",
                    help="Regenerate data/hqmpsd_utt_categories.csv from the released directory tree, then exit.")
    args = ap.parse_args()
    if args.build_categories:
        build_category_csv(Path(args.build_categories))
    else:
        main()
