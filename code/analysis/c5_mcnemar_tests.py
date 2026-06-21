from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

_BASE = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR     = Path(__file__).resolve().parents[2] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTORS = ("bam", "cfprf", "mrm")

PAIRED_DATASETS = {
    "PartialSpoof":      ("raw_e1_baseline",      ""),
    "LlamaPartialSpoof": ("raw_e5_cross_dataset", "llamapartialspoof_"),
    "HQ-MPSD":           ("raw_e5_cross_dataset", "hqmpsd_"),
}

PARTIALEDIT = ("raw_e5_cross_dataset", "partialedit_")

with open(RESULTS_DIR / "raw_e1_baseline" / "results.json") as fh:
    e1 = json.load(fh)
THRESHOLDS = {det: float(e1[det]["utt_eer_threshold"]) for det in DETECTORS}

def load_arrays(subdir: str, infix: str, det: str):
    base = RESULTS_DIR / subdir
    ids    = np.load(base / f"{det}_{infix}utt_ids.npy",    allow_pickle=True)
    scores = np.load(base / f"{det}_{infix}utt_scores.npy")
    labels = np.load(base / f"{det}_{infix}utt_labels.npy")
    return ids, scores, labels

def align_by_id(ds_label: str) -> tuple[list, dict[str, np.ndarray], np.ndarray]:
    subdir, infix = PAIRED_DATASETS[ds_label]

    per_det: dict[str, tuple[dict, dict]] = {}
    id_sets = []
    for det in DETECTORS:
        ids, scores, labels = load_arrays(subdir, infix, det)
        preds = (scores >= THRESHOLDS[det]).astype(np.int8)
        pred_map  = {}
        label_map = {}
        for i, uid in enumerate(ids):
            if uid in pred_map:
                raise RuntimeError(f"Unexpected ID collision in {ds_label}/{det}: {uid}")
            pred_map[uid]  = int(preds[i])
            label_map[uid] = int(labels[i])
        per_det[det] = (pred_map, label_map)
        id_sets.append(set(pred_map.keys()))

    common = sorted(id_sets[0] & id_sets[1] & id_sets[2])

    labels_bam = np.array([per_det["bam"][1][i] for i in common], dtype=np.int8)
    for det in ("cfprf", "mrm"):
        labels_other = np.array([per_det[det][1][i] for i in common], dtype=np.int8)
        if not np.array_equal(labels_bam, labels_other):
            raise RuntimeError(f"Label disagreement across detectors on {ds_label}/{det}")

    preds = {det: np.array([per_det[det][0][i] for i in common], dtype=np.int8) for det in DETECTORS}
    return common, preds, labels_bam

def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, int, int, float]:
    both_correct = int(np.sum( correct_a &  correct_b))
    both_wrong   = int(np.sum(~correct_a & ~correct_b))
    b            = int(np.sum( correct_a & ~correct_b))
    c            = int(np.sum(~correct_a &  correct_b))
    n_discord = b + c
    if n_discord == 0:
        p = 1.0
    else:
        p = float(binomtest(k=min(b, c), n=n_discord, p=0.5, alternative="two-sided").pvalue)
    return both_correct, both_wrong, b, c, p

def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom  = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)

def run_paired_mcnemar():
    rows = []
    for ds_label in PAIRED_DATASETS:
        ids, preds, labels = align_by_id(ds_label)
        n_total = len(ids)
        max_native = max(
            len(np.load(RESULTS_DIR / PAIRED_DATASETS[ds_label][0]
                        / f"{det}_{PAIRED_DATASETS[ds_label][1]}utt_ids.npy",
                        allow_pickle=True))
            for det in DETECTORS
        )
        n_dropped = max_native - n_total

        correct = {det: (preds[det] == labels) for det in DETECTORS}
        for det_a, det_b in combinations(DETECTORS, 2):
            both_c, both_w, b, c, p = mcnemar_exact(correct[det_a], correct[det_b])
            rows.append({
                "dataset":             ds_label,
                "comparison":          f"{det_a.upper()}_vs_{det_b.upper()}",
                "n_total":             n_total,
                "n_dropped_intersect": n_dropped,
                "both_correct":        both_c,
                "both_wrong":          both_w,
                "b":                   b,
                "c":                   c,
                "p_value":             p,
                "significant_005":     bool(p < 0.05),
            })

    out = OUT_DIR / "mcnemar_tests.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out, rows

def run_partialedit_wilson():
    subdir, infix = PARTIALEDIT
    rows = []
    for det in DETECTORS:
        _, scores, labels = load_arrays(subdir, infix, det)
        if not (labels == 1).all():
            raise RuntimeError(f"PartialEdit/{det}: expected all spoof (label=1), found {set(labels.tolist())}")
        preds = (scores >= THRESHOLDS[det]).astype(np.int8)
        n = int(len(preds))
        k = int(preds.sum())
        lo, hi = wilson_ci_95(k, n)
        rows.append({
            "detector":        det.upper(),
            "n":               n,
            "n_correct":       k,
            "detection_rate":  k / n,
            "wilson_lower":    lo,
            "wilson_upper":    hi,
        })

    out = OUT_DIR / "mcnemar_partialedit_wilson.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out, rows

def main():
    print("Thresholds (PartialSpoof in-domain EER threshold from raw_e1_baseline/results.json):")
    for det, t in THRESHOLDS.items():
        print(f"  {det.upper():6s}  thr = {t:.6f}")
    print()

    print("=" * 78)
    print("Exact McNemar's tests (3 datasets x 3 detector pairs = 9 tests)")
    print("=" * 78)
    mc_path, mc_rows = run_paired_mcnemar()
    last_ds = None
    for r in mc_rows:
        if r["dataset"] != last_ds:
            print(f'\n  {r["dataset"]}  (n_total={r["n_total"]}, dropped to intersect={r["n_dropped_intersect"]})')
            last_ds = r["dataset"]
        print(
            f'    {r["comparison"]:14s}  '
            f'b={r["b"]:>6d}  c={r["c"]:>6d}  '
            f'b+c={r["b"]+r["c"]:>6d}  '
            f'p={r["p_value"]:.3e}  '
            f'sig@0.05={"yes" if r["significant_005"] else "no"}'
        )
    print(f"\n  -> {mc_path}")
    print()

    print("=" * 78)
    print("PartialEdit Wilson 95% CIs (3 detectors, single-class spoof-only)")
    print("=" * 78)
    pe_path, pe_rows = run_partialedit_wilson()
    for r in pe_rows:
        print(
            f'  {r["detector"]:6s}  '
            f'n={r["n"]:>6d}  '
            f'detected={r["n_correct"]:>6d}  '
            f'rate={r["detection_rate"]:.4f}  '
            f'CI=[{r["wilson_lower"]:.4f}, {r["wilson_upper"]:.4f}]'
        )
    print(f"\n  -> {pe_path}")

if __name__ == "__main__":
    main()
