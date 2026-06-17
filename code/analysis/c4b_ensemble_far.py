from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_E1 = DATA_DIR / "raw_e1_baseline"
RAW_E5 = DATA_DIR / "raw_e5_cross_dataset"
OUT_CSV = DATA_DIR / "ensemble_far_operating_points.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
FAR_TARGET = 0.01

LLAMA_CAT = DATA_DIR / "llamaps_utt_categories.csv"
HQMPSD_CAT = DATA_DIR / "hqmpsd_utt_categories.csv"

def load_det(prefix: str, key: str):
    base = RAW_E1 if key == "PartialSpoof" else RAW_E5
    ids = np.load(base / f"{prefix}_utt_ids.npy", allow_pickle=True)
    s = np.load(base / f"{prefix}_utt_scores.npy")
    l = np.load(base / f"{prefix}_utt_labels.npy")
    ids = np.array([str(x) for x in ids])
    return ids, s, l

def aligned(dataset: str):
    suffix = {"PartialSpoof": "", "LlamaPS": "_llamapartialspoof", "HQ-MPSD": "_hqmpsd"}[dataset]
    raw = {}
    for d in DETECTORS:
        prefix = d + suffix
        ids, s, l = load_det(prefix, dataset)
        raw[d] = {"id": ids, "s": s, "l": l, "pos": {u: k for k, u in enumerate(ids)}}
    common = set(raw["bam"]["id"]) & set(raw["cfprf"]["id"]) & set(raw["mrm"]["id"])
    order = [i for i in raw["bam"]["id"] if i in common]
    S = {d: np.array([raw[d]["s"][raw[d]["pos"][i]] for i in order]) for d in DETECTORS}
    L = np.array([raw["bam"]["l"][raw["bam"]["pos"][i]] for i in order])
    return order, S, L

def partial_mask(dataset: str, order: list) -> np.ndarray:
    if dataset == "LlamaPS":
        tern = {}
        for r in csv.DictReader(open(LLAMA_CAT)):
            tern[r["utterance_id"]] = int(r["ternary_label"])
        return np.array([tern[i] in (0, 1) for i in order])
    if dataset == "HQ-MPSD":
        cat = {}
        for r in csv.DictReader(open(HQMPSD_CAT)):
            cat[r["utterance_id"]] = r["category"]
        return np.array([cat[i] != "fully_fake" for i in order])
    return np.ones(len(order), bool)

def rates(pred: np.ndarray, lab: np.ndarray):
    ng, ns = int((lab == 0).sum()), int((lab == 1).sum())
    bp = ((pred == 1) & (lab == 0)).sum() / ng if ng else float("nan")
    ap = ((pred == 0) & (lab == 1)).sum() / ns if ns else float("nan")
    return 100 * bp, 100 * ap, 100 * (bp + ap) / 2

def main():
    e1 = json.load(open(RAW_E1 / "results.json"))
    eer_thr = {d: float(e1[d]["utt_eer_threshold"]) for d in DETECTORS}

    order_id, S_id, L_id = aligned("PartialSpoof")
    far_thr = {}
    for d in DETECTORS:
        g = S_id[d][L_id == 0]
        far_thr[d] = float(np.percentile(g, 100 * (1 - FAR_TARGET)))

    print("Per-detector thresholds (fixed from in-domain):")
    print(f"{'det':<6}{'EER':>12}{'FAR=1%':>12}")
    for d in DETECTORS:
        print(f"{d.upper():<6}{eer_thr[d]:>12.6f}{far_thr[d]:>12.6f}")
    for d, exp in [("bam", 0.999615), ("cfprf", 0.729442), ("mrm", 0.902043)]:
        assert abs(far_thr[d] - exp) < 5e-4, f"FAR thresh {d}={far_thr[d]:.6f} != c4 {exp}"

    rows = []
    guard = {}
    for dataset in ["PartialSpoof", "LlamaPS", "HQ-MPSD"]:
        order, S, L = aligned(dataset)
        for subset in ["full", "partial_only"]:
            pm = partial_mask(dataset, order)
            if subset == "full":
                m = np.ones(len(order), bool)
            else:
                m = pm
            if dataset == "PartialSpoof" and subset == "partial_only":
                continue
            lab = L[m]
            for pt_name, thr in [("EER", eer_thr), ("FAR1", far_thr)]:
                votes = sum((S[d] >= thr[d]).astype(int) for d in DETECTORS)
                maj = (votes >= 2).astype(int)[m]
                bp, ap, ac = rates(maj, lab)
                rows.append({
                    "Dataset": dataset, "Subset": subset, "OperatingPoint": pt_name,
                    "N_genuine": int((lab == 0).sum()), "N_spoof": int((lab == 1).sum()),
                    "BPCER": f"{bp:.2f}", "APCER": f"{ap:.2f}", "ACER": f"{ac:.2f}",
                })
                guard[(dataset, subset, pt_name)] = (bp, ap, ac)
                print(f"  {dataset:<13} {subset:<13} {pt_name:<5} "
                      f"BPCER={bp:6.2f}  APCER={ap:6.2f}  ACER={ac:6.2f}  "
                      f"(Ng={int((lab==0).sum())}, Ns={int((lab==1).sum())})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

    def chk(key, exp, tol=0.35):
        got = guard[key]
        for name, g, e in zip("BPCER APCER ACER".split(), got, exp):
            assert abs(g - e) < tol, f"GUARD FAIL {key} {name}: got {g:.2f}, expected {e}"

    chk(("PartialSpoof", "full", "EER"), (0.4, 0.8, 0.6))
    chk(("LlamaPS", "full", "EER"), (68.7, 2.4, 35.6))
    chk(("LlamaPS", "partial_only", "EER"), (68.7, 0.2, 34.4))
    print("Reproduction guards passed: EER-point ensemble matches published values.")
    print("=> FAR=1% ensemble numbers above are produced by the same machinery and are trustworthy.")

if __name__ == "__main__":
    main()
