"""c12: Partial-only McNemar discordance (BAM-correct fraction on discordant utterances).

For each detector pair (A, B), restrict to utterances where A and B disagree on
correctness (exactly one of them is right) and report the fraction on which A is the
correct one: pct = b / (b + c), with b = (A correct, B wrong), c = (A wrong, B correct).

This makes traceable the paper's BAM-vs-CFPRF statement (results.tex 5.2):
"BAM is correct on 96% of the discordant utterances in-domain but only 53% on
LlamaPartialSpoof and 81% on HQ-MPSD."

Method (matches c5 / c10 / c6):
  - Threshold: each detector's PartialSpoof in-domain `utt_eer_threshold`
    (data/raw_e1_baseline/results.json); preds = score >= threshold.
  - correct = (preds == label); align the three detectors by utterance-ID intersection.
  - PartialSpoof: full in-domain eval set.
  - LlamaPartialSpoof / HQ-MPSD: PARTIAL-ONLY subset = bonafide + partial-spoof,
    excluding fully-fake (same masks as c10 and c6).

Outputs data/partial_discordance.csv and asserts the BAM-vs-CFPRF values reproduce
the published 96.3 / 52.9 / 81.4.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_E1 = DATA_DIR / "raw_e1_baseline"
RAW_E5 = DATA_DIR / "raw_e5_cross_dataset"
LLAMA_CAT = DATA_DIR / "llamaps_utt_categories.csv"
HQMPSD_CAT = DATA_DIR / "hqmpsd_utt_categories.csv"
OUT_CSV = DATA_DIR / "partial_discordance.csv"

DETECTORS = ["bam", "cfprf", "mrm"]
PAIRS = [("bam", "cfprf"), ("bam", "mrm"), ("cfprf", "mrm")]

THR_JSON = json.load(open(RAW_E1 / "results.json"))
THR = {d: float(THR_JSON[d]["utt_eer_threshold"]) for d in DETECTORS}


def load_arrays(base: Path, infix: str) -> dict:
    out = {}
    for d in DETECTORS:
        ids = np.load(base / f"{d}_{infix}utt_ids.npy", allow_pickle=True)
        out[d] = {
            "id": np.array([str(x) for x in ids]),
            "s": np.load(base / f"{d}_{infix}utt_scores.npy"),
            "l": np.load(base / f"{d}_{infix}utt_labels.npy"),
        }
    return out


def align3(raw: dict):
    """Intersect the three detectors by utterance ID, ordered by BAM's ordering."""
    common = set(raw["bam"]["id"]) & set(raw["cfprf"]["id"]) & set(raw["mrm"]["id"])
    order = [i for i in raw["bam"]["id"] if i in common]
    pos = {d: {u: k for k, u in enumerate(raw[d]["id"])} for d in DETECTORS}
    scores = {d: np.array([raw[d]["s"][pos[d][i]] for i in order]) for d in DETECTORS}
    labels = np.array([raw["bam"]["l"][pos["bam"][i]] for i in order], dtype=int)
    # Labels must agree across detectors after alignment.
    for d in ("cfprf", "mrm"):
        ld = np.array([raw[d]["l"][pos[d][i]] for i in order], dtype=int)
        if not np.array_equal(labels, ld):
            raise RuntimeError(f"Label disagreement after alignment for {d}")
    return order, scores, labels


def llama_partial_mask(order: list) -> np.ndarray:
    tmap = {r["utterance_id"]: int(r["ternary_label"]) for r in csv.DictReader(open(LLAMA_CAT))}
    tern = np.array([tmap[i] for i in order])
    return (tern == 0) | (tern == 1)  # bonafide + partial, exclude fully-fake (==2)


def hqmpsd_partial_mask(order: list) -> np.ndarray:
    cmap = {r["utterance_id"]: r["category"] for r in csv.DictReader(open(HQMPSD_CAT))}
    cats = np.array([cmap[i] for i in order])
    return np.isin(cats, ["bonafide", "partial_clean", "partial_noisy"])


def discordance(correct_a: np.ndarray, correct_b: np.ndarray):
    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B correct
    n = b + c
    pct = 100.0 * b / n if n > 0 else float("nan")
    p = float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue) if n > 0 else 1.0
    return b, c, n, pct, p


def main() -> None:
    print("Thresholds (PartialSpoof in-domain utt_eer_threshold):")
    for d in DETECTORS:
        print(f"  {d.upper():6s} {THR[d]:.6f}")

    ps = load_arrays(RAW_E1, "")
    order_ps, S_ps, L_ps = align3(ps)
    lp = load_arrays(RAW_E5, "llamapartialspoof_")
    order_lp, S_lp, L_lp = align3(lp)
    hq = load_arrays(RAW_E5, "hqmpsd_")
    order_hq, S_hq, L_hq = align3(hq)

    datasets = [
        ("PartialSpoof", "full", S_ps, L_ps, np.ones(len(order_ps), dtype=bool)),
        ("LlamaPartialSpoof", "partial_only", S_lp, L_lp, llama_partial_mask(order_lp)),
        ("HQ-MPSD", "partial_only", S_hq, L_hq, hqmpsd_partial_mask(order_hq)),
    ]

    rows = []
    print("\nDiscordance: A-correct fraction on A-vs-B disagreements")
    for label, subset, S, L, mask in datasets:
        correct = {d: ((S[d] >= THR[d]).astype(int) == L) for d in DETECTORS}
        print(f"\n  {label}  (subset={subset}, n={int(mask.sum())} of {len(L)})")
        for a, bdet in PAIRS:
            b, c, n, pct, p = discordance(correct[a][mask], correct[bdet][mask])
            print(f"    {a.upper()}_vs_{bdet.upper():6s}  b={b:>6d} c={c:>6d} "
                  f"n_disc={n:>6d}  A-correct={pct:5.1f}%  p={p:.2e}")
            rows.append({
                "dataset": label, "subset": subset,
                "pair": f"{a.upper()}_vs_{bdet.upper()}",
                "n_discord": n, "b_first_correct": b, "c_second_correct": c,
                "first_correct_pct": f"{pct:.1f}", "p_value": p,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUT_CSV} ({len(rows)} rows)")

    def bam_cfprf(ds):
        return float(next(r["first_correct_pct"] for r in rows
                          if r["dataset"] == ds and r["pair"] == "BAM_vs_CFPRF"))
    assert abs(bam_cfprf("PartialSpoof") - 96.3) < 0.5, bam_cfprf("PartialSpoof")
    assert abs(bam_cfprf("LlamaPartialSpoof") - 52.9) < 0.5, bam_cfprf("LlamaPartialSpoof")
    assert abs(bam_cfprf("HQ-MPSD") - 81.4) < 0.5, bam_cfprf("HQ-MPSD")
    print("Reproduction guard passed: BAM-vs-CFPRF discordance reproduces 96 / 53 / 81.")


if __name__ == "__main__":
    main()
