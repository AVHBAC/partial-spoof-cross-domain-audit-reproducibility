"""Verify E1 baseline data integrity: .npy arrays, JSON, and CSV consistency."""
import numpy as np
import json
import csv
from pathlib import Path
from sklearn.metrics import roc_curve

NPY_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e1_baseline")
JSON_FILE = NPY_DIR / "results.json"
CSV_FILE = (Path(__file__).resolve().parents[1] / "data" / "e1_baseline_partialspoof.csv")

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def main():
    errors = []
    with open(JSON_FILE) as f:
        results = json.load(f)
    with open(CSV_FILE) as f:
        csv_rows = {r['Detector'].lower(): r for r in csv.DictReader(f)}

    for det in ['bam', 'cfprf', 'mrm']:
        print(f"\n--- {det.upper()} ---")
        scores = np.load(NPY_DIR / f"{det}_utt_scores.npy", allow_pickle=True)
        labels = np.load(NPY_DIR / f"{det}_utt_labels.npy", allow_pickle=True)
        ids = np.load(NPY_DIR / f"{det}_utt_ids.npy", allow_pickle=True)
        frame_s = np.load(NPY_DIR / f"{det}_frame_scores.npy", allow_pickle=True)
        frame_l = np.load(NPY_DIR / f"{det}_frame_labels.npy", allow_pickle=True)

        # Array lengths
        lengths = [len(scores), len(labels), len(ids), len(frame_s), len(frame_l)]
        if len(set(lengths)) != 1:
            errors.append(f"{det}: array length mismatch {lengths}")
        print(f"  Utterances: {len(scores)}")
        print(f"  Spoof: {(labels==1).sum()}, Genuine: {(labels==0).sum()}")

        # NaN/Inf
        if np.isnan(scores).any(): errors.append(f"{det}: NaN in scores")
        if np.isinf(scores).any(): errors.append(f"{det}: Inf in scores")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Unique IDs
        n_unique = len(set(ids))
        if n_unique != len(ids):
            errors.append(f"{det}: {len(ids)-n_unique} duplicate IDs")
        print(f"  Unique IDs: {n_unique}/{len(ids)}")

        # JSON count match
        json_n = results[det]['n_utterances']
        if json_n != len(scores):
            errors.append(f"{det}: JSON n={json_n} vs npy n={len(scores)}")
        print(f"  JSON count: {json_n} - {'MATCH' if json_n==len(scores) else 'MISMATCH'}")

        # EER recomputation
        eer = compute_eer(labels, scores)
        eer_json = results[det]['utt_eer']
        delta = abs(eer - eer_json)
        print(f"  EER: computed={eer:.6f}, JSON={eer_json:.6f}, delta={delta:.6f} - {'MATCH' if delta<0.001 else 'DIFF'}")
        if delta >= 0.001: errors.append(f"{det}: EER delta={delta:.6f}")

        # CSV match
        csv_eer = float(csv_rows[det]['Utt_EER'])
        if abs(csv_eer - eer_json) > 1e-10:
            errors.append(f"{det}: CSV EER {csv_eer} != JSON {eer_json}")
        print(f"  CSV EER: {csv_eer:.10f} - {'MATCH' if abs(csv_eer-eer_json)<1e-10 else 'MISMATCH'}")

    print(f"\n{'='*40}")
    print(f"ERRORS: {len(errors)}")
    for e in errors: print(f"  {e}")
    if not errors: print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
