"""Verify HQ-MPSD data integrity: .npy arrays, JSON, CSV consistency."""
import numpy as np
import json
import csv
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

NPY_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e5_cross_dataset")
JSON_FILE = NPY_DIR / "results.json"
CSV_FILE = (Path(__file__).resolve().parents[1] / "data" / "hqmpsd_results.csv")

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def main():
    errors = []
    with open(JSON_FILE) as f:
        results = json.load(f)['hqmpsd']
    with open(CSV_FILE) as f:
        csv_rows = {r['Detector'].lower(): r for r in csv.DictReader(f)}

    for det in ['bam', 'cfprf', 'mrm']:
        prefix = f"{det}_hqmpsd_"
        print(f"\n--- {det.upper()} ---")

        scores = np.load(NPY_DIR / f"{prefix}utt_scores.npy", allow_pickle=True)
        labels = np.load(NPY_DIR / f"{prefix}utt_labels.npy", allow_pickle=True)
        ids = np.load(NPY_DIR / f"{prefix}utt_ids.npy", allow_pickle=True)

        n = len(scores)
        n_spoof = (labels == 1).sum()
        n_genuine = (labels == 0).sum()
        print(f"  N={n} (spoof={n_spoof}, genuine={n_genuine})")

        # EER
        eer = compute_eer(labels, scores)
        eer_json = results[det]['utt_eer']
        delta = abs(eer - eer_json)
        print(f"  EER: computed={eer:.6f}, JSON={eer_json:.6f} - {'MATCH' if delta<0.001 else 'DIFF'}")
        if delta >= 0.001: errors.append(f"{det}: EER delta={delta:.6f}")

        # AUC
        auc = roc_auc_score(labels, scores)
        auc_json = results[det]['utt_roc_auc']
        delta_auc = abs(auc - auc_json)
        print(f"  AUC: computed={auc:.6f}, JSON={auc_json:.6f} - {'MATCH' if delta_auc<0.001 else 'DIFF'}")
        if delta_auc >= 0.001: errors.append(f"{det}: AUC delta={delta_auc:.6f}")

        # Accuracy/F1 at EER threshold
        thresh = results[det]['utt_eer_threshold']
        preds = (scores >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        print(f"  Acc: computed={acc:.6f}, JSON={results[det]['utt_accuracy']:.6f}")
        print(f"  F1:  computed={f1:.6f}, JSON={results[det]['utt_f1']:.6f}")

        # CSV match
        csv_eer = float(csv_rows[det]['Utt_EER'])
        csv_auc = float(csv_rows[det]['Utt_ROC_AUC'])
        if abs(csv_eer - eer_json) > 1e-10: errors.append(f"{det}: CSV EER mismatch")
        if abs(csv_auc - auc_json) > 1e-10: errors.append(f"{det}: CSV AUC mismatch")
        print(f"  CSV EER: {'MATCH' if abs(csv_eer-eer_json)<1e-10 else 'MISMATCH'}")
        print(f"  CSV AUC: {'MATCH' if abs(csv_auc-auc_json)<1e-10 else 'MISMATCH'}")

    print(f"\nERRORS: {len(errors)}")
    for e in errors: print(f"  {e}")
    if not errors: print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
