"""Verify PartialEdit data integrity: .npy arrays, JSON, CSV consistency."""
import numpy as np
import json
import csv
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

NPY_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e5_cross_dataset")
JSON_FILE = NPY_DIR / "results.json"
CSV_FILE = (Path(__file__).resolve().parents[1] / "data" / "partialedit_results.csv")

def main():
    errors = []
    with open(JSON_FILE) as f:
        results = json.load(f)['partialedit']
    with open(CSV_FILE) as f:
        csv_rows = {r['Detector'].lower(): r for r in csv.DictReader(f)}

    for det in ['bam', 'cfprf', 'mrm']:
        prefix = f"{det}_partialedit_"
        print(f"\n--- {det.upper()} ---")

        scores = np.load(NPY_DIR / f"{prefix}utt_scores.npy", allow_pickle=True)
        labels = np.load(NPY_DIR / f"{prefix}utt_labels.npy", allow_pickle=True)
        ids = np.load(NPY_DIR / f"{prefix}utt_ids.npy", allow_pickle=True)
        frame_s = np.load(NPY_DIR / f"{prefix}frame_scores.npy", allow_pickle=True)
        frame_l = np.load(NPY_DIR / f"{prefix}frame_labels.npy", allow_pickle=True)

        n = len(scores)
        n_spoof = (labels == 1).sum()
        n_genuine = (labels == 0).sum()
        print(f"  N={n} (spoof={n_spoof}, genuine={n_genuine})")

        # Verify single class
        if n_genuine > 0:
            errors.append(f"{det}: unexpected {n_genuine} genuine utterances")
        else:
            print(f"  Single class confirmed (all spoof)")

        # Verify utt_eer is null in JSON
        utt_eer = results[det]['utt_eer']
        if utt_eer is not None:
            errors.append(f"{det}: utt_eer should be null, got {utt_eer}")
        else:
            print(f"  utt_eer: null (correct - single class)")

        # Unique IDs check. PartialEdit concatenates the E1 and E2 subsets, whose
        # utt_ids drop the E1/E2 prefix and so collide 2:1 (88,330 -> 44,165 unique).
        # The arrays stay row-aligned, so per-utterance metrics are unaffected
        # (registry section 4.3). Any other duplicate pattern is a real error.
        n_unique = len(set(ids))
        if n_unique == len(ids):
            print(f"  Unique IDs: {n_unique}/{len(ids)}")
        elif n_unique == len(ids) // 2:
            print(f"  Unique IDs: {n_unique}/{len(ids)} - EXPECTED E1/E2 2:1 collision (row-aligned)")
        else:
            print(f"  Unique IDs: {n_unique}/{len(ids)}")
            errors.append(f"{det}: {len(ids)-n_unique} duplicate IDs ({n_unique} unique of {len(ids)})")

        # Frame-level metrics
        seg_metrics = results[det].get('partialedit_segment_metrics', {})
        if seg_metrics:
            json_f1 = seg_metrics['f1']
            json_acc = seg_metrics['accuracy']
            json_prec = seg_metrics['precision']
            json_rec = seg_metrics['recall']
            json_auc = seg_metrics['roc_auc']
            n_frames = seg_metrics['n_frames']
            spoof_ratio = seg_metrics['spoof_frame_ratio']
            print(f"  Frame metrics (JSON): F1={json_f1:.4f} Acc={json_acc:.4f} Prec={json_prec:.4f} Rec={json_rec:.4f} AUC={json_auc:.4f}")
            print(f"  N_frames={n_frames}, spoof_ratio={spoof_ratio:.4f}")

            # CSV match
            csv_f1 = float(csv_rows[det]['Seg_F1'])
            csv_acc = float(csv_rows[det]['Frame_Accuracy'])
            if abs(csv_f1 - json_f1) > 1e-10: errors.append(f"{det}: CSV Seg_F1 mismatch")
            if abs(csv_acc - json_acc) > 1e-10: errors.append(f"{det}: CSV Frame_Accuracy mismatch")
            print(f"  CSV Seg_F1: {'MATCH' if abs(csv_f1-json_f1)<1e-10 else 'MISMATCH'}")
            print(f"  CSV Frame_Acc: {'MATCH' if abs(csv_acc-json_acc)<1e-10 else 'MISMATCH'}")

        # Seg EER CSV check
        seg_eer_key = 'seg_eer_160ms' if det == 'bam' else 'seg_eer_20ms'
        json_seg_eer = results[det].get(seg_eer_key)
        csv_seg_eer_col = 'Seg_EER_160ms' if det == 'bam' else 'Seg_EER_20ms'
        csv_seg_eer = csv_rows[det][csv_seg_eer_col]
        if csv_seg_eer != 'NA':
            delta = abs(float(csv_seg_eer) - json_seg_eer)
            print(f"  Seg EER ({seg_eer_key}): CSV={csv_seg_eer} JSON={json_seg_eer:.6f} - {'MATCH' if delta<1e-10 else 'DIFF'}")

    print(f"\nERRORS: {len(errors)}")
    for e in errors: print(f"  {e}")
    if not errors: print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
