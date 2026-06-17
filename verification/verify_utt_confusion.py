"""Verify utterance-level confusion matrix CSV against .npy arrays."""
import numpy as np
import json
import csv
from pathlib import Path

NPY_BASE = (Path(__file__).resolve().parents[1] / "data")
CSV_FILE = (Path(__file__).resolve().parents[1] / "data" / "utt_confusion_matrices.csv")

DATASETS = {
    'PartialSpoof': {'dir': 'raw_e1_baseline', 'prefix': '', 'json_dir': 'raw_e1_baseline', 'json_key': None},
    'LlamaPartialSpoof': {'dir': 'raw_e5_cross_dataset', 'prefix': 'llamapartialspoof_', 'json_dir': 'raw_e5_cross_dataset', 'json_key': 'llamapartialspoof'},
    'HQ-MPSD': {'dir': 'raw_e5_cross_dataset', 'prefix': 'hqmpsd_', 'json_dir': 'raw_e5_cross_dataset', 'json_key': 'hqmpsd'},
}

def main():
    errors = []

    # Load CSV
    with open(CSV_FILE) as f:
        csv_rows = list(csv.DictReader(f))
    csv_lookup = {(r['Dataset'], r['Detector']): r for r in csv_rows}

    # Load JSONs for thresholds
    with open(NPY_BASE / "raw_e1_baseline" / "results.json") as f:
        e1_json = json.load(f)
    with open(NPY_BASE / "raw_e5_cross_dataset" / "results.json") as f:
        e5_json = json.load(f)

    for ds_name, cfg in DATASETS.items():
        npy_dir = NPY_BASE / cfg['dir']

        for det in ['bam', 'cfprf', 'mrm']:
            det_upper = det.upper()
            print(f"\n--- {ds_name}/{det_upper} ---")

            prefix = f"{det}_{cfg['prefix']}"
            scores = np.load(npy_dir / f"{prefix}utt_scores.npy", allow_pickle=True)
            labels = np.load(npy_dir / f"{prefix}utt_labels.npy", allow_pickle=True)

            # Get threshold
            if cfg['json_key'] is None:
                thresh = e1_json[det]['utt_eer_threshold']
            else:
                thresh = e5_json[cfg['json_key']][det]['utt_eer_threshold']

            preds = (scores >= thresh).astype(int)

            # Recompute confusion matrix
            tp = int(((preds == 1) & (labels == 1)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())

            # Check CSV values
            csv_row = csv_lookup[(ds_name, det_upper)]
            csv_tp = int(csv_row['TP'])
            csv_fn = int(csv_row['FN'])
            csv_tn = int(csv_row['TN'])
            csv_fp = int(csv_row['FP'])

            # Verify each cell
            checks = [
                ('TP', tp, csv_tp),
                ('FN', fn, csv_fn),
                ('TN', tn, csv_tn),
                ('FP', fp, csv_fp),
            ]
            for name, computed, from_csv in checks:
                if computed != from_csv:
                    errors.append(f"{ds_name}/{det_upper}: {name} computed={computed} csv={from_csv}")
                    print(f"  {name}: computed={computed}, csv={from_csv} - MISMATCH")
                else:
                    print(f"  {name}: {computed} - MATCH")

            # Verify counts sum to total
            total = tp + fn + tn + fp
            csv_total = int(csv_row['N_total'])
            if total != csv_total:
                errors.append(f"{ds_name}/{det_upper}: sum={total} vs N_total={csv_total}")
            if total != len(labels):
                errors.append(f"{ds_name}/{det_upper}: sum={total} vs npy length={len(labels)}")

            # Verify spoof/genuine split
            n_spoof = int((labels == 1).sum())
            n_genuine = int((labels == 0).sum())
            if n_spoof != int(csv_row['N_spoof']):
                errors.append(f"{ds_name}/{det_upper}: N_spoof mismatch")
            if n_genuine != int(csv_row['N_genuine']):
                errors.append(f"{ds_name}/{det_upper}: N_genuine mismatch")

    print(f"\n{'='*40}")
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  {e}")
    if not errors:
        print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
