"""Verify E5 cross-dataset data integrity per detector per dataset."""
import numpy as np
import json
import csv
from pathlib import Path
from sklearn.metrics import roc_curve

NPY_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e5_cross_dataset")
JSON_FILE = NPY_DIR / "results.json"
CSV_DET = (Path(__file__).resolve().parents[1] / "data" / "e5_cross_dataset_detection.csv")
CSV_LOC = (Path(__file__).resolve().parents[1] / "data" / "e5_cross_dataset_localization.csv")

DATASETS = {
    'llamapartialspoof': 'LlamaPartialSpoof',
    'hqmpsd': 'HQ-MPSD',
    'partialedit': 'PartialEdit',
}

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def main():
    errors = []
    with open(JSON_FILE) as f:
        results = json.load(f)

    for ds_key, ds_name in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"{ds_name} ({ds_key})")
        print(f"{'='*50}")

        for det in ['bam', 'cfprf', 'mrm']:
            prefix = f"{det}_{ds_key}_"
            print(f"\n  --- {det.upper()} ---")

            scores = np.load(NPY_DIR / f"{prefix}utt_scores.npy", allow_pickle=True)
            labels = np.load(NPY_DIR / f"{prefix}utt_labels.npy", allow_pickle=True)
            ids = np.load(NPY_DIR / f"{prefix}utt_ids.npy", allow_pickle=True)
            frame_s = np.load(NPY_DIR / f"{prefix}frame_scores.npy", allow_pickle=True)
            frame_l = np.load(NPY_DIR / f"{prefix}frame_labels.npy", allow_pickle=True)

            n = len(scores)
            n_spoof = (labels == 1).sum()
            n_genuine = (labels == 0).sum()
            print(f"    Utterances: {n} (spoof={n_spoof}, genuine={n_genuine})")
            print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")

            # Array consistency
            lengths = [len(scores), len(labels), len(ids), len(frame_s), len(frame_l)]
            if len(set(lengths)) != 1:
                errors.append(f"{ds_key}/{det}: array lengths {lengths}")

            # NaN/Inf
            if np.isnan(scores).any(): errors.append(f"{ds_key}/{det}: NaN")
            if np.isinf(scores).any(): errors.append(f"{ds_key}/{det}: Inf")

            # Unique IDs. PartialEdit concatenates the E1 and E2 subsets, whose
            # utt_ids drop the E1/E2 prefix and so collide 2:1 (88,330 -> 44,165
            # unique). The arrays stay row-aligned, so per-utterance metrics are
            # unaffected (registry section 4.3). Any other duplicate pattern is real.
            n_unique = len(set(ids))
            if n_unique == len(ids):
                print(f"    Unique IDs: {n_unique}/{len(ids)}")
            elif ds_key == "partialedit" and n_unique == len(ids) // 2:
                print(f"    Unique IDs: {n_unique}/{len(ids)} - EXPECTED E1/E2 2:1 collision (row-aligned)")
            else:
                errors.append(f"{ds_key}/{det}: {len(ids)-n_unique} duplicate IDs")
                print(f"    Unique IDs: {n_unique}/{len(ids)}")

            # JSON count. MRM failed to process 9 LlamaPartialSpoof utterances, so its
            # npy holds 140,607 of the 140,616 scored; the analysis uses that
            # intersection. That specific 9-gap is expected; any other mismatch is real.
            json_n = results[ds_key][det]['n_utterances']
            gap = json_n - n
            if json_n == n:
                print(f"    JSON count: {json_n} vs npy={n} - MATCH")
            elif ds_key == "llamapartialspoof" and det == "mrm" and gap == 9:
                print(f"    JSON count: {json_n} vs npy={n} - EXPECTED (MRM dropped {gap}; analysis uses {n})")
            else:
                print(f"    JSON count: {json_n} vs npy={n} - DIFF ({gap})")
                errors.append(f"{ds_key}/{det}: JSON={json_n} npy={n}")

            # EER (skip PartialEdit - single class)
            if n_genuine > 0:
                eer = compute_eer(labels, scores)
                eer_json = results[ds_key][det]['utt_eer']
                delta = abs(eer - eer_json)
                print(f"    EER: computed={eer:.6f}, JSON={eer_json:.6f} - {'MATCH' if delta<0.001 else 'DIFF'}")
                if delta >= 0.001: errors.append(f"{ds_key}/{det}: EER delta={delta:.6f}")
            else:
                print(f"    EER: N/A (single class)")

    print(f"\n{'='*50}")
    print(f"ERRORS: {len(errors)}")
    for e in errors: print(f"  {e}")
    if not errors: print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
