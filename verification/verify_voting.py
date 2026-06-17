"""Verify voting analysis data: APCER/BPCER/ACER from .npy, cross-check with CSV."""
import numpy as np
import json
import csv
from pathlib import Path

E1_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e1_baseline")
E5_DIR = (Path(__file__).resolve().parents[1] / "data" / "raw_e5_cross_dataset")
CSV_DET = (Path(__file__).resolve().parents[1] / "data" / "voting_detection.csv")
CSV_LOC = (Path(__file__).resolve().parents[1] / "data" / "voting_localization.csv")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_voting(npy_dir, prefix, thresholds):
    """Load scores, apply thresholds, compute majority vote APCER/BPCER."""
    preds = {}
    gt = None
    for det in ['bam', 'cfprf', 'mrm']:
        labels = np.load(npy_dir / f"{det}_{prefix}utt_labels.npy", allow_pickle=True)
        scores = np.load(npy_dir / f"{det}_{prefix}utt_scores.npy", allow_pickle=True)
        ids = np.load(npy_dir / f"{det}_{prefix}utt_ids.npy", allow_pickle=True)
        if gt is None:
            gt = labels
        preds[det] = {'ids': ids, 'labels': labels, 'preds': (scores >= thresholds[det]).astype(int)}

    # Align by common IDs
    common = set(preds['bam']['ids']) & set(preds['cfprf']['ids']) & set(preds['mrm']['ids'])
    idx = {d: {uid: i for i, uid in enumerate(preds[d]['ids'])} for d in ['bam','cfprf','mrm']}

    gt_aligned, pb, pc, pm = [], [], [], []
    for uid in sorted(common):
        bi = idx['bam'][uid]
        gt_aligned.append(preds['bam']['labels'][bi])
        pb.append(preds['bam']['preds'][bi])
        pc.append(preds['cfprf']['preds'][idx['cfprf'][uid]])
        pm.append(preds['mrm']['preds'][idx['mrm'][uid]])

    gt_a = np.array(gt_aligned)
    votes = np.stack([np.array(pb), np.array(pc), np.array(pm)])
    maj = (votes.sum(0) >= 2).astype(int)
    sp = gt_a == 1
    gn = gt_a == 0

    apcer = float((maj[sp] == 0).sum() / sp.sum())
    bpcer = float((maj[gn] == 1).sum() / gn.sum())
    acer = (apcer + bpcer) / 2
    agree = float((votes.std(0) == 0).mean())

    return {
        'n': len(gt_a), 'n_spoof': int(sp.sum()), 'n_genuine': int(gn.sum()),
        'apcer': apcer, 'bpcer': bpcer, 'acer': acer, 'agreement': agree,
    }

def main():
    errors = []
    e1_json = load_json(E1_DIR / "results.json")
    thresholds_e1 = {d: e1_json[d]['utt_eer_threshold'] for d in ['bam','cfprf','mrm']}

    # PartialSpoof
    print("PartialSpoof (in-domain thresholds):")
    ps = compute_voting(E1_DIR, '', thresholds_e1)
    print(f"  N={ps['n']} (spoof={ps['n_spoof']}, genuine={ps['n_genuine']})")
    print(f"  APCER={ps['apcer']*100:.1f}% BPCER={ps['bpcer']*100:.1f}% ACER={ps['acer']*100:.1f}% Agreement={ps['agreement']*100:.1f}%")

    # LlamaPartialSpoof (E1 thresholds = deployment scenario)
    print("\nLlamaPartialSpoof (E1 in-domain thresholds):")
    lps = compute_voting(E5_DIR, 'llamapartialspoof_', thresholds_e1)
    print(f"  N={lps['n']} (spoof={lps['n_spoof']}, genuine={lps['n_genuine']})")
    print(f"  APCER={lps['apcer']*100:.1f}% BPCER={lps['bpcer']*100:.1f}% ACER={lps['acer']*100:.1f}% Agreement={lps['agreement']*100:.1f}%")

    # Cross-check with CSV
    print("\nCSV Cross-Check:")
    with open(CSV_DET) as f:
        csv_data = {r['Metric']: r for r in csv.DictReader(f)}

    checks = [
        ('Majority_APCER', 'PartialSpoof', ps['apcer']*100),
        ('Majority_BPCER', 'PartialSpoof', ps['bpcer']*100),
        ('Majority_ACER', 'PartialSpoof', ps['acer']*100),
        ('Majority_APCER', 'LlamaPartialSpoof', lps['apcer']*100),
        ('Majority_BPCER', 'LlamaPartialSpoof', lps['bpcer']*100),
        ('Majority_ACER', 'LlamaPartialSpoof', lps['acer']*100),
    ]
    for metric, ds, computed in checks:
        csv_val = float(csv_data[metric][ds].strip('%'))
        delta = abs(csv_val - computed)
        status = "MATCH" if delta < 0.5 else "MISMATCH"
        if delta >= 0.5: errors.append(f"{metric}/{ds}: csv={csv_val} computed={computed:.1f}")
        print(f"  {metric}/{ds}: csv={csv_val}% computed={computed:.1f}% - {status}")

    print(f"\nERRORS: {len(errors)}")
    for e in errors: print(f"  {e}")
    if not errors: print("  ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
