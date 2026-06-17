import numpy as np
import json
import csv
from pathlib import Path

_BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = _BASE / "data"
OUT_CSV     = _BASE / "data" / "far_operating_points.csv"

with open(RESULTS_DIR / "raw_e1_baseline" / "results.json") as f:
    e1_json = json.load(f)

DETECTORS = ["bam", "cfprf", "mrm"]

DATASETS = {
    "PartialSpoof": {
        "bam":   (RESULTS_DIR / "raw_e1_baseline/bam_utt_scores.npy",   RESULTS_DIR / "raw_e1_baseline/bam_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "raw_e1_baseline/cfprf_utt_scores.npy", RESULTS_DIR / "raw_e1_baseline/cfprf_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "raw_e1_baseline/mrm_utt_scores.npy",   RESULTS_DIR / "raw_e1_baseline/mrm_utt_labels.npy"),
    },
    "LlamaPS": {
        "bam":   (RESULTS_DIR / "raw_e5_cross_dataset/bam_llamapartialspoof_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/bam_llamapartialspoof_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "raw_e5_cross_dataset/cfprf_llamapartialspoof_utt_scores.npy", RESULTS_DIR / "raw_e5_cross_dataset/cfprf_llamapartialspoof_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "raw_e5_cross_dataset/mrm_llamapartialspoof_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/mrm_llamapartialspoof_utt_labels.npy"),
    },
    "HQ-MPSD": {
        "bam":   (RESULTS_DIR / "raw_e5_cross_dataset/bam_hqmpsd_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/bam_hqmpsd_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "raw_e5_cross_dataset/cfprf_hqmpsd_utt_scores.npy", RESULTS_DIR / "raw_e5_cross_dataset/cfprf_hqmpsd_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "raw_e5_cross_dataset/mrm_hqmpsd_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/mrm_hqmpsd_utt_labels.npy"),
    },
}

FAR_TARGET = 0.01

def metrics_at_threshold(scores, labels, thresh):
    preds     = (scores >= thresh).astype(int)
    n_genuine = int((labels == 0).sum())
    n_spoof   = int((labels == 1).sum())
    fp        = int(((preds == 1) & (labels == 0)).sum())
    fn        = int(((preds == 0) & (labels == 1)).sum())
    bpcer     = fp / n_genuine if n_genuine > 0 else float("nan")
    apcer     = fn / n_spoof   if n_spoof   > 0 else float("nan")
    acer      = (bpcer + apcer) / 2
    return {
        "N_genuine": n_genuine, "N_spoof": n_spoof,
        "FP": fp, "FN": fn,
        "BPCER": bpcer, "APCER": apcer, "ACER": acer,
    }

def run():
    far_thresholds = {}
    for det in DETECTORS:
        s, l = [np.load(p) for p in DATASETS["PartialSpoof"][det]]
        genuine_scores = s[l == 0]
        far_thresholds[det] = float(np.percentile(genuine_scores, 100 * (1 - FAR_TARGET)))

    eer_thresholds = {det: e1_json[det]["utt_eer_threshold"] for det in DETECTORS}

    print(f"{'Detector':<8} {'EER thresh':>12} {'FAR=1% thresh':>14}")
    for det in DETECTORS:
        print(f"{det.upper():<8} {eer_thresholds[det]:>12.6f} {far_thresholds[det]:>14.6f}")
    print()

    rows = []
    for ds_name, det_paths in DATASETS.items():
        for det in DETECTORS:
            s, l = [np.load(p) for p in det_paths[det]]

            eer_thresh = eer_thresholds[det]
            far_thresh = far_thresholds[det]

            m_eer = metrics_at_threshold(s, l, eer_thresh)
            m_far = metrics_at_threshold(s, l, far_thresh)

            row = {
                "Dataset":          ds_name,
                "Detector":         det.upper(),
                "N_genuine":        m_eer["N_genuine"],
                "N_spoof":          m_eer["N_spoof"],
                "EER_threshold":    f"{eer_thresh:.6f}",
                "EER_BPCER":        f"{100*m_eer['BPCER']:.2f}",
                "EER_APCER":        f"{100*m_eer['APCER']:.2f}",
                "EER_ACER":         f"{100*m_eer['ACER']:.2f}",
                "FAR1_threshold":   f"{far_thresh:.6f}",
                "FAR1_BPCER":       f"{100*m_far['BPCER']:.2f}",
                "FAR1_APCER":       f"{100*m_far['APCER']:.2f}",
                "FAR1_ACER":        f"{100*m_far['ACER']:.2f}",
            }
            rows.append(row)

            print(f"{ds_name:<14} {det.upper():<6} "
                  f"EER→ BPCER={m_eer['BPCER']*100:.1f}% APCER={m_eer['APCER']*100:.1f}%  |  "
                  f"FAR=1%→ BPCER={m_far['BPCER']*100:.1f}% APCER={m_far['APCER']*100:.1f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

if __name__ == "__main__":
    run()
