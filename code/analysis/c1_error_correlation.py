import numpy as np
import json
import csv
from pathlib import Path

_BASE = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_CSV = Path(__file__).resolve().parents[2] / "data" / "error_correlation.csv"

with open(RESULTS_DIR / "raw_e1_baseline" / "results.json") as f:
    e1_json = json.load(f)

THRESHOLDS = {
    "bam":   e1_json["bam"]["utt_eer_threshold"],
    "cfprf": e1_json["cfprf"]["utt_eer_threshold"],
    "mrm":   e1_json["mrm"]["utt_eer_threshold"],
}

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
    "PartialEdit": {
        "bam":   (RESULTS_DIR / "raw_e5_cross_dataset/bam_partialedit_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/bam_partialedit_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "raw_e5_cross_dataset/cfprf_partialedit_utt_scores.npy", RESULTS_DIR / "raw_e5_cross_dataset/cfprf_partialedit_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "raw_e5_cross_dataset/mrm_partialedit_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/mrm_partialedit_utt_labels.npy"),
    },
    "HQ-MPSD": {
        "bam":   (RESULTS_DIR / "raw_e5_cross_dataset/bam_hqmpsd_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/bam_hqmpsd_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "raw_e5_cross_dataset/cfprf_hqmpsd_utt_scores.npy", RESULTS_DIR / "raw_e5_cross_dataset/cfprf_hqmpsd_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "raw_e5_cross_dataset/mrm_hqmpsd_utt_scores.npy",   RESULTS_DIR / "raw_e5_cross_dataset/mrm_hqmpsd_utt_labels.npy"),
    },
}

PAIRS = [("bam", "cfprf"), ("bam", "mrm"), ("cfprf", "mrm")]

def cohen_kappa(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    n = len(pred_a)
    p_obs = float(np.mean(pred_a == pred_b))
    p1_a = float(np.mean(pred_a))
    p1_b = float(np.mean(pred_b))
    p_exp = p1_a * p1_b + (1.0 - p1_a) * (1.0 - p1_b)
    if abs(1.0 - p_exp) < 1e-12:
        return 1.0
    return (p_obs - p_exp) / (1.0 - p_exp)

def run():
    rows = []

    for ds_name, det_paths in DATASETS.items():
        data = {}
        for det, (score_path, label_path) in det_paths.items():
            scores = np.load(score_path)
            labels = np.load(label_path)
            thresh = THRESHOLDS[det]
            preds = (scores >= thresh).astype(np.int8)
            errs  = (preds != labels).astype(np.int8)
            data[det] = {"preds": preds, "errs": errs, "N": len(scores)}

        for det_a, det_b in PAIRS:
            n = min(data[det_a]["N"], data[det_b]["N"])
            truncated = (data[det_a]["N"] != data[det_b]["N"])

            err_a  = data[det_a]["errs"][:n]
            err_b  = data[det_b]["errs"][:n]
            pred_a = data[det_a]["preds"][:n]
            pred_b = data[det_b]["preds"][:n]

            n_err_a    = int(err_a.sum())
            n_err_b    = int(err_b.sum())
            n_both_err = int((err_a & err_b).sum())

            overlap_a_given_b = n_both_err / n_err_b if n_err_b > 0 else float("nan")
            overlap_b_given_a = n_both_err / n_err_a if n_err_a > 0 else float("nan")
            union_err = n_err_a + n_err_b - n_both_err
            jaccard = n_both_err / union_err if union_err > 0 else float("nan")

            kappa = cohen_kappa(pred_a, pred_b)

            row = {
                "Dataset":       ds_name,
                "Detector_A":    det_a.upper(),
                "Detector_B":    det_b.upper(),
                "N":             n,
                "Truncated":     truncated,
                "N_err_A":       n_err_a,
                "N_err_B":       n_err_b,
                "N_both_err":    n_both_err,
                "Pct_err_A":     f"{100*n_err_a/n:.2f}",
                "Pct_err_B":     f"{100*n_err_b/n:.2f}",
                "Overlap_A_given_B_wrong": f"{overlap_a_given_b:.4f}",
                "Overlap_B_given_A_wrong": f"{overlap_b_given_a:.4f}",
                "Jaccard_err":   f"{jaccard:.4f}",
                "Cohen_kappa":   f"{kappa:.4f}",
            }
            rows.append(row)

            trunc_note = f" [TRUNCATED to {n}]" if truncated else ""
            print(
                f"{ds_name:<12} | {det_a.upper()}/{det_b.upper()}{trunc_note}\n"
                f"  N={n}, err_A={n_err_a} ({100*n_err_a/n:.1f}%), "
                f"err_B={n_err_b} ({100*n_err_b/n:.1f}%), "
                f"both_err={n_both_err} ({100*n_both_err/n:.1f}%)\n"
                f"  overlap(A|B_wrong)={overlap_a_given_b:.3f}, "
                f"overlap(B|A_wrong)={overlap_b_given_a:.3f}, "
                f"Jaccard={jaccard:.3f}, kappa={kappa:.4f}"
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")
    print("\n--- INTERPRETATION GUIDE ---")
    print("Cohen's kappa:  <0.2=slight, 0.2-0.4=fair, 0.4-0.6=moderate, 0.6-0.8=substantial, >0.8=near-perfect")
    print("Shared SSL lens hypothesis confirmed if:")
    print("  PartialSpoof kappa is LOW (complementary in-domain errors)")
    print("  LlamaPS kappa is HIGH (correlated cross-domain errors)")

if __name__ == "__main__":
    run()
