import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import json

_BASE = Path(__file__).resolve().parents[3]
RESULTS_DIR = _BASE / "xps_forensic" / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "e1_baseline" / "results.json") as f:
    e1_json = json.load(f)

THRESHOLDS = {
    "bam":   e1_json["bam"]["utt_eer_threshold"],
    "cfprf": e1_json["cfprf"]["utt_eer_threshold"],
    "mrm":   e1_json["mrm"]["utt_eer_threshold"],
}

DETECTORS  = ["bam", "cfprf", "mrm"]
DET_LABELS = {"bam": "BAM", "cfprf": "CFPRF", "mrm": "MRM"}

DATASETS = {
    "PartialSpoof\n(in-domain)": {
        "bam":   (RESULTS_DIR / "e1_baseline/bam_utt_scores.npy",   RESULTS_DIR / "e1_baseline/bam_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "e1_baseline/cfprf_utt_scores.npy", RESULTS_DIR / "e1_baseline/cfprf_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "e1_baseline/mrm_utt_scores.npy",   RESULTS_DIR / "e1_baseline/mrm_utt_labels.npy"),
    },
    "LlamaPS\n(cross-domain)": {
        "bam":   (RESULTS_DIR / "e5_cross_dataset/bam_llamapartialspoof_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/bam_llamapartialspoof_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "e5_cross_dataset/cfprf_llamapartialspoof_utt_scores.npy", RESULTS_DIR / "e5_cross_dataset/cfprf_llamapartialspoof_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "e5_cross_dataset/mrm_llamapartialspoof_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/mrm_llamapartialspoof_utt_labels.npy"),
    },
    "HQ-MPSD\n(cross-domain)": {
        "bam":   (RESULTS_DIR / "e5_cross_dataset/bam_hqmpsd_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/bam_hqmpsd_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "e5_cross_dataset/cfprf_hqmpsd_utt_scores.npy", RESULTS_DIR / "e5_cross_dataset/cfprf_hqmpsd_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "e5_cross_dataset/mrm_hqmpsd_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/mrm_hqmpsd_utt_labels.npy"),
    },
    "PartialEdit\n(cross-domain)": {
        "bam":   (RESULTS_DIR / "e5_cross_dataset/bam_partialedit_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/bam_partialedit_utt_labels.npy"),
        "cfprf": (RESULTS_DIR / "e5_cross_dataset/cfprf_partialedit_utt_scores.npy", RESULTS_DIR / "e5_cross_dataset/cfprf_partialedit_utt_labels.npy"),
        "mrm":   (RESULTS_DIR / "e5_cross_dataset/mrm_partialedit_utt_scores.npy",   RESULTS_DIR / "e5_cross_dataset/mrm_partialedit_utt_labels.npy"),
    },
}

DS_KEYS = list(DATASETS.keys())

BINS = np.arange(0.0, 1.01, 0.01)

COLORS = {"bonafide": "#2196F3", "spoof": "#F44336"}
ALPHA  = 0.55

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        7,
    "axes.labelsize":   7,
    "axes.titlesize":   7.5,
    "xtick.labelsize":  6,
    "ytick.labelsize":  6,
    "lines.linewidth":  0.8,
    "axes.linewidth":   0.6,
})

def _add_composition_cue(ax, n_bona, n_spoof):
    total = n_bona + n_spoof
    if total == 0:
        return
    pct_b = 100.0 * n_bona / total
    pct_s = 100.0 * n_spoof / total
    if n_bona > 0:
        txt = (f"n: {n_bona:,} bona ({pct_b:.0f}%)\n"
               f"{n_spoof:,} spoof ({pct_s:.0f}%)")
    else:
        txt = f"n: {n_spoof:,} spoof (100%)"
    ax.text(0.5, 0.97, txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.0, color="0.2",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

fig, axes = plt.subplots(
    nrows=len(DS_KEYS),
    ncols=len(DETECTORS),
    figsize=(7.0, 3.6),
    sharey=True,
    sharex="col",
)
fig.subplots_adjust(hspace=0.36, wspace=0.30)

import csv as _csv
_LLAMA_TERNARY = {
    r["utterance_id"]: int(r["ternary_label"])
    for r in _csv.DictReader(open(
        _BASE / "partial-deepfake-analysis" / "data" / "llamaps_utt_categories.csv"))
}
_HQMPSD_KEEP = {
    r["utterance_id"]
    for r in _csv.DictReader(open(
        _BASE / "partial-deepfake-analysis" / "data" / "hqmpsd_utt_categories.csv"))
    if r["binary_label"] == "0" or r["is_partial_spoof"] == "1"
}

for row_i, ds_label in enumerate(DS_KEYS):
    for col_j, det in enumerate(DETECTORS):
        ax = axes[row_i, col_j]
        thresh = THRESHOLDS[det]
        score_path, label_path = DATASETS[ds_label][det]

        scores = np.load(score_path)
        labels = np.load(label_path)

        if "llamapartialspoof" in str(score_path):
            ids = np.load(str(score_path).replace("_utt_scores.npy", "_utt_ids.npy"),
                          allow_pickle=True)
            keep = np.array([_LLAMA_TERNARY.get(str(i), 0) in (0, 1) for i in ids])
            scores, labels = scores[keep], labels[keep]

        if "hqmpsd" in str(score_path):
            ids = np.load(str(score_path).replace("_utt_scores.npy", "_utt_ids.npy"),
                          allow_pickle=True)
            keep = np.array([str(i) in _HQMPSD_KEEP for i in ids])
            scores, labels = scores[keep], labels[keep]

        bona_scores  = scores[labels == 0]
        spoof_scores = scores[labels == 1]

        if len(bona_scores) > 0:
            ax.hist(bona_scores,  bins=BINS,
                    weights=np.full(len(bona_scores), 100.0 / len(bona_scores)),
                    color=COLORS["bonafide"], alpha=ALPHA, label="Bonafide",
                    edgecolor="none")
        if len(spoof_scores) > 0:
            ax.hist(spoof_scores, bins=BINS,
                    weights=np.full(len(spoof_scores), 100.0 / len(spoof_scores)),
                    color=COLORS["spoof"],    alpha=ALPHA, label="Spoof",
                    edgecolor="none")

        ax.axvline(thresh, color="black", linestyle="--", linewidth=0.9, alpha=0.8)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelleft=True)

        if row_i == len(DS_KEYS) - 1:
            ax.set_xlabel("Score", labelpad=2)
            ax.tick_params(axis="x", labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if row_i == 0:
            ax.set_title(DET_LABELS[det], pad=4)
        if col_j == 0:
            ax.set_ylabel("% within class")
            _add_composition_cue(ax, len(bona_scores), len(spoof_scores))
        if col_j == len(DETECTORS) - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ds_label, rotation=270, labelpad=28, va="bottom")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D([0], [0], color=COLORS["bonafide"], linewidth=5, alpha=ALPHA, label="Bonafide"),
    Line2D([0], [0], color=COLORS["spoof"],    linewidth=5, alpha=ALPHA, label="Spoof"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7)

pdf_path = OUT_DIR / "score_distributions.pdf"
png_path = OUT_DIR / "score_distributions.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=200)
plt.close()

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
