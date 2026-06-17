import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_BASE = Path(__file__).resolve().parents[3]
DATA_DIR = _BASE / "xps_forensic" / "deliverables" / "data"
OUT_DIR  = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTORS  = ["bam", "cfprf", "mrm"]
DET_LABELS = {"bam": "BAM", "cfprf": "CFPRF", "mrm": "MRM"}

DATASETS = [
    ("partialspoof",       "PartialSpoof\n(in-domain)"),
    ("llamapartialspoof",  "LlamaPS\n(cross-domain)"),
    ("partialedit",        "PartialEdit\n(cross-domain)"),
]

BINS  = np.linspace(0.0, 1.0, 21)
COLOR = "#4C72B0"

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       7,
    "axes.labelsize":  7,
    "axes.titlesize":  7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 0.8,
    "axes.linewidth":  0.6,
})

fig, axes = plt.subplots(
    nrows=len(DATASETS),
    ncols=len(DETECTORS),
    figsize=(7.0, 4.6),
    sharey=False,
)
fig.subplots_adjust(hspace=0.38, wspace=0.30)

for row_i, (ds_key, ds_label) in enumerate(DATASETS):
    for col_j, det in enumerate(DETECTORS):
        ax = axes[row_i, col_j]

        fpath = DATA_DIR / f"iou_dist_{det}_{ds_key}.npy"
        iou   = np.load(fpath)

        mean_iou   = iou.mean()
        frac_hi    = (iou >= 0.7).mean() * 100.0
        frac_zero  = (iou == 0.0).mean() * 100.0

        print(f"{det:5s} / {ds_key:20s}:  N={len(iou):7d}  "
              f"mean={mean_iou:.3f}  IoU≥0.7={frac_hi:.1f}%  IoU=0={frac_zero:.1f}%")

        ax.hist(iou, bins=BINS, density=True,
                color=COLOR, alpha=0.75, edgecolor="white", linewidth=0.4)

        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([])

        ha  = "left"  if ds_key != "partialedit" else "right"
        xpos = 0.03   if ds_key != "partialedit" else 0.97
        ann = f"mean={mean_iou:.3f}\nIoU≥0.7: {frac_hi:.1f}%"
        ax.text(xpos, 0.94, ann, transform=ax.transAxes,
                ha=ha, va="top", fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.7", alpha=0.85))

        if row_i == 0:
            ax.set_title(DET_LABELS[det], pad=4)
        if col_j == len(DETECTORS) - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ds_label, rotation=270, labelpad=32, va="bottom")
        if row_i == len(DATASETS) - 1:
            ax.set_xlabel("IoU", labelpad=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

pdf_path = OUT_DIR / "iou_distributions.pdf"
png_path = OUT_DIR / "iou_distributions.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=200)
plt.close()

print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")
