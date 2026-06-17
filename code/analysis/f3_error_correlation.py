import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "error_correlation_partial.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

PAIR_ORDER = ["BAM / CFPRF", "BAM / MRM", "CFPRF / MRM"]
DATASET_ORDER = ["PartialSpoof", "LlamaPS", "HQ-MPSD", "PartialEdit"]

df["Pair"] = df["Detector_A"] + " / " + df["Detector_B"]

df["Jaccard_err"] = df["Jaccard_err"].astype(float)

pivot = df.pivot(index="Pair", columns="Dataset", values="Jaccard_err")

pivot = pivot.loc[PAIR_ORDER, DATASET_ORDER]

print("=== Jaccard error overlap matrix (rows=pairs, cols=datasets) ===")
print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print(f"Highest value: {pivot.values.max():.4f}  "
      f"at pair='{pivot.stack().idxmax()[0]}', "
      f"dataset='{pivot.stack().idxmax()[1]}'")
print()

# Fig 3 is single-column (\begin{figure}) but drawn on a 7.0in canvas scaled down
# to \linewidth (~2x reduction). Sizes are therefore set to ~2x those in Fig 1/2
# (the full-width figure* plots) so the printed text matches them: ~7pt labels,
# ~6pt ticks, ~5.5pt cell values.
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        14,
    "axes.labelsize":   14,
    "axes.titlesize":   15,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "lines.linewidth":  0.8,
    "axes.linewidth":   0.6,
})

matrix = pivot.values

fig, ax = plt.subplots(figsize=(7.0, 3.5))

im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=0.7, aspect="auto")

col_labels = list(DATASET_ORDER)
row_labels  = PAIR_ORDER

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, ha="center", va="top")

ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)

ax.tick_params(axis="both", which="both", length=0)

for row_i in range(matrix.shape[0]):
    for col_j in range(matrix.shape[1]):
        val = matrix[row_i, col_j]
        text_color = "white" if val > 0.30 else "black"
        ax.text(
            col_j, row_i,
            f"{val:.3f}",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color=text_color,
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Jaccard error overlap", fontsize=14)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()

pdf_path = OUT_DIR / "error_correlation.pdf"
png_path = OUT_DIR / "error_correlation.png"

fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=150)
plt.close()

print(f"Saved: {pdf_path}  ({pdf_path.stat().st_size:,} bytes)")
print(f"Saved: {png_path}  ({png_path.stat().st_size:,} bytes)")
