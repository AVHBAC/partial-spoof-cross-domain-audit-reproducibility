import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

FONT = 'Arial'
FONT_SIZE = 10
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT, 'DejaVu Sans', 'Helvetica'],
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

BAM_C = '#7bafd4'
CFPRF_C = '#f4a582'
MRM_C = '#92c5a9'
DET_COLORS = [BAM_C, CFPRF_C, MRM_C]
DETECTORS = ['BAM', 'CFPRF', 'MRM']
IN_C = '#a6bddb'
CROSS_C = '#fc8d59'
ANNOT_SIZE = FONT_SIZE - 1
APCER_C = '#fc8d59'
BPCER_C = '#91bfdb'
ACER_C = '#bdbdbd'

def read_csv(name):
    with open(DATA_DIR / name) as f:
        return list(csv.DictReader(f))

def annotate_bars(ax, bars, fmt='{:.1f}', offset=0.5, size=ANNOT_SIZE, **kwargs):
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                fmt.format(val), ha='center', va='bottom', fontsize=size, **kwargs)

def fig1_baseline():
    rows = read_csv("e1_baseline_partialspoof.csv")
    det = [r['Detector'] for r in rows]
    eer = [float(r['Utt_EER']) * 100 for r in rows]
    f1 = [float(r['Seg_F1']) * 100 for r in rows]
    iou = [float(r['Mean_IoU']) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    b = axes[0].bar(det, eer, color=DET_COLORS, width=0.55)
    axes[0].set_ylabel('Utt EER (%)')
    axes[0].set_title('Detection')
    annotate_bars(axes[0], b, fmt='{:.2f}%', offset=0.08)

    b = axes[1].bar(det, f1, color=DET_COLORS, width=0.55)
    axes[1].set_ylabel('Seg F1 (%)')
    axes[1].set_title('Localization')
    axes[1].set_ylim(0, 100)
    annotate_bars(axes[1], b, fmt='{:.1f}', offset=1)

    b = axes[2].bar(det, iou, color=DET_COLORS, width=0.55)
    axes[2].set_ylabel('Mean IoU')
    axes[2].set_title('Temporal Overlap')
    axes[2].set_ylim(0, 1)
    axes[2].axhline(0.5, color='#999', ls=':', lw=0.7)
    annotate_bars(axes[2], b, fmt='{:.3f}', offset=0.015)

    fig.suptitle('In-Domain Baseline (PartialSpoof)', fontweight='bold', fontsize=FONT_SIZE)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "baseline.png")
    plt.close()
    print("  baseline.png")

def fig2_cross_detection():
    e5 = read_csv("e5_cross_dataset_detection.csv")
    e1 = read_csv("e1_baseline_partialspoof.csv")

    datasets = ['PartialSpoof']
    bam, cfprf, mrm = [float(e1[0]['Utt_EER'])*100], [float(e1[1]['Utt_EER'])*100], [float(e1[2]['Utt_EER'])*100]
    for r in e5:
        if r['BAM_Utt_EER'] == 'N/A':
            continue
        datasets.append(r['Dataset'])
        bam.append(float(r['BAM_Utt_EER'])*100)
        cfprf.append(float(r['CFPRF_Utt_EER'])*100)
        mrm.append(float(r['MRM_Utt_EER'])*100)

    x = np.arange(len(datasets))
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 3.5))
    b1 = ax.bar(x - w, bam, w, label='BAM', color=BAM_C)
    b2 = ax.bar(x, cfprf, w, label='CFPRF', color=CFPRF_C)
    b3 = ax.bar(x + w, mrm, w, label='MRM', color=MRM_C)
    ax.set_ylabel('Utterance EER (%)')
    ax.set_title('Cross-Dataset Detection EER', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=False)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, fmt='{:.1f}', offset=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cross_detection_eer.png")
    plt.close()
    print("  cross_detection_eer.png")

def fig3_cross_localization():
    e5 = read_csv("e5_cross_dataset_localization.csv")
    e1 = read_csv("e1_baseline_partialspoof.csv")

    datasets = ['PartialSpoof']
    bam = [float(e1[0]['Seg_F1'])*100]
    cfprf = [float(e1[1]['Seg_F1'])*100]
    mrm = [float(e1[2]['Seg_F1'])*100]
    for r in e5:
        if r['BAM_Seg_F1'] == 'N/A':
            continue
        datasets.append(r['Dataset'])
        bam.append(float(r['BAM_Seg_F1'])*100)
        cfprf.append(float(r['CFPRF_Seg_F1'])*100)
        mrm.append(float(r['MRM_Seg_F1'])*100)

    x = np.arange(len(datasets))
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 3.5))
    b1 = ax.bar(x - w, bam, w, label='BAM', color=BAM_C)
    b2 = ax.bar(x, cfprf, w, label='CFPRF', color=CFPRF_C)
    b3 = ax.bar(x + w, mrm, w, label='MRM', color=MRM_C)
    ax.axhline(50, color='#999', ls=':', lw=0.7, label='Random')
    ax.set_ylabel('Segment F1 (%)')
    ax.set_title('Cross-Dataset Localization F1', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, fmt='{:.1f}', offset=1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cross_localization_f1.png")
    plt.close()
    print("  cross_localization_f1.png")

def fig4_voting():
    rows = read_csv("voting_detection.csv")
    data = {r['Metric']: r for r in rows}
    pct = lambda s: float(s.strip('%'))

    metrics = ['Majority_APCER', 'Majority_BPCER', 'Majority_ACER']
    labels = ['APCER', 'BPCER', 'ACER']
    ps = [pct(data[m]['PartialSpoof']) for m in metrics]
    lps = [pct(data[m]['LlamaPartialSpoof']) for m in metrics]

    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    b1 = ax.bar(x - w/2, ps, w, label='PartialSpoof (in-domain)', color=IN_C)
    b2 = ax.bar(x + w/2, lps, w, label='LlamaPS (cross-domain)', color=CROSS_C)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Majority Voting: APCER / BPCER / ACER', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    annotate_bars(ax, b1, fmt='{:.1f}%', offset=0.5)
    annotate_bars(ax, b2, fmt='{:.1f}%', offset=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "voting_apcer_bpcer.png")
    plt.close()
    print("  voting_apcer_bpcer.png")

def fig5_voting_iou():
    rows = read_csv("voting_localization.csv")
    data = {r['Metric']: r for r in rows}

    labels = ['Mean IoU', 'Median IoU']
    ps = [float(data[m]['PartialSpoof']) for m in ['MajVote_IoU_mean', 'MajVote_IoU_median']]
    lps = [float(data[m]['LlamaPartialSpoof']) for m in ['MajVote_IoU_mean', 'MajVote_IoU_median']]

    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    b1 = ax.bar(x - w/2, ps, w, label='PartialSpoof', color=IN_C)
    b2 = ax.bar(x + w/2, lps, w, label='LlamaPS', color=CROSS_C)
    ax.axhline(0.5, color='#999', ls=':', lw=0.7)
    ax.set_ylabel('IoU')
    ax.set_title('Majority Vote Localization IoU', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False)

    annotate_bars(ax, b1, fmt='{:.3f}', offset=0.01)
    annotate_bars(ax, b2, fmt='{:.3f}', offset=0.01)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "voting_iou.png")
    plt.close()
    print("  voting_iou.png")

def fig6_afnext_apcer_bpcer():
    rows = read_csv("afnext_main_results.csv")
    data = {r['Metric']: r for r in rows}

    modes = ['Blind', 'Guided_BAM', 'Guided_CFPRF', 'Guided_MRM']
    labels = ['Blind', '+BAM', '+CFPRF', '+MRM']
    apcer = [float(data['APCER'][m]) * 100 for m in modes]
    bpcer = [float(data['BPCER'][m]) * 100 for m in modes]
    acer = [float(data['ACER'][m]) * 100 for m in modes]

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 3.5))
    b1 = ax.bar(x - w, apcer, w, label='APCER', color=APCER_C)
    b2 = ax.bar(x, bpcer, w, label='BPCER', color=BPCER_C)
    b3 = ax.bar(x + w, acer, w, label='ACER', color=ACER_C)
    ax.axhline(50, color='#999', ls=':', lw=0.7)
    ax.set_ylabel('Rate (%)')
    ax.set_title('AF-Next: APCER / BPCER / ACER (N=300)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, ncol=3)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, fmt='{:.1f}', offset=1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "afnext_apcer_bpcer.png")
    plt.close()
    print("  afnext_apcer_bpcer.png")

def fig7_afnext_bias():
    rows = read_csv("afnext_main_results.csv")
    data = {r['Metric']: r for r in rows}

    modes = ['Blind', 'Guided_BAM', 'Guided_CFPRF', 'Guided_MRM']
    labels = ['Blind', '+BAM', '+CFPRF', '+MRM']
    br = [float(data['Boundary_Recall'][m]) * 100 for m in modes]
    iou = [float(data['Temporal_IoU'][m]) for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    colors_br = ['#fc8d59'] + ['#92c5a9'] * 3
    b1 = ax1.bar(labels, br, color=colors_br, width=0.55)
    ax1.set_ylabel('Boundary Recall (%)')
    ax1.set_title('Boundary Recall (\u00b1500ms)', fontweight='bold')
    ax1.set_ylim(0, 100)
    annotate_bars(ax1, b1, fmt='{:.1f}%', offset=1)

    colors_iou = ['#fc8d59'] + ['#fdd49e'] * 3
    b2 = ax2.bar(labels, iou, color=colors_iou, width=0.55)
    ax2.axhline(0.5, color='#999', ls=':', lw=0.7)
    ax2.set_ylabel('Temporal IoU')
    ax2.set_title('Temporal IoU', fontweight='bold')
    ax2.set_ylim(0, 0.6)
    annotate_bars(ax2, b2, fmt='{:.3f}', offset=0.01)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "afnext_bias_iou.png")
    plt.close()
    print("  afnext_bias_iou.png")

def fig8_afnext_confusion():
    rows = read_csv("afnext_confusion_matrix.csv")

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, row in zip(axes, rows):
        mode = row['Mode'].replace('_', ' ')
        tp, fn, tn, fp = int(row['TP']), int(row['FN']), int(row['TN']), int(row['FP'])
        matrix = np.array([[tn, fp], [fn, tp]])
        ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=225, aspect='equal')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Genuine', 'Spoof'], fontsize=FONT_SIZE - 1)
        ax.set_yticklabels(['Genuine', 'Spoof'], fontsize=FONT_SIZE - 1)
        if ax == axes[0]:
            ax.set_ylabel('Ground Truth')
        ax.set_xlabel('Predicted')
        ax.set_title(mode, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        fontsize=FONT_SIZE + 1, fontweight='bold',
                        color='white' if matrix[i, j] > 150 else 'black')

    fig.suptitle('AF-Next Confusion Matrices (N=300)', fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "afnext_confusion.png")
    plt.close()
    print("  afnext_confusion.png")

def fig9_hqmpsd():
    rows = read_csv("hqmpsd_results.csv")
    det = [r['Detector'] for r in rows]
    eer = [float(r['Utt_EER']) * 100 for r in rows]
    f1 = [float(r['Utt_F1']) * 100 for r in rows]
    auc = [float(r['Utt_ROC_AUC']) * 100 for r in rows]

    x = np.arange(len(det))
    w = 0.22

    fig, ax = plt.subplots(figsize=(6, 3.5))
    b1 = ax.bar(x - w, eer, w, label='EER (%)', color=APCER_C)
    b2 = ax.bar(x, f1, w, label='F1 (%)', color=MRM_C)
    b3 = ax.bar(x + w, auc, w, label='AUC (%)', color=BPCER_C)
    ax.set_ylabel('Value (%)')
    ax.set_title('HQ-MPSD Utterance-Level Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(det)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, fmt='{:.1f}', offset=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "hqmpsd.png")
    plt.close()
    print("  hqmpsd.png")

def fig10_partialedit():
    rows = read_csv("partialedit_results.csv")
    det = [r['Detector'] for r in rows]
    seg_f1 = [float(r['Seg_F1']) * 100 for r in rows]
    prec = [float(r['Frame_Precision']) * 100 for r in rows]
    rec = [float(r['Frame_Recall']) * 100 for r in rows]

    x = np.arange(len(det))
    w = 0.22

    fig, ax = plt.subplots(figsize=(6, 3.5))
    b1 = ax.bar(x - w, seg_f1, w, label='Seg F1', color=BAM_C)
    b2 = ax.bar(x, prec, w, label='Precision', color=CFPRF_C)
    b3 = ax.bar(x + w, rec, w, label='Recall', color=MRM_C)
    ax.axhline(50, color='#999', ls=':', lw=0.7)
    ax.set_ylabel('Value (%)')
    ax.set_title('PartialEdit Frame-Level Localization', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(det)
    ax.set_ylim(0, 75)
    ax.legend(frameon=False)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, fmt='{:.1f}', offset=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "partialedit.png")
    plt.close()
    print("  partialedit.png")

def fig11_seg_eer():
    rows = read_csv("seg_eer_verified.csv")

    datasets = ['PartialSpoof', 'LlamaPartialSpoof', 'PartialEdit']

    data = {}
    for row in rows:
        ds, det = row['Dataset'], row['Detector']
        val = row['Seg_EER_pct']
        data[(ds, det)] = float(val) if val != 'N/A' else None

    x = np.arange(len(datasets))
    w = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for j, (det, color) in enumerate(zip(DETECTORS, DET_COLORS)):
        vals = [data.get((ds, det)) for ds in datasets]
        positions = x + (j - 1) * w
        bars = ax.bar(positions, [v if v is not None else 0 for v in vals],
                      w, label=det, color=color)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=ANNOT_SIZE)

    ax.axhline(50, color='#999', ls=':', lw=0.7, label='Random (50%)')
    ax.set_ylabel('Segment EER (%)')
    ax.set_title('Segment-Level EER at Native Resolution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 65)
    ax.legend(frameon=False, loc='upper left')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "seg_eer_comparison.png")
    plt.close()
    print("  seg_eer_comparison.png")

def fig12_boundary_metrics():
    rows = read_csv("boundary_metrics.csv")

    datasets = ['PartialSpoof', 'LlamaPartialSpoof', 'PartialEdit']
    metrics = ['Boundary_Precision_pct', 'Boundary_Recall_pct', 'Boundary_F1_pct']
    metric_labels = ['Precision', 'Recall', 'F1']

    data = {}
    for row in rows:
        ds, det = row['Dataset'], row['Detector']
        for m in metrics:
            data[(ds, det, m)] = float(row[m])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (m, m_label) in zip(axes, zip(metrics, metric_labels)):
        x = np.arange(len(datasets))
        w = 0.22

        for j, (det, color) in enumerate(zip(DETECTORS, DET_COLORS)):
            vals = [data.get((ds, det, m), 0) for ds in datasets]
            positions = x + (j - 1) * w
            bars = ax.bar(positions, vals, w, label=det, color=color)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=ANNOT_SIZE)

        ax.set_ylabel(f'{m_label} (%)')
        ax.set_title(f'Boundary {m_label}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=FONT_SIZE - 1)
        ax.set_ylim(0, 105)
        if ax == axes[0]:
            ax.legend(frameon=False, loc='upper right')

    fig.suptitle('Boundary Detection Metrics (tolerance = ±500ms)',
                 fontweight='bold', fontsize=FONT_SIZE)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "boundary_metrics.png")
    plt.close()
    print("  boundary_metrics.png")

def fig13_iou():
    rows = read_csv("iou_metrics.csv")

    datasets = ['PartialSpoof', 'LlamaPartialSpoof', 'PartialEdit']
    metrics = ['Mean_IoU', 'Median_IoU']
    metric_labels = ['Mean IoU', 'Median IoU']

    data = {}
    for row in rows:
        ds, det = row['Dataset'], row['Detector']
        for m in metrics:
            data[(ds, det, m)] = float(row[m])
        for m in ['IoU_ge_0.5_pct', 'IoU_ge_0.7_pct', 'IoU_ge_0.9_pct']:
            data[(ds, det, m)] = float(row[m])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (m, m_label) in zip(axes[:2], zip(metrics, metric_labels)):
        x = np.arange(len(datasets))
        w = 0.22
        for j, (det, color) in enumerate(zip(DETECTORS, DET_COLORS)):
            vals = [data.get((ds, det, m), 0) for ds in datasets]
            positions = x + (j - 1) * w
            bars = ax.bar(positions, vals, w, label=det, color=color)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=ANNOT_SIZE)

        ax.axhline(0.5, color='#999', ls=':', lw=0.7)
        ax.set_ylabel(m_label)
        ax.set_title(m_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=FONT_SIZE - 1)
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.legend(frameon=False, loc='upper right')

    ax3 = axes[2]
    x = np.arange(len(datasets))
    w = 0.22
    thresh_metrics = ['IoU_ge_0.5_pct', 'IoU_ge_0.7_pct', 'IoU_ge_0.9_pct']
    thresh_labels = ['>=0.5', '>=0.7', '>=0.9']
    thresh_colors = ['#a6bddb', '#3690c0', '#034e7b']

    for j, (tm, tl, tc) in enumerate(zip(thresh_metrics, thresh_labels, thresh_colors)):
        vals = [data.get((ds, 'BAM', tm), 0) for ds in datasets]
        bars = ax3.bar(x + (j - 1) * w, vals, w, label=f'BAM {tl}', color=tc)
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                     f'{val:.0f}', ha='center', va='bottom',
                     fontsize=ANNOT_SIZE)

    ax3.set_ylabel('Utterances (%)')
    ax3.set_title('BAM: IoU Threshold', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontsize=FONT_SIZE - 1)
    ax3.set_ylim(0, 105)
    ax3.legend(frameon=False, loc='upper right', fontsize=FONT_SIZE - 2)

    fig.suptitle('Temporal IoU (frame-level overlap)',
                 fontweight='bold', fontsize=FONT_SIZE)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "iou_metrics.png")
    plt.close()
    print("  iou_metrics.png")

def fig14_iou_distribution():
    datasets = ['PartialSpoof', 'LlamaPartialSpoof', 'PartialEdit']
    ds_keys = ['partialspoof', 'llamapartialspoof', 'partialedit']
    detectors_lc = ['bam', 'cfprf', 'mrm']

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))

    global_ymax = 0
    hist_data = {}
    bins = np.linspace(0, 1, 51)

    for i, (ds, ds_key) in enumerate(zip(datasets, ds_keys)):
        for j, det in enumerate(detectors_lc):
            arr = np.load(DATA_DIR / f"iou_dist_{det}_{ds_key}.npy")
            counts, _ = np.histogram(arr, bins=bins)
            density = counts / len(arr) * 100
            hist_data[(i, j)] = (arr, density)
            global_ymax = max(global_ymax, density.max())

    global_ymax = int(np.ceil(global_ymax / 5) * 5) + 2

    for i, (ds, ds_key) in enumerate(zip(datasets, ds_keys)):
        for j, (det, color) in enumerate(zip(detectors_lc, DET_COLORS)):
            ax = axes[i][j]
            arr, density = hist_data[(i, j)]

            ax.bar(bins[:-1], density, width=bins[1] - bins[0],
                   align='edge', color=color, alpha=0.85, edgecolor='white',
                   linewidth=0.3)

            mean_val = np.mean(arr)
            median_val = np.median(arr)
            ax.axvline(mean_val, color='#333333', ls='-', lw=1.2,
                       label=f'Mean={mean_val:.2f}')
            ax.axvline(median_val, color='#333333', ls='--', lw=1.0,
                       label=f'Median={median_val:.2f}')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, global_ymax)

            if i == 2:
                ax.set_xlabel('IoU')
            if j == 0:
                ax.set_ylabel('Utterances (%)')

            if i == 0:
                ax.set_title(det.upper(), fontweight='bold')

            ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper left')

    for i, ds in enumerate(datasets):
        axes[i][2].annotate(ds, xy=(1.08, 0.5), xycoords='axes fraction',
                            fontsize=FONT_SIZE, fontweight='bold', rotation=-90,
                            ha='left', va='center')

    fig.suptitle('IoU Score Distribution by Detector and Dataset',
                 fontweight='bold', fontsize=FONT_SIZE + 1, y=0.99)
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    fig.savefig(FIG_DIR / "iou_distribution.png")
    plt.close()
    print("  iou_distribution.png")

def fig15_voting_agreement():
    utt_rows = read_csv("voting_utterance_agreement.csv")
    seg_rows = read_csv("voting_segment_agreement.csv")

    datasets = ['PartialSpoof', 'LlamaPartialSpoof', 'PartialEdit']
    categories = ['Only_BAM', 'Only_CFPRF', 'Only_MRM',
                   'BAM_CFPRF', 'BAM_MRM', 'CFPRF_MRM', 'All_three', 'None_flagged']
    cat_labels = ['BAM\nonly', 'CFPRF\nonly', 'MRM\nonly',
                  'BAM+\nCFPRF', 'BAM+\nMRM', 'CFPRF+\nMRM', 'All\nthree', 'None']
    cat_colors = [BAM_C, CFPRF_C, MRM_C,
                  '#d4a0c4', '#7bc8c4', '#c4b896', '#999999', '#e0e0e0']

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for col, ds in enumerate(datasets):
        utt = [r for r in utt_rows if r['Dataset'] == ds][0]
        n_utt = int(utt['N_utterances'])
        utt_vals = [int(utt[c]) / n_utt * 100 for c in categories]

        ax_u = axes[0][col]
        bars = ax_u.bar(range(len(categories)), utt_vals, color=cat_colors)
        for bar, val in zip(bars, utt_vals):
            if val >= 1.0:
                ax_u.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        ax_u.set_xticks(range(len(categories)))
        ax_u.set_xticklabels(cat_labels, fontsize=7)
        ax_u.set_title(ds, fontweight='bold')
        ax_u.set_ylim(0, 100)
        if col == 0:
            ax_u.set_ylabel('Utterances (%)')

        seg = [r for r in seg_rows if r['Dataset'] == ds][0]
        n_seg = int(seg['N_segments'])
        seg_vals = [int(seg[c]) / n_seg * 100 for c in categories]

        ax_s = axes[1][col]
        bars = ax_s.bar(range(len(categories)), seg_vals, color=cat_colors)
        for bar, val in zip(bars, seg_vals):
            if val >= 1.0:
                ax_s.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        ax_s.set_xticks(range(len(categories)))
        ax_s.set_xticklabels(cat_labels, fontsize=7)
        ax_s.set_ylim(0, 100)
        if col == 0:
            ax_s.set_ylabel('Segments (%)')

    axes[0][0].annotate('Utterance\nLevel', xy=(-0.35, 0.5),
                         xycoords='axes fraction', fontsize=FONT_SIZE,
                         fontweight='bold', ha='right', va='center')
    axes[1][0].annotate('Segment\nLevel', xy=(-0.35, 0.5),
                         xycoords='axes fraction', fontsize=FONT_SIZE,
                         fontweight='bold', ha='right', va='center')

    fig.suptitle('Detector Agreement: Which Detectors Flag Spoof?',
                 fontweight='bold', fontsize=FONT_SIZE + 1, y=0.99)
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "voting_agreement.png")
    plt.close()
    print("  voting_agreement.png")

if __name__ == "__main__":
    print(f"Data: {DATA_DIR}")
    print(f"Output: {FIG_DIR}")
    print(f"Font: {FONT} {FONT_SIZE}pt\n")

    fig1_baseline()
    fig2_cross_detection()
    fig3_cross_localization()
    fig4_voting()
    fig5_voting_iou()
    fig6_afnext_apcer_bpcer()
    fig7_afnext_bias()
    fig8_afnext_confusion()
    fig9_hqmpsd()
    fig10_partialedit()
    fig11_seg_eer()
    fig12_boundary_metrics()
    fig13_iou()
    fig14_iou_distribution()
    fig15_voting_agreement()

    print(f"\n15 figures saved to {FIG_DIR}/")
