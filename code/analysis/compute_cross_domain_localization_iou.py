import sys
import numpy as np
import csv
import os

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_BASE, 'lib'))
from xps_forensic.utils.metrics import compute_tIoU, upsample_binary_predictions_to_label_grid

RESULTS_DIR = os.path.join(_BASE, 'data', 'raw_e5_cross_dataset')
OUT_DIR     = os.path.join(_BASE, 'data')
OUT_CSV     = os.path.join(OUT_DIR, 'cross_domain_localization_iou.csv')

FRAME_THRESHOLDS = {
    'bam':   0.9999083280563354,
    'cfprf': 0.3542732000350952,
    'mrm':   0.5038838982582092,
}

FRAME_SHIFTS_MS = {
    'bam':   160.0,
    'cfprf':  20.0,
    'mrm':    20.0,
}
LABEL_FRAME_SHIFT_MS = 10.0

E1_RESULTS_DIR = os.path.join(_BASE, 'data', 'raw_e1_baseline') + os.sep

DETECTORS = ['bam', 'cfprf', 'mrm']
DATASETS  = ['llamapartialspoof', 'partialedit']

def load_object_npy(path):
    return np.load(path, allow_pickle=True)

def compute_iou_per_utterance(frame_scores, frame_labels, threshold, frame_shift_ms):
    pred = (frame_scores >= threshold).astype(int)
    pred_up = upsample_binary_predictions_to_label_grid(
        pred,
        pred_frame_shift_ms=float(frame_shift_ms),
        label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
    )
    min_len = min(len(pred_up), len(frame_labels))
    return compute_tIoU(pred_up[:min_len], frame_labels[:min_len])

rows = []

for det in DETECTORS:
    thresh    = FRAME_THRESHOLDS[det]
    fs_ms     = FRAME_SHIFTS_MS[det]
    for ds in DATASETS:
        print(f'Processing {det.upper()} on {ds} ...')

        frame_scores_path = os.path.join(RESULTS_DIR, f'{det}_{ds}_frame_scores.npy')
        frame_labels_path = os.path.join(RESULTS_DIR, f'{det}_{ds}_frame_labels.npy')
        utt_labels_path   = os.path.join(RESULTS_DIR, f'{det}_{ds}_utt_labels.npy')

        frame_scores_all = load_object_npy(frame_scores_path)
        frame_labels_all = load_object_npy(frame_labels_path)
        utt_labels       = np.load(utt_labels_path)

        n_total = len(utt_labels)
        spoof_idx = np.where(utt_labels == 1)[0]
        n_spoof = len(spoof_idx)
        print(f'  N total={n_total}, N spoof={n_spoof}')

        iou_values = np.zeros(n_spoof, dtype=np.float32)
        for i, idx in enumerate(spoof_idx):
            fs = frame_scores_all[idx].astype(np.float32)
            fl = frame_labels_all[idx].astype(np.int32)
            iou_values[i] = compute_iou_per_utterance(fs, fl, thresh, fs_ms)

        mean_iou   = float(np.mean(iou_values))
        median_iou = float(np.median(iou_values))
        iou_ge_07  = float(np.mean(iou_values >= 0.7)) * 100
        iou_ge_05  = float(np.mean(iou_values >= 0.5)) * 100
        iou_ge_09  = float(np.mean(iou_values >= 0.9)) * 100

        print(f'  mean_IoU={mean_iou:.4f}, median={median_iou:.4f}, '
              f'IoU>=0.7: {iou_ge_07:.1f}%, IoU>=0.5: {iou_ge_05:.1f}%')

        out_npy = os.path.join(OUT_DIR, f'iou_dist_{det}_{ds}_localization.npy')
        np.save(out_npy, iou_values)
        print(f'  Saved: {out_npy}')

        rows.append({
            'Dataset':    ds,
            'Detector':   det.upper(),
            'Condition':  'localization-conditional',
            'N_spoof':    n_spoof,
            'Frame_thresh': thresh,
            'Mean_IoU':   round(mean_iou, 6),
            'Median_IoU': round(median_iou, 6),
            'IoU_ge_0.5_pct': round(iou_ge_05, 2),
            'IoU_ge_0.7_pct': round(iou_ge_07, 2),
            'IoU_ge_0.9_pct': round(iou_ge_09, 2),
        })

print('\nComputing in-domain PartialSpoof localization-conditional IoU ...')
for det in DETECTORS:
    thresh = FRAME_THRESHOLDS[det]
    fs_ms  = FRAME_SHIFTS_MS[det]
    print(f'  Processing {det.upper()} on partialspoof ...')
    frame_scores_all = load_object_npy(E1_RESULTS_DIR + f'{det}_frame_scores.npy')
    frame_labels_all = load_object_npy(E1_RESULTS_DIR + f'{det}_frame_labels.npy')
    utt_labels       = np.load(E1_RESULTS_DIR + f'{det}_utt_labels.npy')

    spoof_idx = np.where(utt_labels == 1)[0]
    n_spoof   = len(spoof_idx)

    iou_values = np.zeros(n_spoof, dtype=np.float32)
    for i, idx in enumerate(spoof_idx):
        fs = frame_scores_all[idx].astype(np.float32)
        fl = frame_labels_all[idx].astype(np.int32)
        iou_values[i] = compute_iou_per_utterance(fs, fl, thresh, fs_ms)

    mean_iou   = float(np.mean(iou_values))
    median_iou = float(np.median(iou_values))
    iou_ge_07  = float(np.mean(iou_values >= 0.7)) * 100
    iou_ge_05  = float(np.mean(iou_values >= 0.5)) * 100
    iou_ge_09  = float(np.mean(iou_values >= 0.9)) * 100

    print(f'    mean_IoU={mean_iou:.4f}, median={median_iou:.4f}, IoU>=0.7: {iou_ge_07:.1f}%')

    out_npy = os.path.join(OUT_DIR, f'iou_dist_{det}_partialspoof_localization.npy')
    np.save(out_npy, iou_values)
    print(f'    Saved: {out_npy}')

    rows.append({
        'Dataset':    'partialspoof',
        'Detector':   det.upper(),
        'Condition':  'localization-conditional',
        'N_spoof':    n_spoof,
        'Frame_thresh': thresh,
        'Mean_IoU':   round(mean_iou, 6),
        'Median_IoU': round(median_iou, 6),
        'IoU_ge_0.5_pct': round(iou_ge_05, 2),
        'IoU_ge_0.7_pct': round(iou_ge_07, 2),
        'IoU_ge_0.9_pct': round(iou_ge_09, 2),
    })

fieldnames = ['Dataset', 'Detector', 'Condition', 'N_spoof', 'Frame_thresh',
              'Mean_IoU', 'Median_IoU', 'IoU_ge_0.5_pct', 'IoU_ge_0.7_pct', 'IoU_ge_0.9_pct']
with open(OUT_CSV, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'\nDone. CSV saved: {OUT_CSV}')
