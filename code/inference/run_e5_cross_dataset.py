"""E5: Cross-Dataset Generalization.

Run available detectors on cross-domain datasets (LlamaPartialSpoof,
PartialEdit, HQ-MPSD-EN) to evaluate generalization under domain shift.

Protocol mirrors E1:
  - Batched GPU inference for throughput.
  - Seg-EER computed by pooling ALL segments across utterances, then EER.
    (Not per-utterance average - that is the wrong protocol, see BUG-E1-A2.)
  - Frame-level EER threshold computed on label grid (10ms), then used for
    binary metrics (F1, accuracy, precision, recall).
  - Resolutions below detector native are skipped (no score interpolation).

For PartialEdit (all-spoof utterances, no bonafide class):
  - Utt-EER is skipped (single class).
  - Segment-level metrics are valid because frame labels have both classes
    (edited regions = 1, unedited = 0 within each utterance).
  - Additional metrics: accuracy, precision, recall, ROC-AUC at segment level.
  - Reference: Zhang et al. (Interspeech 2025) used frame-level EER at 20ms.

Output layout (under results/e5_cross_dataset/):
    results.json                          - all computed metrics
    {det}_{ds}_utt_scores.npy             - utterance-level scores
    {det}_{ds}_utt_labels.npy             - utterance-level binary labels
    {det}_{ds}_frame_scores.npy           - per-utterance frame score arrays
    {det}_{ds}_frame_labels.npy           - per-utterance frame label arrays
    {det}_{ds}_utt_ids.npy                - utterance IDs

Reference: Tibshirani et al. (NeurIPS 2019) for covariate-shift CP context.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_f1,
    upsample_binary_predictions_to_label_grid,
    _pool_scores_to_windows,
    _pool_labels_to_windows,
)
from xps_forensic.data.partialedit import PartialEditDataset
from xps_forensic.data.hqmpsd import HQMPSDDataset
from xps_forensic.data.llamapartialspoof import LlamaPartialSpoofDataset
from xps_forensic.detectors.bam import BAMDetector
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector

logger = logging.getLogger(__name__)

DETECTOR_MAP = {
    "bam": BAMDetector,
    "sal": SALDetector,
    "cfprf": CFPRFDetector,
    "mrm": MRMDetector,
}

# Label frame shift in all cross-datasets (10 ms).
LABEL_FRAME_SHIFT_MS = 10.0

# Cross-dataset registry: (config_key, DatasetClass, constructor_kwargs_fn)
_CROSS_DATASETS = [
    (
        "llamapartialspoof",
        LlamaPartialSpoofDataset,
        lambda cfg: {"root": cfg.data.llamapartialspoof.path},
    ),
    (
        "partialedit",
        PartialEditDataset,
        lambda cfg: {"root": cfg.data.partialedit.path},
    ),
    (
        "hqmpsd",
        HQMPSDDataset,
        lambda cfg: {
            "root": cfg.data.hqmpsd.path,
            "language": cfg.data.hqmpsd.get("language", "en"),
        },
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="E5: Cross-Dataset Generalization")
    parser.add_argument('--detector', type=str, default=None,
                        choices=['bam', 'sal', 'cfprf', 'mrm'],
                        help='Run single detector. If omitted, runs all.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run single dataset. If omitted, runs all available.')
    return parser.parse_args()


def _build_detector(det_name: str, det_cfg: dict, device: str):
    """Instantiate a detector from config, returning None if checkpoint missing."""
    DetClass = DETECTOR_MAP[det_name]
    checkpoint = det_cfg.get("checkpoint")

    if checkpoint is None:
        logger.warning("Detector %s: checkpoint is null - skipping.", det_name.upper())
        return None

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        logger.warning("Detector %s: checkpoint not found at %s - skipping.",
                        det_name.upper(), ckpt_path)
        return None

    kwargs: dict = {"checkpoint": str(ckpt_path), "device": device}

    external_dir = det_cfg.get("external_dir")
    if external_dir is not None:
        kwargs["external_dir"] = external_dir

    ssl_ckpt = det_cfg.get("ssl_ckpt")
    ssl_path = det_cfg.get("ssl_path")

    if det_name == "bam":
        if ssl_ckpt:
            kwargs["ssl_ckpt"] = ssl_ckpt
    elif det_name == "sal":
        if ssl_ckpt:
            kwargs["ssl_ckpt"] = ssl_ckpt
    elif det_name == "cfprf":
        if ssl_path:
            kwargs["ssl_path"] = ssl_path
    elif det_name == "mrm":
        if ssl_path:
            kwargs["ssl_path"] = ssl_path

    try:
        detector = DetClass(**kwargs)
        detector.load_model()
        return detector
    except Exception as exc:
        logger.warning("Detector %s: failed to load - %s. Skipping.", det_name.upper(), exc)
        return None


def _json_serializer(obj):
    """JSON default serializer for numpy / tuple types."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_cross_datasets(cfg, filter_dataset: str | None = None) -> dict:
    """Load available cross-domain datasets, skipping those without data."""
    datasets: dict = {}
    for ds_name, DatasetClass, kwargs_fn in _CROSS_DATASETS:
        if filter_dataset is not None and ds_name != filter_dataset:
            continue
        try:
            kwargs = kwargs_fn(cfg)
        except AttributeError:
            logger.info("Dataset %s: config entry missing - skipping.", ds_name)
            continue

        root = Path(kwargs["root"])
        if not root.exists():
            logger.info("Dataset %s: path %s does not exist - skipping.", ds_name, root)
            continue

        try:
            ds = DatasetClass(**kwargs)
        except Exception as exc:
            logger.warning("Dataset %s: failed to instantiate - %s. Skipping.", ds_name, exc)
            continue

        n = len(ds)
        if n == 0:
            logger.info("Dataset %s: 0 utterances - skipping.", ds_name)
            continue

        datasets[ds_name] = ds
        print(f"  Loaded {ds_name}: {n} utterances")

    return datasets


def _compute_partialedit_segment_metrics(
    all_frame_scores: list[np.ndarray],
    all_frame_labels: list[np.ndarray],
    det_frame_shift_ms: float,
) -> dict:
    """Compute segment-level accuracy, precision, recall, ROC-AUC for PartialEdit.

    PartialEdit has no bonafide utterances, so utterance-level EER is undefined.
    Frame labels have both classes (edited=1, unedited=0), so segment-level
    metrics are valid.

    Uses the frame-level EER threshold (computed on the 10ms label grid) for
    binary metrics. ROC-AUC uses continuous scores directly.

    Reference: Zhang et al. (Interspeech 2025) used frame-level EER at 20ms.
    """
    # Upsample scores to 10ms label grid and compute frame-level EER threshold
    all_scores_at_grid: list[float] = []
    all_labels_at_grid: list[int] = []
    for fs, fl in zip(all_frame_scores, all_frame_labels):
        if len(fs) == 0 or len(fl) == 0:
            continue
        fs_up = upsample_binary_predictions_to_label_grid(
            fs, pred_frame_shift_ms=det_frame_shift_ms,
            label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
        )
        min_len = min(len(fs_up), len(fl))
        all_scores_at_grid.extend(fs_up[:min_len].tolist())
        all_labels_at_grid.extend(fl[:min_len].tolist())

    scores_arr = np.array(all_scores_at_grid)
    labels_arr = np.array(all_labels_at_grid)

    if len(scores_arr) == 0 or len(np.unique(labels_arr)) < 2:
        return {}

    frame_eer, frame_thresh = compute_eer(scores_arr, labels_arr)
    preds = (scores_arr > frame_thresh).astype(int)

    metrics = {
        "frame_eer": float(frame_eer),
        "frame_eer_threshold": float(frame_thresh),
        "n_frames": len(scores_arr),
        "spoof_frame_ratio": float(labels_arr.mean()),
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "precision": float(precision_score(labels_arr, preds, zero_division=0.0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0.0)),
        "f1": float(f1_score(labels_arr, preds, zero_division=0.0)),
        "roc_auc": float(roc_auc_score(labels_arr, scores_arr)),
    }
    return metrics


def run_e5(cfg=None, filter_detector: str | None = None, filter_dataset: str | None = None):
    """Run E5 cross-dataset generalization experiment.

    For each available detector x dataset:
      1. Batched GPU inference.
      2. Save all raw scores (.npy) for future analysis.
      3. Compute metrics (protocol depends on dataset).

    Returns dict of all results.
    """
    if cfg is None:
        cfg = load_config()

    seed = cfg.get('seed', 42) if hasattr(cfg, 'get') else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    output_dir = Path(cfg.experiments.output_dir) / "e5_cross_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolutions = cfg.experiments.resolutions_ms

    # ── Load cross-domain datasets ────────────────────────────────
    print("Loading cross-domain datasets...")
    datasets = _load_cross_datasets(cfg, filter_dataset=filter_dataset)

    if not datasets:
        logger.warning("No cross-domain datasets available.")
        return {}

    # ── Build detectors ───────────────────────────────────────────
    det_names = [filter_detector] if filter_detector else ["bam", "cfprf", "mrm"]
    det_dict: dict = {}
    for det_name in det_names:
        det_cfg = cfg.detectors.get(det_name, {})
        if not det_cfg:
            continue
        detector = _build_detector(det_name, det_cfg, cfg.device)
        if detector is not None:
            det_dict[det_name] = detector

    if not det_dict:
        logger.warning("No detectors available.")
        return {}

    # ── Main evaluation loop ──────────────────────────────────────
    all_results: dict = {}
    BATCH_SIZE = 16

    for ds_name, dataset in datasets.items():
        n_utterances = len(dataset)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({n_utterances} utterances)")
        print(f"{'='*60}")

        ds_results: dict = {"n_utterances": n_utterances}

        for det_name, detector in det_dict.items():
            print(f"\n  Detector: {det_name.upper()}")

            det_frame_shift_ms = float(detector.frame_shift_ms)
            print(f"    Frame shift: {det_frame_shift_ms} ms")

            # ── Batched inference ─────────────────────────────────
            all_utt_ids: list[str] = []
            all_utt_scores: list[float] = []
            all_utt_labels: list[int] = []
            all_frame_scores: list[np.ndarray] = []
            all_frame_labels: list[np.ndarray] = []

            batch_wavs, batch_ids, batch_labels, batch_frame_labels = [], [], [], []
            t0 = time.time()
            n_errors = 0
            n_total_processed = 0

            def _flush_batch():
                nonlocal n_errors
                if not batch_wavs:
                    return
                try:
                    outputs = detector.predict_batch(
                        batch_wavs, batch_ids,
                        sample_rate=dataset.sample_rate,
                    )
                    for output, lbl, fl in zip(outputs, batch_labels, batch_frame_labels):
                        if np.isnan(output.utterance_score) or np.any(np.isnan(output.frame_scores)):
                            n_errors += 1
                            continue
                        all_utt_ids.append(output.utterance_id)
                        all_utt_scores.append(output.utterance_score)
                        all_utt_labels.append(lbl)
                        all_frame_scores.append(output.frame_scores)
                        all_frame_labels.append(fl)
                except Exception:
                    # Fallback: single-sample inference
                    for wav, uid, lbl, fl in zip(batch_wavs, batch_ids, batch_labels, batch_frame_labels):
                        try:
                            output = detector.predict(wav, dataset.sample_rate, utterance_id=uid)
                            if np.isnan(output.utterance_score) or np.any(np.isnan(output.frame_scores)):
                                n_errors += 1
                                continue
                            all_utt_ids.append(uid)
                            all_utt_scores.append(output.utterance_score)
                            all_utt_labels.append(lbl)
                            all_frame_scores.append(output.frame_scores)
                            all_frame_labels.append(fl)
                        except Exception:
                            n_errors += 1

            for idx in range(n_utterances):
                try:
                    sample = dataset[idx]
                    batch_wavs.append(sample.waveform)
                    batch_ids.append(sample.utterance_id)
                    batch_labels.append(min(sample.utterance_label, 1))
                    batch_frame_labels.append(sample.frame_labels)
                except Exception as exc:
                    n_errors += 1
                    if n_errors <= 5:
                        logger.warning("  Sample %d failed to load: %s", idx, exc)
                    continue

                if len(batch_wavs) >= BATCH_SIZE:
                    _flush_batch()
                    batch_wavs, batch_ids, batch_labels, batch_frame_labels = [], [], [], []
                    n_total_processed = len(all_utt_scores)

                if (idx + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    rate = n_total_processed / elapsed if elapsed > 0 else 0
                    print(f"    Processed {idx + 1}/{n_utterances} "
                          f"({elapsed:.1f}s, {rate:.1f} utt/s, {n_errors} errors)")

            # Flush remaining
            _flush_batch()

            elapsed = time.time() - t0
            n_processed = len(all_utt_scores)
            print(f"    Inference complete: {n_processed}/{n_utterances} in {elapsed:.1f}s "
                  f"({n_errors} errors)")

            utt_scores = np.array(all_utt_scores)
            utt_labels = np.array(all_utt_labels)

            # ── Save raw scores for future analysis ───────────────
            np.save(output_dir / f"{det_name}_{ds_name}_utt_scores.npy", utt_scores)
            np.save(output_dir / f"{det_name}_{ds_name}_utt_labels.npy", utt_labels)
            np.save(
                output_dir / f"{det_name}_{ds_name}_frame_scores.npy",
                np.array(all_frame_scores, dtype=object), allow_pickle=True,
            )
            np.save(
                output_dir / f"{det_name}_{ds_name}_frame_labels.npy",
                np.array(all_frame_labels, dtype=object), allow_pickle=True,
            )
            np.save(
                output_dir / f"{det_name}_{ds_name}_utt_ids.npy",
                np.array(all_utt_ids, dtype=object), allow_pickle=True,
            )

            # ── Metrics ───────────────────────────────────────────
            det_result: dict = {
                "frame_shift_ms": det_frame_shift_ms,
                "n_utterances": n_utterances,
                "n_processed": n_processed,
                "n_errors": n_errors,
            }

            # ── Utterance-level EER ───────────────────────────────
            has_both_utt_classes = len(np.unique(utt_labels)) >= 2
            if has_both_utt_classes:
                utt_eer, utt_thresh = compute_eer(utt_scores, utt_labels)
                det_result["utt_eer"] = float(utt_eer)
                det_result["utt_eer_threshold"] = float(utt_thresh)
                print(f"    Utt-EER: {utt_eer:.4f}")
            else:
                det_result["utt_eer"] = None
                det_result["utt_eer_note"] = (
                    "single class in utterance labels - "
                    f"all labels={utt_labels[0] if len(utt_labels) > 0 else 'empty'}"
                )
                print(f"    Utt-EER: N/A (single class)")

            # ── Utterance-level extra metrics (for datasets without frame labels) ─
            has_frame_labels = any(len(fl) > 0 for fl in all_frame_labels)
            if has_both_utt_classes:
                from sklearn.metrics import (
                    accuracy_score as utt_accuracy_score,
                    precision_score as utt_precision_score,
                    recall_score as utt_recall_score,
                    roc_auc_score as utt_roc_auc_score,
                )
                utt_preds = (utt_scores > utt_thresh).astype(int) if has_both_utt_classes else np.zeros_like(utt_labels)
                det_result["utt_accuracy"] = float(utt_accuracy_score(utt_labels, utt_preds))
                det_result["utt_precision"] = float(utt_precision_score(utt_labels, utt_preds, zero_division=0.0))
                det_result["utt_recall"] = float(utt_recall_score(utt_labels, utt_preds, zero_division=0.0))
                det_result["utt_f1"] = float(f1_score(utt_labels, utt_preds, zero_division=0.0))
                det_result["utt_roc_auc"] = float(utt_roc_auc_score(utt_labels, utt_scores))
                print(f"    Utt-Accuracy:  {det_result['utt_accuracy']:.4f}")
                print(f"    Utt-Precision: {det_result['utt_precision']:.4f}")
                print(f"    Utt-Recall:    {det_result['utt_recall']:.4f}")
                print(f"    Utt-F1:        {det_result['utt_f1']:.4f}")
                print(f"    Utt-ROC-AUC:   {det_result['utt_roc_auc']:.4f}")

            if not has_frame_labels:
                print(f"    [INFO] No frame-level labels - skipping segment metrics.")
                ds_results[det_name] = det_result
                continue

            # ── Segment-level EER at each resolution ──────────────
            # Pool all segments across ALL utterances, then compute EER.
            # Same correct protocol as E1 (fix for BUG-E1-A2).
            for res in resolutions:
                if res < det_frame_shift_ms:
                    print(f"    Seg-EER@{res}ms: SKIP (below native {det_frame_shift_ms}ms)")
                    continue

                pooled_scores: list[float] = []
                pooled_labels: list[int] = []

                for fs, fl in zip(all_frame_scores, all_frame_labels):
                    if len(fl) == 0 or len(fs) == 0:
                        continue
                    s_win = _pool_scores_to_windows(
                        fs, det_frame_shift_ms, float(res), agg="min"
                    )
                    l_win = _pool_labels_to_windows(
                        fl, LABEL_FRAME_SHIFT_MS, float(res), rule="any"
                    )
                    min_len = min(len(s_win), len(l_win))
                    pooled_scores.extend(s_win[:min_len].tolist())
                    pooled_labels.extend(l_win[:min_len].tolist())

                if pooled_scores and len(set(pooled_labels)) > 1:
                    pooled_eer, _ = compute_eer(
                        np.array(pooled_scores), np.array(pooled_labels)
                    )
                    det_result[f"seg_eer_{res}ms"] = float(pooled_eer)
                    print(f"    Seg-EER@{res}ms: {pooled_eer:.4f}")
                else:
                    print(f"    Seg-EER@{res}ms: N/A (single class in pooled set)")

            # ── Frame-level EER threshold on label grid ───────────
            all_fs_at_label_grid: list[float] = []
            all_fl_for_eer: list[int] = []
            for fs, fl in zip(all_frame_scores, all_frame_labels):
                if len(fs) == 0 or len(fl) == 0:
                    continue
                fs_up = upsample_binary_predictions_to_label_grid(
                    fs, pred_frame_shift_ms=det_frame_shift_ms,
                    label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                )
                min_len = min(len(fs_up), len(fl))
                all_fs_at_label_grid.extend(fs_up[:min_len].tolist())
                all_fl_for_eer.extend(fl[:min_len].tolist())

            if all_fs_at_label_grid and len(set(all_fl_for_eer)) > 1:
                frame_eer, frame_thresh = compute_eer(
                    np.array(all_fs_at_label_grid), np.array(all_fl_for_eer)
                )
                print(f"    Frame-level EER: {frame_eer:.4f}, threshold: {frame_thresh:.4f}")
            else:
                frame_thresh = 0.5
                print(f"    Frame-level EER: N/A, using default threshold=0.5")

            # ── Segment F1 at native resolution ───────────────────
            all_preds_aligned: list[int] = []
            all_gts_aligned: list[int] = []
            for fs, fl in zip(all_frame_scores, all_frame_labels):
                if len(fs) == 0 or len(fl) == 0:
                    continue
                pred_binary = (fs > frame_thresh).astype(int)
                pred_at_grid = upsample_binary_predictions_to_label_grid(
                    pred_binary, pred_frame_shift_ms=det_frame_shift_ms,
                    label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                )
                min_len = min(len(pred_at_grid), len(fl))
                all_preds_aligned.extend(pred_at_grid[:min_len].tolist())
                all_gts_aligned.extend(fl[:min_len].tolist())

            if all_preds_aligned and len(set(all_gts_aligned)) > 1:
                seg_f1 = compute_segment_f1(
                    np.array(all_preds_aligned), np.array(all_gts_aligned)
                )
                det_result[f"seg_f1_{det_frame_shift_ms}ms"] = float(seg_f1)
                print(f"    Seg-F1@{det_frame_shift_ms}ms (native): {seg_f1:.4f}")

            # ── PartialEdit-specific segment metrics ──────────────
            if ds_name == "partialedit":
                pe_metrics = _compute_partialedit_segment_metrics(
                    all_frame_scores, all_frame_labels, det_frame_shift_ms,
                )
                if pe_metrics:
                    det_result["partialedit_segment_metrics"] = pe_metrics
                    print(f"    --- PartialEdit segment metrics (at 10ms label grid) ---")
                    print(f"    Frame-EER: {pe_metrics['frame_eer']:.4f}")
                    print(f"    Accuracy:  {pe_metrics['accuracy']:.4f}")
                    print(f"    Precision: {pe_metrics['precision']:.4f}")
                    print(f"    Recall:    {pe_metrics['recall']:.4f}")
                    print(f"    F1:        {pe_metrics['f1']:.4f}")
                    print(f"    ROC-AUC:   {pe_metrics['roc_auc']:.4f}")

            ds_results[det_name] = det_result

        all_results[ds_name] = ds_results

    # ── Persist results (merge with existing per-detector) ────────
    output_file = output_dir / "results.json"
    existing = {}
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Merge: new results take precedence
    for ds_name, ds_data in all_results.items():
        if ds_name not in existing:
            existing[ds_name] = {}
        existing[ds_name].update(ds_data)

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {output_file} "
          f"(datasets: {list(all_results.keys())})")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_e5(filter_detector=args.detector, filter_dataset=args.dataset)
