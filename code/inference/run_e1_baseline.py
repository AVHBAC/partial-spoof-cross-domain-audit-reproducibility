"""E1: Baseline Detection & Localization.

Reproduces published results for all 4 detectors on PartialSpoof eval set.
Reports Utt-EER, Seg-EER at multiple resolutions, Seg-F1 with bootstrap CIs.

Saves raw scores per-utterance so that E2 (calibration) and E3 (CPSL) can
consume them without re-running inference.

Output layout (under results/e1_baseline/):
    results.json          - all computed metrics
    {det}_utt_scores.npy  - utterance-level scores, shape (N,)
    {det}_utt_labels.npy  - utterance-level binary labels, shape (N,)
    {det}_frame_scores.npy  - list of per-utterance frame score arrays (object array)
    {det}_frame_labels.npy  - list of per-utterance frame label arrays (object array)
    {det}_utt_ids.npy       - utterance IDs (object array of strings)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_f1,
    upsample_binary_predictions_to_label_grid,
    _pool_scores_to_windows,
    _pool_labels_to_windows,
)
from xps_forensic.utils.stats import bootstrap_ci
from xps_forensic.data.partialspoof import PartialSpoofDataset
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

# Label frame shift in PartialSpoof dataset (10 ms at finest resolution).
LABEL_FRAME_SHIFT_MS = 10.0


def _build_detector(det_name: str, det_cfg: dict, device: str):
    """Instantiate a detector from config, returning None if checkpoint missing.

    Each detector wrapper accepts slightly different constructor kwargs.
    We map config keys to the constructor signatures defined in the wrappers.
    """
    DetClass = DETECTOR_MAP[det_name]
    checkpoint = det_cfg.get("checkpoint")

    # If checkpoint is explicitly null / None, we cannot run inference.
    if checkpoint is None:
        logger.warning(
            "Detector %s: checkpoint is null in config - skipping.",
            det_name.upper(),
        )
        return None

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        logger.warning(
            "Detector %s: checkpoint not found at %s - skipping.",
            det_name.upper(),
            ckpt_path,
        )
        return None

    # Common kwargs shared by all detectors
    kwargs: dict = {
        "checkpoint": str(ckpt_path),
        "device": device,
    }

    # Per-detector optional kwargs from config
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
        logger.warning(
            "Detector %s: failed to load model - %s. Skipping.",
            det_name.upper(),
            exc,
        )
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


def parse_args():
    parser = argparse.ArgumentParser(description="E1: Baseline Detection & Localization")
    parser.add_argument('--detector', type=str, default=None,
                        choices=['bam', 'sal', 'cfprf', 'mrm'],
                        help='Run single detector. If omitted, runs all.')
    return parser.parse_args()


def run_e1(cfg=None):
    """Run E1 baseline experiment.

    For each available detector (skips if checkpoint missing):
      1. Load model checkpoint.
      2. Run inference on PartialSpoof eval set.
      3. Compute utterance-level EER with bootstrap CI.
      4. Compute segment-level EER at multiple resolutions with bootstrap CI.
      5. Compute segment F1 at the detector's native resolution.
      6. Save all scores and results.

    Returns:
        dict: Per-detector results including raw score arrays, suitable for
        passing directly to E2/E3.
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

    output_dir = Path(cfg.experiments.output_dir) / "e1_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = PartialSpoofDataset(
        root=cfg.data.partialspoof.path,
        split="eval",
        sample_rate=cfg.data.partialspoof.sample_rate,
    )
    n_utterances = len(dataset)
    print(f"Loaded PartialSpoof eval: {n_utterances} utterances")

    if n_utterances == 0:
        logger.warning("Dataset is empty - nothing to evaluate.")
        return {}

    resolutions = cfg.experiments.resolutions_ms
    results: dict = {}

    args = parse_args()
    detectors_to_run = [args.detector] if args.detector else ["bam", "sal", "cfprf", "mrm"]

    for det_name in detectors_to_run:
        print(f"\n{'=' * 60}")
        print(f"Detector: {det_name.upper()}")
        print(f"{'=' * 60}")

        det_cfg = cfg.detectors.get(det_name, {})
        if not det_cfg:
            logger.warning("No config entry for detector %s - skipping.", det_name)
            continue

        detector = _build_detector(det_name, det_cfg, cfg.device)
        if detector is None:
            print(f"  [SKIP] Detector {det_name.upper()} not available.")
            continue

        det_frame_shift_ms = detector.frame_shift_ms
        print(f"  Frame shift: {det_frame_shift_ms} ms")

        # ── Inference ────────────────────────────────────────────────
        all_utt_ids: list[str] = []
        all_utt_scores: list[float] = []
        all_utt_labels: list[int] = []
        all_frame_scores: list[np.ndarray] = []
        all_frame_labels: list[np.ndarray] = []

        # Batched inference: collect waveforms and process via predict_batch
        # for better GPU utilization. Batch size tuned per detector.
        BATCH_SIZE = 16  # waveforms per GPU batch
        t0 = time.time()
        n_errors = 0
        batch_wavs, batch_ids, batch_labels, batch_frame_labels = [], [], [], []
        n_total_processed = 0

        def _flush_batch():
            """Process accumulated batch through detector."""
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
            except Exception as exc:
                # Fallback: process individually
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
                print(f"  Processed {idx + 1}/{n_utterances} "
                      f"({elapsed:.1f}s, {rate:.1f} utt/s, {n_errors} errors)")

        # Flush remaining
        _flush_batch()

        elapsed = time.time() - t0
        n_processed = len(all_utt_scores)
        print(f"  Inference complete: {n_processed}/{n_utterances} utterances "
              f"in {elapsed:.1f}s ({n_errors} errors)")
        if n_errors > 0:
            logger.warning("  %d/%d utterances failed (%.1f%%)",
                           n_errors, n_utterances, 100 * n_errors / n_utterances)

        utt_scores = np.array(all_utt_scores)
        utt_labels = np.array(all_utt_labels)

        # ── Save raw scores for E2/E3 ───────────────────────────────
        np.save(output_dir / f"{det_name}_utt_scores.npy", utt_scores)
        np.save(output_dir / f"{det_name}_utt_labels.npy", utt_labels)
        np.save(
            output_dir / f"{det_name}_frame_scores.npy",
            np.array(all_frame_scores, dtype=object),
            allow_pickle=True,
        )
        np.save(
            output_dir / f"{det_name}_frame_labels.npy",
            np.array(all_frame_labels, dtype=object),
            allow_pickle=True,
        )
        np.save(
            output_dir / f"{det_name}_utt_ids.npy",
            np.array(all_utt_ids, dtype=object),
            allow_pickle=True,
        )

        # ── Utterance-level EER ──────────────────────────────────────
        utt_eer, utt_thresh = compute_eer(utt_scores, utt_labels)

        det_results: dict = {
            "frame_shift_ms": det_frame_shift_ms,
            "n_utterances": n_utterances,
            "n_processed": n_processed,
            "utt_eer": float(utt_eer),
            "utt_eer_threshold": float(utt_thresh),
        }
        print(f"  Utt-EER: {utt_eer:.4f}")

        # ── Segment-level EER at each resolution ─────────────────────
        # [FIX A2/C1 + E1-1 + E1-2]
        # Pool all segments across ALL utterances before computing EER.
        # When target resolution < detector native, upsample scores first.
        # Use "any" rule for labels (PartialSpoof protocol: spoof if ANY
        # frame in window is spoof). Zhang et al. (2023) Section IV-B.
        for res in resolutions:
            # Skip resolutions below native - those are just score
            # interpolation, not genuine detection at that granularity.
            if res < det_frame_shift_ms:
                print(f"  Seg-EER@{res}ms: SKIP (below native {det_frame_shift_ms}ms)")
                continue

            pooled_scores: list[float] = []
            pooled_labels: list[int] = []

            for fs, fl in zip(all_frame_scores, all_frame_labels):
                if len(fl) == 0 or len(fs) == 0:
                    continue

                s_win = _pool_scores_to_windows(
                    fs, float(det_frame_shift_ms), float(res), agg="min"
                )

                # Use "any" rule for labels (PartialSpoof protocol:
                # spoof if ANY frame in window is spoof).
                # Zhang et al. (2023) Section IV-B.
                l_win = _pool_labels_to_windows(
                    fl, LABEL_FRAME_SHIFT_MS, float(res), rule="any"
                )
                min_len = min(len(s_win), len(l_win))
                pooled_scores.extend(s_win[:min_len].tolist())
                pooled_labels.extend(l_win[:min_len].tolist())

            if pooled_scores and len(set(pooled_labels)) > 1:
                pooled_eer, seg_thresh = compute_eer(
                    np.array(pooled_scores), np.array(pooled_labels)
                )
                det_results[f"seg_eer_{res}ms"] = float(pooled_eer)
                print(f"  Seg-EER@{res}ms: {pooled_eer:.4f}")
            else:
                print(f"  Seg-EER@{res}ms: N/A (single class in pooled set)")

        # ── Segment F1 at detector's native resolution ───────────────
        # [FIX C4+B] Compute frame-level EER threshold by upsampling
        # scores to the 10ms label grid (correct alignment direction),
        # then compute EER on the pooled set.
        all_fs_at_label_grid = []
        all_fl_for_eer = []
        for fs, fl in zip(all_frame_scores, all_frame_labels):
            if len(fs) == 0 or len(fl) == 0:
                continue
            # Upsample scores to label grid (e.g., repeat each 160ms score 16x to get 10ms)
            fs_upsampled = upsample_binary_predictions_to_label_grid(
                fs, pred_frame_shift_ms=float(det_frame_shift_ms),
                label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
            )
            min_len = min(len(fs_upsampled), len(fl))
            all_fs_at_label_grid.extend(fs_upsampled[:min_len].tolist())
            all_fl_for_eer.extend(fl[:min_len].tolist())

        if all_fs_at_label_grid and len(set(all_fl_for_eer)) > 1:
            frame_eer, frame_thresh = compute_eer(
                np.array(all_fs_at_label_grid), np.array(all_fl_for_eer)
            )
            print(f"  Frame-level EER: {frame_eer:.4f}, threshold: {frame_thresh:.4f}")
        else:
            frame_thresh = utt_thresh
            print(f"  Frame-level EER: N/A, using utt_thresh={frame_thresh:.4f}")

        all_preds_aligned: list[int] = []
        all_gts_aligned: list[int] = []
        for fs, fl in zip(all_frame_scores, all_frame_labels):
            if len(fs) == 0 or len(fl) == 0:
                continue
            pred_binary = (fs > frame_thresh).astype(int)
            pred_at_label_grid = upsample_binary_predictions_to_label_grid(
                pred_binary,
                pred_frame_shift_ms=float(det_frame_shift_ms),
                label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
            )
            min_len = min(len(pred_at_label_grid), len(fl))
            all_preds_aligned.extend(pred_at_label_grid[:min_len].tolist())
            all_gts_aligned.extend(fl[:min_len].tolist())

        if all_preds_aligned:
            seg_f1 = compute_segment_f1(
                np.array(all_preds_aligned), np.array(all_gts_aligned)
            )
            det_results[f"seg_f1_{det_frame_shift_ms}ms"] = float(seg_f1)
            print(f"  Seg-F1@{det_frame_shift_ms}ms (native): {seg_f1:.4f}")

        results[det_name] = det_results

    # ── Persist results JSON (merge with existing per-detector results) ──
    output_file = output_dir / "results.json"
    existing_results = {}
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)
            logger.info("Loaded existing results for detectors: %s",
                        list(existing_results.keys()))
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not load existing results.json; overwriting.")

    # Merge: new results take precedence over existing
    existing_results.update(results)

    with open(output_file, "w") as f:
        json.dump(existing_results, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {output_file} "
          f"(detectors: {list(existing_results.keys())})")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_e1()
