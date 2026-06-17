"""Core evaluation metrics for spoof detection and localization.

Implements:
- Equal Error Rate (EER) computation via ROC interpolation
- Segment-level EER at configurable temporal resolution
- Segment F1 score
- Temporal False Negative Rate (tFNR)
- Temporal False Discovery Rate (tFDR)
- Temporal Intersection over Union (tIoU)

References
----------
Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem
Assessment of Spoofing Countermeasures and Automatic Speaker Verification",
Proc. Odyssey 2018.

Zhang et al., "The PartialSpoof Database and Countermeasures for the
Detection of Short Fake Speech Segments Embedded in Real Speech",
IEEE/ACM TASLP, 2023.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, f1_score


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute the Equal Error Rate and the corresponding threshold.

    Parameters
    ----------
    scores : array-like, shape (n_samples,)
        Detector scores.  Higher values indicate *spoof* (positive class).
    labels : array-like, shape (n_samples,)
        Binary ground truth: 1 = spoof, 0 = bonafide.

    Returns
    -------
    eer : float
        Equal error rate in [0, 1].
    threshold : float
        Score threshold at the EER operating point.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Guard: EER is undefined with a single class
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return (0.0, float(np.median(scores))) if unique_labels[0] == 0 else (1.0, float(np.median(scores)))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Interpolate to find the crossing point where FPR == FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        # Fallback: pick the point closest to FPR == FNR
        diff = np.abs(fpr - fnr)
        if np.all(np.isnan(diff)):
            eer = 0.5  # degenerate case
            return eer, float(np.median(scores))
        idx = np.nanargmin(diff)
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

    # Threshold at EER: find the threshold closest to the EER operating point
    diff = np.abs(fpr - fnr)
    idx_best = np.nanargmin(diff)
    threshold = float(thresholds[idx_best])

    return float(eer), threshold


def compute_segment_eer(
    frame_scores: np.ndarray,
    frame_labels: np.ndarray,
    resolution_ms: float,
    frame_shift_ms: float = 20.0,
) -> tuple[float, float]:
    """Compute segment-level EER at a given temporal resolution.

    Frames are aggregated into non-overlapping segments of
    ``resolution_ms / frame_shift_ms`` frames, using mean pooling for
    scores and majority vote for labels.

    Parameters
    ----------
    frame_scores : array-like, shape (n_frames,)
        Per-frame spoof scores.
    frame_labels : array-like, shape (n_frames,)
        Per-frame binary labels (1 = spoof).
    resolution_ms : float
        Segment duration in milliseconds.
    frame_shift_ms : float
        Frame shift in milliseconds (default 10 ms).

    Returns
    -------
    eer : float
    threshold : float
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    frame_labels = np.asarray(frame_labels, dtype=int)

    seg_len = max(1, int(round(resolution_ms / frame_shift_ms)))
    n_frames = len(frame_scores)
    n_segments = n_frames // seg_len

    if n_segments == 0:
        return compute_eer(frame_scores, frame_labels)

    # Truncate to exact multiples
    scores_trunc = frame_scores[: n_segments * seg_len].reshape(n_segments, seg_len)
    labels_trunc = frame_labels[: n_segments * seg_len].reshape(n_segments, seg_len)

    seg_scores = scores_trunc.mean(axis=1)
    seg_labels = (labels_trunc.mean(axis=1) >= 0.5).astype(int)

    return compute_eer(seg_scores, seg_labels)


def compute_segment_f1(
    frame_preds: np.ndarray,
    frame_labels: np.ndarray,
) -> float:
    """Compute frame-level F1 score for spoof localization.

    Parameters
    ----------
    frame_preds : array-like, shape (n_frames,)
        Binary frame-level predictions (1 = spoof).
    frame_labels : array-like, shape (n_frames,)
        Binary frame-level ground truth (1 = spoof).

    Returns
    -------
    float
        F1 score.  Returns 1.0 when both predictions and labels are all-zero
        (no spoof to detect and none predicted).
    """
    frame_preds = np.asarray(frame_preds, dtype=int)
    frame_labels = np.asarray(frame_labels, dtype=int)

    if frame_preds.sum() == 0 and frame_labels.sum() == 0:
        return 1.0

    return float(f1_score(frame_labels, frame_preds, zero_division=0.0))


def compute_tFNR(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal False Negative Rate.

    Fraction of truly spoofed frames that are *not* detected.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tFNR in [0, 1].  Returns 0.0 if there are no spoofed frames
        in the ground truth.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    total_spoof = true_binary.sum()
    if total_spoof == 0:
        return 0.0

    missed = np.sum((true_binary == 1) & (pred_binary == 0))
    return float(missed / total_spoof)


def compute_tFDR(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal False Discovery Rate.

    Fraction of predicted-spoof frames that are actually bonafide.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tFDR in [0, 1].  Returns 0.0 if no frames are predicted as spoof.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    total_pred = pred_binary.sum()
    if total_pred == 0:
        return 0.0

    false_alarms = np.sum((pred_binary == 1) & (true_binary == 0))
    return float(false_alarms / total_pred)


def compute_tIoU(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal Intersection over Union.

    Measures the overlap between predicted and true spoof regions.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tIoU in [0, 1].  Returns 1.0 if both are all-zero (no spoof present
        and none predicted).  Returns 0.0 if union is non-empty but
        intersection is empty.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    intersection = np.sum((pred_binary == 1) & (true_binary == 1))
    union = np.sum((pred_binary == 1) | (true_binary == 1))

    if union == 0:
        return 1.0

    return float(intersection / union)


# ── Mixed-resolution helpers ─────────────────────────────────────────────---

def _pool_scores_to_windows(
    frame_scores: np.ndarray,
    score_frame_shift_ms: float,
    window_ms: float,
    agg: str = "min",
) -> np.ndarray:
    """Aggregate scores into non-overlapping windows of ``window_ms``.

    Default aggregation is ``min`` per the PartialSpoof evaluation protocol
    (Zhang et al., Interspeech 2023, Section 4.3): when downsampling scores
    to a coarser resolution, adjacent segments are aggregated by selecting
    their minimum value, because lower scores indicate higher spoof
    likelihood, consistent with the any-spoof label rule.

    Args:
        frame_scores: Shape (n_frames,).
        score_frame_shift_ms: Frame shift of scores in ms (e.g., 20).
        window_ms: Target window size in ms (e.g., 160).
        agg: Aggregation for scores ("min" per PartialSpoof protocol,
             "mean" is an alternative).

    Returns:
        Array of windowed scores.
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    frames_per_window = max(1, int(round(window_ms / score_frame_shift_ms)))
    n = len(frame_scores)
    n_windows = n // frames_per_window
    if n_windows == 0:
        return frame_scores.copy()
    trunc = frame_scores[: n_windows * frames_per_window].reshape(
        n_windows, frames_per_window
    )
    if agg == "min":
        return trunc.min(axis=1)
    if agg == "mean":
        return trunc.mean(axis=1)
    raise ValueError(f"Unknown agg: {agg}")


def _pool_labels_to_windows(
    frame_labels: np.ndarray,
    label_frame_shift_ms: float,
    window_ms: float,
    rule: str = "majority",
) -> np.ndarray:
    """Aggregate binary labels into non-overlapping ``window_ms`` windows.

    Args:
        frame_labels: Binary labels, shape (n_frames,).
        label_frame_shift_ms: Label frame shift in ms (e.g., 10).
        window_ms: Target window size in ms.
        rule: Aggregation rule for labels ("majority" standard in PartialSpoof).

    Returns:
        Binary window labels.
    """
    frame_labels = np.asarray(frame_labels, dtype=int)
    frames_per_window = max(1, int(round(window_ms / label_frame_shift_ms)))
    n = len(frame_labels)
    n_windows = n // frames_per_window
    if n_windows == 0:
        return frame_labels.copy()
    trunc = frame_labels[: n_windows * frames_per_window].reshape(
        n_windows, frames_per_window
    )
    if rule == "majority":
        return (trunc.mean(axis=1) >= 0.5).astype(int)
    elif rule == "any":
        return (trunc.sum(axis=1) > 0).astype(int)
    else:
        raise ValueError(f"Unknown rule: {rule}")


def compute_segment_eer_mixed(
    frame_scores: np.ndarray,
    score_frame_shift_ms: float,
    frame_labels: np.ndarray,
    label_frame_shift_ms: float,
    resolution_ms: float,
) -> tuple[float, float]:
    """Segment-level EER when scores and labels use different frame shifts.

    Pools scores by minimum and labels by "any" rule into windows of
    ``resolution_ms``, then computes EER on the pooled series.

    Score aggregation uses minimum per Zhang et al. (Interspeech 2023,
    Section 4.3): "when down-sampling scores to a coarser-grained
    resolution, we aggregated adjacent segments by selecting their
    minimum value." This is consistent with the any-spoof label rule,
    since lower scores indicate higher spoof likelihood.

    Label aggregation uses the "any" rule per PartialSpoof evaluation
    protocol (Zhang et al. 2023, Section IV-B): a segment is labeled
    as spoof if ANY frame within the window is spoofed.

    Returns:
        (eer, threshold) as in ``compute_eer``.
    """
    seg_scores = _pool_scores_to_windows(
        frame_scores, score_frame_shift_ms, resolution_ms, agg="min"
    )
    seg_labels = _pool_labels_to_windows(
        frame_labels, label_frame_shift_ms, resolution_ms, rule="any"
    )

    # Truncate to common length (can differ by <= 1 due to rounding)
    m = min(len(seg_scores), len(seg_labels))
    return compute_eer(seg_scores[:m], seg_labels[:m])


def upsample_binary_predictions_to_label_grid(
    pred_binary: np.ndarray,
    pred_frame_shift_ms: float,
    label_frame_shift_ms: float,
) -> np.ndarray:
    """Upsample frame-level values to the label frame grid by repetition.

    Accepts both continuous scores and binary predictions.  Values are
    preserved as float so that continuous scores survive upsampling
    (required when computing EER thresholds on upsampled score grids).

    Example: 20 ms predictions to 10 ms labels -> repeat each frame twice.
    If the ratio is not an integer, uses nearest repeat count rounding.
    Truncates to align edges without padding.
    """
    ratio = pred_frame_shift_ms / label_frame_shift_ms
    if ratio <= 0:
        raise ValueError("Frame shift ratio must be positive")
    r = int(round(ratio))
    if abs(ratio - r) > 1e-6:
        # Fallback: approximate by nearest integer repeat
        r = max(1, r)
    up = np.repeat(np.asarray(pred_binary, dtype=float), r)
    return up
