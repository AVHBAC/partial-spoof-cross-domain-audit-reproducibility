"""Memory management utilities for XPS-Forensic pipeline."""
from __future__ import annotations

import numpy as np
import tempfile
import shutil
from pathlib import Path


def auto_batch_size(
    model_vram_gb: float = 1.3,
    device: str = "cuda",
    activation_per_elem_gb: float = 0.4,
    safety_margin_gb: float = 2.0,
    max_batch_size: int = 32,
) -> int:
    """Calculate optimal batch size from available VRAM.

    With process isolation (one model per process), this function
    determines how much of the free VRAM can be used for batching.

    Args:
        model_vram_gb: Estimated model size on GPU in GB.
        device: Device string ('cuda' or 'cpu').
        activation_per_elem_gb: Estimated activation memory per batch element.
        safety_margin_gb: Reserved VRAM for variable-length inputs.
        max_batch_size: Upper cap on batch size.

    Returns:
        Optimal batch size (minimum 1).
    """
    try:
        import torch
    except ImportError:
        return 1
    if device == "cpu" or not torch.cuda.is_available():
        return 1
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    free = total - reserved
    usable = free - model_vram_gb - safety_margin_gb
    bs = max(1, int(usable / activation_per_elem_gb))
    return min(bs, max_batch_size)


def atomic_np_save(path, data, **kwargs):
    """Save numpy array atomically via temp file + rename.

    Prevents half-written .npy files if process is killed mid-write.

    Args:
        path: Destination file path (str or Path).
        data: Numpy array to save.
        **kwargs: Additional keyword arguments passed to ``np.save``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        np.save(tmp.name, data, **kwargs)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), str(path))


def log_memory(tag: str = ""):
    """Log current GPU and CPU memory usage.

    Gracefully handles missing ``psutil`` or ``torch`` packages by
    returning silently when either is unavailable.

    Args:
        tag: Label included in the log line for identification.
    """
    try:
        import torch
    except ImportError:
        return
    try:
        import psutil
    except ImportError:
        # Fall back to torch-only logging if psutil is missing
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[MEM|{tag}] GPU Alloc: {alloc:.2f} GB, Peak: {peak:.2f} GB")
        return
    import os

    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    msg = f"[MEM|{tag}] CPU RSS: {rss_gb:.2f} GB"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        msg += f" | GPU Alloc: {alloc:.2f} GB, Peak: {peak:.2f} GB"
    print(msg)
