"""Configuration loading and management for XPS-Forensic.

Loads YAML configs with variable interpolation (``${data.root}`` style),
deep-merge overrides, and auto device detection.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# configs/ directory is two parents up from utils/config.py:
# xps_forensic/xps_forensic/utils/config.py  ->  xps_forensic/configs/
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


class DotDict(dict):
    """Dictionary subclass that supports dot-notation attribute access.

    Nested dicts are recursively converted to DotDict on access.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = DotDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Values in *override* take precedence.  Nested dicts are merged
    recursively; all other types are replaced outright.
    """
    merged = dict(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_refs(cfg: dict, root: dict | None = None) -> dict:
    r"""Resolve ``${dotted.path}`` references inside string values.

    Parameters
    ----------
    cfg : dict
        The (possibly nested) config dict whose string values may contain
        ``${...}`` interpolation markers.
    root : dict or None
        The top-level config dict used for look-ups.  Defaults to *cfg*
        itself on the initial call.

    Returns
    -------
    dict
        A new dict with all ``${...}`` references replaced by their
        resolved values.
    """
    if root is None:
        root = cfg
    resolved: dict = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            resolved[key] = _resolve_refs(val, root)
        elif isinstance(val, str):
            resolved[key] = _interpolate_string(val, root)
        elif isinstance(val, list):
            resolved[key] = [
                _interpolate_string(v, root) if isinstance(v, str) else v
                for v in val
            ]
        else:
            resolved[key] = val
    return resolved


def _interpolate_string(s: str, root: dict) -> str:
    """Replace all ``${dotted.path}`` tokens in *s* with looked-up values."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def _replacer(match: re.Match) -> str:
        path = match.group(1)
        parts = path.split(".")
        obj: Any = root
        for part in parts:
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                raise KeyError(f"Cannot resolve reference '${{{path}}}'")  # noqa: E501
        return str(obj)

    return pattern.sub(_replacer, s)


def _auto_device() -> str:
    """Return ``'cuda'`` if a CUDA GPU is available, else ``'cpu'``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_config(
    overrides: dict | None = None,
    config_path: str | Path | None = None,
) -> DotDict:
    """Load the default YAML config, apply *overrides*, and resolve refs.

    Parameters
    ----------
    overrides : dict, optional
        Key-value overrides merged on top of the default config.
    config_path : str or Path, optional
        Explicit path to a YAML file.  Defaults to
        ``configs/default.yaml``.

    Returns
    -------
    DotDict
        Fully resolved configuration accessible via dot-notation.
    """
    if config_path is None:
        config_path = _CONFIG_DIR / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # Resolve ${...} variable interpolation
    cfg = _resolve_refs(cfg)

    # Auto-detect device
    if cfg.get("device") == "auto":
        cfg["device"] = _auto_device()

    return DotDict(cfg)
