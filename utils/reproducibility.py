"""Reproducibility primitives. Never serialize API keys or the environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = "2"


def seed_everything(seed: int) -> None:
    """Seed local RNGs; this does not promise cross-device bitwise equality."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_verifier_data(data_chunks, n_vars: int):
    """Reserve the last 20% for graph scoring, before fitting any weights.

    The model makes its own early-stopping split inside the returned fit data.
    The scoring split is reused by graph search, so it is NOT a final test set.
    One continuous chunk is split along time; multiple chunks along chunk order.
    """
    xs = np.asarray(data_chunks, dtype=np.float32)
    if xs.ndim != 3 or xs.shape[0] == 0 or xs.shape[2] != n_vars:
        raise ValueError("data must have shape (chunks, time, variables)")
    if n_vars < 2 or xs.shape[1] < 20:
        raise ValueError("provide at least two variables and 20 time points per chunk")
    if not np.isfinite(xs).all():
        raise ValueError("data contains NaN or infinity")
    if len(xs) == 1:
        cut = int(0.8 * xs.shape[1])
        return xs[:, :cut].copy(), xs[:, cut:].copy()
    cut = min(len(xs) - 1, max(1, int(0.8 * len(xs))))
    return xs[:cut].copy(), xs[cut:].copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_metadata(seed: int, inputs=()) -> dict:
    """Return an allowlisted, JSON-safe provenance record."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    packages = {}
    for name in ("numpy", "torch", "pandas", "scipy", "scikit-learn", "causclass"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "inputs": {Path(p).name: sha256_file(Path(p)) for p in inputs},
    }
