"""Small reproducibility utilities shared by the public NAS entry points.

The full experiment archive is intentionally kept outside this trimmed public
repository.  This module only exposes deterministic seed derivation and run
metadata helpers needed by the released code.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPORTED_SEARCH_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_PROXY_BASE_SEED = 20260722


def stable_int(text: str, bits: int = 32) -> int:
    """Map text to a stable unsigned integer using BLAKE2b."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << bits) - 1)


def search_rng_seed(dataset: str, reported_seed: int) -> int:
    """Derive the proposal RNG seed used for a dataset/run pair."""
    return stable_int(f"e2-search|{dataset}|{reported_seed}")


def proxy_rng_seed(base_seed: int, dataset: str, genotype: str) -> int:
    """Derive an order-independent proxy seed for one architecture."""
    return stable_int(f"e2-zico|{base_seed}|{dataset}|{genotype}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON record, creating its parent directory if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
