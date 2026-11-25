"""Resolve the process-wide inference configuration without loading checkpoints."""

import os
from functools import lru_cache

BACKENDS = ("auto", "metal", "cuda", "cpu")


@lru_cache(maxsize=1)
def selected():
    choice = os.environ.get("TEXT_TO_GRAPH_BACKEND", "auto")
    if choice not in BACKENDS:
        raise ValueError(f"Unknown inference backend {choice!r}")
    if choice == "cpu":
        return choice
    import torch

    if choice in ("auto", "metal") and torch.backends.mps.is_available():
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            if mx.metal.is_available():
                return "metal"
    if choice in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if choice == "auto":
        return "cpu"
    raise RuntimeError(f"Requested inference backend {choice!r} is unavailable")


def devices():
    backend = selected()
    return {
        "backend": backend,
        "gliner_encoder": "mlx" if backend == "metal" else backend,
        "gliner_heads": "cpu" if backend == "metal" else backend,
        "english_coreference": "mps" if backend == "metal" else backend,
        "parser": "cpu",
    }
