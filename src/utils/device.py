"""Device selection: prefer CUDA (remote GPU), then Apple MPS, then CPU.

Honors an explicit override via the NF_DEVICE env var or a passed string. The
remote GPU server therefore needs no code change -- `auto_device()` picks cuda.
"""
from __future__ import annotations

import os

import torch


def auto_device(prefer: str | None = None) -> str:
    choice = prefer or os.environ.get("NF_DEVICE")
    if choice:
        # Fail early with an actionable message instead of a deep CUDA assertion later.
        if choice.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {choice!r} but this PyTorch has no CUDA "
                f"(torch.cuda.is_available() == False; torch {torch.__version__}). "
                "You are likely not in the project's uv environment whose torch is the "
                "CUDA build -- run via `uv run ...` (e.g. `uv run make train DEVICE=cuda:1`), "
                "or install a CUDA-enabled torch in your current env.")
        if choice.startswith("cuda:"):
            idx = int(choice.split(":", 1)[1])
            n = torch.cuda.device_count()
            if idx >= n:
                raise RuntimeError(
                    f"Requested {choice!r} but only {n} CUDA device(s) visible "
                    f"(indices {list(range(n))}). Use a valid index, or "
                    f"CUDA_VISIBLE_DEVICES to remap GPUs.")
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
