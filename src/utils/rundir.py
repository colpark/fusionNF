"""Run-directory convention (operating rule 3 / Phase 0).

Every run writes, to its own directory:
  - config.yaml      : the fully-resolved ExperimentConfig
  - meta.json        : git SHA, seed, timestamp, device, lib versions, config hash
  - metrics.jsonl    : append-only per-step log (written by utils.logging)
  - figures/, ...    : artifacts

The directory name encodes tag + config hash + seed so reruns are discoverable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

from .. import config as cfg


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        sha = out.decode().strip()
        # mark dirty tree so artifacts can't masquerade as a clean commit
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return sha + ("-dirty" if dirty != 0 else "")
    except Exception:
        return "nogit"


def _lib_versions() -> dict:
    import numpy
    import torch
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def create_run_dir(experiment: cfg.ExperimentConfig, when: str | None = None) -> str:
    """Create and populate a run directory; return its path.

    `when` (timestamp string) is injected by the caller rather than read from the
    clock here, to keep this module import-deterministic for tests.
    """
    h = cfg.config_hash(experiment)
    name = f"{experiment.tag}_{h}_seed{experiment.seed}"
    path = os.path.join(experiment.out_root, name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)

    cfg.save_yaml(experiment, os.path.join(path, "config.yaml"))
    meta = {
        "git_sha": git_sha(),
        "seed": experiment.seed,
        "tag": experiment.tag,
        "config_hash": h,
        "timestamp": when,
        "device": experiment.train.device,
        "libs": _lib_versions(),
    }
    with open(os.path.join(path, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return path
