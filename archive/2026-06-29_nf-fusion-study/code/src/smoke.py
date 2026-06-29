"""End-to-end smoke loop for Phase 0.

This exercises the *scaffolding* (seeding, config round-trip, run-dir, logging,
param/FLOP accounting, determinism) -- NOT the real science. The data generator
(Phase 1) and the four fusion families (Phase 4) do not exist yet, so this uses a
deterministic synthetic stand-in: a tiny binary task where a 2-feature signal
predicts a label. Later phases replace `make_placeholder_batch` and
`PlaceholderModel` with the real generator and FusionModels while keeping this
harness identical.

Run: python -m src.smoke --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import datetime as _dt

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg
from .utils.seeding import set_seed, enable_determinism, np_rng
from .utils.rundir import create_run_dir
from .utils.logging import RunLogger
from .utils.flops import count_params, measure_flops


def make_placeholder_batch(rng: np.random.Generator, n: int, dim: int):
    """A trivially-learnable, fully-seeded synthetic batch.

    x in R^dim; label = 1 if a fixed linear projection of x is positive. This is a
    stand-in ONLY to prove the training/logging loop runs deterministically.
    """
    x = rng.standard_normal((n, dim)).astype(np.float32)
    w_true = np.linspace(-1, 1, dim, dtype=np.float32)
    logits = x @ w_true
    y = (logits > 0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


class PlaceholderModel(nn.Module):
    """Minimal MLP standing in for a FusionModel until Phase 4."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_smoke(experiment: cfg.ExperimentConfig) -> dict:
    set_seed(experiment.seed)
    enable_determinism()

    when = _dt.datetime.now(_dt.timezone.utc).isoformat()
    run_dir = create_run_dir(experiment, when=when)
    logger = RunLogger(run_dir)

    device = torch.device(experiment.train.device)
    dim = 8
    model = PlaceholderModel(dim, experiment.model.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=experiment.train.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # FLOP accounting for one forward pass (the harness the real split will reuse).
    probe_x, _ = make_placeholder_batch(np_rng(experiment.seed), experiment.train.batch_size, dim)
    fwd_flops = measure_flops(lambda: model(probe_x.to(device)))
    n_params = count_params(model)

    data_rng = np_rng(experiment.seed + 1)
    final_loss = float("nan")
    for step in range(experiment.train.steps):
        x, y = make_placeholder_batch(data_rng, experiment.train.batch_size, dim)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        if step % experiment.train.log_every == 0 or step == experiment.train.steps - 1:
            logger.log(step=step, loss=final_loss)

    summary = {
        "final_loss": final_loss,
        "n_params": n_params,
        "forward_flops": fwd_flops,
        "run_dir": run_dir,
        "git_dependent": False,
    }
    logger.summary(**summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    args = ap.parse_args()
    experiment = cfg.load_experiment(args.config)
    summary = run_smoke(experiment)
    print("SMOKE OK")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
