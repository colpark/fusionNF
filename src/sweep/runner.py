"""Phase 6 -- the sweep (headline result).

Runs the matrix {difficulty knob values} x {fusion families} x {seeds}, recording
test accuracy, probe-R^2, and the encode/fuse FLOP split per cell. Also runs the
required controls (unimodal-at-chance reuse of C1, and a shuffled-pairs control).
Everything is config-driven and GPU-auto-detecting so the heavy run lives on the
remote server.

Outputs JSON to reports/phase6_sweep.json; figures are produced by sweep.pareto.

Run examples:
  uv run python -m src.sweep.runner --base hard --knob snr_db --values -6 -3 0 6 12 \
       --families late early nf_lainr nf_omnifield --seeds 0 1 2 --steps 3000
  uv run python -m src.sweep.runner --base hard --pareto-only         # just the Pareto cells
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from ..config import DataConfig, ModelConfig, TrainConfig, load_experiment
from ..data.dataset import make_loaders
from ..data.transforms import apply_norm, fit_norm
from ..utils.device import auto_device
from ..models.registry import build_model, FAMILIES
from ..models.train import train_model, seeded_build, matched_plan, _flop_split, _prep
from ..probe.frequency_probe import probe_model

_REPO = Path(__file__).resolve().parents[2]

# Knob name -> setter on a DataConfig (returns a new DataConfig).
def _set_knob(dc: DataConfig, knob: str, value) -> DataConfig:
    if knob == "snr_db":
        return dataclasses.replace(dc, snr_db=float(value))
    if knob == "jitter":
        return dataclasses.replace(dc, jitter=float(value))
    if knob == "nonstationarity":   # max trajectory-modulation rate
        return dataclasses.replace(dc, trajectory=dataclasses.replace(
            dc.trajectory, nu_max=float(value)))
    if knob == "n_components":
        return dataclasses.replace(dc, trajectory=dataclasses.replace(
            dc.trajectory, n_components=int(value)))
    if knob == "rate_ratio":        # B rate as a fraction of A rate (grid mismatch)
        new_rb = float(value) * dc.modality_a.rate
        return dataclasses.replace(dc, modality_b=dataclasses.replace(
            dc.modality_b, rate=new_rb))
    raise ValueError(f"unknown knob {knob!r}")


def run_cell(family: str, dc: DataConfig, exp_model: ModelConfig, tc: TrainConfig,
             seed: int, recon_weight: float, probe_n: int,
             hidden: int | None = None) -> dict:
    """Train one family on one config/seed; return accuracy, probe-R^2, FLOPs."""
    mc = ModelConfig(family=family, hidden=hidden or exp_model.hidden,
                     depth=exp_model.depth, latent_dim=exp_model.latent_dim)
    model = seeded_build(family, mc, dc, seed)
    res = train_model(model, dc, mc, tc, seed, recon_weight=recon_weight)
    pr = probe_model(model, dc, seed, min(tc.n_train, probe_n), tc.n_test, device=tc.device)
    # FLOP split on one normalized batch
    stats = fit_norm(dc, seed, tc.n_train)
    batch = _prep(next(iter(make_loaders(dc, seed, tc.n_train, tc.n_val, tc.n_test,
                                         tc.batch_size)["test"])), stats, tc.device)
    enc_f, fuse_f = _flop_split(model, {k: batch[k] for k in ("A", "t_A", "B", "t_B")})
    return {"test_acc": res["test_acc"], "val_acc": res["val_acc"],
            "linear_r2": pr["linear_r2"], "mlp_r2": pr["mlp_r2"],
            "n_params": res["n_params"], "enc_flops": enc_f, "fuse_flops": fuse_f}


def shuffled_pairs_control(family: str, dc: DataConfig, exp_model: ModelConfig,
                           tc: TrainConfig, seed: int) -> float:
    """Train with B shuffled across the batch (correspondence destroyed).

    Marginals are preserved but the label no longer matches the (A, B') pairing, so
    any above-chance accuracy would signal leakage. Returns test accuracy.
    """
    import torch
    import torch.nn as nn
    from itertools import islice
    from ..utils.seeding import set_seed
    set_seed(seed)
    mc = ModelConfig(family=family, hidden=exp_model.hidden, depth=exp_model.depth,
                     latent_dim=exp_model.latent_dim)
    model = build_model(family, mc, dc).to(tc.device)
    loaders = make_loaders(dc, seed, tc.n_train, tc.n_val, tc.n_test, tc.batch_size)
    stats = fit_norm(dc, seed, tc.n_train)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)
    bce = nn.BCEWithLogitsLoss()

    def shuffle_b(b):
        perm = torch.randperm(b["B"].shape[0])
        b = dict(b); b["B"] = b["B"][perm]; b["t_B"] = b["t_B"][perm]
        return b

    def inf(l):
        while True:
            yield from l
    model.train()
    for batch in islice(inf(loaders["train"]), tc.steps):
        b = shuffle_b(_prep(batch, stats, tc.device))
        opt.zero_grad()
        loss = bce(model(b["A"], b["t_A"], b["B"], b["t_B"]), b["label"])
        loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loaders["test"]:
            b = shuffle_b(_prep(batch, stats, tc.device))
            pred = (torch.sigmoid(model(b["A"], b["t_A"], b["B"], b["t_B"])) > 0.5).float()
            correct += int((pred == b["label"]).sum()); total += b["label"].numel()
    return correct / max(1, total)


def sweep(base: str, knob: str, values: list, families: list, seeds: list,
          steps: int, n_train: int, device: str | None, recon_weight: float,
          controls: bool, match_params: bool = True) -> dict:
    exp = load_experiment(str(_REPO / "configs" / f"{base}.yaml"))
    dev = auto_device(device)
    tc0 = dict(steps=steps, batch_size=32, lr=1e-3, optimizer="adam",
               n_train=n_train, n_val=128, n_test=256, log_every=max(1, steps),
               device=dev)
    # parameter-match all families to the largest (computed once on the base config)
    if match_params:
        plan, target = matched_plan(tuple(families), exp.model, exp.data)
        print(f"param-matched (target≈{target:,}): " +
              ", ".join(f"{f} h{h}→{p:,}" for f, (h, p) in plan.items()))
    else:
        plan = {f: (exp.model.hidden, None) for f in families}
    print(f"Sweep base={base} knob={knob} values={values} families={families} "
          f"seeds={seeds} device={dev} match_params={match_params}")

    cells = []
    for value in values:
        dc = _set_knob(exp.data, knob, value)
        tc = TrainConfig(**tc0)
        for family in families:
            accs, lin, mlp, encf, fusef = [], [], [], [], []
            for seed in seeds:
                r = run_cell(family, dc, exp.model, tc, seed, recon_weight, n_train,
                             hidden=plan[family][0])
                accs.append(r["test_acc"]); lin.append(r["linear_r2"])
                mlp.append(r["mlp_r2"]); encf.append(r["enc_flops"]); fusef.append(r["fuse_flops"])
            cell = {"knob": knob, "value": value, "family": family,
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "linear_r2_mean": float(np.mean(lin)), "linear_r2_std": float(np.std(lin)),
                    "mlp_r2_mean": float(np.mean(mlp)),
                    "enc_flops": int(np.median(encf)), "fuse_flops": int(np.median(fusef)),
                    "n_seeds": len(seeds)}
            cells.append(cell)
            print(f"  {knob}={value} {family:<14} acc={cell['acc_mean']:.3f}"
                  f"±{cell['acc_std']:.3f}  linR2={cell['linear_r2_mean']:.3f}")

    out = {"base": base, "knob": knob, "device": dev, "steps": steps,
           "n_train": n_train, "seeds": seeds, "cells": cells}

    if controls:
        ctrl = {}
        dc = exp.data
        tc = TrainConfig(**tc0)
        for family in families:
            ctrl[family] = shuffled_pairs_control(family, dc, exp.model, tc, seeds[0])
            print(f"  [control] shuffled-pairs {family:<14} acc={ctrl[family]:.3f} (expect ~chance)")
        out["shuffled_pairs_control"] = ctrl

    (_REPO / "reports" / "phase6_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {_REPO / 'reports' / 'phase6_sweep.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 6 difficulty x family sweep.")
    ap.add_argument("--base", default="hard")
    ap.add_argument("--knob", default="snr_db",
                    choices=["snr_db", "jitter", "nonstationarity", "n_components", "rate_ratio"])
    ap.add_argument("--values", nargs="+", default=["-6", "-3", "0", "6", "12"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--recon-weight", type=float, default=0.3)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-match-params", action="store_true")
    a = ap.parse_args()
    # numeric knob values
    vals = [float(v) if a.knob != "n_components" else int(v) for v in a.values]
    sweep(a.base, a.knob, vals, a.families, a.seeds, a.steps, a.n_train,
          a.device, a.recon_weight, controls=not a.no_controls,
          match_params=not a.no_match_params)


if __name__ == "__main__":
    main()
