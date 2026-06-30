"""smFRET experiment: sweep photon brightness (the SNR knob) across all arms.

Falsifiable predictions (the structural-win claim):
  - CLEAN (high photon rate): nf_fourier TIES the cheap comparator (no recovery problem left).
  - INTERMEDIATE: nf_fourier WINS (raw-time Fourier rep keeps the fast structure the binned
    CNNs lose); margin should PEAK here.
  - UNRECOVERABLE (few photons): everyone floors to chance (~0.5); nf_fourier must NOT win.
  - nf_relu ~ binned CNNs (isolates the Fourier basis, not "neural field", as the mechanism).
A monotone nf_fourier advantage everywhere would FALSIFY the structural claim.

Run:  python -m src.smfret.run --device cuda:0           (self-contained simulator)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .data import SimConfig, make_split, fourier_freqs
from .models import FAMILIES, build_model
from ..utils.device import auto_device
from ..utils.seeding import set_seed

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def train_eval(model, tr, te, steps, lr, dev, seed, batch=64):
    set_seed(seed)
    model = model.to(dev)
    tr = {k: v.to(dev) for k, v in tr.items()}; te = {k: v.to(dev) for k, v in te.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr); bce = nn.BCEWithLogitsLoss()
    n = tr["y"].shape[0]; warm = max(1, int(0.1 * steps))
    model.train()
    for step in range(1, steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, step / warm)
        idx = torch.randint(0, n, (batch,), device=dev)
        b = {k: v[idx] for k, v in tr.items()}
        loss = bce(model(b), b["y"])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(te)) > 0.5).float() == te["y"]).float().mean().item()
    return acc


def run(rates, families, n_train, n_test, steps, seeds, device, lr=1e-3):
    dev = auto_device(device)
    base = SimConfig()
    freqs = fourier_freqs(base)
    print(f"smFRET sweep | dev={dev} | rates(photons/s)={rates} | families={list(families)} "
          f"| n_train={n_train} n_test={n_test} steps={steps} seeds={seeds}")
    grid = {f: {} for f in families}
    for rate in rates:
        cfg = SimConfig(photon_rate=rate)
        for f in families:
            accs = []
            for seed in seeds:
                tr = make_split(cfg, n_train, 1000 + seed)
                te = make_split(cfg, n_test, 9000 + seed)
                accs.append(train_eval(build_model(f, cfg, freqs), tr, te, steps, lr, dev, seed))
            grid[f][rate] = (float(np.mean(accs)), float(np.std(accs)))
            print(f"  rate={rate:>6.0f}  {f:<12} acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    # table
    print("\nphotons/s |" + "".join(f"{f:>14}" for f in families))
    print("-" * (11 + 14 * len(families)))
    for rate in rates:
        print(f"{rate:>9.0f} |" + "".join(f"{grid[f][rate][0]:>10.3f}±{grid[f][rate][1]:.2f}"
                                          for f in families))
    out = {"task": "smfret_kinetics", "rates": rates, "n_train": n_train, "steps": steps,
           "seeds": seeds, "results": {f: {str(r): grid[f][r] for r in rates} for f in families}}
    os.makedirs(os.path.join(_REPO, "reports"), exist_ok=True)
    p = os.path.join(_REPO, "reports", "smfret_sweep.json")
    open(p, "w").write(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    print("Read: nf_fourier should TIE comparator when clean, WIN at intermediate rates "
          "(peak margin), and TIE-at-floor when unrecoverable; nf_relu ~ binned CNNs. "
          "A monotone nf_fourier win everywhere FALSIFIES the structural claim.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", type=float, default=[1000, 300, 100, 50, 20, 10])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run(a.rates, a.families, a.n_train, a.n_test, a.steps, a.seeds, a.device)


if __name__ == "__main__":
    main()
