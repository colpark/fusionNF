"""Train/evaluate the fusion families on REAL ECG+PPG (BIDMC) correspondence.

Reuses the model registry, training loop, and ridge probe. GPU auto-detected. Heavy
runs belong on the remote; on CPU pass --steps small.

Run:
  uv run python -m src.real.train_real --steps 4000 --seeds 0 1 2
  uv run python -m src.real.train_real --snr 0 --steps 4000        # inject motion noise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import (DataConfig, ModelConfig, TrainConfig, TrajectoryConfig,
                      ModalityAConfig, ModalityBConfig)
from ..utils.device import auto_device
from ..data.transforms import apply_norm
from ..models.registry import FAMILIES
from ..models.train import train_model, seeded_build, matched_plan
from ..probe.frequency_probe import _ridge_r2
from .bidmc import RealConfig, make_real_loaders

_REPO = Path(__file__).resolve().parents[2]


def _data_cfg(cfg: RealConfig) -> DataConfig:
    """A DataConfig mirroring the real setup so build_model sizes encoders correctly."""
    return DataConfig(name="bidmc", duration=cfg.window_s,
                      trajectory=TrajectoryConfig(f_min=0.6, f_max=3.0),
                      modality_a=ModalityAConfig(rate=cfg.ecg_rate),
                      modality_b=ModalityBConfig(rate=cfg.ppg_rate, carrier=999.0))


@torch.no_grad()
def _probe(model, loader, stats, device, M=64):
    model.eval(); Z, F = [], []
    for b in loader:
        bn = apply_norm(b, stats)
        z = model.representation(bn["A"].to(device), bn["t_A"].to(device),
                                 bn["B"].to(device), bn["t_B"].to(device)).cpu().numpy()
        Z.append(z)
        fa = b["f_A"].numpy(); ta = b["t_A"].numpy()
        for k in range(len(fa)):
            F.append(np.interp(np.linspace(0, ta[k][-1], M), ta[k], fa[k]))
    return np.concatenate(Z), np.stack(F).astype(np.float32)


def run(cfg: RealConfig, steps, seeds, families, batch, device, match, per_split):
    dev = auto_device(device)
    dcfg = _data_cfg(cfg)
    base_model = ModelConfig(hidden=64, depth=2, latent_dim=32)
    plan = (matched_plan(tuple(families), base_model, dcfg)[0] if match
            else {f: (64, None) for f in families})
    print(f"REAL ECG+PPG (BIDMC) | device={dev} | match_params={match}")
    if match:
        print("  " + ", ".join(f"{f} h{h}→{p:,}" for f, (h, p) in plan.items()))

    results = {f: {"acc": [], "probe_r2": []} for f in families}
    for seed in seeds:
        rc = RealConfig(**{**cfg.__dict__, "seed": seed})
        loaders, stats, counts, n_rec = make_real_loaders(rc, batch_size=batch, per_split=per_split)
        if seed == seeds[0]:
            print(f"  records={n_rec}  windows {counts}")
        tc = TrainConfig(steps=steps, batch_size=batch, lr=1e-3, optimizer="adam",
                         n_test=counts["test"], device=dev)
        for fam in families:
            mc = ModelConfig(family=fam, hidden=plan[fam][0], depth=2, latent_dim=32)
            model = seeded_build(fam, mc, dcfg, seed)
            res = train_model(model, dcfg, mc, tc, seed, recon_weight=0.3,
                              loaders=loaders, stats=stats)
            Ztr, Ftr = _probe(model, loaders["train"], stats, dev)
            Zte, Fte = _probe(model, loaders["test"], stats, dev)
            r2 = _ridge_r2(Ztr, Ftr, Zte, Fte)
            results[fam]["acc"].append(res["test_acc"])
            results[fam]["probe_r2"].append(r2)
            print(f"  seed {seed} {fam:<14} test_acc={res['test_acc']:.3f} probe_R2={r2:.3f}")

    print(f"\n{'family':<16}{'test_acc':>16}{'probe_R2(HR)':>16}")
    print("-" * 48)
    summ = {}
    for f in families:
        a, r = np.array(results[f]["acc"]), np.array(results[f]["probe_r2"])
        summ[f] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
                   "probe_r2_mean": float(r.mean())}
        print(f"{f:<16}{a.mean():>10.3f}±{a.std():.3f}{r.mean():>16.3f}")
    out = {"dataset": "bidmc", "snr_db": cfg.snr_db, "ppg_rate": cfg.ppg_rate,
           "window_s": cfg.window_s, "steps": steps, "seeds": seeds, "results": summ}
    (_REPO / "reports").mkdir(exist_ok=True)
    tag = "real_ecg_ppg" + ("" if cfg.snr_db is None else f"_snr{cfg.snr_db:g}")
    (_REPO / "reports" / f"{tag}_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote reports/{tag}_results.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bidmc")
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--ppg-rate", type=float, default=64.0)
    ap.add_argument("--snr", type=float, default=None, help="inject motion noise at this dB; omit for real noise only")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-match-params", action="store_true")
    ap.add_argument("--per-split", type=int, default=None, help="cap windows/split (speed)")
    a = ap.parse_args()
    cfg = RealConfig(data_dir=a.data_dir, window_s=a.window, ppg_rate=a.ppg_rate, snr_db=a.snr)
    run(cfg, a.steps, a.seeds, a.families, a.batch, a.device,
        match=not a.no_match_params, per_split=a.per_split)


if __name__ == "__main__":
    main()
