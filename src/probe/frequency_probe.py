"""Phase 5 -- the probe diagnostic (the real measurement).

The hypothesis is that a reconstruction-trained field latent *preserves* the fine
matching factor f(t), whereas a pooled late-fusion embedding discards it. We test
this directly: freeze a trained model's representation z, then see how well f(t) can
be recovered from z with a linear probe and a small-MLP probe. R^2 of the recovered
instantaneous-frequency trajectory is the measurement.

Three-way diagnostic (per the brief):
  probe high & fusion high -> mechanism confirmed
  probe high & fusion low  -> representation fine; fusion head is the bottleneck
  probe low                -> the field discarded the relevant info (representation failure)

We probe the trajectory of BOTH modalities (f_A from z, f_B from z) and report the
average test R^2, plus each. The target is f(t) resampled to a fixed length so a
fixed-width probe applies across configs with different sample rates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import DataConfig, TrainConfig
from ..data.dataset import SignalPairDataset
from ..data.transforms import NormStats, apply_norm, fit_norm
from ..utils.seeding import set_seed
from .. models.common import BaseFusion

_PROBE_M = 64          # trajectory resample length (probe output width)


def _resample(traj: np.ndarray, t: np.ndarray, m: int, duration: float) -> np.ndarray:
    grid = np.linspace(0.0, duration, m)
    return np.interp(grid, t, traj).astype(np.float32)


@torch.no_grad()
def collect(model: BaseFusion, data_cfg: DataConfig, base_seed: int, n: int,
            split: str, stats: NormStats, device: str, m: int = _PROBE_M):
    """Return (Z, F_A, F_B): frozen representations and resampled trajectories."""
    model.eval()
    ds = SignalPairDataset(data_cfg, base_seed, split, n)
    Z, FA, FB = [], [], []
    bs = 64
    buf = []
    raws = [ds.raw(i) for i in range(n)]
    for s in raws:
        FA.append(_resample(s.f_A, s.t_A, m, data_cfg.duration))
        FB.append(_resample(s.f_B, s.t_B, m, data_cfg.duration))
    for start in range(0, n, bs):
        chunk = raws[start:start + bs]
        batch = {
            "A": torch.from_numpy(np.stack([c.A for c in chunk])),
            "t_A": torch.from_numpy(np.stack([c.t_A for c in chunk])),
            "B": torch.from_numpy(np.stack([c.B for c in chunk])),
            "t_B": torch.from_numpy(np.stack([c.t_B for c in chunk])),
            "f_A": torch.zeros(1), "f_B": torch.zeros(1), "label": torch.zeros(1),
        }
        batch = apply_norm(batch, stats)
        batch = {k: v.to(device) for k, v in batch.items() if k in ("A", "t_A", "B", "t_B")}
        z = model.representation(batch["A"], batch["t_A"], batch["B"], batch["t_B"])
        Z.append(z.cpu().numpy())
    return np.concatenate(Z), np.stack(FA), np.stack(FB)


def _r2(y_true: np.ndarray, y_pred: np.ndarray, y_mean: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _ridge_r2(Ztr, Ytr, Zte, Yte, lam: float = 1.0) -> float:
    """Closed-form ridge linear probe; returns test R^2 (deterministic)."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Xtr = np.concatenate([(Ztr - mu) / sd, np.ones((len(Ztr), 1))], axis=1)
    Xte = np.concatenate([(Zte - mu) / sd, np.ones((len(Zte), 1))], axis=1)
    d = Xtr.shape[1]
    reg = lam * np.eye(d); reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Ytr)
    return _r2(Yte, Xte @ W, Ytr.mean(0, keepdims=True))


def _mlp_r2(Ztr, Ytr, Zte, Yte, device: str, hidden: int = 128,
            steps: int = 800, seed: int = 0) -> float:
    """Small-MLP probe trained with Adam; returns test R^2."""
    set_seed(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Xtr = torch.from_numpy(((Ztr - mu) / sd).astype(np.float32)).to(device)
    Xte = torch.from_numpy(((Zte - mu) / sd).astype(np.float32)).to(device)
    ytr = torch.from_numpy(Ytr.astype(np.float32)).to(device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (min(128, len(Xtr)),), generator=g)
        opt.zero_grad()
        loss = lossf(net(Xtr[idx]), ytr[idx])
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).cpu().numpy()
    return _r2(Yte, pred, Ytr.mean(0, keepdims=True))


def probe_model(model: BaseFusion, data_cfg: DataConfig, base_seed: int,
                n_train: int, n_test: int, device: str = "cpu") -> dict:
    """Freeze model, fit linear + MLP probes for f_A and f_B from z, return R^2s."""
    stats = fit_norm(data_cfg, base_seed, n_train)
    Ztr, FAtr, FBtr = collect(model, data_cfg, base_seed, n_train, "train", stats, device)
    Zte, FAte, FBte = collect(model, data_cfg, base_seed, n_test, "test", stats, device)
    out = {
        "linear_r2_fA": _ridge_r2(Ztr, FAtr, Zte, FAte),
        "linear_r2_fB": _ridge_r2(Ztr, FBtr, Zte, FBte),
        "mlp_r2_fA": _mlp_r2(Ztr, FAtr, Zte, FAte, device),
        "mlp_r2_fB": _mlp_r2(Ztr, FBtr, Zte, FBte, device),
        "latent_dim": int(Ztr.shape[1]),
    }
    out["linear_r2"] = 0.5 * (out["linear_r2_fA"] + out["linear_r2_fB"])
    out["mlp_r2"] = 0.5 * (out["mlp_r2_fA"] + out["mlp_r2_fB"])
    return out


def run(config: str = "easy", steps: int = 3000, n_train: int = 1024,
        device: str | None = None) -> dict:
    """Train each family, then probe f(t) from its frozen representation.

    Writes reports/phase5_probe.json. The diagnostic pairs probe-R^2 with the
    fusion test-accuracy from the same trained model (the three-way diagnostic).
    """
    import json
    from pathlib import Path
    from ..config import ModelConfig, TrainConfig, load_experiment
    from ..utils.device import auto_device
    from ..models.registry import FAMILIES
    from ..models.train import train_model, seeded_build, matched_plan

    repo = Path(__file__).resolve().parents[2]
    exp = load_experiment(str(repo / "configs" / f"{config}.yaml"))
    dc = exp.data
    dev = auto_device(device)
    base_seed = exp.seed
    tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, n_train=n_train,
                     n_val=128, n_test=256, log_every=max(1, steps // 2), device=dev)
    plan, target = matched_plan(FAMILIES, exp.model, dc)
    print(f"Phase-5 probe on '{config}' device={dev} (steps={steps}, n_train={n_train}); "
          f"param-matched target≈{target:,}\n")
    header = f"{'family':<16}{'fusion_acc':>11}{'lin_R2':>9}{'mlp_R2':>9}"
    print(header); print("-" * len(header))
    results = {}
    for fam in FAMILIES:
        mc = ModelConfig(family=fam, hidden=plan[fam][0], depth=exp.model.depth,
                         latent_dim=exp.model.latent_dim)
        model = seeded_build(fam, mc, dc, base_seed)
        res = train_model(model, dc, mc, tc, base_seed, recon_weight=0.3)
        pr = probe_model(model, dc, base_seed, n_train, tc.n_test, device=dev)
        results[fam] = {"fusion_acc": res["test_acc"], **pr}
        print(f"{fam:<16}{res['test_acc']:>11.3f}{pr['linear_r2']:>9.3f}{pr['mlp_r2']:>9.3f}")
    out = {"config": config, "device": dev, "steps": steps, "results": results}
    (repo / "reports" / "phase5_probe.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {repo / 'reports' / 'phase5_probe.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 5 probe diagnostic.")
    ap.add_argument("--config", default="easy")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run(config=a.config, steps=a.steps, n_train=a.n_train, device=a.device)
