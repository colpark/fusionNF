"""Real fusion TASK A: respiratory-rate (RR) estimation from ECG + PPG on BIDMC.

This is the faithful real instantiation of the hypothesis: the target (RR) lives in a
*buried, low-energy shared factor* -- respiration -- that modulates BOTH ECG (respiratory
sinus arrhythmia: R-R interval/amplitude) and PPG (baseline/amplitude modulation), encoded
differently in each. Unlike the HR correspondence probe, respiration is genuinely hard to
detect, so "does fusion preserve the buried factor?" is tested by "does it lower RR error?".

Ground truth RR comes from BIDMC's RESP channel (FFT peak in the respiratory band).
Regression target = breaths/min; metric = MAE (bpm). We compare unimodal ECG / unimodal
PPG / late / early / NF fusion, and report a classical RR bracket (RR estimated from each
modality's respiratory modulation).

Run:  python -m src.real.bidmc_rr --steps 4000 --seeds 0 1 2 --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from numpy.fft import rfft, rfftfreq
from scipy.signal import resample, find_peaks

from ..config import ModelConfig, TrainConfig, DataConfig, TrajectoryConfig, ModalityAConfig, ModalityBConfig
from ..utils.device import auto_device
from ..utils.seeding import set_seed
from ..data.transforms import apply_norm
from ..models.common import mlp
from ..models.late_fusion import SignalCNN
from ..models.registry import build_model
from ..models.train import seeded_build, matched_plan

FS = 125.0
_REPO_FAMILIES = ("late", "early", "nf_lainr", "nf_omnifield")


@dataclass
class RRConfig:
    data_dir: str = "data/bidmc"
    window_s: float = 32.0
    hop_s: float = 8.0
    ecg_rate: float = 64.0
    ppg_rate: float = 32.0
    resp_lo: float = 0.1          # respiratory band (Hz) ~6-36 bpm
    resp_hi: float = 0.6
    seed: int = 0


def _load(path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    cols = arr.dtype.names
    g = lambda key: np.asarray(arr[next(c for c in cols if key in c.upper())], dtype=np.float64)
    return g("II"), g("PLETH"), g("RESP")


def _dominant_freq(x, fs, lo, hi):
    x = x - x.mean()
    if x.std() < 1e-8:
        return np.nan
    X = np.abs(rfft(x)); f = rfftfreq(len(x), 1.0 / fs)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return np.nan
    return float(f[band][np.argmax(X[band])])


def rr_from_resp(resp, cfg):                       # ground-truth RR (bpm) by breath counting
    from scipy.ndimage import uniform_filter1d
    x = resp - resp.mean()
    x = uniform_filter1d(x, size=max(3, int(0.5 * FS)), mode="reflect")  # smooth
    if x.std() < 1e-8:
        return np.nan
    pk, _ = find_peaks(x, distance=int(FS / cfg.resp_hi), prominence=0.3 * x.std())
    if len(pk) < 2:
        return np.nan
    dur = (pk[-1] - pk[0]) / FS                     # span of counted breaths (s)
    return (len(pk) - 1) / dur * 60.0               # breaths/min (continuous, not FFT-binned)


def rr_from_ppg(ppg, rate, cfg):                   # classical PPG bracket (resp-band power)
    fr = _dominant_freq(ppg, rate, cfg.resp_lo, cfg.resp_hi)
    return fr * 60.0 if np.isfinite(fr) else np.nan


def rr_from_ecg(ecg, rate, cfg):                   # classical ECG bracket via R-R series (RSA)
    x = (ecg - np.median(ecg)) / (np.std(ecg) + 1e-8)
    pk, _ = find_peaks(x, distance=int(0.4 * rate), prominence=1.0)
    if len(pk) < 6:
        return np.nan
    tp = pk / rate; rr = np.diff(tp)                # R-R interval series (carries RSA)
    grid = np.linspace(tp[1], tp[-1], 256)
    series = np.interp(grid, 0.5 * (tp[1:] + tp[:-1]), rr)
    fs_series = 255.0 / (grid[-1] - grid[0])
    fr = _dominant_freq(series, fs_series, cfg.resp_lo, cfg.resp_hi)
    return fr * 60.0 if np.isfinite(fr) else np.nan


def build_windows(cfg: RRConfig):
    paths = sorted(glob.glob(os.path.join(cfg.data_dir, "*_Signals.csv")))
    if not paths:
        raise FileNotFoundError(f"no BIDMC CSVs in {cfg.data_dir} (run scripts/download_bidmc.sh)")
    rng = np.random.default_rng(cfg.seed); order = rng.permutation(len(paths))
    n = len(paths); n_tr = max(1, int(0.6 * n)); n_va = max(1, int(0.2 * n))
    split = {idx: ("train" if j < n_tr else "val" if j < n_tr + n_va else "test")
             for j, idx in enumerate(order)}
    w = int(cfg.window_s * FS); hop = int(cfg.hop_s * FS)
    We = int(cfg.window_s * cfg.ecg_rate); Wp = int(cfg.window_s * cfg.ppg_rate)
    t_e = (np.arange(We) / cfg.ecg_rate).astype(np.float32)
    t_p = (np.arange(Wp) / cfg.ppg_rate).astype(np.float32)
    wins = {"train": [], "val": [], "test": []}
    for idx, path in enumerate(paths):
        ecg, ppg, resp = _load(path)
        m = min(len(ecg), len(ppg), len(resp))
        for s in range(0, m - w + 1, hop):
            e, p, r = ecg[s:s + w], ppg[s:s + w], resp[s:s + w]
            rr = rr_from_resp(r, cfg)
            if not np.isfinite(rr):
                continue
            wins[split[idx]].append(dict(
                ecg=resample(e, We).astype(np.float32), ppg=resample(p, Wp).astype(np.float32),
                t_ecg=t_e, t_ppg=t_p, rr=np.float32(rr), record=idx))
    return wins, paths


@dataclass
class Norm:
    a_mean: float; a_std: float; b_mean: float; b_std: float; duration: float


def fit_norm(train, cfg):
    a = np.concatenate([w["ecg"] for w in train]); b = np.concatenate([w["ppg"] for w in train])
    rr = np.array([w["rr"] for w in train])
    return (Norm(float(a.mean()), float(a.std() + 1e-8), float(b.mean()), float(b.std() + 1e-8),
                 cfg.window_s), float(rr.mean()), float(rr.std() + 1e-8))


class RRset(torch.utils.data.Dataset):
    def __init__(self, windows): self.W = windows
    def __len__(self): return len(self.W)
    def __getitem__(self, i):
        w = self.W[i]
        return {"A": torch.from_numpy(w["ecg"]), "t_A": torch.from_numpy(w["t_ecg"]),
                "B": torch.from_numpy(w["ppg"]), "t_B": torch.from_numpy(w["t_ppg"]),
                "rr": torch.tensor(w["rr"])}


def _loaders(wins, batch):
    coll = lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]}
    mk = lambda ws, sh: torch.utils.data.DataLoader(RRset(ws), batch_size=batch, shuffle=sh,
                                                    collate_fn=coll, drop_last=sh)
    return {"train": mk(wins["train"], True), "val": mk(wins["val"], False),
            "test": mk(wins["test"], False)}


class UnimodalReg(nn.Module):
    """ECG-only or PPG-only CNN regressor (the single-modality baseline)."""
    def __init__(self, which, hidden=64, latent=32):
        super().__init__(); self.which = which
        self.enc = SignalCNN(hidden, latent, pooling="mean"); self.head = mlp([latent, hidden, 1])
    def forward(self, A, t_A, B, t_B):
        x = A if self.which == "ecg" else B
        return self.head(self.enc(x)).squeeze(-1)


def _dcfg(cfg):
    return DataConfig(name="bidmc_rr", duration=cfg.window_s,
                      trajectory=TrajectoryConfig(f_min=0.6, f_max=3.0),
                      modality_a=ModalityAConfig(rate=cfg.ecg_rate),
                      modality_b=ModalityBConfig(rate=cfg.ppg_rate, carrier=999.0))


def _build(family, hidden, dcfg):
    if family in ("uni_ecg", "uni_ppg"):
        return UnimodalReg("ecg" if family == "uni_ecg" else "ppg", hidden=hidden)
    return build_model(family, ModelConfig(family=family, hidden=hidden, depth=2, latent_dim=32), dcfg)


def train_regressor(model, loaders, norm, rr_mean, rr_std, steps, lr, device,
                    recon_weight=0.3, warmup_frac=0.1, grad_clip=1.0, seed=0):
    """Train one model to regress (standardized) RR; return test MAE in bpm."""
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(); has_recon = hasattr(model, "recon_loss")
    warm = max(1, int(warmup_frac * steps))

    def prep(b):
        b = apply_norm(b, norm)
        return {k: v.to(device) for k, v in b.items()}

    def stream():
        while True:
            yield from loaders["train"]
    model.train()
    for step, batch in enumerate(islice(stream(), steps), 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / warm)
        b = prep(batch)
        y = (b["rr"] - rr_mean) / rr_std
        if has_recon:
            enc = model.encode(b["A"], b["t_A"], b["B"], b["t_B"])
            pred = model.fuse(enc); loss = mse(pred, y) + recon_weight * model.recon_loss(enc)
        else:
            pred = model(b["A"], b["t_A"], b["B"], b["t_B"]); loss = mse(pred, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip); opt.step()

    model.eval(); errs = []
    with torch.no_grad():
        for batch in loaders["test"]:
            b = prep(batch)
            pred = model(b["A"], b["t_A"], b["B"], b["t_B"]) * rr_std + rr_mean
            errs.append((pred - b["rr"]).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def classical_bracket(wins, cfg):
    """Median MAE (bpm) of RR estimated from each modality alone vs RESP ground truth."""
    e, p = [], []
    for w in wins["test"]:
        re = rr_from_ecg(w["ecg"], cfg.ecg_rate, cfg)
        rp = rr_from_ppg(w["ppg"], cfg.ppg_rate, cfg)
        if np.isfinite(re): e.append(abs(re - w["rr"]))
        if np.isfinite(rp): p.append(abs(rp - w["rr"]))
    return {"ecg_mae": float(np.median(e)), "ppg_mae": float(np.median(p))}


def run(cfg, families, steps, seeds, batch, device, match, lr=1e-3):
    dev = auto_device(device); dcfg = _dcfg(cfg)
    fusion = [f for f in families if f in _REPO_FAMILIES]
    plan = (matched_plan(tuple(fusion), ModelConfig(hidden=64, depth=2, latent_dim=32), dcfg)[0]
            if match and fusion else {})
    res = {f: [] for f in families}
    rr_baseline = None
    for seed in seeds:
        c = RRConfig(**{**cfg.__dict__, "seed": seed})
        wins, paths = build_windows(c)
        norm, rr_mean, rr_std = fit_norm(wins["train"], c)
        loaders = _loaders(wins, batch)
        if seed == seeds[0]:
            br = classical_bracket(wins, c)
            # predict-the-mean baseline MAE (chance level for regression)
            te_rr = np.array([w["rr"] for w in wins["test"]])
            rr_baseline = {"predict_mean_mae": float(np.abs(te_rr - rr_mean).mean()),
                           "classical": br, "rr_mean_bpm": rr_mean, "rr_std_bpm": rr_std}
            print(f"REAL RR (BIDMC) | dev={dev} | records={len(paths)} | "
                  f"windows {[ (k,len(v)) for k,v in wins.items()]}")
            print(f"  bracket: predict-mean MAE={rr_baseline['predict_mean_mae']:.2f} bpm | "
                  f"classical ECG MAE={br['ecg_mae']:.2f}, PPG MAE={br['ppg_mae']:.2f} bpm")
        for f in families:
            hidden = plan.get(f, (64, None))[0]
            model = (UnimodalReg("ecg" if f == "uni_ecg" else "ppg", hidden=hidden)
                     if f in ("uni_ecg", "uni_ppg")
                     else seeded_build(f, ModelConfig(family=f, hidden=hidden, depth=2, latent_dim=32), dcfg, seed))
            mae = train_regressor(model, loaders, norm, rr_mean, rr_std, steps, lr, dev, seed=seed)
            res[f].append(mae); print(f"  seed {seed} {f:<14} test_MAE={mae:.2f} bpm")

    print(f"\n{'family':<16}{'RR MAE (bpm) mean±std':>26}")
    print("-" * 42)
    summ = {}
    for f in families:
        a = np.array(res[f]); summ[f] = {"mae_mean": float(a.mean()), "mae_std": float(a.std())}
        print(f"{f:<16}{a.mean():>18.2f} ± {a.std():.2f}")
    out = {"task": "bidmc_respiratory_rate", "window_s": cfg.window_s, "steps": steps,
           "seeds": seeds, "bracket": rr_baseline, "results": summ}
    os.makedirs("reports", exist_ok=True)
    open("reports/real_rr_results.json", "w").write(json.dumps(out, indent=2))
    print("\nwrote reports/real_rr_results.json")
    print("Read: fusion arms should beat the best UNIMODAL and approach the classical "
          "bracket; if NF > late/early as respiration gets fainter, the buried-factor "
          "claim transfers. predict-mean MAE is the no-skill floor.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=["uni_ecg", "uni_ppg", "late", "early", "nf_lainr", "nf_omnifield"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=float, default=32.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-match-params", action="store_true")
    a = ap.parse_args()
    cfg = RRConfig(window_s=a.window)
    run(cfg, a.families, a.steps, a.seeds, a.batch, a.device, match=not a.no_match_params)


if __name__ == "__main__":
    main()
