"""Generic training loop for all four fusion families (Phase 4).

Every family shares the same ``forward = fuse(encode(...))`` contract, so a single
loop trains them all. The only family-specific behaviour is the neural-field
reconstruction auxiliary loss: if a model exposes ``recon_loss(encoded)`` it is
added to the classification loss so the field actually learns to reconstruct (an
unconstrained field would otherwise produce junk latents).

Run ``uv run python -m src.models.train`` to train all four families on the easy
config for a short self-check and print test accuracy, parameter counts, and the
encode-vs-fuse FLOP split per family.
"""
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from ..config import DataConfig, ModelConfig, TrainConfig, load_experiment
from ..data.dataset import make_loaders
from ..data.transforms import apply_norm, fit_norm, NormStats
from ..utils.device import auto_device
from ..utils.flops import count_params, flop_scope
from ..utils.seeding import set_seed
from .common import BaseFusion
from .registry import build_model, FAMILIES

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_optimizer(model: nn.Module, tc: TrainConfig) -> torch.optim.Optimizer:
    opt = tc.optimizer.lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=tc.lr,
                                weight_decay=tc.weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tc.lr,
                                 weight_decay=tc.weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tc.lr, momentum=0.9,
                               weight_decay=tc.weight_decay)
    raise ValueError(f"unknown optimizer {tc.optimizer!r}")


def _infinite(loader) -> Iterator[dict]:
    while True:
        yield from loader


def _prep(batch: dict, stats: NormStats, device: str) -> dict:
    batch = apply_norm(batch, stats)
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: BaseFusion, loader, stats: NormStats, device: str) -> float:
    """Classification accuracy over a loader (encode runs even under no_grad)."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        b = _prep(batch, stats, device)
        logit = model(b["A"], b["t_A"], b["B"], b["t_B"])
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += int((pred == b["label"]).sum().item())
        total += b["label"].numel()
    return correct / max(1, total)


def train_model(model: BaseFusion, data_cfg: DataConfig, model_cfg: ModelConfig,
                train_cfg: TrainConfig, base_seed: int,
                run_dir: str | None = None, recon_weight: float = 1.0,
                logger=None) -> dict:
    """Train one model and return ``{test_acc, val_acc, n_params, train_loss}``."""
    set_seed(base_seed)
    device = train_cfg.device

    loaders = make_loaders(data_cfg, base_seed, train_cfg.n_train,
                           train_cfg.n_val, train_cfg.n_test, train_cfg.batch_size)
    stats = fit_norm(data_cfg, base_seed, train_cfg.n_train)

    model = model.to(device)
    opt = _make_optimizer(model, train_cfg)
    bce = nn.BCEWithLogitsLoss()
    has_recon = hasattr(model, "recon_loss")

    model.train()
    running = 0.0
    last_loss = float("nan")
    stream = _infinite(loaders["train"])
    for step, batch in enumerate(islice(stream, train_cfg.steps), start=1):
        b = _prep(batch, stats, device)
        encoded = model.encode(b["A"], b["t_A"], b["B"], b["t_B"])
        logit = model.fuse(encoded)
        loss = bce(logit, b["label"])
        if has_recon:
            loss = loss + recon_weight * model.recon_loss(encoded)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().item())
        running += last_loss
        if step % max(1, train_cfg.log_every) == 0:
            avg = running / train_cfg.log_every
            running = 0.0
            if logger is not None:
                logger.log(step=step, train_loss=avg)

    val_acc = evaluate(model, loaders["val"], stats, device)
    test_acc = evaluate(model, loaders["test"], stats, device)
    n_params = count_params(model)
    result = {
        "test_acc": test_acc,
        "val_acc": val_acc,
        "n_params": n_params,
        "train_loss": last_loss,
    }
    if logger is not None:
        logger.summary(**result)
    return result


def _flop_split(model: BaseFusion, batch: dict) -> tuple[int, int]:
    """Measure encode-FLOPs and fuse-FLOPs separately for one batch."""
    model.eval()
    with flop_scope() as get_enc:
        encoded = model.encode(batch["A"], batch["t_A"], batch["B"], batch["t_B"])
    enc_flops = get_enc()
    with flop_scope() as get_fuse:
        model.fuse(encoded)
    fuse_flops = get_fuse()
    return enc_flops, fuse_flops


def gate_run(config: str = "easy", steps: int = 3000, n_train: int = 1024,
             device: str | None = None, recon_weight: float = 0.3,
             seed: int | None = None) -> dict:
    """Train all four families on a config and report acc + FLOP split.

    Defaults are sized for a GPU run (the Phase 4 gate). On CPU, pass a smaller
    --steps/--n-train. Writes reports/phase4_results.json and returns the dict.
    """
    exp = load_experiment(str(_REPO_ROOT / "configs" / f"{config}.yaml"))
    data_cfg = exp.data
    base_seed = exp.seed if seed is None else seed
    dev = auto_device(device)

    tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, optimizer="adam",
                     n_train=n_train, n_val=128, n_test=256,
                     log_every=max(1, steps // 4), device=dev)

    print(f"Phase-4 gate on '{data_cfg.name}' device={dev} "
          f"(steps={tc.steps}, n_train={tc.n_train}, n_test={tc.n_test})\n")
    header = f"{'family':<16}{'test_acc':>9}{'val_acc':>9}{'params':>10}" \
             f"{'enc_FLOPs':>14}{'fuse_FLOPs':>14}"
    print(header)
    print("-" * len(header))

    # One normalized batch (for FLOP measurement), shared across families' shapes.
    stats = fit_norm(data_cfg, base_seed, tc.n_train)
    probe_loader = make_loaders(data_cfg, base_seed, tc.n_train, tc.n_val,
                                tc.n_test, tc.batch_size)["test"]
    probe_batch = _prep(next(iter(probe_loader)), stats, tc.device)

    results = {}
    for family in FAMILIES:
        model_cfg = ModelConfig(family=family, hidden=exp.model.hidden,
                                depth=exp.model.depth,
                                latent_dim=exp.model.latent_dim)
        model = build_model(family, model_cfg, data_cfg)
        # recon weight < 1 so the field's reconstruction objective shapes the latent
        # without starving the correspondence classifier (NF arms only).
        res = train_model(model, data_cfg, model_cfg, tc, base_seed,
                          recon_weight=recon_weight)
        enc_f, fuse_f = _flop_split(
            model,
            {k: probe_batch[k] for k in ("A", "t_A", "B", "t_B")},
        )
        res["enc_flops"], res["fuse_flops"] = enc_f, fuse_f
        results[family] = res
        print(f"{family:<16}{res['test_acc']:>9.3f}{res['val_acc']:>9.3f}"
              f"{res['n_params']:>10,}{enc_f:>14,}{fuse_f:>14,}")

    ok = all(r["test_acc"] > 0.55 for r in results.values())
    floor = min(r["test_acc"] for r in results.values())
    print(f"\nAll four > 0.55 (above chance): {ok}  (min test_acc = {floor:.3f})")

    out = {"config": config, "device": dev, "steps": steps, "n_train": n_train,
           "seed": base_seed, "gate_pass": bool(ok), "results": results}
    out_path = _REPO_ROOT / "reports" / "phase4_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 4 gate: train the four fusion families.")
    ap.add_argument("--config", default="easy")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--recon-weight", type=float, default=0.3)
    args = ap.parse_args()
    gate_run(config=args.config, steps=args.steps, n_train=args.n_train,
             device=args.device, recon_weight=args.recon_weight)
