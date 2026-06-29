"""High-capacity single-modality classifiers for criterion C1.

If a classifier that sees ONLY modality A (or only B) can predict the label above
chance, the label has leaked into a single modality and any 'fusion win' would be a
unimodal shortcut. C1 asserts these classifiers sit at chance. Self-contained (does
not import the Phase 4 models) so Phase 2 stands alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import DataConfig
from ..data.dataset import SignalPairDataset
from ..utils.seeding import set_seed


class UnimodalCNN(nn.Module):
    """Deliberately high-capacity 1D CNN -> global pool -> MLP head."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x):                 # x: (B, N)
        h = self.net(x.unsqueeze(1))      # (B, width, L)
        h = h.mean(dim=-1)                # global average pool
        return self.head(h).squeeze(-1)   # (B,)


def _load_modality(ds: SignalPairDataset, modality: str, n: int):
    xs, ys = [], []
    for i in range(n):
        s = ds.raw(i)
        xs.append(s.A if modality == "A" else s.B)
        ys.append(float(s.label))
    return np.stack(xs), np.array(ys, dtype=np.float32)


def unimodal_accuracy(data_cfg: DataConfig, modality: str, base_seed: int,
                      n_train: int, n_test: int, steps: int = 400,
                      batch_size: int = 64, lr: float = 1e-3, width: int = 64,
                      device: str = "cpu") -> dict:
    """Train a one-modality classifier; return test accuracy + binomial p-value
    for H0: accuracy == 0.5 (one-sided, alternative > 0.5)."""
    set_seed(base_seed)
    tr = SignalPairDataset(data_cfg, base_seed, "train", n_train)
    te = SignalPairDataset(data_cfg, base_seed, "test", n_test)
    Xtr, ytr = _load_modality(tr, modality, n_train)
    Xte, yte = _load_modality(te, modality, n_test)

    # normalize using TRAIN stats only
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    dev = torch.device(device)
    model = UnimodalCNN(width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()

    g = torch.Generator().manual_seed(base_seed)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr_t), (batch_size,), generator=g)
        xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(torch.from_numpy(Xte).float().to(dev))) > 0.5)
        pred = pred.cpu().numpy().astype(np.float32)
    acc = float((pred == yte).mean())
    n_correct = int((pred == yte).sum())

    from scipy.stats import binomtest
    p = binomtest(n_correct, n_test, 0.5, alternative="greater").pvalue
    # train accuracy too, to confirm the classifier had the capacity to learn *if*
    # there were signal (a sanity check that "chance" isn't just an untrained net).
    with torch.no_grad():
        tr_pred = (torch.sigmoid(model(Xtr_t.to(dev))) > 0.5).cpu().numpy().astype(np.float32)
    train_acc = float((tr_pred == ytr).mean())
    return {"modality": modality, "test_acc": acc, "train_acc": train_acc,
            "p_value_above_chance": float(p), "n_test": n_test}
