"""Dataset and splits (no leakage: splits are by *generative seed*, not index).

Each split owns a disjoint seed range, so a sample can never appear in two splits
and normalization stats (transforms.py) computed on train never see val/test
generative seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import DataConfig
from .generator import generate, Sample

# Disjoint seed offsets per split (each split must have < OFFSET samples).
SPLIT_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000}
Split = Literal["train", "val", "test"]


def split_seed(base_seed: int, split: Split, index: int) -> int:
    off = SPLIT_OFFSET[split]
    assert index < SPLIT_OFFSET["val"], "split size exceeds disjoint seed range"
    # multiply base by a large stride so different base seeds never collide
    return base_seed * 100_000_000 + off + index


class SignalPairDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, base_seed: int, split: Split, n: int,
                 return_components: bool = False):
        self.cfg = data_cfg
        self.base_seed = base_seed
        self.split = split
        self.n = n
        self.return_components = return_components

    def __len__(self) -> int:
        return self.n

    def raw(self, index: int) -> Sample:
        seed = split_seed(self.base_seed, self.split, index)
        return generate(self.cfg, seed, return_components=self.return_components)

    def __getitem__(self, index: int) -> dict:
        s = self.raw(index)
        return {
            "A": torch.from_numpy(s.A), "t_A": torch.from_numpy(s.t_A),
            "B": torch.from_numpy(s.B), "t_B": torch.from_numpy(s.t_B),
            "f_A": torch.from_numpy(s.f_A), "f_B": torch.from_numpy(s.f_B),
            "label": torch.tensor(float(s.label)),
        }


def collate(batch: list[dict]) -> dict:
    # Within a config, all A share length and all B share length, so plain stack works.
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def make_loaders(data_cfg: DataConfig, base_seed: int,
                 n_train: int, n_val: int, n_test: int,
                 batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        ds = SignalPairDataset(data_cfg, base_seed, split, n)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=collate, drop_last=False,
        )
    return loaders
