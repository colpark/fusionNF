"""Config schema (dataclasses) with YAML round-trip.

Design goals:
- Every knob in the dataset-difficulty spec (criterion C6) is a field here so the
  generator (Phase 1) is fully config-driven.
- Round-trip is lossless: from_yaml(to_yaml(c)) == c. This is required so a run's
  resolved config can be re-loaded and reproduced bit-for-bit (operating rule 3).
- No defaults are "tuned to advantage NF fusion" -- these are neutral generative
  knobs. Difficulty presets live in configs/*.yaml, not in code.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Any, get_type_hints

import yaml


# --------------------------------------------------------------------------- #
# Dataset generator config (implements the knobs in the brief, section 2).
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryConfig:
    """Prior over the shared instantaneous-frequency trajectory f(t).

    f(t) = clip(f0 + sum_k a_k * sin(2*pi*nu_k*t + phi_k), f_min, f_max)
    """
    f0: float = 5.0                 # Hz, center frequency of the trajectory
    f_min: float = 1.0              # Hz, clip floor
    f_max: float = 12.0             # Hz, clip ceiling
    n_components: int = 3           # number of slow sinusoids (K)
    nu_min: float = 0.05            # Hz, slowest trajectory-modulation rate
    nu_max: float = 0.5             # Hz, fastest trajectory-modulation rate (nonstationarity knob)
    amp_min: float = 0.5            # Hz, min per-component amplitude a_k
    amp_max: float = 2.0            # Hz, max per-component amplitude a_k


@dataclass
class ModalityAConfig:
    """Modality A: phase/frequency modulation (chirp)."""
    rate: float = 128.0             # r_A, samples/sec
    signal_amp: float = 1.0         # c, carries the matching component
    noise_beta: float = 1.0         # 1/f^beta background exponent
    n_octaves: int = 5              # S, number of background scales/bands


@dataclass
class ModalityBConfig:
    """Modality B: amplitude modulation (AM)."""
    rate: float = 96.0              # r_B (!= r_A by default, criterion C4)
    carrier: float = 40.0           # f_carrier, Hz
    am_depth: float = 1.0           # m, AM depth (carries matching component)
    noise_beta: float = 1.0         # 1/f^beta background exponent
    n_octaves: int = 5              # S, number of background scales/bands


@dataclass
class DataConfig:
    name: str = "tiny"
    generator: str = "chirp"        # "chirp" (FM/AM) | "ecg_ppg" (cardiac)
    duration: float = 4.0           # T, seconds
    snr_db: float = 0.0             # matching-band SNR relative to background (difficulty)
    jitter: float = 0.0             # fractional inter-modal timing jitter/warp (0 = none)
    p_positive: float = 0.5         # fraction of label-1 pairs
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    modality_a: ModalityAConfig = field(default_factory=ModalityAConfig)
    modality_b: ModalityBConfig = field(default_factory=ModalityBConfig)


# --------------------------------------------------------------------------- #
# Model / training / experiment config.
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    family: str = "placeholder"     # late | early | nf_autodecode | nf_amortized | placeholder
    hidden: int = 64
    depth: int = 2
    latent_dim: int = 32
    # Budget-matching fields (operating rule 4): comparisons must match these.
    param_budget: int = 0           # 0 = unconstrained; >0 asserts approx param parity


@dataclass
class TrainConfig:
    steps: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    n_train: int = 256
    n_val: int = 64
    n_test: int = 64
    log_every: int = 10
    device: str = "cpu"             # cpu | mps | cuda ; smoke runs on cpu


@dataclass
class ExperimentConfig:
    seed: int = 0
    tag: str = "default"
    out_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------- #
# Nested-dataclass <-> dict <-> YAML helpers.
# --------------------------------------------------------------------------- #
def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def from_dict(cls: type, data: dict) -> Any:
    """Reconstruct a (possibly nested) dataclass from a plain dict.

    Unknown keys raise -- a typo in a config must fail loudly, not be ignored.
    """
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = hints[f.name]
        val = data[f.name]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def to_yaml(obj: Any) -> str:
    return yaml.safe_dump(to_dict(obj), sort_keys=False, default_flow_style=False)


def save_yaml(obj: Any, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(to_yaml(obj))


def load_experiment(path: str) -> ExperimentConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return from_dict(ExperimentConfig, data)


def config_hash(obj: Any) -> str:
    """Stable short hash of a config, for run-dir naming and dedup."""
    import hashlib
    blob = yaml.safe_dump(to_dict(obj), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
