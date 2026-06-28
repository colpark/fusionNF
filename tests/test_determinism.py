"""Phase 0 gate tests: config round-trip + bit-reproducibility of a seeded run."""
from __future__ import annotations

import math

from src import config as cfg
from src.smoke import run_smoke


def _tiny(seed: int = 0, tag: str = "tinytest", out_root: str = "runs/_test") -> cfg.ExperimentConfig:
    c = cfg.load_experiment("configs/tiny.yaml")
    c.seed = seed
    c.tag = tag
    c.out_root = out_root
    return c


def test_config_round_trip(tmp_path):
    c = _tiny()
    p = tmp_path / "c.yaml"
    cfg.save_yaml(c, str(p))
    assert cfg.load_experiment(str(p)) == c


def test_unknown_key_rejected():
    import pytest
    with pytest.raises(ValueError):
        cfg.from_dict(cfg.ModelConfig, {"nonexistent_field": 1})


def test_same_seed_bit_reproducible():
    a = run_smoke(_tiny(seed=123, tag="rep_a"))
    b = run_smoke(_tiny(seed=123, tag="rep_b"))
    # Exact match expected on CPU; allow tiny tolerance for platform float quirks.
    assert math.isclose(a["final_loss"], b["final_loss"], rel_tol=0, abs_tol=1e-6)
    assert a["n_params"] == b["n_params"]


def test_different_seed_differs():
    a = run_smoke(_tiny(seed=1, tag="diff_a"))
    b = run_smoke(_tiny(seed=2, tag="diff_b"))
    assert a["final_loss"] != b["final_loss"]
