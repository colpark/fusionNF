"""Phase 3 brackets: chance < classical < oracle, for easy and hard.

Runnable as ``python -m src.baselines.brackets``. For each difficulty tag it
computes three bracketing accuracies on the held-out test split:

* **chance**    -- majority-class accuracy (the trivial floor, ~0.5).
* **classical** -- the model-structure-aware coherence baseline (no ground truth).
* **oracle**    -- the ground-truth correspondence ceiling (~1.0).

Results are printed as a table, written to ``reports/phase3_baselines.md``, and
plotted to ``reports/phase3_baselines.png``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import load_experiment
from ..data.dataset import SignalPairDataset
from .oracle import oracle_accuracy
from .wavelet_coherence import coherence_accuracy

# Repo root is two levels up from this file (src/baselines/brackets.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = {"easy": _REPO_ROOT / "configs" / "easy.yaml",
            "hard": _REPO_ROOT / "configs" / "hard.yaml"}
_REPORTS = _REPO_ROOT / "reports"
_BASE_SEED = 0
_CAP = 256  # cap train (threshold fit) and test (eval) sizes for speed


def chance_accuracy(data_cfg, base_seed: int, n: int, split: str = "test") -> float:
    """Majority-class accuracy: predict the more frequent label, always."""
    ds = SignalPairDataset(data_cfg, base_seed, split, n)
    labels = np.array([int(ds.raw(i).label) for i in range(n)], dtype=np.int64)
    p_pos = labels.mean()
    return float(max(p_pos, 1.0 - p_pos))


def evaluate_tag(tag: str) -> dict:
    """Compute chance/classical/oracle accuracy for one difficulty tag."""
    exp = load_experiment(str(_CONFIGS[tag]))
    data_cfg = exp.data
    n_eval = min(exp.train.n_test, _CAP)
    n_fit = min(exp.train.n_train, _CAP)
    # The accuracy helpers fit a threshold on `n` train samples and evaluate on `n`
    # test samples; use the smaller cap so both stages stay within budget.
    n = min(n_eval, n_fit)

    chance = chance_accuracy(data_cfg, _BASE_SEED, n_eval, split="test")
    classical = coherence_accuracy(data_cfg, _BASE_SEED, n, split="test")
    oracle = oracle_accuracy(data_cfg, _BASE_SEED, n, split="test")
    return {
        "tag": tag,
        "n": n,
        "chance": chance,
        "classical": classical["accuracy"],
        "classical_threshold": classical["threshold"],
        "oracle": oracle["accuracy"],
        "oracle_threshold": oracle["threshold"],
    }


def _print_table(rows: list[dict]) -> None:
    header = f"{'tag':<6} {'n':>5} {'chance':>8} {'classical':>10} {'oracle':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['tag']:<6} {r['n']:>5} {r['chance']:>8.3f} "
              f"{r['classical']:>10.3f} {r['oracle']:>8.3f}")


def _write_report(rows: list[dict]) -> Path:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / "phase3_baselines.md"
    lines = [
        "# Phase 3 -- Baselines & Brackets",
        "",
        "Correspondence accuracy bracketed by a trivial floor (chance), a classical "
        "signal-processing anchor (FM/AM demodulation + coherence, no ground truth), "
        "and a ground-truth ceiling (oracle).",
        "",
        f"- Base seed: `{_BASE_SEED}`",
        f"- Samples per stage: capped at `{_CAP}` (train for threshold fit, test for eval)",
        "- Threshold fit on the train split, evaluated on the disjoint test split.",
        "",
        "| Tag | n | Chance | Classical | Oracle |",
        "|-----|---|--------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['n']} | {r['chance']:.3f} | "
            f"{r['classical']:.3f} | {r['oracle']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Chance** ~ 0.5 confirms balanced classes; nothing is learnable for free.",
        "- **Oracle** ~ 1.0 confirms the matching signal is fully present: if a model "
        "could recover the latent trajectories, the task is solved.",
        "- **Classical** sits between the two. On `easy` it clears chance comfortably, "
        "showing hand-built demodulation already extracts real cross-modal structure. "
        "On `hard` (low SNR, timing jitter, mismatched rates, richer trajectories) the "
        "classical anchor degrades -- this is the gap a learned fusion model must close.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _write_figure(rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / "phase3_baselines.png"

    tags = [r["tag"] for r in rows]
    series = {
        "chance": [r["chance"] for r in rows],
        "classical": [r["classical"] for r in rows],
        "oracle": [r["oracle"] for r in rows],
    }
    colors = {"chance": "#9e9e9e", "classical": "#1f77b4", "oracle": "#2ca02c"}

    x = np.arange(len(tags))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[name])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Phase 3 baselines: chance < classical < oracle")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    rows = [evaluate_tag(tag) for tag in ("easy", "hard")]
    _print_table(rows)
    report_path = _write_report(rows)
    figure_path = _write_figure(rows)
    print()
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
