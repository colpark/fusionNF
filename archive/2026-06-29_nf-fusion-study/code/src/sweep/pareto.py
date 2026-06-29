"""Phase 6 figures: accuracy-vs-compute Pareto + accuracy/probe-R^2 vs difficulty.

Reads reports/phase6_sweep.json (written by sweep.runner) and writes PNGs to
reports/. The Pareto is plotted twice -- vs fusion FLOPs and vs total
(encode+fuse) FLOPs -- because the central tension lives in that split.

Run: uv run python -m src.sweep.pareto
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_MARK = {"late": "o", "early": "s", "nf_lainr": "^", "nf_omnifield": "D"}


def _by_family(cells):
    fam = defaultdict(list)
    for c in cells:
        fam[c["family"]].append(c)
    for f in fam:
        fam[f] = sorted(fam[f], key=lambda c: c["value"])
    return fam


def pareto_figure(data: dict, out: Path):
    fam = _by_family(data["cells"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, flop_key, title in [(axes[0], "fuse_flops", "Fusion FLOPs"),
                                (axes[1], "enc_flops", "Total (encode+fuse) FLOPs")]:
        for f, cs in fam.items():
            # accuracy averaged across difficulty values; compute is the median FLOPs
            acc = np.mean([c["acc_mean"] for c in cs])
            if flop_key == "enc_flops":
                comp = np.median([c["enc_flops"] + c["fuse_flops"] for c in cs])
            else:
                comp = np.median([c["fuse_flops"] for c in cs])
            ax.scatter(comp, acc, s=90, marker=_MARK.get(f, "o"), label=f)
            ax.annotate(f, (comp, acc), fontsize=8, xytext=(5, 4),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(f"{title} (log)")
        ax.set_ylabel("test accuracy (mean over sweep)")
        ax.set_title(f"Accuracy vs {title}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Pareto: accuracy vs compute  (base={data['base']}, knob={data['knob']})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def knob_figure(data: dict, out: Path):
    fam = _by_family(data["cells"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for f, cs in fam.items():
        x = [c["value"] for c in cs]
        acc = [c["acc_mean"] for c in cs]
        err = [c["acc_std"] for c in cs]
        r2 = [c["linear_r2_mean"] for c in cs]
        axes[0].errorbar(x, acc, yerr=err, marker=_MARK.get(f, "o"), capsize=3, label=f)
        axes[1].plot(x, r2, marker=_MARK.get(f, "o"), label=f)
    axes[0].set_xlabel(data["knob"]); axes[0].set_ylabel("test accuracy")
    axes[0].set_title(f"Accuracy vs {data['knob']}"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel(data["knob"]); axes[1].set_ylabel("linear probe R² of f(t)")
    axes[1].set_title(f"Probe-R² vs {data['knob']}"); axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main():
    path = _REPO / "reports" / "phase6_sweep.json"
    if not path.exists():
        raise SystemExit("run src.sweep.runner first (no reports/phase6_sweep.json)")
    data = json.loads(path.read_text())
    pareto_figure(data, _REPO / "reports" / "phase6_pareto.png")
    knob_figure(data, _REPO / "reports" / "phase6_knob.png")


if __name__ == "__main__":
    main()
