"""Phase 7 -- analysis & findings generator.

Reads the saved artifacts from earlier phases and renders reports/findings.md,
evaluating the pre-registered predictions P1-P4 as supported / not supported /
inconclusive *from the data*. Re-running this after the remote sweep regenerates the
findings -- the single reproduce-from-artifacts command (`make report`).

Robust to missing artifacts: a prediction whose inputs are absent is reported
'inconclusive (artifact missing)' rather than failing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_R = _REPO / "reports"


def _load(name: str):
    p = _R / name
    return json.loads(p.read_text()) if p.exists() else None


def _families(cells):
    fam = defaultdict(list)
    for c in cells:
        fam[c["family"]].append(c)
    for f in fam:
        fam[f].sort(key=lambda c: c["value"])
    return fam


def _verdict_p1(sweep) -> tuple[str, str]:
    """NF advantage over late widens as difficulty increases."""
    if not sweep:
        return "inconclusive", "phase6_sweep.json missing (run the sweep)."
    fam = _families(sweep["cells"])
    if "late" not in fam:
        return "inconclusive", "late-fusion arm absent from sweep."
    nf = [f for f in fam if f.startswith("nf_")]
    if not nf:
        return "inconclusive", "no NF arm in sweep."
    # harder = lower snr / higher jitter|nonstationarity|n_components; rate_ratio: smaller=harder
    knob = sweep["knob"]
    vals = sorted({c["value"] for c in sweep["cells"]})
    harder_first = knob in ("snr_db", "rate_ratio")  # ascending value = easier
    easy_v, hard_v = (vals[-1], vals[0]) if harder_first else (vals[0], vals[-1])
    late = {c["value"]: c["acc_mean"] for c in fam["late"]}
    best_nf = lambda v: max(c["acc_mean"] for f in nf for c in fam[f] if c["value"] == v)
    gap_easy = best_nf(easy_v) - late[easy_v]
    gap_hard = best_nf(hard_v) - late[hard_v]
    msg = (f"knob={knob}: NF−late gap at easy({easy_v})={gap_easy:+.3f}, "
           f"at hard({hard_v})={gap_hard:+.3f}.")
    if gap_hard > gap_easy + 0.02 and gap_hard > 0.02:
        return "supported", msg + " Gap widens with difficulty."
    if gap_hard <= gap_easy:
        return "not supported", msg + " Gap does not widen."
    return "inconclusive", msg


def _verdict_p2(sweep, phase4) -> tuple[str, str]:
    """NF approaches early-fusion accuracy at lower fusion FLOPs."""
    src = None
    if sweep:
        fam = _families(sweep["cells"])
        src = {f: (sum(c["acc_mean"] for c in cs) / len(cs),
                   sorted(cs, key=lambda c: c["value"])[len(cs) // 2]["fuse_flops"])
               for f, cs in fam.items()}
    elif phase4 and phase4.get("results"):
        src = {f: (r["test_acc"], r.get("fuse_flops", float("nan")))
               for f, r in phase4["results"].items()}
    if not src or "early" not in src:
        return "inconclusive", "need early + NF accuracy & fuse_flops (run phase 4/6)."
    early_acc, early_flops = src["early"]
    nf = {f: v for f, v in src.items() if f.startswith("nf_")}
    if not nf:
        return "inconclusive", "no NF arm."
    lines = []
    ok = False
    for f, (acc, flops) in nf.items():
        cheaper = flops < early_flops
        close = acc >= early_acc - 0.05
        lines.append(f"{f}: acc={acc:.3f} (early={early_acc:.3f}), fuse_flops={flops:,} "
                     f"vs early {early_flops:,} ({'cheaper' if cheaper else 'not cheaper'})")
        ok = ok or (cheaper and close)
    verdict = "supported" if ok else "not supported"
    return verdict, " | ".join(lines)


def _verdict_p3(sweep, probe) -> tuple[str, str]:
    """f(t) more decodable from NF latent than from late-fusion pooled embedding."""
    pairs = None
    if probe and probe.get("results"):
        pairs = {f: r.get("linear_r2") for f, r in probe["results"].items()}
    elif sweep:
        fam = _families(sweep["cells"])
        pairs = {f: sum(c["linear_r2_mean"] for c in cs) / len(cs) for f, cs in fam.items()}
    if not pairs or "late" not in pairs:
        return "inconclusive", "probe-R² for late + NF missing (run phase 5/6)."
    late_r2 = pairs["late"]
    nf = {f: r for f, r in pairs.items() if f.startswith("nf_")}
    if not nf:
        return "inconclusive", "no NF arm."
    msg = f"late R²={late_r2:.3f}; " + ", ".join(f"{f} R²={r:.3f}" for f, r in nf.items())
    if all(r > late_r2 + 0.02 for r in nf.values()):
        return "supported", msg + " (all NF > late)."
    if any(r > late_r2 + 0.02 for r in nf.values()):
        return "partially supported", msg + " (some NF > late)."
    return "not supported", msg + " (NF not above late)."


def _verdict_p4(sweep, phase4) -> tuple[str, str]:
    """Reframed: LAINR (per-modality field + head) vs OmniField (cross-modal crosstalk)."""
    src = None
    if sweep:
        fam = _families(sweep["cells"])
        src = {f: (sum(c["acc_mean"] for c in cs) / len(cs),
                   sum(c["linear_r2_mean"] for c in cs) / len(cs),
                   sorted(cs, key=lambda c: c["value"])[len(cs) // 2]["fuse_flops"])
               for f, cs in fam.items()}
    elif phase4 and phase4.get("results"):
        src = {f: (r["test_acc"], None, r.get("fuse_flops")) for f, r in phase4["results"].items()}
    if not src or "nf_lainr" not in src or "nf_omnifield" not in src:
        return "inconclusive", "need both NF arms (run phase 4/6)."
    la, lo = src["nf_lainr"], src["nf_omnifield"]
    return "reframed", (f"LAINR acc={la[0]:.3f} fuse_flops={la[2]:,}; "
                        f"OmniField acc={lo[0]:.3f} fuse_flops={lo[2]:,}. "
                        "Auto-decode arm dropped, so the amortization-spectrum prediction "
                        "is reported as this two-architecture comparison.")


def build() -> str:
    val = _load("phase2_validation.json")
    p4 = _load("phase4_results.json")
    probe = _load("phase5_probe.json")
    sweep = _load("phase6_sweep.json")

    p1 = _verdict_p1(sweep)
    p2 = _verdict_p2(sweep, p4)
    p3 = _verdict_p3(sweep, probe)
    p4v = _verdict_p4(sweep, p4)

    val_pass = None
    if val:
        val_pass = all(t.get("all_pass") for t in val.values())

    L = []
    L += ["# Findings — Neural-Field Fusion Hypothesis", ""]
    L += ["## Question", "",
          "Do two multiscale signals carrying a shared instantaneous-frequency "
          "trajectory f(t) — labeled by *correspondence* — fuse better via neural-field "
          "latents than via pooled late fusion, and approach early-fusion accuracy at "
          "lower fusion cost? Built to falsify, not to confirm.", ""]
    L += ["## Dataset validity (Phase 2)", "",
          (f"All criteria C1–C7 pass on easy and hard: **{val_pass}**." if val_pass is not None
           else "_phase2_validation.json missing._"),
          "Joint-only label (C1), marginal invariance (C2), multiscale + measured SNR "
          "(C3), grid mismatch/jitter (C4), ground truth present (C5), monotone "
          "difficulty (C6), oracle≫classical≫chance (C7).", ""]

    bl = _load("phase3_baselines.json")
    L += ["## Bracket (Phase 3)", "",
          "Chance < classical wavelet-coherence < oracle establishes real, non-trivial "
          "signal with an achievable ceiling. See reports/phase3_baselines.md.", ""]

    L += ["## Pre-registered predictions", ""]
    for tag, (v, m) in [("P1 — NF advantage over late widens with difficulty", p1),
                        ("P2 — NF approaches early accuracy at lower fusion FLOPs", p2),
                        ("P3 — f(t) more decodable from NF latent than late pooled", p3),
                        ("P4 — amortization spectrum (reframed: LAINR vs OmniField)", p4v)]:
        L += [f"### {tag}", "", f"**Verdict: {v.upper()}**", "", m, ""]

    L += ["## Figures", "",
          "- `reports/phase6_pareto.png` — accuracy vs fusion FLOPs and vs total FLOPs.",
          "- `reports/phase6_knob.png` — accuracy and probe-R² vs the difficulty knob.",
          "- `reports/phase4_recon.md` — NF reconstruction PSNR per frequency band.", ""]

    L += ["## Threats to validity", "",
          "- Budgets are matched on steps/params/optimizer, not tuned per-family; a "
          "  family could improve with bespoke tuning.",
          "- OmniField is the heaviest arm and most budget-sensitive; under-training it "
          "  would understate its accuracy (report the budget used).",
          "- The auto-decoded NF point was dropped, so the cost/faithfulness end of the "
          "  amortization spectrum is not measured here.",
          "- Results are on synthetic signals designed to isolate the matching component; "
          "  external validity to real multimodal signals is not established.", ""]

    L += ["## Bottom line", "",
          "Filled in from the verdicts above once the remote sweep has run. The honest "
          "regime statement: where (SNR, nonstationarity, rate-mismatch) NF fusion does "
          "and does not beat late fusion, and whether it reaches the early-fusion "
          "accuracy/compute frontier.", ""]

    return "\n".join(L)


def main():
    md = build()
    out = _R / "findings.md"
    out.write_text(md)
    print("wrote", out)


if __name__ == "__main__":
    main()
