"""Phase 2 dataset validation suite (criteria C1-C7).

This is the most important gate: it protects every downstream conclusion. Each
criterion returns a structured result {name, passed, detail}. If a criterion fails,
the FIX is to the generator (falsification-first) -- never to relax the test.

Run: python -m src.validation.criteria_tests          # easy + hard
     python -m src.validation.criteria_tests --tag easy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..config import load_experiment, DataConfig
from ..data.dataset import SignalPairDataset
from ..data.generator import generate
from . import stats as S
from .unimodal import unimodal_accuracy


# --------------------------------------------------------------------------- #
# C1 -- joint-only label: unimodal classifiers at chance.
# --------------------------------------------------------------------------- #
def c1_unimodal_at_chance(dc: DataConfig, base_seed: int, n_train: int,
                          n_test: int, alpha: float = 0.01) -> dict:
    res = {}
    for m in ("A", "B"):
        res[m] = unimodal_accuracy(dc, m, base_seed, n_train, n_test)
    # Pass iff neither modality is significantly above chance (p > alpha both).
    passed = all(res[m]["p_value_above_chance"] > alpha for m in ("A", "B"))
    return {"name": "C1_joint_only_label", "passed": bool(passed),
            "detail": res, "alpha": alpha}


# --------------------------------------------------------------------------- #
# C2 -- marginal invariance across classes (per modality).
# --------------------------------------------------------------------------- #
def c2_marginal_invariance(dc: DataConfig, base_seed: int, n_per_class: int = 128,
                           alpha: float = 0.05, seed: int = 0) -> dict:
    ds = SignalPairDataset(dc, base_seed, "train", 100 * n_per_class)
    rates = {"A": dc.modality_a.rate, "B": dc.modality_b.rate}
    feats = {"A": {0: [], 1: []}, "B": {0: [], 1: []}}
    i = 0
    while min(len(feats["A"][0]), len(feats["A"][1])) < n_per_class:
        s = ds.raw(i); i += 1
        if len(feats["A"][s.label]) < n_per_class:
            feats["A"][s.label].append(S.summary_stats(s.A, rates["A"]))
            feats["B"][s.label].append(S.summary_stats(s.B, rates["B"]))
        if i > 1000 * n_per_class:
            break
    out = {}
    passed = True
    for m in ("A", "B"):
        X0 = np.stack(feats[m][0]); X1 = np.stack(feats[m][1])
        mmd = S.mmd_permutation_test(X0, X1, n_perm=200, seed=seed)
        out[m] = {"mmd2": mmd["mmd2"], "p_value": mmd["p_value"],
                  "n_per_class": n_per_class}
        # non-rejection at alpha = invariance holds
        if mmd["p_value"] <= alpha:
            passed = False
    return {"name": "C2_marginal_invariance", "passed": bool(passed),
            "detail": out, "alpha": alpha}


# --------------------------------------------------------------------------- #
# C3 -- multiscale structure + measured matching-band SNR.
# --------------------------------------------------------------------------- #
def c3_multiscale_and_snr(dc: DataConfig, base_seed: int, n: int = 64,
                          snr_tol_db: float = 1.0) -> dict:
    S_a, S_b = dc.modality_a.n_octaves, dc.modality_b.n_octaves
    band_pow = {"A": np.zeros(S_a), "B": np.zeros(S_b)}
    snr = {"A": [], "B": []}
    # regenerate with components (the default loader path doesn't request them)
    from ..data.dataset import split_seed
    for i in range(n):
        seed = split_seed(base_seed, "train", i)
        smp = generate(dc, seed, return_components=True)
        band_pow["A"] += S.octave_band_powers(smp.noise_A, dc.modality_a.rate, S_a)
        band_pow["B"] += S.octave_band_powers(smp.noise_B, dc.modality_b.rate, S_b)
        snr["A"].append(10 * np.log10(smp.clean_A.var() / (smp.noise_A.var() + 1e-12)))
        snr["B"].append(10 * np.log10(smp.clean_B.var() / (smp.noise_B.var() + 1e-12)))
    out = {}
    passed = True
    for m, n_oct in (("A", S_a), ("B", S_b)):
        bp = band_pow[m] / n
        # "resolvable" = band power within 40 dB of the strongest band
        strongest = bp.max()
        resolvable = int(np.sum(bp > strongest * 1e-4))
        measured_snr = float(np.mean(snr[m]))
        snr_ok = abs(measured_snr - dc.snr_db) <= snr_tol_db
        bands_ok = resolvable >= n_oct
        out[m] = {"resolvable_bands": resolvable, "configured_octaves": n_oct,
                  "measured_snr_db": measured_snr, "configured_snr_db": dc.snr_db,
                  "band_powers": bp.tolist(), "snr_ok": snr_ok, "bands_ok": bands_ok}
        passed = passed and snr_ok and bands_ok
    return {"name": "C3_multiscale_and_snr", "passed": bool(passed), "detail": out,
            "snr_tol_db": snr_tol_db}


# --------------------------------------------------------------------------- #
# C4 -- cross-modal heterogeneity: grid mismatch + jitter.
# --------------------------------------------------------------------------- #
def c4_grid_mismatch(dc: DataConfig, base_seed: int) -> dict:
    s = generate(dc, base_seed)
    rate_mismatch = dc.modality_a.rate != dc.modality_b.rate
    len_mismatch = s.A.shape[0] != s.B.shape[0]
    dtB = np.diff(s.t_B)
    jitter_present = float(dtB.std())
    jitter_ok = (jitter_present > 1e-9) if dc.jitter > 0 else (jitter_present < 1e-6)
    # If rates differ, lengths must differ. If jitter configured, it must be present.
    passed = (len_mismatch == rate_mismatch) and jitter_ok
    return {"name": "C4_grid_mismatch", "passed": bool(passed),
            "detail": {"rate_A": dc.modality_a.rate, "rate_B": dc.modality_b.rate,
                       "len_A": int(s.A.shape[0]), "len_B": int(s.B.shape[0]),
                       "rate_mismatch": rate_mismatch, "len_mismatch": len_mismatch,
                       "configured_jitter": dc.jitter,
                       "measured_jitter_std_dt": jitter_present, "jitter_ok": jitter_ok}}


# --------------------------------------------------------------------------- #
# C5 -- known ground-truth factor present and aligned.
# --------------------------------------------------------------------------- #
def c5_ground_truth(dc: DataConfig, base_seed: int, n: int = 32) -> dict:
    ds = SignalPairDataset(dc, base_seed, "train", n)
    ok = True
    detail = {}
    for i in range(n):
        s = ds.raw(i)
        ok = ok and s.f_A.shape == s.A.shape and s.f_B.shape == s.B.shape
        ok = ok and np.all(np.isfinite(s.f_A)) and np.all(np.isfinite(s.f_B))
    detail = {"f_A_shape_matches_A": True, "f_B_shape_matches_B": True,
              "all_finite": bool(ok), "checked": n}
    return {"name": "C5_ground_truth_present", "passed": bool(ok), "detail": detail}


# --------------------------------------------------------------------------- #
# C6 -- controllable difficulty: SNR sweep -> oracle flat, classical monotone.
# C7 -- in-principle solvable: oracle ~100%, classical > chance.
# Both use the Phase 3 baselines.
# --------------------------------------------------------------------------- #
def c6_difficulty_monotone(dc: DataConfig, base_seed: int, n: int = 200,
                           snr_grid=(-6.0, 0.0, 6.0, 12.0)) -> dict:
    from ..baselines.oracle import oracle_accuracy
    from ..baselines.wavelet_coherence import coherence_accuracy
    import dataclasses
    oracle_acc, coh_acc = [], []
    for snr in snr_grid:
        dc_s = dataclasses.replace(dc, snr_db=snr)
        oracle_acc.append(oracle_accuracy(dc_s, base_seed, n)["accuracy"])
        coh_acc.append(coherence_accuracy(dc_s, base_seed, n)["accuracy"])
    oracle_flat = (max(oracle_acc) - min(oracle_acc)) < 0.05 and min(oracle_acc) > 0.95
    # monotone non-decreasing within tolerance
    diffs = np.diff(coh_acc)
    coh_monotone = bool(np.all(diffs > -0.05))
    passed = oracle_flat and coh_monotone
    return {"name": "C6_difficulty_monotone", "passed": bool(passed),
            "detail": {"snr_grid": list(snr_grid), "oracle_acc": oracle_acc,
                       "coherence_acc": coh_acc, "oracle_flat": oracle_flat,
                       "coherence_monotone": coh_monotone}}


def c7_solvable(dc: DataConfig, base_seed: int, n: int = 256) -> dict:
    from ..baselines.oracle import oracle_accuracy
    from ..baselines.wavelet_coherence import coherence_accuracy
    oracle = oracle_accuracy(dc, base_seed, n)["accuracy"]
    coh = coherence_accuracy(dc, base_seed, n)["accuracy"]
    passed = oracle >= 0.98 and coh > 0.55
    return {"name": "C7_solvable", "passed": bool(passed),
            "detail": {"oracle_acc": oracle, "coherence_acc": coh}}


# --------------------------------------------------------------------------- #
# Runner.
# --------------------------------------------------------------------------- #
def run_all(tag: str, base_seed: int = 0, include_baselines: bool = True) -> dict:
    exp = load_experiment(f"configs/{tag}.yaml")
    dc = exp.data
    n_train = min(exp.train.n_train, 512)
    n_test = min(exp.train.n_test, 256)
    results = []
    results.append(c1_unimodal_at_chance(dc, base_seed, n_train, n_test))
    results.append(c2_marginal_invariance(dc, base_seed))
    results.append(c3_multiscale_and_snr(dc, base_seed))
    results.append(c4_grid_mismatch(dc, base_seed))
    results.append(c5_ground_truth(dc, base_seed))
    if include_baselines:
        results.append(c6_difficulty_monotone(dc, base_seed))
        results.append(c7_solvable(dc, base_seed))
    all_pass = all(r["passed"] for r in results)
    return {"tag": tag, "all_pass": all_pass, "results": results}


def _print(report: dict):
    print(f"\n===== Dataset validation: {report['tag']}  "
          f"(ALL PASS = {report['all_pass']}) =====")
    for r in report["results"]:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{flag}] {r['name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="easy|hard|tiny; default runs easy+hard")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    tags = [args.tag] if args.tag else ["easy", "hard"]
    os.makedirs("reports", exist_ok=True)
    all_reports = {}
    for tag in tags:
        rep = run_all(tag, include_baselines=not args.no_baselines)
        _print(rep)
        all_reports[tag] = rep
    with open("reports/phase2_validation.json", "w") as fh:
        json.dump(all_reports, fh, indent=2)
    print("\nwrote reports/phase2_validation.json")


if __name__ == "__main__":
    main()
