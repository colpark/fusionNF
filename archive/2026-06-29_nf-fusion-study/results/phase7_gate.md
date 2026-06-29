# Phase 7 Gate — Analysis & Findings

**Built:** `src/findings.py` — a data-driven generator that reads the saved
artifacts (phase2_validation, phase3_baselines, phase4_results, phase5_probe,
phase6_sweep) and renders `reports/findings.md`, evaluating each pre-registered
prediction as **supported / not supported / inconclusive** *from the numbers*, with
the evidence inline. Robust to missing artifacts (reports 'inconclusive' rather than
crashing). This is the single reproduce-from-artifacts command: `make report`.

**Predictions evaluated:**
- **P1** — NF advantage over late widens as difficulty rises (computed from the
  sweep gap at easy vs hard knob values).
- **P2** — NF approaches early-fusion accuracy at lower **fusion** FLOPs.
- **P3** — f(t) more decodable from the NF latent than the late pooled embedding
  (probe-R²). *Already trending supported even in smoke data: late R²≈0.50 <
  LAINR R²≈0.59.*
- **P4** — reframed (auto-decode arm dropped per design decision) as **LAINR vs
  OmniField**: per-modality field+head vs cross-modal-crosstalk field.

**Status:** generator complete and verified to run end-to-end. Final verdicts await
the **remote GPU** Phase 4 gate + Phase 6 sweep; `make report` regenerates
`findings.md` from whatever real artifacts are present. The findings template already
includes threats-to-validity and an honest regime-boundary bottom line.

**How to run (after remote train+sweep):**
```
uv run make report      # -> reports/findings.md with P1-P4 verdicts from real data
```

**Decision needed:** none for the code; this closes the harness build. Run the remote
GPU pipeline (train → recon → probe → sweep → report) to produce the scientific result.
