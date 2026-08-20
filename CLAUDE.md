# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick start

```powershell
# Symbolic separation (seconds, numpy only)
python run_phase_a.py --graphs 25 --nodes 24

# Parameterised family — fastest falsification
python run_family_spike.py --gate 0

# Full Phase F ladder (~10 min, CPU only)
python run_family_spike.py --gate all

# Identifiability boundary vs graph depth (minutes)
python run_identifiability.py --graphs 12 --nodes 24

# E3 architecturally-distinct procedure test
python run_e3.py

# Train 6 MPNNs and fingerprint each (needs CUDA, ~25 min)
python run_seed_control.py --seeds 6 --epochs 1500 --probe-graphs 10 --budget 24

# E1+E2 suite with smoke test (plumbing check, ~2 min)
python run_e1_e2.py --smoke

# E5 sensitivity normalisation (no CUDA, loads existing checkpoints, ~1 min)
python run_e5.py

# E6 Dijkstra/Prim key-dissociation probe (numpy only, seconds)
python run_e6.py

# G reference-free discrimination: networks vs each other (no CUDA, ~5 min)
python run_g.py --stage 1
python run_g.py --stage 2        # trains 12 fresh checkpoints, needs CUDA (~40 min)
```

Environment: Windows, PowerShell. Python 3.14, numpy 2.4, PyTorch 2.11+cu128 with CUDA. Run all scripts from the repo root. JAX and DeepMind CLRS are not installed and are not needed.

## Architecture

The project has two layers — a symbolic layer (`fpid/` minus `neural.py`/`train.py`) and a neural layer — sharing a common observable state `(d, pi)`.

### Core library (`fpid/`)

| Module | Role |
|---|---|
| `graphs.py` | Mutable graph container, generators, ground-truth SSSP |
| `algorithms.py` | Steppable `Executor` subclasses: BF, truncated-BF, SPFA, Dijkstra, Prim. All expose the same `(d, pi)` state so trajectories are comparable. |
| `interventions.py` | Nine probes, targeting rule, progress-relative firing time (`settle_round`), `settle_round` |
| `fingerprint.py` | Predicate extraction + aggregation across graph instances |
| `separation.py` | Pairwise separation, minimal separating set |
| `identify.py` | Nearest-reference classifier, margin |
| `noisy.py` | Four noise axes for predicate robustness testing |
| `neural.py` | `MPNN`, `GatedMPNN`, `NeuralExecutor` — state-encoded recurrence required by intervention harness |
| `train.py` | Batched training, hint/no-hint regimes, dataset generation |
| `response.py` | Round Jacobian estimator. **Old `width`/`row_sum` claims were wrong** — read the docstring before using those statistics. |
| `family.py` | Parameterised relaxation family `(α, τ, γ, c, k, selection)` — all five reference algorithms are exact limit points in `REFERENCE_LIMITS` |
| `estimate.py` | Closed-form estimators for α and γ, temperature calibration, `FitReport.withheld` |

### Intervention pipeline

1. Draw graph from distribution with propagation depth ≥ 5 (depth < 4 is provably non-separating)
2. Run algorithm to produce **control** trajectory
3. Fire intervention at `t = clip(T/2, 1, budget-1)` where `T` is the settling round — progress-relative, never absolute
4. Compare **intervened** vs **control** (not vs the reference run)
5. Evaluate qualitative predicates; aggregate modal values across instances
6. Compute Hamming distance over jointly-stable predicates (stability is pairwise, not intersected globally)

### Experiment runners

`run_e1_e2.py` contains the shared gate machinery (Fisher exact, admission, family-collapse, predicate ablation) and is imported by the other runners. Start there to understand the experimental scaffolding before reading a later runner.

Pre-registration files (`PREREGISTRATION_*.md`) are written before each experiment runs. `GATE1_DISQUALIFICATIONS.md` is the Phase F companion for confounds discovered between gates.

## Key constraints (violations silently corrupt results)

1. **Probe graphs must have propagation depth ≥ 5.** Below depth ~4 no intervention set separates all pairs; dense shallow graphs return a null about nothing.
2. **Interventions fire relative to each algorithm's own progress** — never at an absolute round number.
3. **Relaxation is monotone** — it only lowers `d`. A model that responds to an edge deletion is not doing relaxation.
4. **`UNREACHED` is graph-family dependent.** Default `1e3` suits density-0.05 n=24 graphs. Depth-17 path graphs need it raised.
5. **Response measures compare intervened vs control**, not intervened vs reference. The reference run defines *recovery*, not the expected response.
6. **Predicate stability is applied pairwise**, not intersected globally.
7. **Never tie test tolerance to the noise amplitude it probes** — `tol = k*σ` produces a flat 100% that is a tautology, not a robustness result.
8. **Model admission is by MAE**, not validation accuracy. At MAE ~5 no tolerance satisfies both the error floor and the edge-weight scale; the model cannot be fingerprinted.
9. **State interventions require state-encoded recurrence.** A latent-only model silently ignores every state poke.

## Methodological rules

These are hard-won; four results in this project evaporated by violating them.

- **Pre-register** — fix the candidate list and decision rule, generate both seed sets up front, select on the first and report on the second in one pass.
- **Verify the manipulation before reading the measurement.** E1 looked like a null about fingerprints; the cross-fit trajectory check showed the training difference had barely been installed, making it a null about MPNN capacity. Opposite next steps.
- **Quote `d_near`, the margin, and the reference-to-reference scale together** or do not quote the label. A nearest-reference label is meaningless unless `d_near` is small relative to the reference-to-reference median (0.25–0.30 here).
- **Direction replicating while resolution does not is the characteristic signature of a dead lead** — this is what both `overshoot` and E3b's truncation statistic looked like. Leave no intermediate rung in the outcome ladder for that state to land on.

## Current project status (as of 2026-08-20)

Symbolic work: complete and positive. Neural work: finished and negative on all tested approaches, with the three remaining objections closed: E5 shows the vacuity gap is not a sensitivity-normalisation artefact; E6's attempt at a sharper Dijkstra/Prim probe failed its own mechanism check informatively (edge-weight raises are as invisible to relaxation as deletions, for the same monotonicity reason); and **G closed the reference-free question** — the fingerprints do not separate the networks from *each other* either (chance purity on 12 fresh seeds, 0/8 on gated-vs-plain, 11 cross-arm pairs bit-identical), which removes the last defence available to E3's null, that the label rather than the instrument was broken. The recommended next action is writing up. See `HANDOFF.md` for the full status table and `ROUTES_TO_POSITIVE.md` for the ranked menu of what remains open — after G its top item is §2.1 + §2.2. The `FINDINGS_RESIDUAL.md` exploratory follow-up carries its own frozen confirmatory design and is explicitly not yet a result.
