# Pre-registration: parametric family spike (Phase F)

Frozen 2026-08-13, before `run_family_spike.py` was executed on anything but Gate 0a.
Every threshold below is fixed. Nothing in this file is refitted after the fact.

Reproduce with:

```bash
python run_family_spike.py --gate all
```

## Why this exists

`FINDINGS_DECOMPOSITION.md` established that nearest-reference identification of a neural
model is vacuous here: every neural model sits 0.37–0.40 from its nearest symbolic reference
while the reference-to-reference median is 0.303, so the classifier reports the least-bad of
five poor matches (`THEORY.md` Proposition 5). The diagnosis is that the hypothesis space —
five discrete algorithms — does not contain the measured object, which is a damped,
soft-aggregating dynamical system whose *limit* is a discrete algorithm.

Phase F replaces classification with parameter estimation over a family that contains both.
This spike does not attempt the science; it tests whether the method can work at all. **If
parameters put in by hand cannot be read back out, the reframe is dead and Phase F stops
here.** Out of scope, and staying out: the crystallisation experiment (does fitted temperature
rise as a model hardens), any model zoo, any training run.

## The family

`fpid/family.py`. Six parameters `θ = (α, τ, γ, c, k, selection)`; update rule and the
justification for each design choice are in that module's docstring. `τ = 1/β`, so the hard
min is the interior point `τ = 0`.

Reference algorithms as points (`REFERENCE_LIMITS`), all at `α=1, τ=0, c=0`:

| reference | γ | k | selection |
|---|---|---|---|
| Bellman-Ford | 1 | n−1 | sync |
| truncated BF (k=3) | 1 | 3 | sync |
| SPFA | 1 | — | fifo |
| Dijkstra | 1 | — | argmin |
| Prim | **0** | — | argmin |

`c = 0` for all five: Dijkstra's and Prim's settling is *selection* commitment, supplied by
`argmin`. The `c` axis is standstill commitment, which no reference algorithm has. That
`c > 0` is uninhabited by the reference set is the point — it is where learned models may sit.

## Frozen constants

| constant | value | why this value |
|---|---|---|
| `eps` (probe amplitude) | 0.25 | `response.jacobian_at` default; well under the 1–10 weight scale |
| `theta` (support threshold) | 0.05 | `response.summarise` default |
| `kappa` (standstill tolerance = `kappa·eps`) | 1.0 | a row moving less than the probe amplitude sits at a kink where the difference is a secant |
| `reached_frac` | 0.9 | `d < 0.9·UNREACHED` counts as reached, matching `GatedMPNN` |
| `UNREACHED` | 80.0 | unchanged from `fpid/algorithms.py` |
| rounds `T` | 8 | profile length |
| n | 24 | graph size, unchanged from every prior phase |
| probe graphs | `random_connected_graph(24, density=0.05)`, min depth 5 | matches `deep_probe_graphs` |

`kappa` is a hand-picked constant and is therefore **swept** over `{0.5, 1.0, 2.0}` and the
sweep reported, rather than defended. The frozen value for all verdicts is 1.0.

## Grid

```
alpha      {0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0}
tau        {0, 0.25, 0.5, 1, 2, 4, 8}          # 0 = hard min
gamma      {0.0, 0.25, 0.5, 0.75, 0.9, 1.0}
c          {0, 0.1, 0.25, 0.5, 1.0}
k          {3, 5, 8, 12, 24, None}
selection  {sync, argmin}
```

`fifo` is in the family and is used in Gate 0a, but is **excluded from the recovery grid**.
Reason, pre-committed as a prediction to be measured in Gate 1: the response profile cannot
separate SPFA from Dijkstra at hard min, so including `fifo` would manufacture a confound
rather than resolve one. In the family's own vocabulary the distinction is commitment, not
queue order — `(argmin, c=0)` revisits nodes as SPFA does. **Prediction: argmin-vs-fifo
SNR < 3.** If it exceeds 3, that is a bonus finding and is reported as one.

## Observable

`fpid/estimate.py`. Profile vector `z(θ)` = concat over `t = 1..8` of
`(diag_frac, self_med, off_row_med, off_pr, off_rows/n, off_width, row_sum_med, n_fit/n)`,
each coordinate standardised by its across-graph-instance SD at the grid reference point.
`nan` medians (an empty mask, or no row with off-diagonal mass) are imputed to 0.0; `n_fit`
and `off_rows` are never `nan` and carry the "no information" signal themselves.

Estimators, all measured on the masked row set (reached **and** moving):

| parameter | statistic | relation |
|---|---|---|
| α | `self_med` | `α = 1 − self_med`, closed form |
| γ | `off_row_med` | `γ = off_row_med / α̂`, closed form |
| τ | `off_pr` | monotone, inverted through a calibration table `off_pr = f(τ; α, γ)` |
| selection | `off_rows` | one sender → few live rows; sync → frontier-sized |
| k | first `t` with `off_rows = 0` | exact step |
| c | shape of the `diag_frac(t)` ramp | calibration |

**Parameter-free membership test, reported for every fit:** the family predicts
`row_sum_med = 1 − α(1−γ)` identically. With α and γ pinned by the diagonal and the
off-diagonal, `row_sum_deviation` falsifies family membership with no fitted parameters.

## Goodness of fit

```
chi2 = mean over (statistic j, round t) of ( (z_obs − z_fit) / s[j,t] )^2
ve   = 1 − Var_resid / Var_grid
```

`s[j,t]` is the across-instance SD measured on the **target itself**, so `chi2 ≈ 1` means the
family contains the object with no reference-to-reference scale needed. `Var_grid` is the
variance of `z` across the grid.

Frozen verdicts, no middle rung:

- **family adequate** iff `chi2 ≤ 4` **and** `ve ≥ 0.90`
- **family still wrong** otherwise; `chi2 > 9` is the headline "does not contain this object"
- high `ve` with high `chi2` means the fit found the right region and the family still cannot
  reach the object. That combination is reported explicitly, not averaged away.

## Gates

### Gate 0a — embedding

Five references vs the family at `reference_params`, 20 graphs (10 `random_connected_graph`
density 0.05, 10 `high_diameter_graph`), budget 24, both initialised from `UNREACHED`.
Compare `d` at every round and every node.

**PASS: `max |d_family − d_ref| < 1e-9` for 5/5.** These are exact special cases; a looser
tolerance would hide a bug. A failure here is a code fix, not a finding.

*Status at freeze time: already run, PASS at exactly 0.0 for all five.*

### Gate 0b — observable adequacy (manipulation check)

For each of the six axes, step it away from the reference point with the others held fixed.
The standardised profile must move by more than the within-θ measurement noise.

**PASS: all 6 axes reach marginal SNR ≥ 2 somewhere along the axis.**

This is the project's own rule — verify the independent variable moved before measuring the
outcome — applied inside the method. Without it, Gate 1 would report a confound where the
truth is a dead observable, which is E1's error pointed inward.

### Gate 1 — identifiability

`SNR(θ,θ′) = D(θ,θ′) / sqrt(S_within(θ)·S_within(θ′))`, where `D` is the standardised profile
distance and `S_within` is the median split-half profile distance over the graph set.
Distinguishable iff **SNR ≥ 3** — three not two, because a split-half median is optimistic and
this project treats an effect comparable to noise as absent.

Instrument: 15 pairwise 2-D slices through the reference point `(α=1, τ=0, γ=1, c=0, k=None,
sync)`, plus a 256-point Latin hypercube over the full box for distant collisions. The full
outer product is 17,640 points and is both unaffordable and the wrong question.

**PASS: (i) every axis pair has min-SNR ≥ 3 at two grid steps; (ii) ≥90% of LHS pairs have
SNR ≥ 3; (iii) sentinel invariance `max |z(θ; U=80) − z(θ; U=160)| < 1e-6` over the grid.**

Reported in full **regardless of verdict**: the 6×6 confound matrix, per-axis marginal SNR,
every LHS pair below SNR 3, the argmin-vs-fifo SNR, and `tau_detectable(α)` — the smallest
grid τ whose profile is 3σ from τ=0, **tabulated per α, not as one global constant**.

A confound found here **disqualifies that parameter from Gate 2's criteria.** Demanding
recovery of something unidentifiable is incoherent. The disqualification is written down
between Gate 1 and Gate 2, before Gate 2 runs.

### Gate 2 — recovery

40 target θ drawn by Latin hypercube over the box with `default_rng(5)`, drawn before the run;
each simulated on 6 graphs from `default_rng(3)`, profiled, fitted.

| parameter | criterion |
|---|---|
| α | median \|Δα\| ≤ 0.05 **and** 90th pct ≤ 0.15 |
| γ | median \|Δγ\| ≤ 0.05 **and** 90th pct ≤ 0.15 |
| τ, where true `τ ≥ tau_detectable(α_true)` | median relative error ≤ 0.40 |
| τ, where true `τ < tau_detectable(α_true)` | `tau_unidentified` flagged in ≥90% of cases; a confident finite estimate is a **failure** |
| k | exact ≥ 90% |
| selection | exact ≥ 95% |
| c | correct bucket of `{0, (0,1), 1}` ≥ 80% |
| γ ≈ 0 cases (`γ̂ < 0.05`) | `gamma_degenerate` flagged and (α, τ) withheld in ≥90% of cases |

**PASS requires α, γ, selection and k.** τ and c are reported separately.

**The reframe is dead only if α or γ fails.** Those are the two axes on which the networks
demonstrably differ from every reference, and the two with exact closed-form estimators.
Failure of τ or c instead yields the scoped result "the family is estimable in
(α, γ, k, selection); aggregation temperature and standstill commitment are not", which is a
live pivot rather than a death. This asymmetry is declared here so it cannot be relitigated
once the numbers are in.

**Predicted outcome, stated in advance so a weaker result cannot be dressed up as the intended
one.** From the calibration already run on the closed forms and the τ sweep:

- α and γ recover essentially exactly (α̂ and γ̂ matched 12/12 hand-set cases to 4 decimals).
- **τ is expected to fail its primary criterion over most of the box.** Median
  best-minus-second-best message gap is 3.9 at α=1 but 21.8 at α=0.7 and 29.5 at α=0.5:
  damping holds the estimate far above its fixed point where the minimum is unambiguous, so
  temperature stops being observable. Measured `off_pr` was flat at 1.0000 for every τ ≤ 2 at
  α=0.7 and every τ ≤ 4 at α=0.5. The expected honest result is that τ is identifiable only
  for α near 1 and τ ≳ 1, and the estimator's job there is to **refuse**, not to guess.
- c is the weakest axis and may fail its bucket criterion.

### Gate 2b — held-out replication

Fresh graph stream `default_rng(11)`, fresh 40-point LHS draw `default_rng(13)`, every
constant frozen from Gate 2 — no refitting of `kappa`, `theta`, grid extents, the β
calibration table, or `tau_detectable`.

**Two outcomes only: REPLICATED or FAILED.** No intermediate rung, per
`PREREGISTRATION_E3B.md`. The estimator's constants will have been chosen against Gate 2's
first draw, and five results in this project have died on exactly that step.

### Gate 3 — losability

Out-of-family generators, no training required:

- **`GlobalMean`** (primary, must fail): `d_v ← (1−α)d_v + α(mean_u d_u + w̄)`. Non-local, so
  `J` is dense and no family member can produce it.
- **`SecondBest`** (stress case, reported not gated): `d_v ← min(d_v, second smallest of
  {d_u + w})`. Identical α, γ and support size to a family member; differs only in *which*
  parent responds. The honest hard case — if the residual cannot see it, the observable has a
  resolution ceiling and that gets stated.
- **`MaxAgg`** (sanity anchor): softmax rather than softmin, wrong monotonicity.
- **The nine `artifacts/*.pt` checkpoints**: real out-of-family objects, free. Fits reported
  **descriptively and not gated** — they are three architectures × three seeds, not a sample.

**PASS iff in-family holdout `chi2 ≤ 4` and `GlobalMean chi2 ≥ 25` (≥5σ), non-overlapping
across instances.**

Gate 3 runs last because its thresholds are in χ² units that only exist once Gate 2 has
established what a good fit looks like. Inventing that scale earlier is the error Proposition
5 diagnoses.

## Leads explicitly not claimed by this spike

An unversioned prototype reading of the `plain_*` checkpoints suggested α ≈ 0.75 with
γ ≈ 0.23 — a discounted relaxation, correct on short paths with error compounding as
`γ^depth`, which would be a mechanism for B1's deep-OOD blow-up. **This spike does not claim
it.** Gate 3 reports checkpoint fits descriptively. Turning it into a result requires its own
pre-registration, a held-out seed split, and a depth sweep predicted in advance. Five results
in this project have died from being reported before that step.
