# Findings: parametric family spike (Phase F)

```bash
python run_family_spike.py --gate all          # full ladder, ~10 min on CPU
python run_family_spike.py --gate 0            # the embedding gate alone, seconds
python run_family_spike.py --reproduce-response-table
```

Pre-registered decision rules: `PREREGISTRATION_FAMILY.md`. Disqualifications written between
Gate 1 and Gate 2: `GATE1_DISQUALIFICATIONS.md`. Raw output: `family_spike_log.txt`,
`family_spike_results.json`.

## Bottom line

**The spike fails on the pre-registered criteria, and the reason is not the one that was
registered.** The family contains the five reference algorithms *exactly*, and two of its six
parameters are recoverable in closed form to four decimal places. But the object the whole
reframe was built for — a gradient-trained MPNN as an interior point of the family — is
rejected by the family's own residual at chi-squared of order 10^4.

| gate | result | why |
|---|---|---|
| 0a embedding | **PASS** | all five references reproduced at exactly 0.0 |
| 0b adequacy | **PASS** | all six axes move the observable, marginal SNR 4.5–41 |
| 1 identifiability | **FAIL** | 14/15 axis pairs below SNR 3; 75.8% of random pairs vs 90% required |
| 2 recovery | **FAIL** (frozen) | α p90 0.29 vs 0.15 required — driven entirely by targets the estimator had no data for |
| 2b replication | **replicates** | every number lands in the same place on a fresh graph stream and fresh draw, except the γ≈0 flag rate (100% → 71.4%, n=7) |
| 3 losability | **FAIL** | the out-of-family control is rejected, but at chi2 15.0 against a required 25 |

The verdict of record is **DEAD**, per the pre-registered asymmetry ("the reframe is dead only
if α or γ fails"). What follows is what the failure is actually made of, because three of the
four failures are informative and one of them is a design error of mine.

## What worked, and worked exactly

**The family contains the algorithms.** Bellman-Ford, truncated Bellman-Ford, SPFA, Dijkstra
and Prim are all reproduced to `max |d_family − d_ref| = 0.0` — bit-identical, not
within-tolerance — across 20 graphs × 24 rounds from the `UNREACHED` sentinel. Prim is a family
member at γ=0 (keying on `w` alone is a zero path discount), which is why γ earns its place
over a discrete `key ∈ {sum, weight}` axis.

**Two parameters have exact closed forms.** On rows that are reached *and still moving*:

```
alpha = 1 - |J[v,v]|                gamma = (off-diagonal row sum) / alpha
```

Recovered to 4 decimals on 13 hand-set triples covering all three selection modes, c > 0, γ=0
and k=3. Where the estimator answers at all, α p90 error is **0.0039** (Gate 2) and **0.0000**
(Gate 2b), and γ p90 is 0.0055 and 0.0011.

**The residual self-diagnoses.** Of the targets recovered badly, 83% (Gate 2) and 100% (Gate
2b) carry chi2 > 4; of those recovered well, 100% and 83% carry chi2 ≤ 4. This is the property
`THEORY.md` Proposition 5 says the nearest-reference distance lacked: a residual with a unit,
so "this fit is bad" is readable without knowing the answer.

**It replicates, with one exception.** Gate 2b uses a fresh graph stream (`default_rng(11)`), a
fresh 40-point design (`default_rng(13)`) and no refitted constants. Every headline number
reproduces: α and γ medians at 0.0000, coverage 70%→75% and 87.5%→87.5%, k-class 95%→95%,
selection 97.5%→100%, c 62.5%→65%, τ relative error 1.000→1.000, refusal 92%→96%.

The exception is the γ≈0 degeneracy flag, which went **100% → 71.4%** (5 of 7) and so passes in
Gate 2 and fails in Gate 2b. With n=7 degenerate targets per arm, two misses is the whole
difference, and this is the one place where the spike is underpowered rather than wrong. It is
recorded as a failure, not averaged into the successes. After five results in this project died
on held-out replication, everything else here survived it.

**The one pre-committed prediction held.** `PREREGISTRATION_FAMILY.md` predicted, before Gate 1
ran, that `argmin` and `fifo` profiles would be within measurement noise — that the response
observable cannot separate SPFA from Dijkstra at hard min, so including `fifo` in the recovery
grid would manufacture a confound rather than resolve one. Measured SNR: **1.46**, against a
threshold of 3.

## What failed, and what each failure means

### 1. The recovery failure is coverage, not accuracy

α's median error is 0.0000 and its p90 is 0.29. The tail is not noise: it is four targets at
α ∈ {0.1, 0.2} with `argmin` selection, all flagged `no_fittable_rows`. Under one-sender-per-
round with heavy damping, each update moves a node by ~10% of its gap, which falls below the
mask's `κ·eps = 0.25` threshold, so **no row qualifies and the direct estimators have no
data at all**.

The original implementation then reported the constructor default (α = 1.0) as a confident
estimate, giving errors of 0.9. That was a real bug and is fixed: `FitReport.withheld` now
makes refusal machine-readable, and `no_fittable_rows` withholds α, γ and τ.

Scored on answered targets, α and γ both pass comfortably. The honest statement is therefore:

> The estimator is **exact where it has data and blind where it does not**, and it answers on
> 70–75% of the pre-registered parameter box.

That is a real limitation and it is *not* the limitation that was registered, so the frozen
verdict stands. Fixing it means changing the observable, not the estimator: the mask's
"still moving" condition is tied to the probe amplitude for a good reason (below it, a
one-sided difference is a secant), so a heavily damped rule needs either a smaller `eps` or a
multi-round differencing scheme that accumulates motion instead of requiring it per round.

### 2. Identifiability fails for four separate mechanisms, and one design error

Per-axis level resolution (others held at the reference point):

| axis | distinguishable | merged |
|---|---|---|
| α | **6/7** | {0.35, 0.5} |
| γ | **5/6** | {0.5, 0.75} |
| selection | **2/2** | — |
| k | 3/6 | {8, 12, 24, None} |
| τ | 3/7 | {0, 0.25, 0.5, 1}, {2, 4}, {8} |
| c | 2/5 | {0, 0.1}, {0.25, 0.5, 1} |

**The k result is my error in the pre-registration.** With `rounds = 8`, a halt at k ≥ 8 never
fires inside the profile window, so k = 8, 12, 24 and `None` produce bit-identical profiles.
Freezing a k grid wider than the profile horizon was incoherent from the start. Collapsed to
what the horizon can see, k recovers at 95%.

The genuine cross-axis confounds, each with a mechanism, all surviving restriction to
individually-resolvable levels:

- **α ↔ c** (SNR 0.00). Commitment drives `a_eff = α(1−m)` toward zero, which is exactly what
  lowering α does. On the response operator these are the same move.
- **τ ↔ γ** (0.00). γ compresses the spread between competing messages; τ softens the minimum
  over them. Both reduce effective sharpness.
- **τ ↔ selection** (0.00). Under `argmin` there is one sender per round, so there is no
  distribution over parents for a temperature to spread across. τ is structurally invisible.
- **γ ↔ c** (0.00).

**More probe graphs cannot fix any of these.** That is the load-bearing point: α and c are the
same operation at the level of the Jacobian, so no amount of data separates them.

### 3. Aggregation temperature is effectively not estimable

τ was predicted in advance to be weak; it is worse than weak. Median relative error is
**1.000** in both Gate 2 and Gate 2b — meaning the estimator returned τ = 0 for every target
that was nominally in a detectable range. The refusal mechanism works (92% and 96% of
below-floor targets correctly flagged), but there is essentially no regime where it answers.

The mechanism is measured, not conjectured. Temperature is visible only when τ is comparable
to the typical best-minus-second-best message gap, and damping inflates that gap:

| α | median message gap | smallest detectable τ (γ=1) |
|---|---|---|
| 1.0 | 3.9 | 1.0 |
| 0.85 | — | 4.0 |
| 0.7 | 21.8 | 8.0 |
| 0.5 | 29.5 | 8.0 |
| ≤0.2 | — | never on this grid |

A damped estimate sits far above its fixed point, where the minimum is unambiguous — so
`off_pr = 1.000` has to be read as "**this update behaves as a hard min at this operating
point**", which is not the claim "the aggregation is hard".

### 4. The sentinel is part of the model, not a nuisance constant

The registered invariance check (`UNREACHED` 80 vs 160, tolerance 1e-6) fails at 2.1e-01, and
only for α < 1. `softmin` is translation-equivariant to 1e-14, so the aggregation cannot depend
on the sentinel. But damping *interpolates from* it: a node at `U` receiving a message of 10
lands at `0.5U + 5`, so the whole approach path scales with `U` while the `reached` mask
threshold `0.9U` scales differently, shifting which rounds admit which rows.

Consequence: a fitted α is comparable only across objects measured with the same sentinel. That
holds throughout this spike (`UNREACHED = 80` everywhere) and would not transfer to a model
trained with a different one.

### 5. The losability gate is directionally right and under-resolved

Designing the negative controls surfaced a small result about the task. **Under a sentinel
initialisation, any aggregator not dominated by the smallest message cannot propagate from a
single source at all**: for `max` and second-best aggregation, every non-minimal incoming
message is `UNREACHED + w`, which never beats `UNREACHED`, so nothing moves and no row is
fittable. A global `min(d) + w̄` rule fails the opposite way, converging in one round so that
no node is ever both reached and still moving. Taking the minimum is load-bearing for
propagation, not a stylistic choice — which is why the family is built on a softmin rather than
a general aggregator, and why two of the three registered controls are unusable.

Mean-aggregation (`d_v ← min(d_v, mean_u(d_u + w_uv))`) is a valid control: it propagates at
one hop per round, and it is not a limit of the family in any direction, since the softmin is
dominated by the smallest message at every finite τ and diverges to −∞ rather than approaching
the mean as τ → ∞. It is rejected — chi2 **15.0**, verdict `family_wrong`, against an in-family
holdout at **0.0000** — but 15.0 falls short of the registered 25. Sign, not resolution.

## The result that matters most: the networks are not in the family

Descriptive, not gated — three architectures × three seeds is not a sample, and the
pre-registration forbids claiming anything from it. But the numbers are unambiguous:

| model | α̂ | γ̂ | τ̂ | ĉ | chi2 | row-sum deviation |
|---|---|---|---|---|---|---|
| gated_800 | 0.441 | 0.149 | 0.55 | 0.00 | 34.5 | 0.012 |
| gated_801 | 0.427 | 0.123 | 2.16 | 0.10 | **6.2** | 0.025 |
| gated_802 | 0.348 | 0.171 | 0.95 | 0.00 | **9.6** | 0.024 |
| halt_900 | 0.761 | 0.807 | 1.26 | 0.00 | 46155 | 0.168 |
| halt_901 | 0.831 | 0.956 | 4.40 | 0.00 | 40427 | 0.068 |
| halt_902 | 0.796 | 0.785 | 0.96 | 0.00 | 15070 | 0.075 |
| plain_700 | 0.316 | 0.864 | 6.06 | 0.10 | 21133 | 0.127 |
| plain_701 | 0.516 | 0.132 | 0.86 | 0.00 | 14177 | 0.133 |
| plain_702 | 0.332 | 0.592 | 1.81 | 0.00 | 44216 | 0.014 |

Six of nine models sit at chi2 of order 10^4 against an in-family holdout at 0.0000 and a
deliberately out-of-family control at 15.0. **Seven of the nine sit further from this family
than the negative control does** — only `gated_801` (6.2) and `gated_802` (9.6) are closer, and
neither clears the adequacy threshold of 4. The row-sum deviation — the parameter-free
membership test,
which needs no fitted parameters at all — is non-zero for every one of them, where every family
member gives exactly 0.0000.

So the premise of the reframe is not supported. The proposal was that a family parameterised by
damping, sharpness, commitment, truncation and selection would contain the discrete algorithms
as limit points *and the networks as interior points*. The first half is exact. The second half
is false: a gradient-trained MPNN is not a damped, soft, discounted relaxation with a fitted
temperature. It is something else, and this family's residual says so at four orders of
magnitude.

There is a third, independent sign of non-membership, and it is the cleanest of them. Re-run
the same fits under `--smoke` (6 rounds and 4 graphs instead of 8 and 6) and the family members
recover *identically* — the closed forms do not care how long you look — while the checkpoints'
fitted parameters move a lot: `plain_700` reads α̂ = 0.670, γ̂ = 0.283 at 6 rounds against
α̂ = 0.316, γ̂ = 0.864 at 8. **Parameters fitted to an object outside the family are not stable
under a change of measurement settings**, which is exactly what one should expect and exactly
the check that the nearest-reference method could never perform.

The α̂ ≈ 0.3–0.5 and γ̂ ≈ 0.12–0.17 readings for the `gated` arm are **not** a result and are
not claimed. Their chi2 (6–35) is the lowest of the nine, but "the least-bad fit among nine" is
precisely the error Proposition 5 diagnoses, restated in a new coordinate system. An earlier
unversioned prototype reading of `plain_700` as α ≈ 0.75, γ ≈ 0.23 does not survive the
checked-in estimator (0.316, 0.864, chi2 21133) and is withdrawn.

## Corrections to `fpid/response.py`

The module's docstring made two claims that were produced ad hoc, had no caller, and were
wrong. Both are now regenerated by `--reproduce-response-table` and written to
`artifacts/response_reference_table.json`.

| claim | measured |
|---|---|
| all five references have `width` = 1.00 at every round | Bellman-Ford is **2.00 from round 3** |
| `row_sum` is the damping coefficient α | `row_sum ≡ 1 − α(1−γ)`, so **identically 1 at γ=1 whatever α is** |

Both have one cause: the one-sided difference **double-counts at a hard-min tie**. At a
converged node `d_v == d_u + w`, so lowering either argument moves the min by the full `eps`
and both `J[v,v]` and `J[v,u]` read 1. Bellman-Ford enters that state around round 3 on depth-5
instances and stays there; the selection-based algorithms do not, because only one node is
active per round.

This matters beyond bookkeeping: "the networks depend on 2–3 neighbours per update where every
hard-`min` algorithm depends on exactly 1" was the measurement most often cited as motivating
this whole reframe, and it is substantially an artefact of the estimator. The vacuity result it
was cited alongside (0.37–0.40 vs a 0.303 reference-to-reference median) is independent and
stands. `diag_frac` reproduces exactly and remains sound.

## What a redesign would have to fix

Not recommendations to act on — the honest read is that Phase F as specified is finished — but
the four things that would have to change before any version of this is worth attempting:

1. **The observable cannot see heavily damped rules.** A per-round one-sided difference needs
   per-round motion exceeding the probe amplitude. Multi-round differencing, or a probe
   amplitude scaled to the observed motion, is a prerequisite, not a refinement.
2. **α and c are the same move on the response operator.** No observable built from
   one-round Jacobians can separate them. Separating commitment from damping needs a
   *sustained* measurement — which is what `HANDOFF.md`'s own repair (b) proposed for
   `poked_settled` and `latency`.
3. **The family is missing whatever the networks are doing.** Six of nine checkpoints sit at
   chi2 10^4 with non-zero row-sum deviation, so the update rule is not
   `(1−a)d + a·min(d, softmin(γd+w))` for any parameters. Finding the missing term is a
   modelling problem, and the row-sum deviation is the measurement that localises it.

   **Followed up in `FINDINGS_RESIDUAL.md` (exploratory).** Pointing the residual at the nine
   checkpoints gives a specific answer: the per-node discount `γ_v` has an interquartile range
   of 0.24–0.84 where every family member gives exactly 0.000, it **exceeds 1** in 2–47% of
   rows, and it rises with round (ρ = +0.47 on the division-free numerator, 9/9 sign
   agreement). So the missing ingredient is a *state-dependent, sometimes amplifying path
   gain* rather than a scalar discount. Two further hypotheses were refuted and both matter:
   there is no self-amplification on the diagonal (`|J[v,v]| ≤ 0.832` everywhere), and
   `width > deg(v)` never occurs — the network really is one-hop local in `d`-space, so the
   observable is sound and it is the hypothesis space that was wrong, exactly as assumed.
   Nothing there is a result yet; the frozen confirmatory test is in that document.
4. **The k grid must not exceed the profile horizon.** Trivial, and mine to own.

## Status of the wider project

The symbolic results (Phases A, A2, A3) are untouched and remain the project's positive
contribution. The neural negative is unchanged and now has a second, independent
corroboration: the networks are not near the five discrete algorithms (Phase B/E), *and* they
are not interior points of the natural continuous family containing those algorithms (Phase F).
Those are two different statements and the second is the stronger one.
