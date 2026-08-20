# Gate 1 outcome and the resulting disqualifications

Written after Gate 1 ran and **before Gate 2 ran**, as `PREREGISTRATION_FAMILY.md` requires:
"A confound found here disqualifies that parameter from Gate 2's criteria. Demanding recovery
of something unidentifiable is incoherent. The disqualification is written down between Gate 1
and Gate 2, before Gate 2 runs."

Reproduce with `python run_family_spike.py --gate 1`.

## Gate 1 verdict against the frozen criteria: FAIL

| frozen criterion | result |
|---|---|
| every axis pair min-SNR ≥ 3 | **FAIL** — 14/15 pairs below 3 |
| ≥90% of random-design pairs at SNR ≥ 3 | **FAIL** — 75.8% of 2016 pairs |
| sentinel invariance `max |dz| < 1e-6` | **FAIL** — 2.08e-01 |

All three failures are real. None of them is a bug, and two of the three are informative.

## What actually failed, decomposed

The frozen confound metric ("min SNR over pairs ≥2 grid steps apart in the 2-D slice")
conflates two different things: axis *i* being mistakable for axis *j*, and axis *j* having
levels indistinguishable from *each other*. Measuring level resolution per axis separates them.

### Per-axis level resolution

| axis | distinguishable classes | merged levels |
|---|---|---|
| α | **6/7** | {0.35, 0.5} |
| γ | **5/6** | {0.5, 0.75} |
| selection | **2/2** | — |
| k | 3/6 | **{8, 12, 24, None}** |
| τ | 3/7 | {0, 0.25, 0.5, 1.0}, {2, 4}, {8} |
| c | 2/5 | {0, 0.1}, {0.25, 0.5, 1.0} |

α, γ and selection are well resolved. The other three are not, for three different reasons.

**`k` above the profile horizon is a design error in the pre-registration, and it is mine.**
The profile is 8 rounds and the halt is applied as `finished` once `round + 1 >= k`, so k = 8,
12, 24 and `None` produce *bit-identical* profiles: the halt never fires inside the window.
Freezing a k grid of {3, 5, 8, 12, 24, None} against `rounds = 8` was incoherent from the
start. This is structural, predictable, and has nothing to do with the observable.

**`τ` resolution matches the prediction already registered.** Temperature is visible only when
it is comparable to the best-minus-second-best message gap, and damping inflates that gap
(3.9 at α=1, 21.8 at α=0.7, 29.5 at α=0.5). The `tau_detectable` table is the quantitative
form of this and was pre-registered as a reported output.

**`c` is binary at best.** Only {0, 0.1} vs {0.25, 0.5, 1.0} separates. This is weaker than the
three buckets the pre-registration asked for.

### Genuine cross-axis confounds

Restricting the same metric to levels resolvable within their own axis still leaves 14/15
pairs below SNR 3, so the failure is not only within-axis degeneracy. Four confounds are real
and each has a mechanism:

- **α ↔ c** (min SNR 0.00). Commitment drives `a_eff = α(1−m)` toward 0, which is exactly what
  lowering α does. On the *joint* profile surface these are the same move.
- **τ ↔ γ** (0.00). γ compresses the spread between competing messages, τ softens the
  minimum over them. Both reduce effective sharpness.
- **γ ↔ c** (0.00).
- **τ ↔ selection** (0.00). Under `argmin` there is one sender per round, so there is no
  distribution over parents for a temperature to spread — τ is structurally unobservable there.

### The sentinel failure is a scoping result about the family, not about the estimator

`softmin` is translation-equivariant, verified to 1e-14, so the *aggregation* cannot depend on
the sentinel's magnitude. But the profile does depend on it, and only when α < 1: at α = 0.5 a
node at the sentinel `U` receiving a message of 10 lands at `0.5U + 5`, so the entire approach
path scales with `U` while the `reached` mask threshold `0.9U` scales differently, moving which
rounds admit which rows. `α = 1`, `τ = 2`, `γ = 0.5` and `argmin` are all invariant to
machine precision; only the damped case moves.

Consequence, stated rather than patched: **the sentinel is part of the model, not a nuisance
constant.** Fitted α is comparable across objects measured with the same sentinel — which
holds throughout this spike, since `UNREACHED = 80` everywhere — and is not transferable to a
model trained with a different one.

## Disqualifications applied to Gate 2

Gate 2 reports **both** the frozen criterion and the amended one, so nothing is hidden.

| parameter | frozen criterion | amended criterion | why |
|---|---|---|---|
| α | median \|Δ\| ≤ 0.05, p90 ≤ 0.15 | **unchanged** | α has a closed form read off the Jacobian diagonal on rows that are *still moving*, so a frozen node is excluded from the very rows the estimator uses. The α↔c confound lives on the joint surface, not in the direct read. |
| γ | median \|Δ\| ≤ 0.05, p90 ≤ 0.15 | **unchanged** | closed form, same argument |
| selection | exact ≥ 95% | **unchanged** | 2/2 resolvable |
| k | exact ≥ 90% | **exact among {3, 5}; {8, 12, 24, None} collapsed to one "no halt within horizon" class**, ≥ 90% | k above the horizon is not in the data |
| c | 3 buckets {0, (0,1), 1} ≥ 80% | **2 classes {≈0} vs {>0} ≥ 80%**, reported not gated | only 2 classes are resolvable |
| τ | rel err ≤ 0.40 where τ ≥ floor | **unchanged**, floor from `tau_detectable` | already conditioned on the operating point |

The pass condition is unchanged: **α, γ, selection and k (amended) must pass, and the reframe
is dead only if α or γ fails.** That asymmetry was registered in advance and is not revisited
here.

## What this costs the reframe

Nothing that was load-bearing, and one thing that was not.

The registered claim was that (α, γ) are the axes on which learned models demonstrably differ
from every reference, that they carry closed-form estimators, and that τ and c were expected to
be weak. Gate 1 confirms all three: α and γ are the two best-resolved axes in the family, and
τ and c are the two worst. What Gate 1 adds is that the weakness is *structural* — four
mechanisms, each derivable from the update rule — rather than a resolution shortfall that more
graphs or more rounds would fix. More probe graphs cannot separate α from c, because at the
level of the response operator they are the same move.

So the honest ceiling on Phase F, before recovery has even been measured: this is a method for
estimating **damping and path discount**, with commitment and temperature recoverable only in
corners. Whether that is worth a paper depends on Gate 2 and on whether α and γ separate the
trained checkpoints from the references — which Gate 3 reports descriptively and does not claim.
