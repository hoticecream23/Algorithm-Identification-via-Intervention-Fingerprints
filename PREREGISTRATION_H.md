# Pre-registration — Experiment H: repair the instrument

Written 2026-08-21, **before `run_h.py` existed.** Frozen. `ROUTES_TO_POSITIVE.md` §2.1 and
§2.2 are the motivation; this file is the contract. The candidate list is complete as written.

Two independent repairs to a **diagnosed** defect, run as one pre-registered pass:

- **H1 (§2.2)** — the orthogonality verdict is computed against the wrong symbolic quantity,
  and on too few points to support the conclusion drawn from it.
- **H2 (§2.1)** — two of seven predicates are pinned constants, contributing ~29% of every
  neural fingerprint as a pure offset.

Neither is a fishing expedition: both targets were named as defects in
`FINDINGS_DECOMPOSITION.md` itself, and neither has been fixed.

---

## H1 — the orthogonality metric

### The claim under test

`FINDINGS_DECOMPOSITION.md`: *"the fingerprint's discriminative axes and its neural-disagreement
axes are orthogonal — in fact slightly anti-correlated, r = −0.233 ... That is a complete account
of the neural null ... No number of seeds or probe graphs fixes an orthogonal measurement."*

That document labels itself "descriptive decomposition, no test, no p-value", and that caveat is
honest. But `HANDOFF.md` and `README.md` both carry `r = −0.23` forward as an established
mechanism that closes the door on probe redesign, so the conclusion is load-bearing even though
the number was never tested. H1 tests it.

### Two independent objections, both pre-specified

**(a) Statistical.** `r = −0.233` is a correlation over **nine points**. Computed before this
file was written, and stated here so it cannot be presented later as a discovery: the exact
two-sided permutation p over all 9! = 362,880 relabellings is **0.5441**, and the Fisher-z 95%
interval is **[−0.777, +0.510]**. The data cannot distinguish orthogonality from strong positive
alignment. This is arithmetic on already-published numbers, not a new measurement, and it is
reported as such.

**(b) Construct validity.** `run_decompose.py:157–176` computes the symbolic column as the
**mean over all ten reference pairs of (disagreeing slots / total slots)** for that intervention
— an average disagreement *rate*. That is not separating power, and here the two demonstrably
diverge:

| intervention | symbolic column | actual role |
|---|---|---|
| `inject_unreached_node` | **0.095** (2nd lowest) | in the size-2 minimal separating set; the **unique** separator of dijkstra/spfa |
| `insert_improving_edge` | 0.458 (highest) | in the minimal separating set; the **unique** separator of dijkstra/prim |

An intervention can separate every pair while disagreeing on few slots, and can disagree on many
slots while separating nothing new. The decomposition's own §1 leans on exactly this cell,
calling `inject_unreached_node` (0.944 neural vs 0.095 symbolic) "nearly uninformative between
symbolic algorithms" — which is false under a separating-power reading, since it is the only
probe that separates dijkstra from spfa.

### The statistic, fixed now

Recompute the same correlation with the symbolic column replaced by a **separation-weighted**
score. Primary and secondary are fixed here so the winner cannot be chosen afterwards:

- **PRIMARY — `sv_pairs`**: for each intervention, the number of the 10 reference pairs that it
  separates **on its own** (`fpid.separation.separates` with that single intervention). This is
  the natural definition of separating power, and it is what the minimal-separating-set
  machinery already uses.
- **SECONDARY — `sv_unique`**: the number of pairs for which it is the **only** separator among
  the nine. Reported, never decisive.
- **BASELINE — `sv_rate`**: the original mean-disagreement-rate column, recomputed here to prove
  the pipeline reproduces `r = −0.233` before anything is changed.

The neural column `nv` is **unchanged** from `run_decompose.py` and is recomputed from the
stored `artifacts/fingerprints.json`, so H1 changes exactly one variable.

Every correlation is reported with its exact permutation p over all 362,880 relabellings and its
Fisher-z 95% interval. No sampling.

### Ladder — two rungs, no middle

Applied to the **primary** `sv_pairs` correlation:

| outcome | condition |
|---|---|
| **UPHELD** | exact p < 0.05 **and** r < 0 — the axes really are opposed, and redesign is futile |
| **REVERSED** | exact p < 0.05 **and** r > 0 — the axes are aligned, and probe redesign is actively promising |
| **UNSUPPORTED** | exact p >= 0.05 — no evidence of any relationship, in either direction |

**Stated in advance so the likely outcome cannot be oversold:** with n = 9 this test has very low
power, so UNSUPPORTED is the probable result and it is *not* evidence of alignment. Its content
is narrow and entirely about the original claim: an argument that closed off probe redesign
would be resting on a correlation consistent with anything from −0.78 to +0.51. UNSUPPORTED
reopens the question; it does not answer it. Any write-up saying more than that is overclaiming.

### Negative control

`sv_rate` must reproduce **r = −0.233 ± 0.005** from the stored fingerprints. If it does not,
the H1 pipeline is not the pipeline that produced the published number and no comparison is
meaningful.

---

## H2 — the pinned predicates

### The defect

`FINDINGS_DECOMPOSITION.md` §3: `latency` and `poked_settled` disagree with Bellman-Ford
**100.0%** of the time for every neural model, and all four slots where every network disagrees
with every reference are `poked_settled`. That is ~29% of every neural fingerprint contributing
**nothing but a constant offset** — and a constant offset is exactly what manufactures a vacuity
gap.

Both predicates ask exact-fixed-point questions of a system that has no exact fixed point.
`fpid/fingerprint.py` computes `latency` by requiring `_eq(intervened[r].d, ref_f, tol)` to hold
**at a single round**, and `poked_settled` from `was_converged`, which tests a single round of
`converged_mask`.

### AMENDMENT, 2026-08-21 — the assumed mechanism was wrong, and the repair below is replaced

**Written after the mechanism check and before any H2 outcome statistic was computed.** The
diagnosis in "The defect" above — that these predicates ask exact-fixed-point questions of a
system with no exact fixed point — is **false**, and the sustained-tolerance repair specified
below would not have fixed anything. Both are kept, struck through, because the reason they were
wrong is the actual finding.

`was_converged` already receives `tol = 2.5`, so tolerance was never the binding constraint.
Measured instead:

| executor | `settle_round` (drives firing) | `_settle_round` (true active life) | fire round | **fire / active** |
|---|---|---|---|---|
| bellman_ford | 5.0 | 6.0 | 2.0 | **0.39** |
| spfa | 22.0 | 23.0 | 11.0 | **0.48** |
| dijkstra | 19.5 | 20.5 | 9.5 | **0.47** |
| prim | 20.5 | 22.0 | 10.0 | **0.45** |
| plain_700 | 2.0 | 8.0 | 1.0 | **0.12** |
| plain_701 | 3.0 | 11.0 | 1.0 | **0.09** |
| gated_800 | 3.0 | 9.0 | 1.0 | **0.11** |

The codebase carries **two different notions of "settled"**. They agree for symbolic executors
and disagree by 3–4× for networks. `fpid/interventions.py:settle_round` drives the firing rule;
`fpid/fingerprint.py:_settle_round` measures when `d` actually stopped moving.

**Root cause.** `settle_round` counts a round as active only if its largest single-node change
exceeds `max(tol·sqrt(2 ln N), 0.05 · max_over_the_run)`. A symbolic algorithm lifts nodes from
the `UNREACHED = 80` sentinel to their true value in one ~70-unit jump *every round*, so it
clears that bar for its whole run. A network makes one large round-1 transient (sentinel →
finite everywhere) and thereafter moves every node by only 2–5 per round — below the ~3.5
threshold its own transient sets. The `got_reach` escape hatch cannot help: under `finite_init`
the sentinel is finite, so `~isfinite(a)` is never true and that branch is dead for every
executor in this pipeline.

**Consequence.** CLAUDE.md constraint 2 — *"interventions fire relative to each algorithm's own
progress, never at an absolute round number"* — is satisfied in code and **not in effect** for
neural executors. Networks are poked at ~11% of their active life; symbolic algorithms at ~45%.
`poked_settled` is then pinned to `no` for every network because at round 1 nothing has settled
yet, and `reached@t` is 1.00 for every network against 0.16–0.73 for the references.

This is a manipulation failure in the project's own sense, and it affects **every neural
fingerprint in the repository** — B0, B1, E2, E3, E3b, the decomposition, E5 and G alike. It is
also not what `ROUTES_TO_POSITIVE.md` §2.1 proposed to fix.

### The repair (amended)

Make the firing rule use a settle notion valid for continuous-output executors: the last round at
which `d` changed by more than `tol`, which is exactly `fingerprint._settle_round` and is what
the symbolic side already effectively gets.

```
firing_round_absolute(control) := clip(_settle_round(control, tol) // 2, 1, budget - 1)
```

Opt-in flag, default off, so every existing number stays reproducible — E5's convention.

**Manipulation check, and it gates H2.** Before any distance is read: median `fire / active` must
land in **[0.35, 0.60] for at least 7 of 9 networks**. If the repair does not actually move the
firing time into the intended window, H2 is INCONCLUSIVE (mechanism) and no `d_near` is reported
— the E1 error, refused in advance.

The ladder, the negative control and the named failure mode in the sections below are
**unchanged** and apply to this repair instead.

### ~~The repair (superseded — see the amendment above)~~

~~A sustained-tolerance basis, with window `W`:~~

```
settled(v, t)  :=  |d_v(t') - d_v(t)| < tol   for all t' in [t, t+W]
latency        :=  first k after which the run stays within tol of the reference
                   for a sustained window W
```

`W = 3` is fixed now, chosen as the smallest window spanning more than one propagation step at
this graph depth, and is **not** swept. Sweeping W and reporting the best is the exact failure
mode this project has retracted five results to.

The repair is implemented as an **opt-in flag**, default off, so every existing number in the
repository stays reproducible by the command at the top of its findings document. That is the
convention E5 established with `normalise=False`.

### Why this is not what E5 already tested

E5's mask drops a slot when *both* fingerprints match their own null-control response. A pinned
constant is pinned under the null control too, so the network side is uninformative there — but
the **reference** side is informative, and the pairwise rule therefore **keeps** the slot. E5
structurally could not remove this offset. This is the orthogonal repair, and it is the one
`HANDOFF.md` names as repair (1) and never ran.

### Statistic and ladder

Recompute `d_near` and the reference-to-reference median on the nine Stage-1 checkpoints under
repaired predicates, exactly as E5 did:

| outcome | condition |
|---|---|
| **CLOSED** | `d_near` < reference-to-reference median for **>= 7/9** checkpoints |
| **UNCHANGED** | the reverse for **>= 7/9** |
| **INDETERMINATE** | 4–6 |

### Negative control, and it can stop H2

The symbolic side must survive the repair: **no unseparated reference pairs, and a minimal
separating set of size <= 3.** If the repaired predicates break symbolic separation, the repair
is malformed and H2 reports a broken instrument, not a result about vacuity.

### The failure mode named in advance

`ROUTES_TO_POSITIVE.md` §2.3 already worked this arithmetic: changing the slot set shrinks the
denominator on **both** sides, and E5 demonstrated that this can raise the reference median
faster than it lowers `d_near` (0.303 → 0.489). H2 may well come out UNCHANGED for exactly that
reason. **If it does, that is the answer.** No third variant of the distance, no window sweep,
no follow-up statistic on the same checkpoints.

---

## Scope limits

- H1 and H2 are **independent**; neither is contingent on the other's outcome.
- H2 runs on the nine Stage-1 checkpoints only. Experiment G established that these are three
  design points rather than nine, so a CLOSED verdict here would be **exploratory** and would
  require the twelve Stage-2 checkpoints as a confirmatory rung before it is written as a
  result. That rung is specified now: same ladder, `>= 10/12`.
- One pass. No follow-up statistic on this data if either misses.

## Commands

```powershell
python run_h.py --part 1      # orthogonality recheck, numpy only, seconds
python run_h.py --part 2      # predicate repair, existing checkpoints, minutes
python run_h.py --part all
```
