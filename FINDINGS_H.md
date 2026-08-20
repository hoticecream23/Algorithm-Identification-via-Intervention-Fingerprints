# Findings — Experiment H: repairing the instrument

```powershell
python run_h.py --part 1     # orthogonality recheck, numpy only, seconds
python run_h.py --part 2     # firing-rule repair, existing checkpoints, minutes
```

Statistic, ladder and controls frozen in `PREREGISTRATION_H.md` before `run_h.py` existed.
Log `h_log.txt`, records `h_results.json`.

**Two results, and the second one includes a defect that affects every neural fingerprint in
this repository.**

---

## H1 — the orthogonality verdict is not supported by its own data

### The claim under test

`FINDINGS_DECOMPOSITION.md`: *"the fingerprint's discriminative axes and its neural-disagreement
axes are orthogonal — in fact slightly anti-correlated, r = −0.233 ... That is a complete account
of the neural null ... No number of seeds or probe graphs fixes an orthogonal measurement."*
`HANDOFF.md` and `README.md` both carry that number forward as the mechanism closing the door on
probe redesign.

### Negative control

`sv_rate` reproduces **r = −0.233** exactly from the stored fingerprints, so this is the pipeline
that produced the published number. (It caught a real scoping error on the way: E6 later added
`raise_tree_edge_weight` and `dissociate_keys` to `INTERVENTIONS`, and including them injects two
all-zero rows into a nine-point correlation, moving it to +0.184. H1 is restricted to the nine
probes the stored fingerprints contain.)

### Result

| symbolic score | r | exact p | Fisher-z 95% CI |
|---|---|---|---|
| `sv_rate` — mean pairwise slot-disagreement (**published**) | **−0.233** | 0.5445 | [−0.777, +0.510] |
| `sv_pairs` — reference pairs separated (**PRIMARY**) | **−0.004** | 0.9930 | [−0.666, +0.662] |
| `sv_unique` — pairs uniquely separated (secondary) | **+0.240** | 0.5972 | [−0.504, +0.780] |

**Verdict: UNSUPPORTED.**

Two independent problems, either of which is sufficient:

1. **Statistical.** `r = −0.233` on nine points has an exact two-sided permutation p of **0.5445**
   over all 362,880 relabellings, and a 95% interval of **[−0.777, +0.510]**. The data cannot
   distinguish orthogonality from strong positive alignment. The decomposition does label itself
   "no test, no p-value" — that caveat is honest — but the conclusion drawn from it, and carried
   into two other documents, is far stronger than the number supports.

2. **Construct validity.** The published symbolic column is a mean *disagreement rate*, not
   separating power, and here the two diverge exactly as predicted:

   | intervention | `sv_rate` | `sv_pairs` | `sv_unique` |
   |---|---|---|---|
   | `inject_unreached_node` | **0.095** (2nd lowest) | 4 | **1** — unique separator of dijkstra/spfa |
   | `insert_improving_edge` | 0.458 (highest) | **9 of 10** | **2** |

   The decomposition's §1 calls `inject_unreached_node` (0.944 neural vs 0.095 symbolic) "nearly
   uninformative between symbolic algorithms". It is in the size-2 minimal separating set and is
   the only probe separating dijkstra from spfa. Under a separating-power reading that sentence
   is false.

The anti-correlation is **not robust to the definition of "carries identity"**: it moves −0.233 →
−0.004 → +0.240 across three defensible choices, with intervals overlapping most of [−1, +1].

**What this does and does not license.** It does *not* show the axes are aligned, and n = 9 has
almost no power. It shows that the argument which closed off probe redesign rests on a
correlation the data cannot resolve. **The question is reopened, not answered.** Probe redesign
(`ROUTES_TO_POSITIVE.md` §4) is no longer excluded by evidence.

---

## H2 — the firing rule did not fire where it claimed to. Fixed. The gap survives anyway.

### The pre-registered mechanism was wrong, and the correction is the finding

`PREREGISTRATION_H.md` originally diagnosed `latency`/`poked_settled` as asking exact-fixed-point
questions of a system with no exact fixed point, and proposed a sustained-tolerance repair. The
mechanism check ran first — the project's own rule — and **refuted that diagnosis before any
outcome statistic was computed.** `was_converged` already receives `tol = 2.5`; tolerance was
never the binding constraint. The pre-registration was amended on the spot, with the superseded
text struck through rather than deleted.

### What is actually broken

The codebase carries **two different notions of "settled"**, which agree for symbolic executors
and disagree by 3–4× for networks:

- `fpid/interventions.py:settle_round` — drives the firing rule.
- `fpid/fingerprint.py:_settle_round` — measures when `d` actually stopped moving.

`settle_round` counts a round as active only if its largest single-node change exceeds
`max(tol·sqrt(2 ln N), 0.05 · max_over_the_run)`. A symbolic algorithm lifts nodes off the
`UNREACHED = 80` sentinel in one ~70-unit jump *every round* and clears that bar for its whole
run. A network makes one large round-1 transient — sentinel → finite everywhere — and thereafter
moves each node by only 2–5 per round, **below the ~3.5 threshold its own transient sets**. The
`got_reach` escape hatch cannot rescue it: under `finite_init` the sentinel is finite, so
`~isfinite(a)` is never true and that branch is dead for every executor in this pipeline.

**Consequence.** CLAUDE.md constraint 2 — *"interventions fire relative to each algorithm's own
progress, never at an absolute round number"* — was satisfied in code and **not in effect** for
neural executors:

| executor | fire / active life, before | after repair |
|---|---|---|
| bellman_ford | 0.39 | 0.46 |
| spfa / dijkstra / prim | 0.48 / 0.47 / 0.45 | 0.48 / 0.48 / 0.50 |
| `plain_700` / `701` / `702` | **0.12 / 0.09 / 0.15** | 0.47 / 0.45 / 0.50 |
| `gated_800` / `801` / `802` | **0.11 / 0.11 / 0.17** | 0.44 / 0.44 / 0.50 |
| `halt_900` / `901` / `902` | 0.31 / 0.25 / 0.38 | 0.50 / 0.50 / 0.50 |

Networks were poked at ~11% of their active life against ~45% for the references. That is why
`poked_settled` is `no` for all nine networks on every intervention while four of five references
are `yes`: at round 1 nothing has settled yet. Relatedly, `reached@t` is **1.00** for every
network against 0.16–0.73 for the references — a decoder emits finite values everywhere
immediately, so a network has no frontier at all.

**This affects every neural fingerprint in the repository** — B0, B1, E2, E3, E3b, the
decomposition, E5 and G alike.

### The repair, and it demonstrably works

`firing_round(..., absolute=True)` measures active life with `settle_round_absolute` (last round
`d` moved by more than `tol`). Opt-in, default off, so every existing number stays reproducible —
E5's convention.

**Manipulation check: 9/9 networks land inside the pre-registered [0.35, 0.60] window** (0.44–0.50),
while symbolic executors barely move. The repair installs exactly the change it claims to.

**Negative control: PASS.** No unseparated reference pairs. The minimal separating set grows from
2 to 3 (`delete_tree_edge_near` joins) — within the pre-registered `<= 3`, and worth recording as
a real cost: firing later makes the symbolic problem slightly harder.

### Primary result

| | before | after repair |
|---|---|---|
| mean `d_near` | 0.4088 | **0.2901** |
| reference-to-reference median | 0.3028 | **0.2158** |
| **ratio `d_near` / median** | **1.35** | **1.34** |

**Verdict: UNCHANGED** — 1 of 9 checkpoints closed, against the 7 required.

Both quantities fell by ~29% **together**. The vacuity gap is scale-invariant: fixing the firing
rule rescales the whole distance space and moves the ratio by 0.01.

This is precisely the failure mode `ROUTES_TO_POSITIVE.md` §2.3 named in advance and that E5
demonstrated once already (0.303 → 0.489) — changing the slot set moves the numerator and the
denominator together. It was written down before the run, so it is the expected answer rather
than a rescue.

---

## What H is worth

**Proposition 5 is now much harder to dismiss.** The vacuity gap has survived three independent
attacks on the measurement: sensitivity normalisation (E5), the reference-free reformulation (G),
and now a firing rule proved to fire at the intended point in the run. It is not an artefact of
the distance, the comparison, or the timing.

**One door is reopened.** H1 removes the evidential basis for "no probe redesign can help". That
claim was never tested, and under a separating-power score the anti-correlation is exactly zero.
`ROUTES_TO_POSITIVE.md` §4 — second-order/compositional response, the probe-amplitude
thermometer, wavefront-speed profiles — is live again on the merits.

**One defect is fixed, and it was load-bearing.** The firing-rule bug is worth reporting on its
own: an intervention-timing rule validated on discrete frontier-propagating algorithms silently
mis-times every continuous-output executor it is applied to, because the relative threshold is
anchored on an initialisation transient the continuous executor has and the discrete one does
not. Anyone building progress-relative probes for learned models will hit this.

### What it does not buy

H2 is UNCHANGED, so no neural claim improves. The nine checkpoints remain three design points
(Experiment G), so nothing here would have been confirmatory even had it closed. And H1's
UNSUPPORTED is a statement about an argument, not about the world: it licenses *trying* probe
redesign, not expecting it to work.
