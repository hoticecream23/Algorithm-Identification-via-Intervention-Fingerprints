# Pre-registration — E3b: held-out replication of the truncation-distance lead

**Written before the run, with the threshold frozen from E3's data. Nothing below may be
changed after seeing E3b's results.**

Date: 2026-08-10. Implemented by `run_e3b.py`. Results go in `FINDINGS_E3.md`.

---

## Why this run exists

E3 was NULL: 17/17 models labelled `bellman_ford` regardless of architecture. One post-hoc
quantity separated two arms perfectly. It is being tested rather than reported, because
the last time this project had a perfectly-separating post-hoc quantity — B2's `overshoot`,
which also had a sound mechanistic argument — it collapsed on held-out seeds with the
effect size sign-flipped.

**This document exists so that the lead can only die honestly or survive honestly.**

## The hypothesis

The fingerprint registers truncation as a shift *toward* the truncation reference without
changing the nearest-reference label. Statistic, fixed:

    s = d(bellman_ford) - d(bf_truncated_k3)

more negative meaning more Bellman-Ford-like. In E3, `plain` ∈ [−0.122, −0.095] and
`halt` ∈ [−0.073, −0.013], disjoint.

## The decision rule, frozen from E3's data before E3b runs

**Threshold τ = −0.084**, the midpoint of E3's observed gap (plain max −0.095, halt min
−0.073). A model is predicted `halt` if `s > τ`, else `plain`.

τ is **not** refitted on E3b under any circumstance. If E3b's own optimal threshold
differs, that is reported as a failure of the frozen rule, not as a corrected result.

## Design

Identical to E3 in every setting — n=24, density 0.05, depth >= 5, 24 train graphs, 8 probe
graphs, hint-free, 1500 epochs, budget 48, tol 2.5, stability 0.8, admission MAE < 2.5.
Only the seeds change, and they are disjoint from every seed used anywhere in this project:

| Arm | Architecture | Train rounds | Deploy `max_rounds` | Seeds |
|---|---|---|---|---|
| `plain` | `MPNN` | 24 | `None` | 500–505 |
| `halt` | `MPNN` | 8 | 8 | 600–605 |

The `gated` arm is not replicated: it overlapped both other arms in E3 and carries no
hypothesis.

**Probe graphs are regenerated from a different RNG stream (`default_rng(7)`).** E3's probe
set is part of what the threshold was fitted to; reusing it would test seed generalisation
only, not probe generalisation, and the statistic is a distance measured *on* those graphs.

## Gates

**Gate 0 — admission.** MAE < 2.5, >= 5/6 per arm, else INCONCLUSIVE (training).

**Gate 1 — mechanism.** `halt` median firing round < 8 on the new probe graphs.

**Gate 2 — the frozen rule.** Apply τ to every admitted model.

| Verdict | Condition |
|---|---|
| **REPLICATED** | >= 11/12 correctly classified by the frozen τ **and** Fisher exact two-sided p < 0.05 |
| **FAILED** | anything else |

There is no intermediate rung. A lead that only half-replicates is dead: that is precisely
the state `overshoot` reached before being written up and retracted.

## Also reported, and pre-committed as descriptive only

- The label distribution (expected: all `bellman_ford` again; if any model lands elsewhere
  that is noted but changes no verdict).
- Mean distance to the nearest reference per arm, against the 0.248 mean / 0.303 median
  symbolic reference-to-reference scale — i.e. whether E3's structural finding, that
  neural models are further from every reference than the references are from each other,
  holds on a fresh probe set. **This is the more important number in the run**, and unlike
  the τ test it is a confirmation of an already-observed effect on new data rather than a
  fitted rule.
- The observed E3b gap and the threshold that would have been optimal, reported *after*
  the frozen-τ verdict and explicitly not substituted for it.

## What each outcome means

- **REPLICATED.** The fingerprint carries graded information about truncation that the
  nearest-reference label discards. That would make the label, not the fingerprint, the
  broken component — and a signed per-reference statistic, reported with margins, becomes
  the method's actual output. This would be a genuine positive worth building on.
- **FAILED.** The neural side is finished. Write up the symbolic contribution and the
  characterised negative, per `HANDOFF.md`'s "Alternative to more experiments".
