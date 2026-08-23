# Pre-registration — Experiment D: is the vacuity gap a resolution artefact?

Written 2026-08-23, **before** the run. Frozen at commit time.

Reproduce:

```bash
python run_depth.py
```

---

## 1. The question

Every neural experiment in this project (E3, E5, G, H) drew probe graphs through
`deep_probe_graphs(...)`, which rejection-samples on **hop** eccentricity with
`min_depth = 5`. `FINDINGS_P2_WITNESS.md` established that hop eccentricity is not the depth
relaxation sees, and `FINDINGS_IDENTIFIABILITY.md` §3 re-cut the identifiability boundary on
the weighted axis:

| weighted depth | `|S_min|` |
|---|---|
| 8.9+ | 2 |
| 6.0 | 3 |
| <= 4.7 | **impossible** |

Measured, the neural probe distribution (density 0.05, `min_depth` 5) gives **weighted depth
mean 6.0, minimum 5**. So every neural null in this project was measured one step above the
regime where the instrument provably cannot resolve anything at all.

**Nothing is invalidated by this.** Weighted depth is provably >= hop depth — for the node
attaining hop eccentricity, every path to it has at least that many edges — so the gate never
admitted a graph that was too shallow. But the headroom is minimal, and "your null was
measured near your own resolution floor" is an objection a reviewer can raise and we currently
cannot answer.

**Question.** Does the vacuity gap survive when the same instrument is applied at weighted
depth ~10, where `|S_min| = 2` and the instrument is at full resolving power?

## 2. What is and is not being changed

**Independent variable: the probe-graph distribution. Nothing else.**

| arm | density | `min_depth` (hop) | expected weighted depth |
|---|---|---|---|
| CONTROL | 0.05 | 5 | ~6.0 (what H2 ran) |
| DEEP | 0.02 | 8 | ~9.9 |

Statistic, thresholds, both ladder rungs, the repaired firing rule (`absolute=True`), `tol =
2.5` and `stability = 0.8` are **inherited verbatim from `PREREGISTRATION_H.md`**. No
candidate statistic is being selected here, which is the failure mode that killed five results
in this project. The primary quantity is the one H2 already reported:

> per checkpoint, `d_near` against the reference-to-reference median; `closed` iff
> `d_near < median`.

Running CONTROL alongside DEEP in the same pass is deliberate: it reproduces H2's published
number under this code path, so a difference between arms cannot be a code difference.

## 3. Cohorts

`load_checkpoints` globs `artifacts/*.pt` and returns all **21** checkpoints — the nine E3
originals (`*_x00`–`*_x02`) plus the twelve Stage-2 checkpoints G added (`*_x10`–`*_x13`).
`run_h.py` filters to the nine by explicit suffix, so its `CLOSED_AT = 7` is correctly
calibrated; this run uses both cohorts and so splits them explicitly:

| cohort | n | ladder | status |
|---|---|---|---|
| E3 originals (`x00`–`x02`) | 9 | `>= 7/9` closed | **exploratory** — G established these are three design points, not nine |
| Stage-2 (`x10`–`x13`) | 12 | `>= 10/12` closed | **confirmatory** — the rung `PREREGISTRATION_H.md` §236 froze |

A CLOSED verdict requires **both** rungs. The exploratory rung alone is not a result.

## 4. Gates, in order

The primary is not read until all four pass. Each is a manipulation or a control, not a
result.

**G1 — the depth manipulation landed.** Mean weighted depth of the DEEP probe graphs
`>= 8.0`, and strictly greater than CONTROL's. This is the new independent variable and the
project's standing rule is to verify the manipulation before reading the measurement; without
it a null here is a null about graph sampling, not about vacuity. FAIL => INCONCLUSIVE
(mechanism).

**G2 — the firing repair still lands at depth.** `>= 7/9` and `>= 10/12` networks fire within
`[0.35, 0.60]` of active life, per arm. Inherited from H2. FAIL => INCONCLUSIVE (mechanism).

**G3 — negative control, symbolic separation.** In each arm, all ten reference pairs separate
and `|S_min| <= 3`. If the instrument cannot separate the symbolic references at this
distribution, it cannot be asked about networks. Expect `|S_min| = 2` in DEEP. FAIL =>
INCONCLUSIVE (negative control).

**G4 — the `FINDINGS_E5.md` artefact guard.** `d_near` and the reference-to-reference median
are both computed over *jointly stable* slots, so a distribution that shrinks the comparable
slot count inflates the denominator and can manufacture a CLOSED. E5 recorded exactly this
failure ("8/9 closed", an artefact of a mask leaving as few as six comparable slots). So:
report the median comparable-slot count per arm, and require DEEP `>= 0.8 x` CONTROL. FAIL =>
INCONCLUSIVE (artefact).

Both terms — `d_near` and the median — are reported separately in every arm, never only as a
ratio, per the standing rule in `CLAUDE.md`.

## 5. Outcome ladder — two rungs, no middle

- **CLOSED.** `>= 7/9` exploratory **and** `>= 10/12` confirmatory close in the DEEP arm.
  Reading: the vacuity gap was a resolution artefact. The neural nulls (E3, G, H) were
  measured too close to the instrument's floor and must be re-examined; §6 of `PAPER.md` needs
  rewriting, and this becomes a positive result.
- **UNCHANGED.** Anything else.
  Reading: the vacuity gap is not a resolution artefact. Proposition 5 stands on materially
  firmer ground, `PAPER.md` §6.4 gains a fourth independent attack that the gap survived, and
  the reviewer objection is closed.

**Explicitly pre-declared:** a ratio that falls without crossing the threshold — say 1.34 to
1.15 — is **UNCHANGED**, not partial success. *Direction replicating while resolution does
not* is this project's characteristic dead-lead signature (`overshoot`, E3b), and it is given
nowhere to land. There is no third rung.

## 5a. Sample size, declared in advance

H2 ran on `--probe-graphs 8`. The primary here uses **8 in both arms**, because comparability
with H2's published number is the point and changing the count would change more than the
distribution.

Eight instances is few, and `FINDINGS_P2_WITNESS.md` documents modal values flipping across
the stability gate at small `n`. So a **secondary pass at 24 graphs in both arms** is declared
here, in advance, and reported alongside. **The verdict is taken from the primary (n=8).** The
secondary exists to show whether the verdict is sample-size fragile; if the two disagree, the
run is reported as INCONCLUSIVE (sample size) rather than resolved in either direction.

## 6. One pass

No follow-up statistic on this data if the first one misses. If DEEP comes back UNCHANGED,
the next move is a different question, not a different statistic on these graphs.

## 7. What this cannot settle

The checkpoints are the same objects G already fingerprinted. This run asks whether the
*instrument* was under-resolved, not whether a different model class would fingerprint
differently. A CLOSED verdict would not show that these networks are identifiable — only that
the measurement that said they were not was taken in the wrong place.
