# Experiment D — the vacuity gap is not a resolution artefact. It widens with depth.

Reproduce:

```bash
python run_depth.py
```

Pre-registered in `PREREGISTRATION_DEPTH.md`, written before the run. Ladder: CLOSED /
UNCHANGED, no middle rung. **Verdict: UNCHANGED**, at both sample sizes, all four gates
passing.

---

## 1. The question

`FINDINGS_P2_WITNESS.md` established that hop eccentricity is not the depth relaxation sees,
and `FINDINGS_IDENTIFIABILITY.md` §3 re-cut the identifiability boundary on the weighted axis:
`|S_min| = 2` above weighted depth ~9, `3` at 6, **impossible** below ~5.

Every neural experiment in this project (E3, E5, G, H) gated probe graphs on *hop* depth >= 5,
which measures at **weighted depth 6.0** — one step above the regime where the instrument
provably cannot resolve anything. Nothing was invalidated by this (weighted depth is provably
>= hop depth, so the gate never admitted a graph that was too shallow), but the headroom was
minimal, and "your null was measured near your own resolution floor" was an objection with no
answer on record.

## 2. Design

One independent variable: the probe-graph distribution. Statistic, both ladder rungs, the
repaired firing rule and every threshold inherited verbatim from `PREREGISTRATION_H.md`.

| arm | density | `min_depth` (hop) | measured weighted depth |
|---|---|---|---|
| CONTROL | 0.05 | 5 | **6.0 – 6.1** |
| DEEP | 0.02 | 8 | **9.8 – 10.1** |

Two cohorts: the nine E3 originals (exploratory rung, `>= 7/9`) and the twelve Stage-2
checkpoints (confirmatory rung, `>= 10/12`, the rung `PREREGISTRATION_H.md` froze). CLOSED
required both.

## 3. The control reproduces H2

CONTROL/exploratory at `n = 8` — the arm that re-runs H2's exact distribution through this
code path:

| quantity | H2 published | Experiment D CONTROL |
|---|---|---|
| `d_near` | 0.290 | **0.2973** |
| reference-to-reference median | 0.216 | **0.2158** |
| ratio | 1.34 | **1.38** |

The residual difference is the probe-graph RNG seed. This is what makes the comparison
readable: a difference between arms cannot be a code difference, because the code reproduces
the published number on the published distribution.

## 4. Result — the gap widens

Vacuity ratio (`median d_near` / reference-to-reference median). Vacuous is `>= 1`.

| cohort | pass | CONTROL (depth 6) | DEEP (depth 10) |
|---|---|---|---|
| exploratory (9) | n=8 | 1.38 | **1.85** |
| exploratory (9) | n=24 | 1.70 | **1.84** |
| confirmatory (12) | n=8 | 1.01 | **1.65** |
| confirmatory (12) | n=24 | 1.33 | **1.68** |

Closed counts in the DEEP arm: **0/9** exploratory and **2/12** confirmatory at `n = 24`
(needed 7/9 and 10/12). Not a near miss in any cell.

Both terms rise with depth, and `d_near` rises faster. At `n = 24`, exploratory: the
reference-to-reference median goes 0.1653 -> 0.1999 while `d_near` goes 0.2812 -> 0.3684. The
references do spread out as the instrument resolves better — and the networks move away from
them faster than the references move apart from each other.

**Reading.** The vacuity is not a resolution artefact. Give the instrument its full resolving
power and the networks are seen to be *further* outside the reference family, not closer to
it. This is consistent with `PAPER.md` §6.5's reading — the networks differ in update-rule
generalisation while running similar control flow — since deeper instances give the update
rule more rounds in which to diverge from relaxation.

## 5. Gates (all passed, both passes)

- **G1 depth manipulation.** DEEP reached weighted 9.8 / 10.1 against a floor of 8.0, and
  strictly above CONTROL. The independent variable moved.
- **G2 firing.** 9/9 and 12/12 networks fired inside `[0.35, 0.60]` of active life in the DEEP
  arm. The H2 repair still lands at this distribution.
- **G3 negative control.** All ten reference pairs separated in both arms.
- **G4 comparable slots.** DEEP within the pre-declared 0.8x floor of CONTROL (38 vs 43, 41 vs
  40), so the `FINDINGS_E5.md` denominator-inflation artefact is excluded. Note this guard was
  protecting against a *false* CLOSED; the verdict went the other way, so it was never load-
  bearing here.

## 6. Two things worth keeping

**G3 independently replicates today's depth re-cut.** The negative control reports
`|S_min| = 3` in CONTROL (weighted 6.0–6.1) and `|S_min| = 2` in DEEP (weighted 9.8–10.1) —
exactly the boundary `FINDINGS_IDENTIFIABILITY.md` §3 re-measured, arrived at from different
graphs, a different seed and a different experiment. The weighted-depth axis reproduces.

**The shallow distribution is the noisy one.** Between `n = 8` and `n = 24` the CONTROL
confirmatory ratio swings 1.01 -> 1.33 (and its closed count 6/12 -> 1/12), while DEEP barely
moves (1.65 -> 1.68, and 1.84/1.85 on the exploratory cohort). A distribution with more
resolving power gives a more stable measurement, which is what one would hope and is worth
knowing before choosing a probe distribution. It also means H2's `n = 8` shallow measurement
sat in the regime where this statistic is least stable.

## 7. What this does not settle

The checkpoints are the same objects G already fingerprinted. This asks whether the
*instrument* was under-resolved, and answers no. It does not show that a different model class
would fingerprint differently, and it is not evidence about any model outside this one.

## 8. Consequences

- `PAPER.md` §6.4 listed three independent attacks the vacuity gap survived. This is the
  fourth, and the only one that attacks the *distribution* rather than the metric or the
  timing.
- The neural results (E3, G, H) stand, and stand on a distribution chosen for the wrong
  reason. Future neural probing should use `--density 0.02 --min-depth 8` (weighted ~10), not
  the `0.05 / 5` default.
- `deep_probe_graphs` still gates on hop depth. It is conservative (weighted >= hop) so it
  cannot admit a too-shallow graph, but it does not control the quantity that matters and
  should be re-keyed on `weighted_depth`.
