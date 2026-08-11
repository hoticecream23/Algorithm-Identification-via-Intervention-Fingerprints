# Pre-registration — E1 (Dijkstra-trained discrimination) + E2 (hint-free regime)

**Written before the run. Nothing below may be changed after seeing fingerprints.**

Date: 2026-08-10. Implemented by `run_e1_e2.py`. Results go in `FINDINGS_E1_E2.md`.

Four results in this project died because a winner was selected and reported before
replication. This document fixes the candidate set, the statistic, the thresholds and the
outcome ladder in advance, so that the run has exactly one degree of freedom left: the data.

---

## Question

**E1.** Every neural model trained so far was supervised on Bellman-Ford, so "all six models
identify as `bellman_ford`" (B0) was never a discrimination test — there was no alternative
for them to be confused with. Does the fingerprint recover *which algorithm a model was
trained on*, when half the models are trained on Dijkstra trajectories instead?

**E2.** Every model so far was hint-supervised, so a matching fingerprint is close to
tautological. What do models supervised only on the final answer fingerprint as?

---

## Design

### Shared setup (fixed, not tunable after the fact)

| Setting | Value | Why |
|---|---|---|
| `n_nodes` | 24 | matches every prior neural phase |
| `density` | 0.05 | gives propagation depth ~5 |
| **probe/train graph filter** | max hop distance from source **>= 5** | constraint 1: below depth ~4 no intervention set separates all pairs. Unfiltered density-0.05 sampling yields depth 3-7 (4/20 below 5), so the filter is applied at generation to **both** train and probe graphs |
| `rounds` | **24** (= n) | see "Round budget" below |
| `train_graphs` | 24 | as B0 |
| `probe_graphs` | 8 | as B0 |
| `epochs` | 1500 | as B0/B1 |
| probe `budget` | 48 (= 2n) | as B0 |
| `NeuralExecutor.max_rounds` | `None` | as B0; the model is not hard-halted during probing |
| `tol` | 2.5 | above the model error floor, below the edge-weight scale ~5 (constraint 9) |
| `stability` | 0.8 | as B0 |
| references | all 5 of `ALGORITHMS`, wrapped in `finite_init` | both sides of a comparison share the measurement setup |
| `UNREACHED` | 80.0 | correct for density-0.05 n=24 (longest path ~39); constraint 10 |

### Round budget: why 24 and not 12

Dijkstra settles one node per round, so it needs `n` rounds to converge; Bellman-Ford
converges in ~depth (~5) and then sits at a fixed point. At `rounds=24` **both algorithms
end at exactly the true SSSP distances** (verified: max deviation 0.0 across 5 sample
graphs; Dijkstra's last active round ~21, Bellman-Ford's ~5).

Two consequences, both load-bearing:

1. The two training groups have **identical final targets** and differ only in the *shape of
   the trajectory*. A single admission criterion (MAE against true distances) is therefore
   meaningful for both groups, and any fingerprint discrimination cannot be explained by the
   two groups computing different answers. This is the cleanest possible form of the claim
   under test: the fingerprint reads procedure, not output.
2. Under `hints=False` the loss touches only the round-24 estimate, which is identical for
   the two algorithms. **Hint-free training is therefore a single condition, not two.** E2 is
   consequently not a 2x2 cell; it is a negative control (below).

### Groups

| Group | Supervision | Seeds | n |
|---|---|---|---|
| `bf_hint` | full hints on Bellman-Ford trajectory | 0-5 | 6 |
| `dj_hint` | full hints on Dijkstra trajectory (`finite_init`) | 100-105 | 6 |
| `nohint` | final answer only (identical data for both algorithms) | 200-205 | 6 |

18 models, ~4.8 min/model measured => ~90 min training plus fingerprinting.

---

## Gates, evaluated in this order. A failed gate stops the interpretation.

### Gate 0 — admission (constraint 9)

A model is **admitted** if its held-out MAE on the final estimate, against true SSSP
distances in original units, is **< 2.5** (= `tol`). Non-admitted models are excluded from
every statistic and reported as excluded with their MAE.

If fewer than **5 of 6** models are admitted in either `bf_hint` or `dj_hint`, E1 is
reported **INCONCLUSIVE (training)**. Same rule applied to `nohint` for E2.

### Gate 1 — manipulation check (E1 only), evaluated before any fingerprint is read

Fingerprints cannot be blamed for a difference that was never installed. A sparse
min-aggregating MPNN has no global readout and may be unable to represent Dijkstra's global
argmin at all; if so, a null E1 is a statement about architecture, not about fingerprints.

For each admitted model, on 8 held-out graphs, compute the mean absolute deviation of its
per-round estimates from the Bellman-Ford hint trajectory (`MAE_bf`) and from the Dijkstra
hint trajectory (`MAE_dj`), averaged over rounds 1..24, in original units. Define

    delta = MAE_bf - MAE_dj      (positive = closer to Dijkstra)

**Pass** requires all three: mean `delta` < 0 for `bf_hint`; mean `delta` > 0 for `dj_hint`;
and the two groups' `delta` ranges **disjoint**.

If Gate 1 fails, E1 is reported **INCONCLUSIVE (manipulation)** whatever the fingerprints
show, and the reported conclusion is about MPNN capacity, not about the method.

`nohint` deltas are reported descriptively; they gate nothing.

### Gate 2 — the primary test (E1)

Each admitted model is classified against the 5 symbolic references by
`fpid.identify.classify` at `stability=0.8`, yielding a label and a margin.

**Primary statistic — family level.** Labels collapse to two families:

- `settling` = {`dijkstra`, `prim`}
- `relaxation` = {`bellman_ford`, `bf_truncated_k3`, `spfa`}

Prediction: `bf_hint` -> `relaxation`, `dj_hint` -> `settling`.

Family collapse is **primary rather than secondary** because the dijkstra/prim distance is
~0.02 while the B0 cross-seed noise floor is 0.094. An exact-label criterion asks the
instrument for resolution it provably does not have; pre-registering it as primary would be
a design error, and it is retained as a secondary readout only.

Test: **Fisher exact, two-sided**, on the 2x2 of (training algorithm) x (predicted family).

**Secondary statistic — exact label.** Full 2 x 5 confusion matrix, reported always.

**Margin reporting.** Every model's margin is reported. A model with margin **< 0.094** (the
B0 worst cross-seed distance) is flagged `unresolved` and its label is not treated as
established. The primary statistic is computed over **all admitted models regardless of
margin**, so that no selection occurs; a sensitivity analysis restricted to resolved models
is reported alongside and is explicitly secondary.

**Runtime ablation.** Bellman-Ford is quiet after ~5 rounds and Dijkstra after ~21, so part
of any discrimination may be pure runtime rather than propagation structure. The
classification is therefore recomputed with the `term_delta` predicate dropped from every
fingerprint, and both results are reported. Runtime is a legitimate observable (A3), so this
is a decomposition, not a disqualification.

### Outcome ladder (E1) — fixed in advance

| Verdict | Condition |
|---|---|
| **STRONG** | Fisher two-sided p < 0.05 on the family 2x2 (with 6v6 this needs >= 11/12 correct) **and** every correctly-labelled model has margin > 0.094 |
| **WEAK** | family split in the predicted direction with p < 0.20 (>= 10/12), **or** p < 0.05 with some margins below the noise floor. Reported as suggestive and explicitly **not established**; requires an independent seed set before any claim is made |
| **NULL** | anything else |

A WEAK outcome is not written up as a positive result. Given this project's history, the
only response to WEAK is a replication run on held-out seeds.

---

## E2 — hint-free regime

No directional prediction is registered; the outcome is genuinely open. Registered in
advance is only what will be **reported**, so that no post-hoc framing is available:

1. Admission rate and MAE of the 6 hint-free models. Hint-free is a 24-step unroll with loss
   at the final step only and has never been run; failure to train is a plausible outcome
   and is reported as such rather than as a fingerprint result.
2. The full label distribution, margins, and distance profiles.
3. Whether hint-free models drift toward a *different* reference than hint-trained ones, or
   toward none (high distance to everything, low margin — the "unidentified algorithm" case
   that `classify`'s distance profile exists to expose).

### E2 as the negative control for E1

The six hint-free models are trained on **byte-identical data** and differ only by
initialisation seed. Any spread in their labels is therefore pure instrument noise measured
on the same instrument E1 uses. Two registered readouts:

- **False-discrimination rate.** Number of distinct labels and the family split within the
  hint-free group. A group that splits across families under identical training means the
  E1 statistic can reach significance from seed noise alone; E1's verdict is then downgraded
  one rung on the ladder.
- **Permutation null.** Over all C(6,3)=20 balanced splits of the hint-free group into two
  arbitrary "groups", the fraction achieving family-perfect separation. This is the empirical
  false-positive rate of the E1 statistic under a true null.

---

## What each outcome means

- **E1 STRONG, Gate 1 passed.** The core identification claim holds on real networks: the
  fingerprint recovers training provenance from causal response alone. Stands independently
  of B1's OOD negative.
- **E1 NULL, Gate 1 passed.** The fingerprint cannot distinguish two networks known to
  differ in procedure. The neural side of the project is finished, and B1's negative was
  never specifically about OOD.
- **E1 INCONCLUSIVE (manipulation).** The architecture cannot learn Dijkstra from hints. Says
  nothing about the method; the next step would be E3's architecturally-distinct models,
  which install the procedural difference by construction rather than by supervision.
