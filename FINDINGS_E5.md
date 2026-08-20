# Findings: E5 — sensitivity normalisation

```bash
python run_e5.py
```

Raw output: `e5_log.txt`, `e5_results.json`. Pre-registered in `PREREGISTRATION_E5E6.md`,
written before this ran.

## The question

`fpid/interventions.py`'s `_delete_nontree_edge` docstring claims "every other response
gets normalised against this." That was never implemented. Proposition 5 — the finding
that ended the neural side (`THEORY.md` §6) — rests on every neural model sitting
`d_near ≈ 0.37–0.40` from its nearest reference against a reference-to-reference median
of `0.303`. If a chunk of that distance is the networks simply being globally jumpy, the
vacuity gap is partly an artefact of an unnormalised measure and Proposition 5's
hypothesis might not actually hold.

## The mask

Implemented as `fpid.identify.informative` / `keys_normalised`: a slot
`(intervention, predicate)` is dropped from a comparison between two fingerprints only
when *both* give the same response there as they give to `delete_nontree_edge`, the null
control. Masking is pairwise, not per-fingerprint — a per-fingerprint mask would drop
`corrupt_node_up` for Bellman-Ford (whose `recovery=exact` there matches its own null
response) and destroy the BF/Dijkstra separation Phase A rests on. `distance` and
`classify` in `fpid/identify.py`, and `separating_predicates` /
`separation_matrix` / `minimal_separating_set` / `unseparated_pairs` in
`fpid/separation.py`, all take an opt-in `normalise` argument, default `False` — every
existing number in every earlier findings document stays reproducible unchanged.

## Negative control — PASS

On `run_phase_a.py`'s own dataset (25 high-diameter graphs, n=24, seed 0): normalised and
raw give the **identical** minimal separating set,
`['insert_improving_edge', 'inject_unreached_node']`, and `unseparated_pairs` is empty
under both. Real relaxation algorithms essentially never respond to a deletion no
shortest path uses, so the mask has almost nothing to remove from the symbolic
fingerprints. Gate passed as predicted; proceeded to the primary.

## Primary — UNCHANGED

Measured on the nine existing E3 checkpoints (`artifacts/*.pt`, no retraining), 8 probe
graphs at depth ≥ 5 (median depth 5.0, matching E3's probe construction), budget 48,
`tol=2.5`:

| checkpoint | raw `d_near` | raw label | norm `d_near` | norm label | outcome |
|---|---|---|---|---|---|
| gated_800 | 0.412 | bellman_ford | 0.706 | bellman_ford | still-vacuous |
| gated_801 | 0.412 | bellman_ford | 0.706 | bellman_ford | still-vacuous |
| gated_802 | 0.367 | bellman_ford | 0.600 | bellman_ford | still-vacuous |
| halt_900 | 0.375 | bellman_ford | 0.333 | bf_truncated_k3 | **CLOSED** |
| halt_901 | 0.439 | bf_truncated_k3 | 0.500 | bf_truncated_k3 | still-vacuous |
| halt_902 | 0.389 | bf_truncated_k3 | 0.333 | bf_truncated_k3 | **CLOSED** |
| plain_700 | 0.438 | bellman_ford | 0.692 | bf_truncated_k3 | still-vacuous |
| plain_701 | 0.424 | bellman_ford | 0.667 | bf_truncated_k3 | still-vacuous |
| plain_702 | 0.424 | bellman_ford | 0.692 | bf_truncated_k3 | still-vacuous |

Reference-to-reference median: raw **0.303** (matches the number quoted throughout the
project), normalised **0.489**.

**7 of 9 checkpoints remain vacuous** (`d_near ≥` the normalised median) — the frozen
ladder's **UNCHANGED** rung, reached exactly at its threshold. The two exceptions are
both `halt` models, whose truncation-driven termination behaviour is genuinely close to
`bf_truncated_k3`'s under either measure; this is a repeat of E3b's finding that the
`halt` arm was always the closest thing to a positive lead in the project, not a new one.

**The normalised median rose (0.303 → 0.489) more than the checkpoints' distances did**,
which is why the ladder still reads UNCHANGED rather than CLOSED even though six of nine
raw-to-normalised deltas are individually large. Masking removes slots that are
uninformative for *both* sides of a comparison; between two different symbolic
algorithms, more slots survive that test than between a network and its nearest
reference, because the network's baseline jumpiness (see secondary, below) makes more of
its own responses match its own null-control response, and once one side of a pair
matches the null everywhere a slot's fate hinges on the reference. The net effect is
that normalisation sharpens the reference family's internal separations by more than it
narrows the network-to-reference gap.

## Secondary — null-control response rate

Fraction of each fingerprint's stable, non-null-control slots where the response differs
from its own null-control response (higher = more of the fingerprint is "real" signal by
this test's own logic, lower = more of it is baseline jumpiness):

| | rate |
|---|---|
| bellman_ford | 0.562 |
| spfa | 0.429 |
| dijkstra | 0.396 |
| bf_truncated_k3 | 0.326 |
| prim | 0.133 |

All five references sit well above 0 — the null control is behaving as a null control,
not accidentally separating real algorithms (which would have invalidated every earlier
fingerprint that included it). Checkpoint-level rates are in `e5_results.json` for anyone
who wants to correlate this against the residual audit's per-checkpoint findings.

## Verdict

**UNCHANGED**, per the pre-registered ladder. Sensitivity normalisation does not close
the vacuity gap — 7 of 9 checkpoints remain further from their nearest reference than two
different references sit from each other, under a measure explicitly designed to
discount baseline jumpiness. Proposition 5 stands, and this closes the specific objection
that the vacuity might have been an artefact of an unnormalised distance rather than a
fact about what the networks execute. Per the pre-registration, this outcome requires no
further action — it is the expected result, and `THEORY.md`'s open list can drop
"sensitivity normalisation is unimplemented" (open item 4) as closed.

`gamma_v` from `FINDINGS_RESIDUAL.md` and the null-control response rates measured here
are two different quantities — one operates on the response Jacobian, this one on
categorical predicates — but both point the same direction: whatever the networks are
doing, it is not well described as "the same procedure, just noisier."
