# Findings: E6 — attempting to close the Dijkstra/Prim boundary

```bash
python run_e6.py
```

Raw output: `e6_log.txt`, `e6_results.json`. Pre-registered in `PREREGISTRATION_E5E6.md`,
written before this ran.

## The question

Dijkstra and Prim are separated by exactly one intervention in the whole suite,
`lower_tree_edge_weight`, which moves both algorithms' selection keys at once
(`d[u]+w` for Dijkstra, `w` for Prim) and infers the boundary from the *manner* of
their responses. `THEORY.md` §7 lists this as open work: "add a probe keyed directly
on `d[u]+w` versus `w`."

`dissociate_keys` (`fpid/interventions.py`) was built to do that directly: raise
`w(u,v)` by `delta` and simultaneously lower `d[u]` by the same `delta`. Algebraically
`d[u]+w` is then exactly invariant (Dijkstra's key does not move) while `w` alone has
strictly risen (Prim's key does move). `raise_tree_edge_weight` is the control arm —
it moves both keys, structurally identical to what `lower_tree_edge_weight` already
does.

## Mechanism check — FAILED

30 probe graphs (density 0.05, n=24, min depth 5, dev seed 0), budget 48. Response
strength measured as the literal quantity the pre-registration names: the number of
nodes whose final `d` differs between the intervened and control runs.

| | raise (control arm) | dissociate |
|---|---|---|
| dijkstra | 0.00 | 1.00 |
| prim | 0.13 | 1.07 |

Required: Dijkstra weaker under `dissociate` than under `raise` on ≥ 80% of instances.
**Observed: 0%** — the exact opposite direction. Prim "comparable within 1" passed at
100%, but not for the intended reason (see below).

## Why it failed

`raise_tree_edge_weight` is **near-invisible to both algorithms**, not just Dijkstra.
Mean response is 0.00 for Dijkstra and 0.13 for Prim — indistinguishable from the
existing deletion probes. The reason is constraint 3, already known and load-bearing
in this project: relaxation only ever **lowers** `d`. Raising an edge's weight can only
repair by *raising* some node's current estimate back up, which no correct relaxation
algorithm does. This is exactly the mechanism that makes `delete_tree_edge_near/far`
near-silent against symbolic algorithms (`fpid/interventions.py`'s own comment on the
deletion probes), and a weight increase is the same move under a different name — it
is monotonically un-actionable for **any** relaxation-based executor, symmetric across
which key that executor happens to use.

So the intended contrast — Dijkstra's key held invariant, Prim's moved — never had a
chance to show up in the weight-change component of `dissociate_keys`, because neither
algorithm responds to that component *at all*. The response that **is** observed
(mean ≈ 1.0 for both) comes entirely from the intervention's other edit: directly
overwriting `ex.d[u]` to `d_true[u] - delta`. That is a state corruption identical in
character to the existing `corrupt_node_down` probe, and it affects Dijkstra and Prim
**symmetrically** because neither algorithm's response to a corrupted neighbour value
depends on which key it uses to select — the corruption propagates or doesn't based on
whether `u` is still live, which is unrelated to the `d[u]+w` vs `w` distinction this
probe was built to isolate.

**The regression check confirms this diagnosis rather than merely being consistent
with it.** With `raise_tree_edge_weight` and `dissociate_keys` registered in
`INTERVENTIONS`, `run_phase_a.py --graphs 25 --nodes 24` still separates every pair,
the minimal separating set is **unchanged**
(`['insert_improving_edge', 'inject_unreached_node']`, size 2), and — the specific
number this experiment was chasing — **`dijkstra` vs `prim` is still separated by
exactly one intervention**, `insert_improving_edge`, not two. The two new probes add
nothing to that boundary, which is what the mechanism-check failure predicts: their
only live component (the `d[u]` corruption) is not differential between the two
algorithms.

## Verdict

**MANIPULATION FAILED**, per the pre-registered gate. Per
`PREREGISTRATION_E5E6.md`'s pre-committed interpretation: this is reported as a
diagnosed construction failure, not read as a null about the Dijkstra/Prim boundary
itself.

What it does establish, and this is worth keeping: **the monotonicity constraint that
makes edge deletion a weak probe (constraint 3) applies equally to edge-weight
*increases*.** Any future attempt at a sharper Dijkstra/Prim probe needs to move a
selection key without relying on a weight raise to do it — for instance, by
constructing an instance where a *second, already-improving* path exists whose
relative ranking under `d[u]+w` versus `w` differs, so the dissociation is visible
through relaxation's one-directional (lowering-only) response rather than through a
raise. That is a genuinely different construction, not a parameter tweak on this one,
and is not attempted here.

`THEORY.md` §7's open item — "add a probe keyed directly on `d[u]+w` versus `w`" — is
therefore **not closed**. It should be restated as a known limitation with the
diagnosis above (monotonicity blocks the direct construction) rather than left as
undifferentiated open work, so a future attempt does not repeat this one.
