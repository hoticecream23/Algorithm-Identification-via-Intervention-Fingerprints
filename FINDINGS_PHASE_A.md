# Phase A findings — symbolic separation gate

Reproduce:

```bash
python run_phase_a.py --graphs 25 --nodes 24
```

5 algorithms (Bellman-Ford, truncated-BF k=3, SPFA, Dijkstra, Prim) × 9 interventions ×
25 graphs. No neural networks. Predicates read only `d` and `pi`.

## Gate result

**PASS on high-diameter graphs. FAIL on dense random graphs.** Stable across seeds
(0 and 7 give identical minimal separating sets).

The premise holds: intervention responses do identify algorithms — but only over the
right input distribution, which is a result in itself (see finding 3).

## 1. Identity is intervention-relative, and the minimal separating set is tiny

Two of nine interventions separate all ten algorithm pairs:

```
Minimal separating set: ['insert_improving_edge', 'inject_unreached_node']
```

Every other probe is redundant given those two. This is the concrete form of P1: the
interesting object is not "the fingerprint" but the *smallest intervention class in which
a family of algorithms stops being equivalent*. Seven probes carry no identity
information once these two are in hand.

Two pairs hang on a single intervention each — `dijkstra vs prim` (only
`insert_improving_edge`) and `dijkstra vs spfa` (only `inject_unreached_node`). Those are
the fragile edges of the classification and where a neural model will be hardest to place.

## 2. The truncation detector works — this is the OOD mechanism probe

`insert_shortcut_from_source` forces an improvement to propagate outward from a distant
node. The `depth` predicate reads how far it actually travelled:

| algorithm | depth reached |
|---|---|
| bellman_ford | 4+ |
| **bf_truncated_k3** | **1** |
| dijkstra / prim / spfa | none |

Truncated-BF is cleanly separated from real BF by depth alone, which is the hypothesis
in P2: a model that looks perfect in-distribution because training graphs were shallow
will show a bounded `depth` here. This predicate is the one to carry into Phase D.

## 3. Behavioural equivalence is also *distribution*-relative

On dense random graphs (diameter ~2), Dijkstra and SPFA become indistinguishable — the
gate fails. The reason is mechanical: `inject_unreached_node` only discriminates while
there still *is* an unreached node, and on a diameter-2 graph everything is reached almost
immediately.

Operational consequence for Phase C: **fingerprints must be extracted on high-diameter
instances.** Running the same protocol on CLRS's default dense graphs would silently lose
most of its discriminative power and produce a null result for the wrong reason.

## 4. Deletion probes are near-silent, for a structural reason

Relaxation only ever *lowers* `d`. Repairing a deleted edge requires *raising* it, so no
relaxation-based algorithm responds to a deletion at all: `spread1 = 0`, `depth = none`
for every algorithm.

This kills edge deletion as a discriminator among symbolic algorithms — but it makes it a
sharp *neural* discriminator, and the probes are kept for that reason. A neural model is
under no monotonicity constraint and can raise an estimate. "Responds at all to a
deletion" therefore separates a learned reasoner from genuine relaxation, and any model
that does respond has learned something that is not Bellman-Ford, however good its
validation accuracy.

## 5. The settled-set discriminator behaves exactly as predicted

`corrupt_node_up` on an already-converged node:

| algorithm | recovery | latency |
|---|---|---|
| bellman_ford | exact | 0 |
| dijkstra / prim / spfa | none | never |

BF keeps relaxing and repairs the damage in one round; the settled set in Dijkstra makes
the error permanent. This was the predicted crux and it needs no tuning to show up.

## Design decisions that turned out to be load-bearing

- **Observational convergence.** Dijkstra has a settled set, BF does not, and a neural
  model has nothing we are allowed to inspect. Defining "settled" as "estimate stopped
  changing" is what makes the same probe well-posed for all three.
- **Relative firing time.** Round 5 is five hops for BF and five settled nodes for
  Dijkstra. Firing at an absolute round produced pure artifacts — interventions landing
  after termination, measuring nothing. Firing at the midpoint of each algorithm's own
  active life fixed it.
- **Algorithm-independent targets.** Targets come from the graph's ground truth, never
  from the running algorithm's own pointers, so every algorithm gets the identical poke.
- **`pi_churn` against the control, not the reference.** Compared against the reference it
  measured whether the poke moved the *right answer* — a fact about the graph, identical
  for all algorithms, and not a response at all. That one bug manufactured separation on
  probes that in truth produced no response.

## Follow-on status

- **Tolerance for continuous outputs — resolved.** It must sit above the model's error
  floor and below the scale of real value gaps (the edge-weight scale, ~5). `tol = 2.5`
  works for models at MAE ~0.3-0.9. Below roughly MAE 5 no tolerance satisfies both bounds
  and fingerprinting is ill-posed, which makes MAE the admission criterion rather than
  validation accuracy. See `FINDINGS_NOISE.md` and `FINDINGS_SEED_CONTROL.md`.
- **Dijkstra vs Prim still rests on one intervention.** Add a probe keyed directly on
  `d[u]+w` versus `w` before trusting that boundary.
- **Sensitivity normaliser still unimplemented.** `delete_nontree_edge` is in place as the
  null control, but responses are not yet reported relative to it, so a globally jumpy
  model can look "different" for free.

Note that the depth-based truncation detector in section 2 did **not** transfer to neural
models: a shallow-trained network does not halt, so its limited generalization is invisible
to a probe that keys on execution structure. See `FINDINGS_OOD_NEGATIVE.md`.
