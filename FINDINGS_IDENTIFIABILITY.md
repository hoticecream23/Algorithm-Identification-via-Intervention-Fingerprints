# Identifiability boundary — which algorithms are distinguishable at which depth

Reproduce:

```bash
python run_identifiability.py --graphs 12 --nodes 24
```

Node count fixed at 24, density swept from 0 (spanning path, depth ~17) to 0.8 (depth 2),
so propagation depth varies with `n` controlled.

## 1. The money result, in symbolic form

**On shallow graphs, truncated Bellman-Ford produces the same answers as real Bellman-Ford
and is still separable by intervention fingerprint.**

At density 0.8 (depth 2.0), truncated-BF with k=3 returns output identical to full
Bellman-Ford on **11 of 12** graphs, and is correct on 11 of 12. No accuracy metric can
tell them apart. The fingerprint separates them anyway, via `insert_shortcut_from_source`:

| predicate | bellman_ford | bf_truncated_k3 |
|---|---|---|
| recovery | exact | partial |
| latency | 2 | never |
| term_delta | + | 0 |

This is exactly the Phase E claim — two models with indistinguishable in-distribution
accuracy, told apart by causal response — demonstrated with no models trained. It also
gives the OOD story its mechanism: at density 0.2 the same truncated algorithm is correct
on **0 of 12** graphs. Same algorithm, same fingerprint, accuracy collapses when the
distribution deepens. The fingerprint saw on shallow graphs what accuracy could only reveal
after the shift.

## 2. The stated prediction was wrong, and the reason matters

Predicted in advance: BF and truncated-BF-k3 would be *inseparable* at depth <= 3, since an
early-halting algorithm is only detectably early on instances deep enough to halt in.

Observed: separable at **every** depth tested, including 2.0.

The error was conflating two distinct consequences of truncation:

- **limited propagation depth** — invisible when the graph is shallower than k, as predicted;
- **limited total runtime** — visible at *any* depth, because the probe fires mid-run and an
  algorithm that has already halted cannot respond to anything.

The second is what separates them at depth 2. This is a better result than the prediction:
truncation is detectable on shallow graphs, so a shortcut model can be caught on exactly the
easy distribution it was trained on, without constructing deep test instances.

Design consequence: interventions must fire while the model still has execution budget left,
which the relative firing rule already guarantees. A protocol that probed only completed
rollouts would lose this entirely.

## 3. Full identification needs depth >= 5

| mean depth | minimal separating set |
|---|---|
| 17.2 | 2 interventions |
| 7.6 | 2 |
| 5.1 | 3 |
| 3.7 | **impossible** |
| 2.9 | impossible |
| 2.1 | impossible |
| 2.0 | impossible |

Degradation is monotone and sharp. Below depth ~4 no intervention set in the suite separates
all pairs: `dijkstra`/`prim` and `dijkstra`/`spfa` both merge.

**Phase C must fingerprint on graphs of depth >= 5.** CLRS's default distribution is dense
and shallow; used unmodified it lands in the region where the method provably cannot work,
and would return a null result that says nothing about any model.

## 4. Which pairs are fragile

- `dijkstra`/`prim` and `dijkstra`/`spfa` rest on a single intervention even at high depth,
  and are the first to merge as graphs get shallower.
- `bellman_ford`/`bf_truncated_k3` stays separable throughout, but drops to one intervention
  below depth 3.
- Everything involving `bf_truncated_k3` against a *different family* stays robust (7-9
  interventions) at all depths.

## Anomaly to treat with suspicion

`dijkstra`/`spfa` reads inseparable at depths 3.7 through 2.1, then separable again at depth
2.0 by one intervention. Non-monotone re-separation is not physically sensible here and is
most likely a stability-threshold artifact at extreme density. It should not be relied on,
and is worth a targeted check before any claim depends on that cell.
