# Identifiability boundary — which algorithms are distinguishable at which depth

Reproduce:

```bash
python run_identifiability.py --graphs 40 --nodes 24 --seed 1
```

(The original sweep used `--graphs 12`. That is too few — see §1 and the resolved anomaly at
the bottom, both of which are `n = 12` artefacts.)

Node count fixed at 24, density swept from 0 (spanning path) to 0.8, so propagation depth
varies with `n` controlled.

> **Read the depth axis carefully.** This sweep originally reported depth as unweighted hop
> eccentricity. That is *not* the depth relaxation sees — see `fpid.graphs.weighted_depth`
> and §3 below. The runner now prints both; quote the weighted one.

## 1. The money result — SUPERSEDED, and strengthened

**Moved to `FINDINGS_P2_WITNESS.md` (`python run_p2_witness.py`).**

The claim — *truncated Bellman-Ford produces the same answers as real Bellman-Ford and is
separable by intervention fingerprint anyway* — is correct and now stated exactly. The
numbers previously given here are withdrawn:

- **"identical on 11 of 12 at density 0.8"** does not survive. It was measured on the hop
  axis, where density 0.8 looks like a homogeneous depth-2 population but is actually a
  mixture: full BF needs 3–5 rounds to settle on those graphs. At `n = 200` the agreement
  rate is 70–76%, at which point truncated-BF is also ~30 points less accurate and the
  "no accuracy metric can tell them apart" claim fails outright.
- **"correct on 0 of 12 at density 0.2"** does not reproduce; the same draw gives 3/12 and
  `n = 200` gives ~12%.

Keyed on `weighted_depth` the mixture resolves into a step function at exactly `k`, and the
claim upgrades from a rate to a guarantee: on instances of weighted depth ≤ k the two are
output-identical on **137/137** instances and the fingerprint still separates them; above k
they agree on **0/463**. Details, predicate table and sample-size caveat in
`FINDINGS_P2_WITNESS.md`.

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

## 3. Full identification needs weighted depth >= 6

Re-measured 2026-08-23 at 40 graphs, seed 1, reporting both axes:

| weighted depth | hop depth | minimal separating set |
|---|---|---|
| 17.1 | 17.1 | 2 interventions |
| 8.9 | 8.4 | 2 |
| 6.0 | 5.0 | 3 |
| 4.7 | 3.6 | **impossible** |
| 4.4 | 3.0 | impossible |
| 3.5 | 2.1 | impossible |
| 3.2 | 2.0 | impossible |

Degradation is monotone and sharp. The `|S_min|` sequence (2, 2, 3, impossible) is exactly
what the original 12-graph sweep found, so **the result replicates**; only the axis it is
plotted against has changed.

The threshold moves, though. On the hop axis the boundary read "impossible below ≈ 4"; on
the weighted axis it is **impossible below ≈ 5, and three interventions are needed at 6**.
The two axes diverge by 1–2 rounds at moderate density and coincide only on the spanning
path (density 0), where every edge is on the shortest path and hops and rounds are the same
thing.

**Fingerprint on graphs of weighted depth >= 6.** CLRS's default distribution is dense and
shallow; used unmodified it lands in the region where the method provably cannot work, and
would return a null result that says nothing about any model. Note that a distribution
selected to have *hop* depth 5 has weighted depth ~6 here — the old constraint happened to
land in roughly the right place for the wrong reason, which is exactly why it went unnoticed.

## 4. Which pairs are fragile

- `dijkstra`/`spfa` merges first, below weighted depth 6.0.
- `dijkstra`/`prim` survives to weighted depth 4.4 and merges at 3.5. Both rest on a single
  intervention even at high depth.
- `bellman_ford`/`bf_truncated_k3` stays separable throughout, dropping to one intervention
  at weighted depth <= 3.5.
- Everything involving `bf_truncated_k3` against a *different family* stays robust (7-9
  interventions) at all depths.

## Anomaly — RESOLVED, it was a small-sample artefact

The original sweep (12 graphs, seed 0) had `dijkstra`/`spfa` reading inseparable at hop
depths 3.7 through 2.1 and then **separable again** at 2.0 by one intervention. Non-monotone
re-separation is not physically sensible, and the doc flagged it as needing a targeted check
before anything depended on it.

Checked, 2026-08-23: re-run at **40 graphs, seed 1**, the cell is `no` at every depth at or
below 3.6. The degradation is monotone, as it should be. The anomaly was a stability-
threshold artefact of `n = 12` and does not reproduce.

```bash
python run_identifiability.py --graphs 40 --nodes 24 --seed 1
```

Nothing in the paper depended on that cell, but it is no longer a live suspicion.
