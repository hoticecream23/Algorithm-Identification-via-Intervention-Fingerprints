# Seed control — does the fingerprint measure the algorithm or the model?

Reproduce:

```bash
python run_seed_control.py --seeds 6 --epochs 1500 --train-graphs 32 --probe-graphs 6 --budget 24 --tol 2.5
```

Six MPNNs, identical architecture and data, hint-supervised on Bellman-Ford, differing
only by seed. Each fingerprinted against the symbolic references.

## Result: seed-stable identity, with a margin caveat

| criterion | value | verdict |
|---|---|---|
| label consistency | 6/6 identify as `bellman_ford` | PASS |
| clustering | mean cross-seed 0.042 vs mean between-algorithm 0.205 (4.9x) | PASS |
| margin headroom | worst cross-seed 0.094 vs smallest margin 0.061 | MARGINAL |

Per-seed identification:

| seed | test MAE | label | margin | d(BF) | d(runner-up) |
|---|---|---|---|---|---|
| 0 | 0.49 | bellman_ford | 0.171 | 0.22 | 0.39 |
| 1 | 0.31 | bellman_ford | 0.061 | 0.18 | 0.24 |
| 2 | **1.59** | bellman_ford | **0.189** | 0.26 | 0.45 |
| 3 | 0.54 | bellman_ford | 0.111 | 0.19 | 0.30 |
| 4 | 0.28 | bellman_ford | 0.184 | 0.30 | 0.48 |
| 5 | 0.88 | bellman_ford | 0.096 | 0.15 | 0.25 |

The fingerprint recovers the trained algorithm from every seed. Cross-seed spread is
about a fifth of the between-algorithm scale, so the clusters are real.

**The caveat is genuine and should not be smoothed over.** Worst-case seed noise (0.094)
exceeds the thinnest classification margin (0.061, seed 1). Identity is stable *in this
run*, but a model sitting close to two references could have its label flipped by seed
alone. Operationally: report the margin alongside every label, and prefer an ensemble of
seeds whenever a model is near a boundary.

## Identity is decoupled from accuracy — direct evidence

Seed 2 is by far the worst model (MAE 1.59, roughly 5x the best seed) and yet identifies
as Bellman-Ford with the **largest** margin of the six (0.189).

This is the differential-measurement property from `FINDINGS_NOISE.md` showing up on real
models rather than on synthetic noise: the fingerprint reads the response operator, not the
values, so a model can be substantially less accurate while still executing recognisably
the same procedure. It is the concrete reason a fingerprint can carry information that
validation accuracy does not — and equally the reason it must never be reported as a
quality score.

## Why the first gate criterion was wrong

The original gate compared worst cross-seed distance against the *minimum* between-algorithm
distance, and failed (0.094 vs 0.021). That comparison is not meaningful: the closest
reference pair is `dijkstra`/`prim` at ~0.02, which `FINDINGS_IDENTIFIABILITY.md` had
already flagged as degenerate — separated by a single intervention and the first pair to
merge as graphs get shallower. Seed noise exceeding the width of a known-degenerate boundary
says nothing about whether a model's own label is stable.

Replaced with three criteria: label consistency, cluster separation, and margin headroom
reported honestly as MARGINAL rather than folded into a pass.

Note also that `min(between)` is itself unstable at 6 probe graphs. More probe graphs would
raise it; the number should not be quoted as a property of the method.

## For Phase B

- **Admission criterion is MAE, not validation accuracy.** Tolerance must sit above the
  model's error floor and below the scale of real value gaps (~5, the edge-weight scale).
  At MAE ~5 those constraints are unsatisfiable and fingerprinting is ill-posed regardless
  of how the model scores. Measured error floor here is ~0.3-0.9, so `tol = 2.5` works.
- **Report margin with every label.** Accuracy of identification hides confidence: all six
  seeds were "correct" while margins varied 3x.
- Seed variance in MAE was large (0.28 to 1.59 at fixed settings). The zoo will need enough
  seeds per configuration for that spread rather than assuming a configuration has one
  quality.
