# In-distribution signals for OOD failure — a null, with one survivor

Reproduce:

```bash
python run_shallow_diagnostics.py --seeds 5 --epochs 1500 --diag-graphs 8
python run_shallow_diagnostics.py --seeds 6 --seed-offset 100 --epochs 1500 --diag-graphs 8
```

Question: is there any quantity measurable **on shallow (in-distribution) graphs** that
separates models which will fail on deep graphs? Four candidates plus validation error as
baseline, tested on two disjoint sets of seeds.

## Result: no candidate reliably beats validation accuracy

Spearman correlation with deep-graph MAE, both runs:

| signal | seeds 0-4 | seeds 100-105 | consistent? |
|---|---|---|---|
| validation MAE (baseline) | +0.261 | +0.427 | — |
| `overshoot` | **+0.758** | **+0.371** | **no — collapsed** |
| `inject_err` | +0.430 | +0.490 | yes, but ~= baseline |
| `nonmono` | +0.321 | +0.469 | yes, but ~= baseline |
| `drift` | -0.479 | -0.434 | yes, wrong direction |

`overshoot` was selected post-hoc as the best of four candidates on the first set of seeds,
where it beat the baseline nearly 3x and ranked all five shallow-trained models in
perfectly monotone order against their OOD error. On held-out seeds it fell below the
baseline and its effect size flipped sign (Cohen's d +0.77 -> -0.31).

That is the fourth time in this project a promising number has evaporated under a proper
check, and the third caused by selecting a winner and reporting it before replication. The
mechanism argument for `overshoot` was sound -- relaxation approaches the answer from above
and cannot undershoot, so undershooting is a violated invariant rather than a small gap --
and it was still not enough. **A mechanism does not substitute for a held-out test.**

## What did survive

**`nonmono` separates the two groups consistently and with a growing effect size.** It
counts rounds in which the model *raises* an estimate, which no correct relaxation ever
does.

| run | shallow-trained | deep-trained | Cohen's d |
|---|---|---|---|
| seeds 0-4 | 1.550 | 0.975 | +0.69 |
| seeds 100-105 | 1.479 | 0.500 | **+1.63** |

Same direction both times, large effect in the second. But group separation is not the same
as predicting OOD magnitude, and its rank correlation (+0.32, +0.47) is no better than
validation error. It distinguishes *which training distribution a model came from* more
reliably than it predicts *how badly it will fail*.

**`drift` is consistently backwards.** The idempotence hypothesis was that a model with a
genuine fixed point would not move under extra iteration, and that drift would signal a
model which merely memorised a shallow input-output map. Both runs show the opposite:
deep-trained models drift more (-0.479, -0.434). Consistent enough to be real and currently
unexplained.

## Caveats on these numbers

- n = 10 and 12. Even the consistent effects are weakly powered.
- `deep-s100` is an outlier (val 2.850, OOD 2.600) — a badly trained model that is bad
  everywhere, which inflates the baseline's correlation in the second run.
- `overshoot`, `nonmono` and `drift` all require ground-truth distances. That is fine here
  (in-distribution labels exist) but they are not label-free diagnostics.

## Standing position

The setup itself is solid and reproduces across both runs: shallow-trained and deep-trained
models match on validation error (0.705/0.758 and 0.655/0.946) while differing 5-8x on deep
graphs. The Model-A/Model-B situation is real and easy to construct.

Nothing yet predicts it from in-distribution data — not intervention fingerprints
(`FINDINGS_OOD_NEGATIVE.md`), and not any of these four diagnostics.

## If continuing

Pre-register before running. The correct protocol, which was not followed here: fix the
candidate list and the decision rule, generate both seed sets up front, select on the first
and report on the second in a single pass. Choosing a winner and then seeking confirmation
invites exactly the failure above.

`nonmono` is the only candidate worth another look, on the strength of its consistent
direction and growing effect size — and it should be tested as a *classifier* of training
provenance rather than as a *regressor* on OOD error, which is what the data actually
supports.
