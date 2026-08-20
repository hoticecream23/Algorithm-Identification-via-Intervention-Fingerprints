# Findings: what the networks do that the family cannot (EXPLORATORY)

```bash
python run_residual_audit.py
```

Raw output: `residual_audit_log.txt`, `residual_audit_results.json`.

> **Nothing in this document is a result.** It is a set of candidate structural violations found
> by pointing the Phase F residual at the nine E3 checkpoints, plus the pre-registration they
> would need. Five findings in this project have died from being reported before a held-out
> replication, and the design here is weaker than it looks: nine checkpoints are three
> architectures × three seeds, i.e. **three independent design points, not nine**. Sign
> agreement across models is reported instead of a p-value for that reason, and every cell of
> every table is printed so that nothing is selected after the fact.

## Why this was worth doing

Phase F's negative was clean but silent about mechanism: the family contains all five reference
algorithms bit-identically and the trained networks not at all, six of nine at chi2 ~10⁴
(`FINDINGS_FAMILY.md`). The per-statistic residual breakdown localises the misfit —
`off_width` carries 36.9% of it and `self_med` 27.0%, two thirds between them — and both are
*bounded* in the family. So the question "is the misfit structural or parametric?" is decidable.

## Stage 1: structural audit

Four properties the family provably cannot have, with family members as the null. All four are
absent from every reference and every family member, as required.

| | plain_700 | plain_701 | plain_702 | gated_800 | gated_801 | gated_802 | halt_900 | halt_901 | halt_902 | family |
|---|---|---|---|---|---|---|---|---|---|---|
| `|J[v,v]| > 1` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| negative entries | **18.8%** | 1.1% | 0% | 0% | 0% | 0.4% | **15.5%** | **7.6%** | 0.3% | 0% |
| `width > deg(v)` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `γ_v > 1` | **26.9%** | 2.1% | 2.9% | **21.8%** | 9.5% | **15.2%** | **46.6%** | 10.9% | **14.7%** | 0–1.6% |
| `γ_v` IQR | **0.759** | **0.718** | 0.341 | **0.812** | **0.682** | **0.836** | 0.306 | 0.239 | 0.321 | **0.000–0.007** |
| `α_v` IQR | 0.416 | 0.191 | 0.047 | 0.076 | 0.061 | 0.038 | 0.061 | 0.109 | 0.141 | **0.000** |

**H1 — self-amplification: REFUTED.** `|J[v,v]| > 1` never occurs; the largest diagonal
anywhere is 0.832. So `self_med` being the second-largest residual is *not* α clipping at the
family's bound. It has to be time-variation instead: a fixed α gives a diagonal that is constant
across rounds (exactly 0.300 at α=0.7), and the networks' is not.

**H2 — anti-relaxation: CONFIRMED, in a third of the models.** `plain_700` (18.8% of
off-diagonal responses), `halt_900` (15.5%) and `halt_901` (7.6%) respond to a *lowered*
neighbour by *raising* their own estimate. Relaxation is monotone by definition, so no member of
the family produces this at any parameter setting — measured 0.0% for all five references and
both family probes. Three other checkpoints are at ~0%, so this is a property of particular
trained models rather than of the architecture.

This is the one place where two independent measurements in this project converge:
`FINDINGS_SHALLOW_SIGNALS.md`'s only surviving lead was `nonmono`, *rounds in which the model
raises an estimate*, which "classifies provenance better than it predicts OOD magnitude". The
signed Jacobian is the same phenomenon measured per-edge instead of per-round.

**H3 — non-local dependence: REFUTED, and this is load-bearing.** `width > deg(v)` never occurs.
One round of the network depends only on graph neighbours, so information really does move one
hop per round and the observable is measuring the right operator. The framing that motivated
Phase F — *the observable is fine, the hypothesis space is broken* — is confirmed, not merely
assumed.

**H4 — heterogeneity: CONFIRMED, decisively.** Every family member gives an `α_v` IQR of exactly
0.000 and a `γ_v` IQR of at most 0.007: one scalar pair, by construction, so the per-node values
are a delta spike. The networks give `γ_v` IQRs of 0.24–0.84 on a quantity nominally bounded in
[0,1]. **The per-node path discount spans essentially its whole range inside a single model.**

This also explains `row_sum_deviation` exactly. It is the gap between three separately-taken
medians, so it vanishes when all live rows share one (α, γ) and is non-zero when they do not.
That is why it reads 0.0000 for every family member and 0.012–0.168 for every checkpoint.

**Unpredicted, and the sharpest of the five: `γ_v > 1`.** In 2.1–46.6% of live rows, the network
propagates *more* than the full parent distance `d_u + w`. The family's γ is a discount on [0,1];
there is no setting at which it amplifies. The family control reads 0.0–1.6%, and the 1.6% is
measurement edge at a true γ of 0.9.

## Stage 2: is the heterogeneity a function of anything?

If the per-node discount is predictable from an observable property, the missing ingredient has a
functional form a redesigned family could carry. If not, the networks have no compact description
in these coordinates.

Spearman correlation with `γ_v`, median over the nine checkpoints, with sign agreement:

| predictor | median ρ | sign agreement |
|---|---|---|
| `round` | **+0.482** | **9/9** |
| `d_now` | **−0.420** | **9/9** |
| `deg` | +0.225 | **9/9** |
| `gap` (= `d_now − d_true`) | −0.344 | 8/9 |
| `hop` | −0.389 | 6/9 — the halt arm flips sign |

The same table for `α_v` is mostly noise: only `deg` reaches 9/9 and it is negligible (−0.045);
`d_now`, `gap` and `round` are 5/9, i.e. coin flips. **So the state dependence lives in γ, not
in α.**

### The division check, which is the decisive part

`γ_v = off_row_v / α_v` is a ratio, so a trend in it can come from the denominator shrinking as
nodes settle rather than from the discount rising. Checked on the undivided numerator:

| | median ρ | sign agreement |
|---|---|---|
| `ρ(round, γ_v)` | +0.482 | 9/9 |
| `ρ(round, γ_v)` restricted to `γ_v ≤ 1` | +0.490 | **7/9** |
| **`ρ(round, off_row_v)` — no division at all** | **+0.468** | **9/9** |
| `ρ(round, α_v)` | −0.060 | **5/9** |

`off_row_v` rises with round in all nine models with no division involved, while `α_v` has no
consistent trend at all. **The rise is in γ and it survives the artefact check.** The one caveat,
recorded rather than buried: restricting to `γ_v ≤ 1` drops sign agreement to 7/9, with
`gated_800` flipping to −0.305 — so part of the trend is carried by the amplifying rows.

**One free robustness check, from `--smoke` (3 graphs and 5 rounds instead of 6 and 8).** This is
a different measurement window on the same weights, so it is not a held-out test of anything —
but it does separate a stable statistic from an unstable one:

| | full run | smoke run |
|---|---|---|
| `ρ(round, off_row_v)` | +0.468, **9/9** | +0.390, **9/9** |
| `ρ(round, α_v)` | −0.060, 5/9 | +0.305, 7/9 |

The division-free γ statistic holds its sign in every model under both windows. The α statistic
reverses its median sign between windows, which is what a non-effect looks like. That is
consistent with the Stage-2 table and is the reason the frozen primary below is stated on
`off_row_v` rather than on anything involving α.

## What this says the missing ingredient is

Concretely, and this is the deliverable:

> The networks implement a **state-dependent path gain, not a constant discount.** The fraction
> of a parent's estimate that propagates varies per node, rises as the run progresses and as the
> node's own estimate falls, rises with degree, and **exceeds 1** for a substantial minority of
> updates. In a third of the models some responses additionally have the wrong sign.

A family that could contain these objects would need `γ` replaced by a gain function
`g(d_v, t, deg(v))` with range extending above 1, and would need signed rather than magnitude
responses in the observable. Note what that costs: a *function* where the family had a scalar is
not a sixth parameter, it is a different kind of model, and fitting it re-opens every
identifiability problem Phase F just measured — plus the α↔c confound, which is structural and
would not go away.

Note also what it explains. `HANDOFF.md`'s root-cause diagnosis was that these networks "differ
in *update-rule generalization*, not in *procedure*". A gain that depends on how settled the
current estimate is, is exactly that: correct where the training distribution put it, drifting
where it did not. The two accounts agree, and this one is quantitative.

## The pre-registration this would need

Nothing above should be believed yet. The confirmatory design, stated so it can be executed
without re-deciding anything:

1. **Fresh seeds are mandatory and are the expensive part.** The nine checkpoints are three
   architectures × three seeds; the correlations could be a property of these seeds. A held-out
   test needs newly trained models (seeds disjoint from 700–702 / 800–802 / 900–902), not fresh
   probe graphs against the same weights. CUDA is currently unavailable, so this is ~25–100 min
   per arm on CPU.
2. **Freeze the primary before running it.** `ρ(round, off_row_v) ≥ 0.30` with sign agreement
   ≥ 8/9 on the new seeds. One statistic, the division-free one, chosen here because it is the
   only Stage-2 cell that is both 9/9 and free of the ratio artefact — and *that choice is
   itself a best-of-N selection over 45 cells*, which is precisely why it must be frozen before
   the replication rather than after.
3. **Negative control in the same run.** Family members must give `ρ ≈ 0` and `γ_v` IQR ≤ 0.01,
   as they do here. If a family member shows the trend, the statistic is measuring the mask.
4. **Secondary, reported not gated:** `γ_v > 1` fraction and negative-entry fraction on the new
   seeds. `H2` is present in only 3 of 9 models, so it needs a larger sample before it is
   anything more than an observation.
5. **Pre-commit the interpretation of failure.** If the trend does not replicate, the conclusion
   is that the networks' per-node gain is heterogeneous but *not systematically* so — which
   closes Phase F permanently rather than suggesting a further redesign.

## Status

Phase F remains DEAD on its pre-registered criteria. This document does not revive it; it
converts "the family does not contain the networks" into "the family lacks a state-dependent,
possibly amplifying path gain, and here is the frozen test of that". Whether that is worth a
Phase G is a judgement about cost — it needs fresh training runs and it re-opens the
identifiability problems Phase F measured — not a judgement about whether the lead is real,
which is currently unknown.
