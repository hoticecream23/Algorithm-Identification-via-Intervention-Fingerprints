# Findings — Experiment G: do fingerprints discriminate networks from each other?

```powershell
python run_g.py --stage 1        # exploratory, existing 9 checkpoints, CPU
python run_g.py --stage 2        # confirmatory, 12 fresh checkpoints, CUDA
```

Decision rule, statistic, null, ladder and every control were frozen in
`PREREGISTRATION_G.md` **before `run_g.py` was written**. Nothing here selects.

---

## The question

Every neural experiment in this project measured **network → symbolic reference**, and
`THEORY.md` Proposition 5 says that comparison is vacuous: `d_near` 0.37–0.40 against a
reference-to-reference median of 0.303, so the nearest reference is the least-bad of five poor
matches.

Nothing in that argument prevents the networks from being separable **from each other**, and the
discrimination claim E3 set out to test never needed the symbolic references at all.
`fpid.identify.distance` is fingerprint-to-fingerprint and symmetric; before this experiment it
had never been called on two neural fingerprints. B0 measured *within*-group cross-seed distance
(mean 0.042, worst 0.094); no findings document reported a distance between two networks from
**different** arms.

That cell is now filled.

---

## Stage 1 — exploratory, the nine existing checkpoints

Stage 1 is exploratory whatever it returns: nine checkpoints are three architectures × three
seeds, i.e. **three independent design points, not nine**, and these fingerprints had already
been looked at by the decomposition, Phase F and E5.

### Controls and manipulation check — all pass

| check | result |
|---|---|
| **NC1** instrument health | no unseparated reference pairs; minimal separating set `['insert_improving_edge', 'inject_unreached_node']`, size 2 — Phase A reproduced exactly |
| **MC1** admission | 9/9 at MAE 0.299–1.738, all < 2.5 |
| **MC2** gate fires | `gated` final frozen fraction 0.81 / 0.93 / 1.00, all > 0.50 |
| **MC3** response room | `halt` median firing round 2 / 2 / 3, all < k = 8 |
| **NC2** non-degeneracy | within-arm distances 0.000–0.089 |
| **NC3** pseudo-arm control | 1/9 — at chance, as required |

The manipulation is real: the architectural differences are active on exactly these
checkpoints, so anything below is a statement about fingerprints and not about training. This
is the check whose absence turned E1 from a fingerprint null into a capacity null.

### Primary — 1-NN arm purity, raw distance

**5/9, exact p = 0.0393 over 280 partitions. Verdict: INDETERMINATE** (the ladder requires
≥ 7/9 *and* p < 0.05).

The descriptive gap statistic agrees that something is there: mean between-arm δ minus mean
within-arm δ = **+0.0369, exact p = 0.0143**.

So both statistics detect arm structure at p < 0.05, and the classification statistic lands
well short of its threshold. **That is direction replicating while resolution does not — this
project's characteristic dead-lead signature**, the one that killed `overshoot` and E3b. The
ladder was built with no middle rung precisely so this state could not be written up as a
partial success, and it is being reported as INDETERMINATE, which is a non-result.

### The critical subset — gated vs plain: NULL

`halt` is deployed at `max_rounds = 8` by the harness, so its trajectories are truncated by
construction rather than by anything it learned. Any three-arm separation it drives is a fact
about `neural_ctor`'s arguments. The load-bearing comparison is `gated` vs `plain`, where the
difference is a local gating rule the network had to learn to use.

**2/6, below the chance level of 2.4/6. Verdict: NULL.** Gap +0.0216, p = 0.30.

Whatever weak structure the three-arm test picked up is not present in the one pair where the
procedural difference is learned rather than imposed.

### Why — the networks are inside each other's seed noise

The mechanism is visible in the distance matrix and is a sharper statement than E3's null.

| quantity | value |
|---|---|
| mean within-arm δ | 0.0318 |
| mean between-arm δ | 0.0688 |
| **between-arm pairs at or below B0's cross-seed noise floor (0.094)** | **21 / 27** |
| max between-arm δ | 0.163 |
| reference-to-reference median δ | 0.303 |
| exactly-identical cross-arm pairs (δ = 0.000) | `gated_800 ≡ plain_702`, `gated_801 ≡ plain_702` |

Two networks from *different arms* have **bit-identical fingerprints**. The most distant pair
of networks in the whole set sits at 0.163 — roughly half the distance between two *different
symbolic algorithms*.

So the finding is not that the arms are mislabelled. It is that **all nine networks occupy a
region of fingerprint space smaller than the spacing between two classical algorithms, and
mostly smaller than the spread between two seeds of the same architecture.** Proposition 5 said
the label was unsupported; this says the geometry is too, on the reference-free question.

### Secondaries — reported, never decisive

**S2, normalised distance (E5's pairwise mask).** 5/9, p = 0.0250 → INDETERMINATE. Agrees with
the raw measure on the verdict, so §7's raw-versus-normalised disagreement clause does not fire.

**S3, the Phase-F (α̂, γ̂) embedding — the strongest-looking lead going in, and it is killed by
its own pre-registered gate.**

`ROUTES_TO_POSITIVE.md` §3.2 flagged this as "the lead is already visible", on the strength of
`FINDINGS_FAMILY.md`'s Gate-3 table showing `gated` and `halt` tightly clustered and disjoint.
Measured here, 1-NN arm purity in (α̂, γ̂) is **8/9 at a 6-round window and 7/9 at 8 rounds** —
much stronger than anything in fingerprint space.

**It fails the window-stability gate.** The pre-registration required the 1-NN arm assignment to
be identical at both windows, because `FINDINGS_FAMILY.md` had already measured `plain_700`
moving from (0.670, 0.283) at 6 rounds to (0.316, 0.864) at 8 while family members recover
identically. Measured here, `plain_700` goes (0.691, 0.196) → (0.510, 0.730) and the assignment
changes. **The embedding is partly measuring the measurement window**, and parameters fitted
outside the family are unstable under it — chi² is of order 10³–10⁵ for every checkpoint, which
is the family saying so.

This is the gate doing its job: the most attractive number in the experiment was disqualified by
a criterion written before it was computed. Had the gate not existed, "8/9 arm purity in
Phase-F coordinates" is exactly the result that would have been reported and later retracted.

**S4, residual-audit features.** Post-hoc reads, weak by construction. `halt` does have the three
tightest `γ_v` IQRs (0.290 / 0.236 / 0.312 against 0.344–0.841 elsewhere), consistent with
`FINDINGS_RESIDUAL.md`. The negative-entry pattern that `ROUTES_TO_POSITIVE.md` §3.3 read as
`gated`-specific does hold here (`gated` 0.0% / 0.0% / 0.0%; `halt` 17.0 / 10.0 / 0.1; `plain`
20.5 / 1.3 / 0.0) — but with one arm at three seeds this is an observation, not evidence.

### Stage 1 verdict

**INDETERMINATE**, with the critical `gated`-vs-`plain` subset **NULL**. Stage 2 is the verdict
of record and it resolves this downward, not upward.

---

## Stage 2 — confirmatory. **NULL.**

Twelve fresh checkpoints (`plain_710–713`, `gated_810–813`, `halt_910–913`), a fresh probe-graph
draw (seed 21, depths 5–7), `run_decompose.py`'s exact training configuration. Thresholds fixed
in `PREREGISTRATION_G.md` §8 before this ran.

Controls again all pass: NC1 reproduces the size-2 minimal separating set; 12/12 admit
(MAE 0.273–2.365); `gated` frozen 0.59–1.00; `halt` fires at rounds 2.5–3 inside k = 8; NC3
pseudo-arm purity 0/12.

| statistic | Stage 1 (9 models) | Stage 2 (12 models) | threshold | verdict |
|---|---|---|---|---|
| **primary** 1-NN arm purity | 5/9, p = 0.039 | **3/12, p = 0.024** | ≥ 10/12 | **NULL** |
| **critical** gated vs plain | 2/6 (chance 2.4) | **0/8, p = 1.00** | 8/8 | **NULL** |
| S1 between − within gap | +0.0369, p = 0.014 | +0.0416, p = 0.017 | descriptive | — |
| S1 gap, gated vs plain only | +0.0216, p = 0.30 | **−0.0078, p = 0.60** | descriptive | sign flips |
| S2 normalised distance | 5/9, p = 0.025 | 1/12, p = 0.50 | ≥ 10/12 | **NULL** |
| S3 Phase-F window stability | UNSTABLE | UNSTABLE (9/12 at 6r, 6/12 at 8r) | must match | disqualified |

**Purity is at chance.** 3/12 against a chance level of 3/11 ≈ 3.3/12. The `gated`-vs-`plain`
subset is **0/8** — not one of the eight networks has a same-arm nearest neighbour — and its gap
statistic **changes sign** between stages, which is what a null looks like when it is measured
twice.

*(The exact p of 0.024 on a chance-level purity is not a contradiction and is not evidence of
anything. Ties dominate this distance matrix, and the pre-registered rule scores a tie against
the hypothesis, so most relabellings score 0; achieving 3 is uncommon under the null but is
still nowhere near the 10/12 the ladder requires. The p-value and the threshold are answering
different questions, and only the threshold is the criterion.)*

### The one thing that does replicate, and what it is

The between-minus-within **gap** replicates cleanly in direction and magnitude: +0.0369
(p = 0.014) then +0.0416 (p = 0.017), on disjoint seeds and disjoint probe graphs. So there is a
real, reproducible arm structure in fingerprint space.

It is not usable and it is not about learned procedure. It vanishes on the `gated`/`plain`
subset — where it flips sign — so it is carried by `halt`, whose trajectories are truncated at
`max_rounds = 8` by the deployment harness. **The replicating signal is the harness argument,
not the network.**

### Why: on fresh seeds, differently-built networks are bit-identical

| quantity | Stage 1 (9) | Stage 2 (12) |
|---|---|---|
| mean within-arm δ | 0.0318 | 0.0555 |
| mean between-arm δ | 0.0688 | 0.0972 |
| between-arm pairs ≤ B0's noise floor (0.094) | 21/27 | 21/48 |
| **exactly-identical pairs (δ = 0.000)** | 5 of 36 | **18 of 66** |
| **of those, cross-arm** | 2 | **11** |

Eleven pairs of networks **from different arms** have bit-identical fingerprints — including
`gated_810 ≡ plain_710 ≡ plain_711 ≡ plain_713` and `halt_910 ≡ plain_710`. Within-arm distances
now reach 0.200, larger than the between-arm mean.

The instrument does not resolve these objects at all. That is a stronger and more specific
statement than "the label is wrong."

---

## Verdict

**Experiment G is NULL at both stages, and the reference-free question is closed.**

E3's null could be answered: *"your label is broken, not your instrument"* — Proposition 5 says
nearest-reference comparison is vacuous, so E3 never tested the fingerprint itself. That defence
is now gone. Asked the question that needs no references and to which Proposition 5 does not
apply, intervention fingerprints **do not** separate networks built to differ in procedure, and
the reason is measurable: their fingerprints sit inside each other's seed noise, and a sixth of
all pairs are identical.

The neural side of this project is finished on strictly stronger grounds than E3 left it.

---

## What this costs and what it buys

**It closes the best remaining objection to the neural negative**, which is worth more than
another qualified null. `ROUTES_TO_POSITIVE.md` ranked §3 as the highest-expected-value route in
the whole document, on a correct observation — the project had been asking a question
Proposition 5 forbids while never asking the neighbouring one that is legitimate. The
neighbouring question has now been asked, twice, with fresh seeds and fresh graphs, and it
answers the same way. Nobody has to wonder about it again.

Three things `ROUTES_TO_POSITIVE.md` ranked highly are downgraded by this run:

- **§3.1** (fingerprint-space clustering) is **NULL** at the confirmatory stage, and **0/8** on
  the subset where the procedural difference is learned rather than imposed.
- **§3.2** (the Phase-F embedding) is window-unstable at **both** stages. Its purity looked
  excellent both times at a 6-round window (8/9, then 9/12) and degraded at 8 rounds (7/9, then
  6/12) with the arm assignment changing. The disjoint `gated`/`halt` clusters in
  `FINDINGS_FAMILY.md`'s Gate-3 table are therefore not usable as an arm classifier until the
  window dependence is fixed. That is a concrete, fixable defect rather than a dead end — but it
  is not a positive result, and the fact that it replicated *as unstable* makes it much less
  likely that a longer window would rescue it.
- **§3.3** (residual features) is unchanged in status: still a post-hoc read, now with no
  primary result to support.

What survives untouched is §2 (the pinned-predicate repair and the orthogonality-metric
recheck), §4 (new observables), §5.1 (Family-2 with a state-dependent gain), and everything
symbolic in §7–§8.

**§2.1 is now considerably better motivated.** An instrument in which a sixth of all model pairs
are *bit-identical* — including eleven cross-arm pairs — is an instrument with far too few
informative slots. That is exactly the signature a fingerprint would have if ~29% of it were a
pinned constant, which is what `FINDINGS_DECOMPOSITION.md` §3 measured and what nothing has yet
fixed. G did not test that repair; it strengthened the case for running it.

The honest ordering after G: **§2.1 + §2.2 first** (repair the instrument, then re-check the
orthogonality metric that closed the door on probe redesign), then **§5.1**, then the symbolic
and theoretical work in §7–§8, which was never blocked on any of this.
