# Pre-registration — Experiment G: do fingerprints discriminate networks from each other?

Written 2026-08-20, **before `run_g.py` was written and before any statistic below was
computed.** Frozen. `ROUTES_TO_POSITIVE.md` §3 is the motivation; this file is the contract.

The candidate list is complete as written. Nothing may be added after Stage 1 is read.

---

## 1. The question, and why it is not a question this project has already answered

Every neural experiment in this repository measured **network → symbolic reference**:
B1, the shallow diagnostics, E1, E2, E3, E3b, the decomposition, Phase F, E5. Proposition 5
says that comparison is vacuous, because `d_near` (0.37–0.40) exceeds the reference-to-reference
median (0.303) — the networks are not in the reference set, so the nearest one is the
least-bad of five poor matches.

**Nothing in that argument prevents the networks from being separable from each other.**
Proposition 5 is a statement about a *label*. The discrimination claim — "does intervention
response tell two differently-built models apart?" — never needed the symbolic references at
all, and it is the claim E3 set out to test and answered only through the label.

`fpid.identify.distance(a, b)` is fingerprint-to-fingerprint and symmetric. It has never been
called on two neural fingerprints. B0 measured *within*-group cross-seed distance for six
identically-trained models (mean 0.042, worst 0.094); no findings document reports a single
distance between two networks from **different** arms. That cell is empty, and this experiment
fills it.

`PREREGISTRATION_E3B.md` anticipated exactly this resolution: if the fingerprints separate but
the labels do not, "that would make the label, not the fingerprint, the broken component."

---

## 2. What is being measured

Nine existing checkpoints in `artifacts/`, from `run_decompose.py`:

| arm | checkpoints | what differs |
|---|---|---|
| `plain` | `plain_700/701/702` | none — plain MPNN, runs to convergence |
| `gated` | `gated_800/801/802` | `GatedMPNN(eps=0.002)`, freezes a reached node once it stops moving |
| `halt`  | `halt_900/901/902`  | trained and deployed for `k = 8` rounds, then stops |

All three were trained hint-free on identical data, so supervision is held constant and
architecture is the only manipulated variable. Their mean MAE is 0.67 / 0.75 / 0.73 — the arms
are **accuracy-matched by design**, so a separation cannot be attributed to one arm simply being
worse. That confound was designed out in `PREREGISTRATION_E3.md` before E3 ran.

Fingerprints are computed by `run_e1_e2.fingerprint_of` on probe graphs from
`deep_probe_graphs`, at `tol = 2.5`, `stability = 0.8`, `budget = 2n`, exactly as E3, E5 and
the decomposition computed them. **No parameter of the fingerprint pipeline is changed by this
experiment.** If any had been, the negative control in §6 would be measuring a different
instrument than the one whose nulls are on record.

---

## 3. Manipulation check — runs first, and can stop the experiment

E1 looked like a null about fingerprints and was a null about MPNN capacity, because nobody
checked that the independent variable had moved before reading the dependent one. So, before
any distance is computed:

- **MC1 — admission.** Every checkpoint must have MAE < 2.5 on a freshly generated admission
  set (constraint 8). A model that cannot be fingerprinted at all is not evidence about
  fingerprints.
- **MC2 — the gate fires.** `gated` checkpoints must reach a final frozen fraction > 0.50.
- **MC3 — response room.** `halt` checkpoints must have median firing round < `k = 8`, or the
  intervention lands after the model has already stopped and there is nothing to observe.

If MC1 fails for any checkpoint, that checkpoint is dropped and the loss of power is reported.
If MC2 or MC3 fails, **Experiment G is INCONCLUSIVE (mechanism)** and the primary statistic is
not reported at all — not reported-and-discounted. That is E3's Gate 1, reused unchanged.

---

## 4. The primary statistic, fixed now

**1-nearest-neighbour arm purity in fingerprint space.**

For each checkpoint `i`, let `nn(i)` be the checkpoint minimising `distance(fp_i, fp_j)` over
`j != i`, at `stability = 0.8`, `normalise = False`. Purity is the count of `i` for which
`arm(nn(i)) == arm(i)`, out of 9.

Ties in the argmin are broken **against the hypothesis**: if any tied nearest neighbour is from
a different arm, the item scores 0. Fingerprint distances are rationals over a small denominator
and exact ties are likely, so this rule matters and is fixed now rather than discovered later.

Chance level is 2/8 = 0.25, i.e. 2.25 of 9.

**Why this and not the within-versus-between gap.** Purity is a classification statistic, so it
answers the question actually being asked, and it is invariant to any monotone rescaling of the
distance — which makes it immune to the entire class of objection E5 had to spend an experiment
disposing of. The gap statistic is reported as a descriptive secondary (§7) and **is not a
decision criterion.**

### Null distribution: exact enumeration, not sampling

Purity depends only on the *partition* of the 9 checkpoints into three groups of 3 — it asks
whether a neighbour shares an item's group, never which group that is — so the null is
enumerated over unordered partitions, of which there are exactly `9! / (3!^3 · 3!) = 280`.
(The 1680 labelled assignments give an identical p-value, since each partition is counted six
times on both sides of the ratio.) All 280 are enumerated and the p-value is the exact fraction
with purity >= observed. No permutation sampling, no random seed, no Monte-Carlo error. The
smallest attainable p is 1/280 = 0.0036, which clears 0.05 with room to spare.

### The ladder — two rungs, no middle

| purity | verdict |
|---|---|
| **>= 7/9 and exact p < 0.05** | **DISCRIMINATES** |
| **<= 4/9** | **NULL** |
| 5–6/9, or p >= 0.05 | **INDETERMINATE** |

INDETERMINATE is a non-result and is reported as one. It is explicitly **not** grounds for a
follow-up statistic, a different stability threshold, a different `tol`, or a normalised
distance. Direction-replicating-without-resolution is this project's characteristic dead-lead
signature (`overshoot`, E3b); this table leaves it nowhere to land.

---

## 5. The critical subset: gated vs plain

**`halt` is expected to separate for an uninteresting reason and must not carry the result.**
It is deployed at `max_rounds = 8` by the harness, so its trajectories are truncated by
construction rather than by anything it learned. A three-arm purity that is driven entirely by
`halt` would be a fact about `neural_ctor`'s `max_rounds` argument.

So the load-bearing comparison is **`gated` vs `plain`**, where the difference is a local gating
rule the network had to learn to use, and where E3's own primary test lived.

Statistic: 1-NN purity restricted to the six `plain`/`gated` checkpoints, out of 6, chance
level 2/5 = 0.4.

**Stated limitation, in advance:** six items admit only `6!/(3!^2 · 2!) = 10` partitions, so the
smallest attainable exact p is 0.1. **A p-value on this subset at Stage 1 cannot reach 0.05 and
will not be quoted as though it could.** Stage 1 reports purity and sign only. This is the
reason Stage 2 exists, and it is a limitation of the sample size, not a result to be softened.

Stage 1 reading: 6/6 is clean separation; <= 3/6 is null; 4–5/6 is indeterminate.

---

## 6. Controls, all in the same run

- **NC1 — instrument health (positive control).** On the same probe graphs, the five symbolic
  references must still have no unseparated pairs and a minimal separating set of size <= 3.
  This is E5's negative control reused. If it fails, the fingerprint pipeline is broken on this
  graph draw and nothing else in the run means anything.
- **NC2 — non-degeneracy.** Within-arm distances must not be identically 0. If all three
  checkpoints in an arm produce bit-identical fingerprints, purity is measuring a constant.
  Reported against B0's cross-seed mean of 0.042 for scale.
- **NC3 — pseudo-arm control.** Regroup the same nine checkpoints by **seed index** rather than
  architecture: `{plain_700, gated_800, halt_900}`, `{plain_701, gated_801, halt_901}`,
  `{plain_702, gated_802, halt_902}`. These groups share no architectural property, so purity
  must sit near chance. **If pseudo-arm purity is >= 7/9, the primary result is void** — the
  statistic would be reading something other than architecture, and the run is reported as a
  failed control, not as a finding.

---

## 7. Secondary analyses — reported, never decisive

Fixed now so that none of them can be promoted to primary afterwards.

- **S1 — within/between gap.** Mean within-arm δ versus mean between-arm δ, with the same exact
  280-partition null. Descriptive.
- **S2 — normalised distance.** The whole primary re-run at `normalise = True` (E5's pairwise
  mask). Reported beside the raw result. **The raw result is the one of record**, because that
  is the distance every existing number in the repository was computed with. If raw and
  normalised disagree, the experiment is INDETERMINATE, not "positive under one measure".
- **S3 — Phase-F embedding (`ROUTES_TO_POSITIVE.md` §3.2).** 1-NN purity in fitted `(α̂, γ̂)`
  under Euclidean distance. `FINDINGS_FAMILY.md`'s Gate 3 shows `gated` at α̂ 0.35–0.44 /
  γ̂ 0.12–0.17 against `halt` at 0.76–0.83 / 0.79–0.96, tight and disjoint.

  **Framing that must be stated wherever this is reported:** chi² of order 10⁴ says the family
  does not *contain* these networks. It does not say the projection onto (α̂, γ̂) is
  uninformative — a rejected generative model can still be a good coordinate system. This is a
  descriptive embedding, never a claim of family membership.

  **Stability gate, pre-registered because the threat is already measured:** `plain_700` reads
  (0.670, 0.283) at 6 rounds and (0.316, 0.864) at 8, while family members recover identically
  under the same change. Parameters fitted outside the family are unstable under the measurement
  window. So S3 is computed at **both 6 and 8 rounds** and is reported as informative **only if
  the 1-NN arm assignment is identical at both windows**. If it moves, the embedding is
  measuring the window and is reported as such.
- **S4 — residual-audit features (§3.3).** `γ_v` IQR, fraction `γ_v > 1`, fraction negative
  Jacobian entries, by arm, via `run_residual_audit.rows_of`. These are post-hoc reads of a
  table built for another purpose and are **weak evidence at best**; they are reported for
  consistency with S3, not as independent confirmation.

---

## 8. Stage 2 — the confirmatory rung

Stage 1 is **exploratory**, without exception, whatever it returns. Nine checkpoints are three
architectures × three seeds: **three independent design points, not nine.** Every retraction in
this project came from reporting a statistic computed on data that had already been looked at,
and the Stage 1 fingerprints have been looked at extensively by the decomposition, Phase F and
E5.

Stage 2 trains **four fresh seeds per arm** — `plain_710–713`, `gated_810–813`, `halt_910–913` —
with `run_decompose.py`'s exact training configuration (hint-free, 1500 epochs, 24 train graphs,
n = 24, `GATE_EPS = 0.002`, `HALT_K = 8`), on **freshly drawn probe graphs**.

Four rather than three is deliberate and is the reason the number is fixed here: the
`plain`/`gated` subset then has 8 items and `8!/(4!^2 · 2!) = 35` partitions, so an exact p of
0.029 is attainable and §5's ceiling is lifted.

Stage 2 recomputes **only the primary statistic of §4 and the critical subset of §5**, under the
ladder already written above. No secondary from §7 is recomputed, and no new statistic is
introduced. The Stage 2 verdict is the verdict of record.

### Stage 2 thresholds, written out explicitly

**Amended 2026-08-20, after Stage 1 ran and before Stage 2 ran.** §4's ladder was stated as
counts out of 9 and never restated for a 12-item sample, which left the Stage 2 thresholds
implicit in `run_g.py` rather than fixed here. They are the §4 proportions carried over
unchanged — `7/9 = 0.778` and `4/9 = 0.444` — and nothing about the criterion is being
loosened. Recording the amendment rather than quietly relying on the code:

| sample | DISCRIMINATES at | NULL at | chance |
|---|---|---|---|
| three arms, 9 models (Stage 1) | >= 7/9 and p < 0.05 | <= 4/9 | 2.25/9 |
| three arms, 12 models (Stage 2) | >= 10/12 and p < 0.05 | <= 5/12 | 3/12 |
| gated vs plain, 6 models (Stage 1) | 6/6 | <= 3/6 | 2.4/6 |
| gated vs plain, 8 models (Stage 2) | 8/8 and p < 0.05 | <= 4/8 | 3.43/8 |

The pseudo-arm void threshold of §6 scales the same way: >= 10/12 at Stage 2.

If Stage 1 is DISCRIMINATES and Stage 2 is not, the result is **failed replication** and is
written up as such — the sixth in this project, and the fourth to die at exactly this step.

---

## 9. What each outcome means

- **Both stages DISCRIMINATES.** Intervention fingerprints separate architectures that differ in
  procedure, even though nearest-reference labelling of those same fingerprints does not. E3's
  null becomes a result about the *label*, exactly as `PREREGISTRATION_E3B.md` foresaw, and
  Proposition 5 is confirmed as a statement about reference sets rather than about the
  instrument. The method has a demonstrated positive capability with a stated scope.
- **Stage 1 DISCRIMINATES, driven only by `halt`.** Reported as a fact about deployment horizon,
  not about learned procedure. Weak.
- **NULL at either stage.** The fingerprint does not separate these networks from each other
  either, and the neural side is finished on strictly stronger grounds than E3 left it — because
  the vacuity objection no longer applies to the question that was asked.
- **INDETERMINATE.** Reported as INDETERMINATE. One pass. No follow-up statistic on this data.

---

## 10. Commands

```powershell
python run_g.py --stage 1              # existing checkpoints, CPU, minutes
python run_g.py --stage 1 --smoke      # plumbing only
python run_g.py --stage 2              # trains 12 fresh models, needs CUDA
```
