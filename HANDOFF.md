# HANDOFF

Cold-start context for resuming this project in a fresh session. Read this first.

**Last updated:** 2026-08-20 — symbolic work complete and positive. **The neural side is
finished twice over and now explained twice.** E3 tested the method on its designed target with
both gates passing and returned a null, E3b failed to replicate the one lead, and the
decomposition shows why — the fingerprint measures neural models along axes orthogonal to the
ones carrying algorithmic identity (r = −0.233), with two of seven predicates pinned constant.
**Phase F then closed the obvious repair**: a parametric family replacing classification with
estimation contains all five reference algorithms *bit-identically* and the trained networks at
chi2 ~10^4 — so the networks are not interior points of the natural continuous family either.
**E5 then closed the last standing objection to that whole chain**: sensitivity normalisation
(the unimplemented claim in `_delete_nontree_edge`'s docstring) does not change the answer — 7
of 9 checkpoints remain vacuous under a measure built to discount baseline jumpiness, so the
vacuity gap is not a normalisation artefact. **E6 attempted to close the Dijkstra/Prim
boundary and its manipulation check failed informatively**: raising an edge weight is
monotonically invisible to relaxation, symmetric across both algorithms, so the boundary still
rests on a single intervention — now a documented limitation rather than open work. **Next
action: write up.** `THEORY.md` holds the definitional layer. Two concrete repairs are
documented under "If someone does resume the neural side" — both candidates, neither claimed,
and one of them is corrected by Phase F. `FINDINGS_RESIDUAL.md` carries an exploratory answer to
"what are the networks doing instead" with its own frozen confirmatory design; it is explicitly
not a result. No further pre-registered items remain queued; what's left is the write-up.

Environment: Windows, PowerShell. Python 3.14, numpy 2.4, PyTorch 2.11+cu128 with CUDA
available, torch_geometric 2.8. JAX and DeepMind CLRS are **not** installed and are not
needed — the neural work is PyTorch-native by choice (CLRS is JAX/haiku and painful on
native Windows). Run scripts from the repo root.

---

## Bottom line

**Established (symbolic, strong).** Algorithm identity is recoverable from causal response
to interventions, and is only well-defined *relative to an intervention class and an input
distribution*. Five algorithms are fully separated by a minimal set of two interventions on
graphs of depth >= 5; below depth ~4 no set in the suite separates all pairs. Truncated
Bellman-Ford is separable from real Bellman-Ford on shallow graphs where the two return
**identical answers on 11/12 instances**.

**Refuted (neural).** The headline claim the project was built on — *fingerprint similarity
predicts OOD generalization better than in-distribution validation accuracy* — is **not
supported**. Models differing 5-8x in OOD error are fingerprint-identical in-distribution.
Four additional in-distribution diagnostics also failed to beat validation accuracy.

**Explained (decomposition, 2026-08-11).** The neural null has a specific cause, and it is
not resolution, seeds or architecture. The fingerprint's discriminative axes and its
neural-disagreement axes are **orthogonal — r = −0.233**. The four interventions carrying
the most symbolic identity are the four where neural models disagree least; the two where
neural models are most distinctive (`inject_unreached_node` 0.944 vs 0.095 symbolic,
`corrupt_node_down` 0.648 vs 0.197) barely separate symbolic algorithms at all. Two of
seven predicates, `latency` and `poked_settled`, disagree **100%** of the time — they ask
exact-fixed-point questions of a system with no exact fixed point, so they are pinned
constants consuming ~29% of every neural fingerprint. See `FINDINGS_DECOMPOSITION.md`.
A predicted deletion/monotonicity offset was **refuted**: deletions are 30.4% of disagreeing
slots against 33.3% of the suite, and removing them makes every distance worse.

**Refuted on its own target (E3).** Three architectures *built* to execute differently —
plain relaxation, Dijkstra-like per-node commitment, hard truncation at k=8 — produce
**17/17 models labelled `bellman_ford`**, primary Fisher p = 1.000, with the mechanism
check passing (commitment spread over 6-9 rounds; truncation firing with 5-6 rounds of
response room). This was the fair test the method never had, and it failed it.

**The structural finding, and the most important result on the neural side.** Every neural
model sits **0.37-0.40 from its nearest symbolic reference, while Bellman-Ford sits 0.306
from Dijkstra** and the reference-to-reference median is 0.303. The models are further from
every reference than the references are from each other. Nearest-reference classification
has therefore been reporting the least-bad of five poor matches — in B0, B1, E2 *and* E3.
`identify.py`'s own docstring names this case: a model roughly equidistant from everything
is an unidentified algorithm, not a Bellman-Ford. **These MPNNs execute something outside
the reference set**, which is a real result and more interesting than the label was.

**Blocked (E1).** The discrimination test could not be run by supervision at all: a
min-aggregating MPNN cannot execute Dijkstra accurately enough to fingerprint (needs a
global argmin, has no global readout). E3 routed around it architecturally.

**Root cause of the negative.** The networks here differ in *update-rule generalization*,
not in *procedure*. They run the same control flow with a rule that degrades off
distribution. Intervention fingerprints probe execution structure, so they are blind to it.
The method should be expected to detect genuinely different procedures — early halting,
premature commitment, depth-limited propagation — and nothing else.

**Tried and closed (Phase F, 2026-08-13).** The obvious repair to the structural finding is
that reading a *discrete label* off a continuous system is the wrong question: replace
classification with **parameter estimation** over a family `(α, τ, γ, c, k, selection)`
containing the discrete algorithms as limit points and the networks as interior points. Half of
that works and is exact. The family contains all five references **bit-identically**, damping
and path discount have exact closed-form estimators, and the whole thing **replicated** on a
fresh graph stream — the first result here to do so. But **the networks are not interior points
of it either**: six of nine E3 checkpoints fit at chi-squared of order 10^4, against an
in-family holdout at 0.0000 and a deliberately out-of-family control at 15.0. They are further
from this family than the negative control is. So a gradient-trained MPNN is not a damped, soft,
discounted relaxation with a fitted temperature; it is something this family does not reach, and
the family's own residual says so at four orders of magnitude. Verdict DEAD on the frozen
criteria. `FINDINGS_FAMILY.md`.

**And an exploratory answer to "so what ARE they doing" (`FINDINGS_RESIDUAL.md`, not a result).**
Pointing the Phase F residual at the nine checkpoints: the per-node path discount `γ_v` has an
IQR of 0.24–0.84 where every family member gives exactly 0.000, it **exceeds 1** in 2–47% of
rows, and it rises through the run (ρ = +0.47 on the division-free numerator, 9/9 sign
agreement) — a *state-dependent, sometimes amplifying* path gain rather than a scalar discount.
Three of nine models also show 7–19% of responses with the **wrong sign** (anti-relaxation),
which no family member can produce and which is the same phenomenon as B2's surviving `nonmono`
lead measured per-edge. Two hypotheses were refuted and both matter: no self-amplification on
the diagonal, and `width > deg(v)` never occurs — the network is genuinely one-hop local in
`d`-space, so **the observable is sound and it was the hypothesis space that was wrong**.
Confirming any of it needs freshly trained seeds; the frozen test is written down.

Phase F also **corrected two wrong claims in `fpid/response.py`** that had been quoted as
motivation for exactly this reframe — see the warning box under "If someone does resume the
neural side". The "networks depend on 2-3 neighbours where every hard-`min` algorithm depends
on exactly 1" separation is largely an estimator artefact. The orthogonality and vacuity results
are independent and stand.

---

## What the project set out to claim

Identify *which algorithm* a neural reasoner learned purely from its causal response to
interventions — no weight inspection, no assumed neuron-to-variable mapping. Poke the model
(delete an edge, corrupt a node's state, inject an estimate), watch the response propagate,
match the signature against reference algorithms.

The high-stakes second claim was that fingerprint similarity predicts OOD generalization
better than validation accuracy, attacking a real open problem in neural algorithmic
reasoning: models all hit near-perfect validation accuracy and behave completely differently
OOD, so validation accuracy is useless for model selection. **That second claim did not
survive** — see B1/B2 below. The problem it targets is real and reproducible; this method
does not solve it.

Original plan and novelty positioning:
`C:\Users\advay\.claude\plans\5-algorithm-identification-tender-flask.md` (now partly
superseded — treat as history).

---

## Status

| Phase | What | State |
|---|---|---|
| A | Symbolic separation gate | **Done, PASS** |
| A2 | Noise calibration — predicates under continuous, noisy output | **Done, PASS** |
| A3 | Identifiability boundary vs graph depth | **Done, POSITIVE** |
| B0 | Neural harness + seed-control gate (PyTorch) | **Done, PASS (caveat)** |
| B1 | Does the fingerprint predict OOD? | **Done, NEGATIVE** |
| B2 | Do other in-distribution signals predict OOD? | **Done, NULL** |
| E1 | Dijkstra-trained discrimination | **Done, INCONCLUSIVE** — MPNN cannot execute Dijkstra |
| E2 | Hint-free regime | **Done, POSITIVE** (but see E3's structural finding) |
| E3 | Architecturally distinct procedures | **Done, NULL** — both gates passed |
| E3b | Held-out replication of E3's truncation lead | **Done, FAILED** — direction held, resolution did not |
| F | Parametric family spike — estimate parameters instead of classifying | **Done, DEAD** — family contains the algorithms exactly; the networks sit at chi2 ~10^4 |
| F+ | Exploratory: what the networks do instead (`run_residual_audit.py`) | **Done, EXPLORATORY** — state-dependent path gain; needs fresh seeds |
| E5 | Sensitivity normalisation vs Proposition 5's vacuity gap | **Done, UNCHANGED** — 7/9 checkpoints stay vacuous under a normalised measure |
| E6 | Direct Dijkstra/Prim key-dissociation probe | **Done, MANIPULATION FAILED** — weight-raise is monotonically invisible; boundary still rests on one intervention |
| — | Full model zoo (60-100 models) | Not started; **not justified** without a new idea |

**Phase F in one line** (`FINDINGS_FAMILY.md`, `PREREGISTRATION_FAMILY.md`,
`GATE1_DISQUALIFICATIONS.md`): replacing nearest-reference classification with parameter
estimation over a family `(alpha, tau, gamma, c, k, selection)` that contains all five
references **exactly** (bit-identical, Gate 0a) and has closed-form estimators for damping and
path discount accurate to 4 decimals — and which **does not contain the trained networks**. Six
of nine E3 checkpoints fit at chi2 of order 10^4 against an in-family holdout at 0.0000 and a
deliberately out-of-family control at 15.0, with non-zero row-sum deviation (a membership test
with no fitted parameters) on all nine. This is a second, independent corroboration of the
neural negative and a stronger one: the networks are not near the five discrete algorithms,
*and* they are not interior points of the natural continuous family containing them. Phase F
also replicated on a fresh graph stream and fresh design — the first result here to survive
that step.

---

## Quick start

```bash
python run_phase_a.py --graphs 25 --nodes 24
```
Symbolic separation gate. Seconds, numpy only.

```bash
python run_identifiability.py --graphs 12 --nodes 24
```
Which algorithm pairs separate at which graph depth. Minutes.

```bash
python run_family_spike.py --gate 0                    # the cheapest falsification, seconds
python run_family_spike.py --gate all                  # full Phase F ladder, ~10 min, CPU only
python run_family_spike.py --reproduce-response-table  # regenerates response.py's numbers
```
Phase F. `--gate 0` checks that the parameterised family reproduces all five reference
algorithms bit-identically and that every axis moves the observable; it is the fastest way to
tell whether anything in `fpid/family.py` has been broken. No training and no CUDA — the
trained checkpoints in `artifacts/*.pt` are loaded, never fitted.

```bash
python run_noise_robustness.py --graphs 10 --nodes 20
```
Predicate robustness to decoder noise. Minutes. See `--fixed-tol`, `--axis`.

```bash
python run_seed_control.py --seeds 6 --epochs 1500 --probe-graphs 10 --budget 24
```
Trains 6 MPNNs, fingerprints each. Needs CUDA. ~25 min.

```bash
python run_ood_prediction.py --seeds 5 --epochs 1500 --probe-graphs 8
```
The negative result. Trains 10 models, fingerprints on shallow *and* deep probes. ~50 min.

```bash
python run_shallow_diagnostics.py --seeds 5 --epochs 1500 --diag-graphs 8
python run_shallow_diagnostics.py --seeds 6 --seed-offset 100 --epochs 1500 --diag-graphs 8
```
Four in-distribution diagnostics vs OOD error, on disjoint seed sets. ~25 min each.

```bash
python run_e1_e2.py
```
E1 + E2. Trains 18 models (BF-hint / Dijkstra-hint / hint-free), fingerprints each, and
evaluates the pre-registered gates in order. ~100 min. `--smoke` runs the plumbing in ~2 min.

---

## Repo map

| Path | Contents |
|---|---|
| `fpid/graphs.py` | Mutable graph container, generators, ground-truth SSSP |
| `fpid/algorithms.py` | Steppable BF, truncated-BF, SPFA, Dijkstra, Prim over shared `(d, pi)`; `UNREACHED`, `d_equal` |
| `fpid/interventions.py` | The 9 probes, target selection, firing-time rule, `settle_round` |
| `fpid/fingerprint.py` | Predicate extraction + aggregation across graph instances |
| `fpid/separation.py` | Pairwise separation, minimal separating set |
| `fpid/identify.py` | Nearest-reference classifier, pairwise-stable distance, margin |
| `fpid/noisy.py` | Four noise axes for testing predicate robustness |
| `fpid/neural.py` | MPNN, `GatedMPNN` (Dijkstra-like commitment), `NeuralExecutor` |
| `fpid/train.py` | Batched training, hint / no-hint regimes, dataset generation |
| `fpid/response.py` | Jacobian of one round; `summarise_masked` / `jacobian_pure` are the Phase F additions. **Its old `width` and `row_sum` claims were wrong — the docstring now records both** |
| `fpid/family.py` | The parameterised relaxation family. All five references are exact points in it (`REFERENCE_LIMITS`); vectorised `family_step`, `softmin`, sender-set selection |
| `fpid/estimate.py` | Parameter estimation with a residual that has a scale: closed forms for α and γ, temperature calibration, `FitReport.withheld` for refusals |

`THEORY.md` holds the definitional layer: `I,D`-equivalence, the minimal separating set,
and five propositions with witnesses — including Proposition 5, why a nearest-reference
label is vacuous once `d_near` approaches the reference-to-reference scale.

Findings, in order written: `FINDINGS_PHASE_A.md`, `FINDINGS_NOISE.md`,
`FINDINGS_IDENTIFIABILITY.md`, `FINDINGS_SEED_CONTROL.md`, `FINDINGS_OOD_NEGATIVE.md`,
`FINDINGS_SHALLOW_SIGNALS.md`, `FINDINGS_E1_E2.md`, `FINDINGS_E3.md` (covers E3 and E3b),
`FINDINGS_DECOMPOSITION.md`, `FINDINGS_FAMILY.md` (Phase F), `FINDINGS_RESIDUAL.md` (exploratory follow-up to Phase F -- carries its own frozen confirmatory design, and is
labelled throughout as not-yet-a-result).

Pre-registrations: `PREREGISTRATION_E1E2.md`, `PREREGISTRATION_E3.md`,
`PREREGISTRATION_E3B.md`, `PREREGISTRATION_FAMILY.md`. Write one for every experiment from
here on. `GATE1_DISQUALIFICATIONS.md` is the companion Phase F turned out to need: the
pre-registration said a confound found at Gate 1 disqualifies that parameter from Gate 2's
criteria, and that file is the disqualification — written between the two gates, before
Gate 2 ran, and recording a k-grid error in the pre-registration itself.

Experiment runners: `run_e1_e2.py` (E1+E2), `run_e3.py` (E3), `run_e3b.py` (E3b),
`run_decompose.py` (orthogonality decomposition), `run_family_spike.py` (Phase F),
`run_residual_audit.py` (exploratory structural audit of the Phase F residual). The
shared gate machinery — admission, Fisher exact, family collapse, probe-graph depth
filtering, predicate ablation — lives in `run_e1_e2.py` and is imported by the others.

---

## Results by phase

**A — symbolic separation.** Two of nine interventions (`insert_improving_edge`,
`inject_unreached_node`) separate all ten algorithm pairs; the other seven are redundant.
`corrupt_node_up` separates BF (`recovery=exact, latency=0`) from Dijkstra (`none, never`)
via the settled set, as predicted. Stable across seeds.

**A2 — noise.** Identification survives input-dependent model error to sigma = 2 against an
edge-weight scale of 1-10 (100% accuracy; margin falls 0.146 -> 0.077). The fingerprint is a
*differential* measurement, so it cancels any error common-mode across control and
intervened rollouts. That is why it is near-orthogonal to accuracy by construction — and why
it must never be reported as a quality metric. Report the **margin**, not just the label.

**A3 — identifiability boundary.** Minimal separating set: 2 interventions at depth 7+, 3 at
depth 5, **impossible below depth ~4**. On shallow graphs truncated-BF returns output
identical to BF on 11/12 instances and is *still* separable; on deeper graphs that same
algorithm is correct on 0/12. A predicted threshold at k=3 was **falsified** — truncation is
detectable at every depth, because limited *runtime* is observable even when limited
*propagation depth* is not.

**B0 — seed control.** Six hint-trained MPNNs differing only by seed all fingerprint as
`bellman_ford`; cross-seed spread ~1/5 of the between-algorithm scale. Seed 2 is the worst
model (MAE 1.59, ~5x the best) and identifies with the **largest** margin — direct evidence
on real models that fingerprint identity is decoupled from accuracy. Caveat: worst
cross-seed noise (0.094) exceeds the thinnest margin (0.061).

**B1 — OOD prediction: NEGATIVE.** 12 models, matched shallow validation MAE (0.815 vs
0.778), **8.6x apart on deep OOD MAE** (6.178 vs 0.719). All 12 label `bellman_ford`; the
signed predictor scores -0.049. On *deep* probes an asymmetry does appear (3/5 shallow-
trained labelled truncated, 0/5 deep-trained, Fisher p ~ 0.083) — but deep probes are OOD
data, and anyone holding those can just evaluate accuracy directly. The selling point was
prediction *without* OOD data; that is the part that failed.

**B2 — other in-distribution signals: NULL.** Four candidates vs validation accuracy on two
disjoint seed sets. None reliably wins. `overshoot` looked excellent on the first set
(+0.758 vs +0.261 baseline, perfectly monotone within a group) and collapsed on held-out
seeds (+0.371, below baseline, effect size sign-flipped). Only surviving lead: `nonmono`
(rounds in which the model *raises* an estimate, which no correct relaxation does) separates
the two training groups in the same direction on both runs with growing effect size (Cohen's
d +0.69 then +1.63) — it classifies *provenance* better than it predicts OOD *magnitude*.

**F — parametric family: DEAD, and informatively so.** Full account in
`FINDINGS_FAMILY.md`; the five things worth carrying forward:

1. **The family contains the algorithms exactly.** All five references reproduced at
   `max |d_family − d_ref| = 0.0` — bit-identical — over 20 graphs × 24 rounds. Prim is the
   γ=0 point (keying on `w` alone is a zero path discount), which is why a continuous discount
   beats a discrete `key ∈ {sum, weight}` axis.
2. **Damping and path discount are exact, closed-form observables.**
   `α = 1 − |J[v,v]|` and `γ = off_row/α` on rows that are reached *and still moving*, verified
   to 4 decimals. Where the estimator answers, α p90 error is 0.0039. And
   `row_sum = 1 − α(1−γ)` is a **membership test with no fitted parameters at all**.
3. **It replicated** — fresh graph stream, fresh design, nothing refitted, every headline
   number reproduced. The first result in this project to survive that step.
4. **Four confounds are structural, not resolution shortfalls.** α ↔ c (commitment drives
   `a_eff → 0`, which is what lowering α does), τ ↔ γ, τ ↔ selection (one sender leaves no
   distribution for a temperature to spread over), γ ↔ c — all at SNR 0.00. **More probe
   graphs cannot fix these**, because at the level of the response operator the moves are
   identical. Temperature is effectively not estimable at all (median relative error 1.000).
5. **The networks are not in the family.** Six of nine E3 checkpoints fit at chi2 of order
   10^4, against an in-family holdout at 0.0000 and a deliberately out-of-family control at
   15.0 — so *seven of the nine sit further from this family than the negative control does* —
   with
   non-zero row-sum deviation on all nine, and fitted parameters that move substantially when
   the measurement window changes from 8 rounds to 6 while family members recover identically.

So the reframe's premise fails at the last step. "A family containing both the algorithms and
the networks" was achieved only for the algorithms. This is a second, independent, and stronger
corroboration of the neural negative: the networks are not near the five discrete algorithms,
*and* they are not interior points of the natural continuous family containing them.

---

## Methodological rules, learned the hard way

Four results here evaporated on checking; three because a winner was selected and reported
before replication.

- **Pre-register.** Fix the candidate list and decision rule, generate both seed sets up
  front, select on the first and report on the second, in one pass.
- **A mechanism is not protection.** `overshoot` had a sound invariant argument behind it
  (relaxation approaches from above and cannot undershoot) and still failed replication.
  E3b's truncation-distance statistic had one too, and perfectly disjoint ranges, and also
  failed. Direction replicating while resolution does not is the *characteristic* signature
  of a dead lead here — it is what both of them looked like. Leave no intermediate rung in
  the outcome ladder for that state to land on.
- **Verify the manipulation before reading the measurement.** E1 would have been written up
  as a null about fingerprints; the cross-fit trajectory check showed the training
  difference had barely been installed, making it a null about MPNN capacity instead.
  Opposite next steps. Every experiment needs a gate that fails loudly when the independent
  variable never moved.
- **A nearest-reference label means nothing without the reference-to-reference scale.**
  Every neural "identifies as bellman_ford" in this project sits 0.30-0.40 from that
  reference, against a 0.25-0.30 median distance between two genuinely different
  algorithms. Quote `d_near`, the margin, and that scale together or do not quote the label.
- **Resolution floor.** When a margin or effect is comparable to seed/probe noise, treat the
  result as absent regardless of how sensible the pattern looks.
- **Flat-in-sigma degradation** means a categorical threshold is being ignored, not exceeded.
  Statistical problems scale with noise; categorical ones do not.

---

## Constraints that will silently invalidate work if forgotten

Each was discovered by a run that produced confident-looking garbage.

1. **Fingerprint on graphs of propagation depth >= 5.** Below depth ~4 no intervention set
   separates all pairs. CLRS's default distribution is dense and shallow — used unmodified
   it lands where the method provably cannot work and returns a null about nothing.
2. **Interventions fire relative to each algorithm's own progress**, never at an absolute
   round. Round 5 is five hops for BF and five settled nodes for Dijkstra; absolute rounds
   land after termination and measure nothing.
3. **Relaxation is monotone** — it only lowers `d` — so edge deletion produces zero response
   in any real relaxation algorithm. Useless for separating symbolic algorithms, but a model
   that responds to a deletion is not doing relaxation, whatever its accuracy says.
4. **"Settled" is defined observationally** (estimate stopped changing). BF has no settled
   set and a network has no inspectable internals, so any probe phrased in terms of an
   algorithm's own bookkeeping is ill-posed across the family.
5. **Response measures compare against the control run, not the reference run.** Comparing
   to the reference measures whether the poke moved the *right answer* — a fact about the
   graph, identical for every algorithm.
6. **Never tie a test's tolerance to the noise amplitude it probes.** `tol = k * sigma` makes
   any perturbation bounded by sigma invisible for every sigma, producing a flat 100% that
   reads as robustness and is a tautology.
7. **Predicate stability must be applied pairwise, not intersected globally.** A global
   intersection discards exactly the probe that separates a single pair.
8. **Check every constant's magnitude against the data it is compared with.** Three failures
   so far: `tol = 4 sigma`, a tolerance-blind `settle_round`, and `UNREACHED = 1e3` sitting
   25x above real distances so the loss optimised the sentinel (fixing it cut MAE 2-3x).
9. **Model admission is by MAE, not validation accuracy.** Tolerance must exceed the model's
   error floor *and* stay below the edge-weight scale (~5). At MAE ~5 no tolerance satisfies
   both and the model cannot be fingerprinted at all.
10. **`UNREACHED` is graph-family dependent.** 80 suits density-0.05 n=24 graphs (longest
   path ~39). Depth-17 path graphs reach ~126 and need it raised.
11. **Hint supervision on Dijkstra needs `rounds >= n`.** It settles one node per round, so
   a shorter budget trains on a half-finished run and its MAE is not comparable with a
   Bellman-Ford model's — the shared admission threshold then means two different things. At
   `rounds = n` both algorithms end on exactly the true distances, which is also what makes
   the two groups differ *only* in trajectory shape. Side effect worth knowing: the
   hint-free loss then cannot see the hint target at all, so hint-free is one condition
   rather than two, and a nominally-split hint-free group is a negative control.
12. **State interventions require state-encoded recurrence.** The model must carry `d`
   through the round loop and re-encode it. A latent-only recurrent model silently ignores
   every state poke and can receive graph edits only — which A3 shows is not enough.

---

## Next experiments — start here

**E1 and E2 have been run** (`run_e1_e2.py`, pre-registered in `PREREGISTRATION_E1E2.md`,
results in `FINDINGS_E1_E2.md`). E1 was blocked at admission: a min-aggregating MPNN cannot
execute Dijkstra accurately enough to be fingerprinted, so the discrimination claim is still
untested. **Start at E3**, which installs the procedural difference by construction and so
does not need the network to learn a global argmin.

### E1. DONE — INCONCLUSIVE (training). Kept for the reasoning; do not re-run as written.

0/6 Dijkstra-supervised models were admitted (MAE 2.78-3.90 against a 2.5 threshold; loss
plateaued at 1.64-2.00 across all six seeds versus 0.002-0.038 for Bellman-Ford). The
manipulation itself *did* take — trajectory-MAE deltas are +3.96 for Dijkstra-trained and
-10.92 for BF-trained, ranges disjoint — so the models learned part of Dijkstra's schedule
and cannot execute it. Tight loss spread across seeds says capacity, not optimisation:
Dijkstra needs a *global* argmin over unsettled nodes and the architecture has no global
readout.

Do not re-run by relaxing `--admit-mae`. At MAE 3.5 with `tol = 2.5` the error floor exceeds
the measurement tolerance and the predicates read model noise (constraint 9). Re-running
would require an architecture with a global readout, at which point E3 is the better
experiment.

<details>
<summary>Original E1 plan (superseded)</summary>

### E1. Can the fingerprint tell two differently-trained models apart at all? (~30 min)

**The biggest gap in the project.** Every neural model trained so far was supervised on
Bellman-Ford. "All six models identify as `bellman_ford`" (B0) was therefore never a
discrimination test — there was no alternative for them to be confused with, so the result
is consistent with a fingerprint that simply always says Bellman-Ford.

Train half the models on Bellman-Ford trajectories and half on **Dijkstra** trajectories,
then check whether the fingerprint recovers which. This is the core identification claim on
neural models, and it is untested.

Implementation:
- `fpid/train.py` has `bf_trajectory()`. Add `dijkstra_trajectory()` alongside it, using
  `Dijkstra` from `fpid.algorithms` with `d_init=UNREACHED` (or wrap with `finite_init`).
  Both algorithms expose the same `(d, pi)` state, so nothing else changes.
- Reuse the `run_seed_control.py` scaffolding — it already trains models, builds symbolic
  references with `finite_init`, and classifies.
- Probe graphs **must** be depth >= 5 (constraint 1). Density 0.05 at n=24 gives depth ~5.
- Keep `tol = 2.5` and admit only models with MAE well under 5 (constraint 9).

**Pre-register the decision rule before running:** BF-trained models should classify as
`bellman_ford` and Dijkstra-trained as `dijkstra`, at margins above the cross-seed noise
floor (~0.094 from B0). Report the confusion matrix and the margins, not just accuracy.

If this fails, the neural side of the project is done and B1's negative was never really
about OOD. If it succeeds, the identification claim holds on real networks, which stands on
its own regardless of the OOD result.

</details>

### E2. DONE — POSITIVE. The strongest neural result in the project.

Hint-free training works: 6/6 admitted at MAE 0.38-0.92, matching hint-supervised accuracy.
All six label `bellman_ford` at mean margin **0.106** — versus **0.028** for hint-supervised
models, every one of which is *below* the 0.094 noise floor. And hint-free models do not
reproduce Bellman-Ford's trajectory at all (trajectory MAE 7.9-10.7 versus 0.3-1.3 for
hint-trained). A model told only the answer, which picked its own procedure, still responds
to interventions like relaxation — the one fingerprint match here that is not close to
tautological.

Dropping `term_delta` flips 4/6 hint-supervised models to `bf_truncated_k3` but leaves all
six hint-free models on `bellman_ford` with *higher* margins. The hint-supervised label
leans on the termination predicate; the hint-free one does not.

Caveat that keeps this from being the discrimination result: six models all saying
`bellman_ford` is still consistent with a fingerprint that always says `bellman_ford`.

### E3. DONE — NULL, with both gates passed. The decisive neural result.

`run_e3.py`, pre-registered in `PREREGISTRATION_E3.md`, results in `FINDINGS_E3.md`.
Three arms, all hint-free on identical data so architecture is the only variable: `plain`
MPNN, `gated` (`GatedMPNN`, freezes a reached node once it stops moving — Dijkstra-like
commitment from a purely *local* rule, which is how it sidesteps E1's global-argmin block),
and `halt` (trained and deployed at k=8). Mechanism check passed: gated models commit
66-100% of nodes spread over 6-9 rounds starting at round 8-12; halt models fire at round
2-3 against k=8. **17/17 labelled `bellman_ford`. Primary p = 1.000.**

Two things to carry forward if anyone resumes this:

1. **Calibrate architectural constants at the real epoch count.** `GatedMPNN`'s `eps` at
   600 epochs looked ideal at 0.05 (committing at round 7, just past propagation depth); at
   1500 epochs the same value commits at round **2**, before information crosses the graph,
   freezing on garbage — a global halt wearing a gate's clothes, i.e. a duplicate of the
   `halt` arm. Calibrating on a short proxy run would have shipped a confounded design.
   Settled on `eps = 0.002` (MAE 0.708, inside plain's range, so the arms are
   accuracy-matched; commitment over rounds 14-19).
2. **`evaluate(model, ds, k)` compares against row `k`.** A model deployed at k < rounds
   must be graded against an admission set with the true answer written into row `k` —
   `true_distance_dataset(ds, at=k)`. Otherwise the halt arm is silently graded against
   Bellman-Ford's round-k state. They nearly agree, which is why it goes unnoticed.

<details>
<summary>Original E3 plan (superseded)</summary>

### E3. Models that genuinely differ in procedure (~1 hr)

The fair test the method never received. B1 failed because shallow-trained and deep-trained
networks differ in update-rule generalization, not procedure. Build architectures that
differ in *how they compute*:

- a gated model that freezes a node's estimate once it stops changing (Dijkstra-like
  commitment);
- a model that hard-halts after k rounds (`NeuralExecutor` already supports `max_rounds`);
- a plain MPNN (Bellman-Ford-like).

If fingerprints separate those three, the method works on its actual target and B1's
negative is specifically about the proxy chosen, not the method.

**E1 promoted this from "the fair test" to "the only remaining route."** Supervision cannot
install Dijkstra-like commitment in this architecture, but a gating rule can, without any
global argmin: freeze a node's estimate once it stops changing. Reuse `run_e1_e2.py`
wholesale — admission, the manipulation check, the family-level Fisher test, the margin
floor and the `term_delta` ablation all transfer unchanged; only the model constructors
differ. Pre-register the same ladder before running.

</details>

### E3b. Held-out replication of E3's one post-hoc lead

`run_e3b.py`, pre-registered in `PREREGISTRATION_E3B.md`. E3's arms were perfectly
separated by `s = d(bellman_ford) - d(bf_truncated_k3)` (plain [-0.122, -0.095], halt
[-0.073, -0.013]) — the fingerprint moves truncated models toward the truncation reference
without ever changing the label. **This is exactly what `overshoot` looked like before it
died**, so it is tested, not reported: threshold frozen at tau = -0.084, fresh seeds
(500-505, 600-605) and a fresh probe stream (`default_rng(7)`, since the statistic is a
distance measured *on* the probe graphs and E3's set is part of what tau was fitted to).
Two outcomes only, REPLICATED or FAILED — a half-replication is the state `overshoot`
reached before retraction.

**DONE — FAILED.** 9/11 correct under the frozen threshold, Fisher p = 0.0801, and the arm
ranges now overlap by 0.067 where they had been disjoint. Both gates passed, so the failure
belongs to the statistic. Direction survived, resolution did not — `overshoot`'s exact
half-alive state, which is why the pre-registration left no rung for it to land on. **That
is the fifth result in this project to die on replication, and the second with a sound
mechanistic argument behind it.** Incidental: `halt` seed 601 produced the first
non-`bellman_ford` neural label in the project (`bf_truncated_k3`) at margin 0.019, a fifth
of the noise floor — noise, not a discovery.

### E4. `nonmono` as a provenance classifier (~1 hr)

The one surviving lead from B2. It counts rounds in which the model *raises* an estimate,
which no correct relaxation does. Same direction on both seed sets with growing effect size
(Cohen's d +0.69 then +1.63).

Test it as a **classifier of training provenance**, which is what the data supports — not as
a regressor on OOD error, which it is not. Pre-register: fix the decision rule, generate both
seed sets up front, select on one and report on the other in a single pass.

### E5. Sensitivity normalisation (~20 min)

`delete_nontree_edge` exists as the null control but responses are never divided by it, so a
globally jumpy model looks "different" for free. Cheap, and may sharpen every earlier result.

### E6. Close the Dijkstra/Prim boundary (~20 min)

It rests on a single intervention. Add a probe keyed directly on `d[u]+w` versus `w`, which
is the actual difference between the two algorithms.

---

## Alternative to more experiments

**This is now the recommended path.** E3 was the method's designed target and it failed
with both gates passing; the distance diagnostic says the failure is structural, not
marginal; and E3b buried the one lead E3 produced. There is no remaining neural experiment
with a live hypothesis behind it.

**Write up as-is.** The symbolic contribution stands alone: intervention-relative algorithm
identity, the minimal separating set, depth-dependent identifiability, and truncated-BF
separable from BF on instances where the two return identical answers — plus a
well-characterised negative on neural transfer, which is genuinely useful to the field.

**Not recommended:** training the full 60-100 model zoo. Every neural result so far says it
would produce a larger null.

**Report the distance profile, never the bare label.** The single most reusable lesson: a
nearest-reference label is meaningless unless the nearest distance is small *relative to
the reference-to-reference scale*. Here it never was — 0.37-0.40 against a 0.303 median.
Any future write-up should quote `d_near`, the margin, and that scale together.

---

## If someone does resume the neural side

The decomposition leaves two concrete repairs. Both are candidates, neither is claimed, and
both need a pre-registered exploratory/confirmatory split on disjoint seeds before belief.

1. **`poked_settled` and `latency` are broken for continuous-output models**, not
   uninformative about them. They ask exact-fixed-point questions ("had this node already
   finished?", "how many rounds until the run matches the reference exactly?") of a system
   that never reaches an exact fixed point, so both are pinned constants. Reformulate on a
   sustained-tolerance basis and ~29% of the fingerprint comes back.
2. **Measure where the networks actually vary.** `fpid/response.py` estimates the round
   Jacobian `J[v,u] = -d(d_v(t+1))/d(d_u(t))` by finite differences, identically for
   symbolic and neural executors.

   > **CORRECTED BY PHASE F — read this before using the statistics below.** This section
   > previously recommended **median nonzeros per active row** ("every hard-`min` reference is
   > exactly 1.00 at every support threshold; networks sit at 2-3"). That is **wrong**.
   > Measured by `run_family_spike.py --reproduce-response-table`, Bellman-Ford sits at
   > **2.00 from round 3**, not 1.00. The one-sided difference double-counts at a hard-min
   > tie: at a converged node `d_v == d_u + w`, so lowering *either* argument moves the min by
   > the full `eps` and both `J[v,v]` and `J[v,u]` read 1. Bellman-Ford enters that state
   > around round 3 on depth-5 instances; the selection-based algorithms do not, because only
   > one node is active per round. The "2-3 neighbours versus exactly 1" separation is
   > therefore substantially an estimator artefact and must not be cited. The vacuity result
   > it was usually cited alongside (0.37-0.40 against a 0.303 reference-to-reference median)
   > is independent and stands.
   >
   > `row_sum` was also mislabelled as the damping coefficient. It is `1 - alpha(1-gamma)`,
   > i.e. **identically 1 whenever gamma = 1, whatever alpha is**, because softmin weights are
   > a partition of unity. Use `summarise_masked` instead of `summarise` and read:
   > `alpha = 1 - |J[v,v]|` and `gamma = (off-diagonal row sum) / alpha`, both exact on rows
   > that are reached *and still moving*; and `row_sum = 1 - alpha(1-gamma)` as a
   > **parameter-free family-membership test**.

   - **diagonal fraction** `||diag J|| / ||J||` — a continuous, purely *observational*
     measure of commitment, which constraint 4 says cannot be read off internals. BF falls
     0.90 -> 0.50 and stays; truncated-BF tracks it then jumps to exactly 1.00 at round `k`;
     Dijkstra/SPFA/Prim sit at 0.88-1.00. **This one reproduces** and is the surviving
     recommendation.
   - Phase F also establishes what this observable *cannot* do, for reasons no amount of extra
     data fixes: damping and commitment are the same move on a one-round Jacobian
     (`alpha <-> c` at SNR 0.00), aggregation temperature is invisible unless it is comparable
     to the best-minus-second-best message gap (which damping inflates from 3.9 at alpha=1 to
     29.5 at alpha=0.5), and a rule damped below ~alpha=0.2 moves too little per round to pass
     the secant threshold at all. Separating commitment from damping needs a *sustained*
     measurement — the same fix repair (1) above proposes for `poked_settled`.

   Two predictions here were falsified and the reasons are documented in the module: Prim's
   Jacobian does **not** vanish (its value update is diagonal, but `d` is also its selection
   key, so a state perturbation can reroute control flow — for selection-based algorithms
   the value *is* the control signal), and those flips are rare rather than pervasive
   (0/24 for Dijkstra, 2/24 for Prim), contaminating means as a heavy tail. Use medians.

---

## Still open (unaffected by the negative results)

- **Formalize behavioural equivalence under an intervention class**, and the minimal
  separating set. Phase A gives the empirical version; the definition is unwritten and is the
  strongest theoretical contribution available.
- **Equivalence is distribution-relative as well as intervention-relative** (A3). Stronger
  than the original plan claimed; deserves stating as a result, not a caveat.
- **Dijkstra vs Prim rests on a single intervention, and stays that way.** E6
  (`FINDINGS_E6.md`) attempted a direct probe (`dissociate_keys`: raise `w(u,v)` while
  lowering `d[u]` by the same amount, so `d[u]+w` is exactly invariant and `w` alone has
  moved) and its mechanism check **failed**: raising an edge weight is monotonically
  invisible to both algorithms (constraint 3, the same reason the deletion probes are
  near-silent), so the only live component of the probe was a `d[u]` corruption
  affecting both algorithms symmetrically. Registering the two new interventions in the
  suite does not change the minimal separating set or add a second Dijkstra/Prim
  separator — confirmed by re-running Phase A. A future attempt needs a construction
  that dissociates the keys without relying on a weight raise.
- ~~Sensitivity normalisation is unimplemented.~~ **Closed, E5 (`FINDINGS_E5.md`).**
  Implemented as a pairwise mask on `delete_nontree_edge`'s response
  (`fpid.identify.informative`, opt-in `normalise=` on `distance`/`classify`/the
  separation functions). Negative control passed; primary was **UNCHANGED** — 7 of 9 E3
  checkpoints stay vacuous under the normalised measure. Proposition 5's vacuity gap is
  not a sensitivity artefact.
