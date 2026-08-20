# Routes to a positive result

Written 2026-08-20, after reading the full corpus. **Nothing here is a result.** It is a ranked
menu of every route I can see to a positive finding, with the mechanism, the cost, and a frozen
decision rule where one is possible.

Read `## The rule` at the bottom first if you intend to run any of the zero-compute items. Five
results in this project died from computing a new statistic on data that had already been looked
at, and roughly half the ideas below are exactly that shape.

---

## 0. What "positive" can mean here

Three different things, and conflating them is how this ends badly.

1. **Reframing** — the project already contains positive results that are currently written as
   the leftovers of a failed neural claim. Zero compute, zero risk, and it is the single largest
   gain available. §1.
2. **Repair** — the instrument is *known* broken for continuous-output models in two specific,
   documented ways that were diagnosed and never fixed. A repair that flips a null is a positive
   about the method, not a fishing expedition. §2–§3.
3. **New evidence** — new observables, new hypothesis spaces, new models, new domains. §4–§8.

The honest position: the *headline OOD claim* (B1) is dead and should stay dead. Everything else
is more open than `HANDOFF.md` currently reads.

---

## 1. Positives already in hand, currently mis-framed (zero compute)

The write-up as planned leads with a negative. It does not have to.

**1.1 — The framework is the contribution, and it is positive.** `≡_{I,D}`, the minimal
separating set, Propositions 1–4 with constructive witnesses, and the depth threshold are all
*established*. Lead with "algorithm identity is well-posed only relative to an intervention class
and an input distribution", prove it, and the neural section becomes a corollary — the reference
family is incomplete — rather than the headline failure.

**1.2 — Proposition 5 is a positive methodological tool, not an obituary.** "A nearest-reference
label is vacuous unless `d_near ≪ median_{A≠B} δ(A,B)`" is a *portable audit test*. Anyone
classifying a learned model against a reference set — probing, circuit-matching, behavioural
cloning comparisons — can run it, and this project has the worked example where it fires. Name
it, give the recipe, show it firing on 17/17 models that a naive pipeline would have labelled
`bellman_ford` with confidence. That is a contribution with teeth.

**1.3 — Phase F's family is a positive result standing alone.** A six-parameter continuous family
containing Bellman-Ford, truncated-BF, SPFA, Dijkstra **and Prim** *bit-identically*
(`max |d_family − d_ref| = 0.0`), with closed-form estimators for two parameters accurate to four
decimals, a parameter-free membership test (`row_sum = 1 − α(1−γ)`), and a held-out replication.
That is "a continuous relaxation of the classical shortest-path algorithms with identifiable
parameters" — publishable without mentioning a neural network. The neural rejection then reads as
the family's *falsification power*, which is a feature of a good model, not a failure.

**1.4 — The residual audit's refutations are positive structural claims.** `width > deg(v)` never
occurs across nine checkpoints: **a trained MPNN is genuinely one-hop local in `d`-space**,
established causally with no weight inspection. Same for no self-amplification (`|J[v,v]| ≤ 0.832`
everywhere). Those are affirmative measurements about what learned reasoners do.

**1.5 — The pre-registration discipline is a contribution.** Five retracted results with the
retractions recorded, plus `GATE1_DISQUALIFICATIONS.md` written *between* two gates, is a
worked example of confirmatory practice in an ML subfield that has almost none.

---

## 2. The instrument is broken in a known way — fix it (cheap, mechanism-backed)

These are the highest-value non-reframing items because they are *repairs to a diagnosed defect*,
not new hypotheses.

**2.1 — Reformulate `poked_settled` and `latency` on a sustained-tolerance basis. ★ TOP TIER**

`FINDINGS_DECOMPOSITION.md` §3: these two predicates disagree with Bellman-Ford **100.0%** of the
time for every neural model, and all four slots where every network disagrees with every reference
are `poked_settled`. That is ~29% of every neural fingerprint contributing **nothing but a constant
offset** — and a constant offset is precisely what manufactures a vacuity gap.

Both predicates ask exact-fixed-point questions of a system with no exact fixed point. Fix:

```
settled(v,t)  := |d_v(t') − d_v(t)| < tol  for all t' ∈ [t, t+W]     # sustained, not exact
latency       := first t after which the run stays within tol of the reference
                 for a sustained window W, bucketed {0, 1–2, 3–5, >5, never}
```

**Why this is not what E5 already tested.** E5's mask drops a slot when *both* fingerprints match
their own null-control response. A pinned constant is pinned under the null control too, so the
network side is uninformative there — but the *reference* side is informative, and the pairwise
rule therefore **keeps** the slot. E5 structurally could not remove this offset. This is the
orthogonal repair, and it is the one `HANDOFF.md` names as repair (1) and never ran.

Frozen rule, reusing E5's two-rung ladder: recompute `d_near` and the reference-to-reference
median on the nine checkpoints under repaired predicates. **CLOSED** if `d_near <` median for
≥ 7/9; **UNCHANGED** if ≥ 7/9 the other way; 4–6 is INDETERMINATE. Negative control first: the
symbolic minimal separating set must be unchanged.

Cost: ~1 hour of implementation, then minutes on CPU with existing checkpoints.

**2.2 — The orthogonality result is measured against the wrong symbolic quantity. ★ TOP TIER**

I checked `run_decompose.py:157–176`. The "symbolic pairs" column is the **mean over all ten
reference pairs of (disagreeing slots / total slots)** for that intervention — an average
disagreement *rate*. That is not separating power, and here the two demonstrably diverge:

| intervention | symbolic column | actual role |
|---|---|---|
| `inject_unreached_node` | **0.095** (2nd lowest) | in the size-2 minimal separating set; the **unique** separator of dijkstra/spfa |
| `insert_improving_edge` | 0.458 (highest) | in the minimal separating set; the **unique** separator of dijkstra/prim |

An intervention can separate every pair while disagreeing on few slots, and can disagree on many
slots while separating nothing new. So `r = −0.233` — the number that "completely accounts for"
the neural null — is a correlation against a proxy that ranks an *indispensable* probe second-last.

Recompute the correlation with a separation-weighted symbolic score: per intervention, the number
of reference pairs for which it is a separator, or its marginal contribution to the minimal set.
**If `r` moves toward zero or flips sign, the "orthogonal by construction, no redesign can help"
verdict is not supported**, and probe redesign (§4) becomes live again.

Cost: an afternoon, numpy only, no models. This is the cheapest idea here with the largest
potential to reopen a closed door.

**2.3 — Drop reference-unanimous slots from the distance.** A slot where all five references agree
contributes 0 to every reference-to-reference distance but 1 to a neural distance whenever the
network differs. The numerator of Proposition 5's comparison is therefore effectively computed
over a different slot set on each side.

Honest caveat, stated now so it cannot be spun later: dropping slots shrinks the denominator on
*both* sides, and E5 showed that arithmetic can raise the reference median faster than it lowers
`d_near` (0.303 → 0.489). This may well come out UNCHANGED for the same reason. Run it, but
expect that, and do not run a third variant afterwards hunting for the one that closes.

**2.4 — Weight the Hamming distance by per-slot discriminative entropy.** The general form of 2.3:
weight each slot by its entropy across the reference family, so zero-information slots cannot
contribute. Validate on the symbolic side (separation must be unchanged) exactly as E5 did.

---

## 3. Change the question: discriminate networks from *each other* ★★ HIGHEST EV

This is the best idea in this document, and the reason is that the project has been asking a
question Proposition 5 says is unanswerable, while never asking the neighbouring one that is.

Every neural experiment measured **network → symbolic reference**. Proposition 5 says that
comparison is vacuous because the networks are not in the reference set. **Nothing about that
argument prevents networks from being separable from each other.** And the discrimination claim —
"does the fingerprint tell two differently-built models apart?" — never needed the symbolic
references at all.

I checked: `fpid.identify.distance(a, b)` is fingerprint-to-fingerprint and symmetric, so this
needs **no new code**. B0 measured within-group cross-seed distance for six identically-trained
models (mean **0.042**, worst **0.094**). **No findings document reports a single distance between
two networks from different arms.** The cell is empty.

**3.1 — Fingerprint-space arm clustering.** Compute the 9×9 distance matrix over
`artifacts/{plain_700-702, gated_800-802, halt_900-902}.pt`. Statistic: mean within-arm δ versus
mean between-arm δ, plus 1-nearest-neighbour arm purity. Null: permute arm labels, 20 000
permutations. E3's arms are **accuracy-matched** (mean MAE 0.67 / 0.75 / 0.73), so a gap cannot
be attributed to one arm simply being worse — that confound was designed out in advance.

If within < between at p < 0.05: **intervention fingerprints discriminate architectures that
differ in procedure, even though nearest-reference labelling does not.** That converts E3's null
into a positive on the same data, and `PREREGISTRATION_E3B.md` already anticipated this exact
resolution — "that would make the label, not the fingerprint, the broken component."

**3.2 — Phase F's fitted parameters as a discriminative embedding. ★ the lead is already visible**

`FINDINGS_FAMILY.md`'s Gate-3 table, never tested as a classifier:

| arm | (α̂, γ̂) | chi² |
|---|---|---|
| `gated` | (0.441, 0.149), (0.427, 0.123), (0.348, 0.171) | 6–35 |
| `halt` | (0.761, 0.807), (0.831, 0.956), (0.796, 0.785) | 1.5–4.6 × 10⁴ |
| `plain` | (0.316, 0.864), (0.516, 0.132), (0.332, 0.592) | 1.4–4.4 × 10⁴ |

**`gated` and `halt` are tightly clustered and completely disjoint in both coordinates.** Three
seeds each, no overlap, and the separation is far cleaner than E3b's lead ever was.

The defensible framing, which must be stated up front: **a rejected generative model can still be
a good coordinate system.** chi² ~10⁴ says the family does not *contain* these objects; it does
not say the projection onto (α̂, γ̂) is uninformative. Compare: PCA is not a generative model of
anything either.

The one measured threat, and it is serious: `plain_700` reads α̂ = 0.670, γ̂ = 0.283 at 6 rounds
against α̂ = 0.316, γ̂ = 0.864 at 8 rounds, while family members recover identically under the same
change. **Parameters fitted outside the family are unstable under the measurement window.** So the
pre-registration must freeze the window *and* report the embedding at two windows, with stability
as a gate rather than a footnote. If the arm clusters survive both windows, that is a real result;
if they move, the embedding is measuring the window.

**3.3 — Residual-audit features as a classifier.** `FINDINGS_RESIDUAL.md` Stage 1, read by arm
rather than by hypothesis:

| feature | plain | gated | halt |
|---|---|---|---|
| `γ_v` IQR | 0.759, 0.718, 0.341 | 0.812, 0.682, 0.836 | **0.306, 0.239, 0.321** |
| `γ_v > 1` | 26.9%, 2.1%, 2.9% | 21.8%, 9.5%, 15.2% | **46.6%**, 10.9%, 14.7% |
| negative entries | **18.8%**, 1.1%, 0% | 0%, 0%, 0.4% | **15.5%, 7.6%**, 0.3% |

`halt` has the three tightest `γ_v` IQRs of the nine. `gated` is the only arm with no model above
0.4% negative entries. Both are post-hoc reads of a table built for another purpose — but they are
*consistent with* 3.2's clusters, which is mild independent support rather than a second bite at
the same apple.

**Do all three as one pre-registered run.** They are three views of the same question, they share
a null, and running them separately is how a best-of-N becomes a "finding".

---

## 4. New observables that dodge Phase F's structural confounds

Phase F declared four confounds structural at SNR 0.00 — but every one of those statements is
scoped to **one-round Jacobians**. That scoping is an invitation.

**4.1 — Second-order / compositional response. ★ strong, genuinely new**

Fire two interventions and test whether the responses **compose additively**. A damped, soft,
linear-in-the-small update composes; a hard `min` is piecewise-linear and does *not* compose across
a kink. Non-additivity is therefore a direct signature of hard selection — and it is a *second
derivative*, which is exactly the information a one-round Jacobian throws away.

Phase F's own verdict says "no observable built from one-round Jacobians can separate α from c."
The Hessian is not one of those. This is the most promising route to separating commitment from
damping, and it plausibly recovers temperature too (§4.2).

**4.2 — Probe-amplitude sweep as a thermometer.** Hard `min` is piecewise linear: response is
constant in `eps` until a kink is crossed, then steps. A softmin's response varies **smoothly**
with `eps`, at a rate set by τ. So the response-vs-amplitude *curve shape* measures temperature
directly, where Phase F's `off_pr` statistic was flat at 1.000 and useless (median relative error
1.000 across the whole box).

Clean symbolic prediction to validate against: all five references show a step, networks a ramp.
Cheap — numpy plus existing checkpoints, no training. If it works, "we can measure the effective
aggregation temperature of a learned reasoner" is a positive result on its own.

**4.3 — Sustained, multi-round response measurement.** Both dead ends point here. `FINDINGS_FAMILY.md`
§"What a redesign would have to fix" item 2 and `HANDOFF.md`'s repair (1) independently conclude
that separating commitment from damping needs a *sustained* measurement — and 2.1's predicate
repair is the same fix in the categorical layer. One implementation, two payoffs. Also fixes the
coverage failure: heavily damped rules (α ≤ 0.2) currently have **no fittable rows at all**,
because per-round motion falls below `κ·eps`.

**4.4 — Wavefront-speed profile.** Measure response arrival time as a function of hop distance
from the poke. Bellman-Ford: exactly one hop per round. Dijkstra: one node per round. Networks:
measured. This is continuous, well-conditioned, and immune to the no-fixed-point problem that
pins `latency`.

**4.5 — Sign as a certificate, not a distance. (This is E4, still unrun.)** `nonmono` is the only
survivor of B2, with the same direction on two disjoint seed sets and a growing effect size
(d +0.69 → +1.63), and the residual audit found the same phenomenon per-edge (negative Jacobian
entries in 3/9 models). Relaxation is monotone **by definition**, so "this model raises an
estimate" is a *proof* that it is not executing any relaxation algorithm — a binary certificate
with no threshold to tune. `HANDOFF.md` §E4 says to test it as a classifier of provenance, which
is what the data supports; that experiment is written up and has never been run.

---

## 5. Enlarge the hypothesis space until something is close

Proposition 5 itself prescribes this: *"enlarge `A` until some member is close, or accept that the
family is incomplete."* Phase F enlarged it in the wrong direction and measured, precisely, which
direction was right.

**5.1 — Family-2: replace scalar γ with a state-dependent gain. ★ high value, and half-built**

`FINDINGS_RESIDUAL.md`'s deliverable is a specification: the missing ingredient is a
**state-dependent, sometimes-amplifying path gain**, not a scalar discount. Replace `γ` with

```
g(d_v, t, deg(v)) = σ(a + b·t + c·d_v/U + e·deg(v))          # range extended above 1
```

The fit is nearly free, because the residual audit **already measures per-node `γ_v`** and its
predictors: ρ(round) = +0.482 (9/9 sign agreement), ρ(d_now) = −0.420 (9/9), ρ(deg) = +0.225 (9/9).
So fitting `g` is a regression on measurements that exist, then simulate under fitted `g` and
recompute chi².

**Frozen criterion, which the family already supplies:** chi² ≤ 4 with ve ≥ 0.90 is "family
adequate" (`PREREGISTRATION_FAMILY.md`). If augmented-family chi² drops from ~10⁴ to ≤ 4, the
result is *"a trained MPNN is a relaxation with a state-dependent path gain, and here are its
coefficients"* — a positive identification of a neural reasoner, which is the thing this project
set out to do and has never achieved.

Negative control, mandatory: `g` must collapse to a constant on family members, which must still
be recovered bit-identically. And note the cost `FINDINGS_RESIDUAL.md` already flags — a *function*
where the family had a scalar re-opens every identifiability problem Gate 1 measured, plus α↔c.

**5.2 — Add references that are neural-plausible but still exactly specifiable.** Damped
relaxation, softmin relaxation, discounted relaxation, and label-correcting variants as *named
references* in the classifier, not just as family points. Even if none is close, the distance
profile becomes informative rather than uniformly bad.

**5.3 — Add more classical algorithms.** Delta-stepping, Johnson, Floyd–Warshall restricted to a
source row, A* with an admissible heuristic, Bellman–Ford with randomised edge order, Gabow's
scaling. Cheap (numpy, seconds) and it directly attacks `THEORY.md` open question 2 — see §7.2.

---

## 6. Build models that actually differ (the E1 that never ran)

**6.1 — Give the architecture a global readout and retrain on Dijkstra. ★**

E1 is the only experiment in this project that failed for a *diagnosed, fixable* reason: a
min-aggregating MPNN has no global readout, so it cannot represent an argmin over unsettled nodes;
loss plateaued at 1.64–2.00 across all six seeds against 0.002–0.038 for Bellman-Ford, with a
seed spread of 0.36 against 0.035 — capacity, not optimisation. `HANDOFF.md` states the fix
outright: *"the fix is architectural."*

Add a global node (or one attention head over all nodes) and retrain Dijkstra-supervised. If it
admits at MAE < 2.5, the discrimination test the project has *never run* becomes runnable, on a
model that genuinely executes one-settle-per-round rather than approximating commitment locally as
E3's `gated` arm did. And the symbolic prediction is sharp and pre-registerable: `corrupt_node_up`
on a settled node should give `recovery = none, latency = never` — Phase A's crux, on a network.

The counter-argument in `HANDOFF.md` — "at which point E3 is the better experiment" — was written
before E3 ran. E3 tested *architectural approximations* of commitment. This tests the real thing.
~30 min on CUDA.

**6.2 — Supervise on a pair the architecture can actually execute.** BF vs truncated-BF-k3 hints.
Both learnable (no global argmin needed), and A3 guarantees they are symbolically separable at
*every* depth. Sidesteps E1's capacity block entirely. Prediction is honestly null — this is close
to E3's `halt` arm installed by supervision instead of deployment — but it is cheap and it closes
the supervision analogue.

**6.3 — Widen the procedural gap toward what the method *provably* detects.** A3/Prop 4: limited
*runtime* is observable at any depth, even where limited propagation depth is not. So train models
that differ in runtime discipline — early-exit heads, adaptive computation time, learned halting —
rather than in aggregation. Aim the instrument at the axis it is proven sharp on.

---

## 7. Guaranteed positives: broaden the symbolic frame

Everything here works, because the symbolic method works. The only question is scope.

**7.1 — Other problem domains. ★ turns a one-domain method into a framework**

MST (Prim / Kruskal / Borůvka — genuinely different control flow), sorting (insertion / bubble /
selection, with array state as the observable), graph search (BFS / DFS / iterative deepening),
max-flow (Ford–Fulkerson / Edmonds–Karp / Dinic), union-find with and without path compression,
topological sort (Kahn / DFS). Each is a fresh symbolic experiment where the machinery already
applies, and *"intervention fingerprints identify algorithms across five problem domains"* is a
much stronger paper than the same claim about SSSP.

Two of these are strictly better neural targets than SSSP as well: sorting and max-flow have
learnable variants whose procedures differ far more structurally than BF-vs-Dijkstra does for a
message-passing network.

**7.2 — Answer open question 2 empirically.** With 8–12 algorithms (§5.3, §7.1), does the minimal
separating set stay at 2, or grow like log|A|? That is a stated open problem in `THEORY.md`, it is
answerable in seconds of numpy, and either answer is a result.

**7.3 — Re-run the depth sweep on the enlarged family.** Does the depth-4 impossibility threshold
move with |A|? Proposition 3 currently rests on five algorithms.

**7.4 — A sharper Dijkstra/Prim probe, given E6's diagnosis.** E6 established that weight
*raises* are as monotonically invisible as deletions, and named the way out: construct an instance
where a *second, already-improving* path exists whose ranking under `d[u]+w` differs from its
ranking under `w`, so the dissociation shows through relaxation's lowering-only response. That is
a different construction, not a parameter tweak, and it would close the one boundary that still
rests on a single intervention.

---

## 8. Theory positives (cheap, and the strongest available)

**8.1 — Write the formal layer.** `HANDOFF.md` calls this "the strongest theoretical contribution
available" and it is unwritten. `THEORY.md` has the definitions and witnesses; it needs the proofs.

**8.2 — Characterise the behavioural equivalence classes in closed form. ★ theorem-shaped**

Gate 1 measured α↔c at SNR 0.00 and gave the mechanism: `a_eff = α(1−m(c))`. That is not a
measurement limitation — it is the statement that the level sets of `a_eff` are **exactly** the
behavioural equivalence classes of the family under this observable. Same for τ↔selection: under
`argmin` there is one sender, so there is no distribution for a temperature to spread over, and τ
is unobservable *by construction*.

Turning Gate 1's four SNR-0.00 cells into four proved propositions converts an identifiability
*failure* into an identifiability *theorem* — a complete characterisation of what the response
operator can and cannot resolve within a family that provably contains five classical algorithms.
This is the single most elegant positive available, and it costs algebra, not compute.

**8.3 — Lower bounds on |I|.** Construct a family requiring Ω(log|A|) interventions, or prove a
constant suffices for the relaxation family. §7.2 gives the empirical side.

**8.4 — Prove monotone coarsening in depth** (open question 3). A3 is consistent with it.

---

## Ranked shortlist

If only three things get done:

1. **§3 — the neural-vs-neural discrimination run** (3.1 + 3.2 + 3.3 as one pre-registered pass).
   Existing checkpoints, no new code for 3.1, a visible lead in 3.2, and it asks the question the
   project actually cares about instead of the one Proposition 5 forbids.
2. **§2.1 + §2.2 — repair the pinned predicates, and recheck the orthogonality metric.** ~29% of
   every neural fingerprint is currently a constant, and the number that closed the door on probe
   redesign is computed against a proxy that ranks an indispensable probe second-last.
3. **§5.1 — Family-2 with a state-dependent gain.** The specification is written, the per-node
   measurements exist, the acceptance threshold is already frozen, and success would be an actual
   positive identification of a neural reasoner.

Then, by cost-effectiveness: §8.2 (algebra only), §7.1–7.2 (guaranteed, grows the paper),
§4.2 and §4.1 (new observables), §6.1 (the one experiment blocked purely on capacity).

---

## The rule

Roughly half of the above — everything in §2, §3, and §4.5 — is **a new statistic computed on data
that has already been looked at**. That is the exact shape of all five results this project has
retracted, and `PREREGISTRATION_E5E6.md` was written specifically because these are "exactly the
kind of thing that has killed five results."

So, non-negotiably:

- **Freeze the statistic and the threshold before running**, in a `PREREGISTRATION_G.md`, with the
  full candidate list written down so the winner cannot be selected afterwards.
- **Two rungs on every ladder, no middle.** Direction-replicating-without-resolution is this
  project's characteristic dead-lead signature (`overshoot`, E3b); leave it nowhere to land.
- **Fresh seeds for anything in §3.** Nine checkpoints are three architectures × three seeds —
  **three independent design points, not nine.** Report sign agreement, not a p-value, on the
  existing set, and treat the whole of §3 as exploratory until new seeds confirm it.
- **A negative control in the same run.** Symbolic separation must be unchanged by any distance
  or predicate repair (§2), and family members must behave as nulls (§5.1).
- **One pass.** No follow-up statistic on the same data if the first one misses.

The failure mode to name in advance: §2.3 and §2.4 change both the numerator and the denominator,
and E5 already demonstrated that this arithmetic can raise the reference median faster than it
lowers `d_near`. If those come out UNCHANGED, that is the answer — not a prompt for a fifth
variant.
