# Behavioural equivalence under an intervention class

The definitional layer under Phase A. Everything here is stated so that the empirical
results — the minimal separating set, the depth threshold, truncated-BF separable from BF
on instances where both return identical answers — are instances of theorems rather than
observations that happened to come out that way.

The central claim is negative and it is the point: **"which algorithm is this?" is not a
well-posed question.** It becomes well-posed only once an intervention class and an input
distribution are named, and different choices give genuinely different, equally correct
answers. Everything below is machinery for saying that precisely.

---

## 1. Executions

An **algorithm** is a map `A` from a problem instance to a sequence of observable states.
Fix a finite state space: instance `G = (V, E, w)` with source `s`, and observable
`x = (d, pi) ∈ (R ∪ {∞})^V × (V ∪ {⊥})^V`. Write

    A(G, s) = (x_0, x_1, x_2, ...)

for the **trajectory**, where `x_{t+1} = step_A(G, x_t)` and the sequence is padded by
`x_{t+1} = x_t` once `A` terminates.

Two commitments are doing real work here.

**Observables are decoded, not internal.** `d` and `pi` are all that is visible. Dijkstra's
settled set, SPFA's queue and a network's hidden activations are all outside the observable.
This is forced rather than chosen: a predicate phrased in terms of one algorithm's private
bookkeeping is ill-posed for algorithms that lack it, and a neural model has no inspectable
internals we are permitted to use. It is also what makes the whole framework apply to
learned models at all.

**Time is rounds of the algorithm's own progress.** Round indices are not comparable across
algorithms — round 5 is five hops of propagation for Bellman-Ford and five settled nodes for
Dijkstra. Section 4 replaces absolute time with a progress-relative clock, and this is not a
technicality: it is the difference between measuring a response and measuring nothing.

## 2. Interventions

An **intervention** is a pair `i = (τ, μ)` where `τ` is a *targeting rule* and `μ` a
*mutation*.

The targeting rule `τ(G, s) ∈ V ∪ E` selects a site **from the instance and its ground truth
only** — never from the running algorithm's state. This restriction is what makes a
fingerprint a comparison rather than a confound: if Bellman-Ford deleted an edge its own
pointers had chosen and Dijkstra deleted a different one, the two would respond differently
for reasons having nothing to do with how they compute. Every algorithm must receive the
identical poke.

The mutation `μ` acts on the pair (instance, state) mid-execution:

- **structural** — edit `G`: delete an edge, insert one, change a weight;
- **state** — edit `x`: raise or lower `d[v]`, give an unreached node a finite estimate.

An **intervention class** `I` is a finite set of such pairs. `I` is the parameter the whole
theory is relative to.

**Intervened execution.** For `i ∈ I` firing at round `t`, write `A[i@t](G, s)` for the
trajectory that follows `A` up to round `t`, applies `μ`, and then continues under `step_A`
on the mutated instance. Three trajectories matter, and confusing them is the most common
way to get a meaningless answer:

| | Definition | Answers |
|---|---|---|
| **control** | `A(G, s)` | what would have happened |
| **intervened** | `A[i@t](G, s)` | what did happen |
| **reference** | `A(μ(G), s)` from scratch | what *should* have happened had the instance always been mutated |

**Response measures compare intervened against control.** Comparing intervened against
reference measures whether the poke moved the *right answer*, which is a fact about the
instance and identical for every algorithm — not a response at all. The reference run has
exactly one job: it defines *recovery*. An algorithm has recovered when the mid-run poke
left it in the state it would have reached had the instance always been that way.

## 3. Predicates and the fingerprint

A **predicate** `p` maps the triple (control, intervened, reference) to a value in a small
finite set. Predicates are qualitative buckets, not real-valued distances, and this is a
substantive choice: a metric over response vectors has many free parameters — which norm,
which weighting, which normalisation — and any separation it reports can be tuned into
existence. *"Does the corruption ever wash out: exact / partial / never"* cannot.

The suite used here reads recovery, latency, first-round spread, propagation depth,
termination shift, pointer churn, and whether the poked node had already converged.

**Restriction to the at-risk set.** Predicates are evaluated only on nodes whose correct
answer the poke actually moved, plus the poked node itself. Without this the untouched
majority washes every difference out.

Given `I` and an instance distribution `D`, the **fingerprint** is

    F_A(I, D) : (i, p) ↦ modal value of p over instances drawn from D

together with a **stability** `σ_A(i, p) ∈ [0, 1]`, the fraction of instances on which the
mode held. Stability is not decoration. A predicate that flips between instances carries no
identity information, and including it manufactures separation out of sampling noise.

## 4. The firing rule

`t` must be defined so that "the same moment" means the same thing across algorithms with
different runtimes. Define the **settling round** `T_A(G, s)` as the last round at which `d`
changed materially, and fire at

    t = clip(T_A(G, s) / 2, 1, budget − 1)

Three properties are needed and each was learned by a run that produced confident-looking
garbage:

1. **Progress-relative.** An absolute round lands after termination for fast algorithms and
   measures nothing.
2. **Half the run remains.** Firing near the end leaves no room to respond, so every
   algorithm scores "no response" and the fingerprint reports the budget, not the algorithm.
3. **Robust settling estimate.** `T_A` must use a threshold relative to the largest change
   the run ever made, with a correction for the number of comparisons (~`|V| × budget`),
   or noise crosses any fixed tolerance somewhere in almost every run and drags `T_A` to the
   end of the budget. Early-terminating algorithms are hit hardest — precisely the shortcut
   procedures the framework exists to detect.

## 5. Equivalence, separation, identifiability

**Definition (I,D-equivalence).** `A ≡_{I,D} B` iff for every `i ∈ I` and every predicate
`p` jointly stable in both, `F_A(I,D)(i,p) = F_B(I,D)(i,p)`.

Distance is Hamming over jointly-stable predicates:

    δ_{I,D}(A, B) = |{(i,p) jointly stable : F_A ≠ F_B}| / |{(i,p) jointly stable}|

so `A ≡_{I,D} B` iff `δ_{I,D}(A, B) = 0`.

**Stability is applied pairwise, never globally.** Intersecting the stable set across every
reference discards exactly the predicates that discriminate a single pair. Under a global
rule Dijkstra and SPFA lose the one probe that separates them, collapse to identical
vectors, and get classified by tie-break.

**Definition (separating set).** `I` separates a family `A` on `D` iff `δ_{I,D}(A,B) > 0`
for all distinct `A, B ∈ A`. `I` is **minimal** if no proper subset separates `A`.

**Proposition 1 (monotonicity).** `I ⊆ I'` implies `A ≡_{I',D} B ⟹ A ≡_{I,D} B`.
Enlarging the intervention class can only refine the equivalence. Immediate from the
definition, and it is what makes minimal separating sets the right object: they are the
cheapest certificate of distinguishability.

**Proposition 2 (equivalence is not implied by agreement on outputs).** There exist `A ≠ B`
and `D` such that `A` and `B` return identical final answers on almost every instance of `D`
while `δ_{I,D}(A,B) > 0`. Truncated Bellman-Ford versus Bellman-Ford on shallow instances is
a witness: identical output on 11 of 12 instances, separated anyway.

This is the load-bearing proposition. Output equivalence is strictly coarser than
behavioural equivalence, so a method that only inspects answers cannot recover procedure,
and a model can be output-perfect in distribution while running a different algorithm.

**Proposition 3 (identifiability is distribution-relative).** There exist `A, B` and
distributions `D, D'` with `A ≡_{I,D} B` and `δ_{I,D'}(A,B) > 0`. Witness: no `I` in the
suite separates all pairs on instances of propagation depth ≤ 4, while two interventions
suffice at depth ≥ 7.

The mechanism is that a depth-limited response is only observable when the instance affords
depth to travel. On a diameter-2 instance "propagated 3 hops" and "propagated everywhere"
are the same observation, and the depth predicate is vacuous. This is stronger than a
caveat: **algorithm identity is a property of a procedure together with an intervention
class and an input distribution, not of the procedure alone.**

A corollary with teeth for practice: evaluating on a dense, shallow default distribution
places the measurement exactly where the method provably cannot work, and returns a null
about nothing.

**Proposition 4 (runtime is observable when propagation is not).** Limited *runtime* and
limited *propagation depth* are distinct properties, and the first remains observable on
instances where the second is not. Witness: truncation is detectable at every depth,
including depths at which the truncated and untruncated algorithms produce identical
trajectories on the unintervened run — because an intervention firing after the halt cannot
be responded to at all, and that absence is itself a signal.

A predicted threshold at `k = 3` was falsified by this. The prediction assumed truncation
becomes invisible once `k` exceeds the instance depth; it does not, because the halt is
observable through response latency rather than through propagation extent.

## 6. Identification and its failure mode

Given reference family `A` with known fingerprints and an unknown `M`, the nearest-reference
rule returns `argmin_{A ∈ A} δ_{I,D}(M, A)` with **margin** the gap between nearest and
second-nearest.

**The label alone is not a result.** A model roughly equidistant from everything is not an
`A`, it is an unidentified procedure, and only the margin distinguishes the two cases. Two
quantities must accompany every label:

- the margin, against the noise floor — the distance between two instances of the *same*
  procedure differing only by an irrelevant parameter such as a seed;
- the nearest distance `δ(M, Â)`, against the **reference-to-reference scale**
  `median_{A≠B} δ(A, B)`.

**Proposition 5 (vacuity of nearest-reference identification).** If
`δ(M, Â) ≳ median_{A≠B} δ(A,B)` then the label carries no information: `M` is as far from
its assigned reference as two genuinely different references are from each other, and the
rule is reporting the least-bad of several poor matches.

This is the regime every neural model in this project occupied — nearest distance 0.30–0.40
against a reference-to-reference median of 0.25–0.30 — across every supervision regime and
every architecture tested, including architectures built to differ in procedure. The
correct reading is not "these models are Bellman-Ford" but **"these models execute a
procedure outside the reference family."**

Proposition 5 also says what would fix it, and the fix is a modelling question rather than a
measurement one: enlarge `A` until some member is close, or accept that the family is
incomplete. A candidate worth stating because it follows from the architecture rather than
from fitting: a message-passing reasoner performs a *damped, soft* relaxation
(`d ← d + α(agg(d) − d)` with a smooth aggregator) rather than a hard `min`. A softmin
responds to deleting an edge that was never on a shortest path — the aggregate changes even
when the argmin does not — whereas hard-min relaxation cannot, since relaxation only lowers
`d` and repairing a deletion requires raising it. That single structural difference predicts
disagreement with *every* hard-relaxation reference on precisely the deletion probes, for
every neural model, independent of which procedure it learned.

## 7. What is proved and what is not

**Established constructively** (Phase A/A2/A3, symbolic, ground truth known exactly):
Propositions 1–4 with explicit witnesses; a minimal separating set of size 2 at depth ≥ 7,
3 at depth 5, and non-existence below depth ≈ 4; and stability of the separation under
input-dependent observation noise to σ = 2 against an edge-weight scale of 1–10.

The noise result deserves its own statement, because it explains why the fingerprint must
never be reported as a quality metric. A fingerprint is a **differential** measurement:
control and intervened rollouts share any error that is common-mode, and the difference
cancels it. Identification is therefore near-orthogonal to accuracy *by construction* — a
property confirmed on real models, where the least accurate seed identified with the largest
margin.

**Not established.** No neural model has been identified. Proposition 5's hypothesis was
satisfied in every neural experiment run, so those experiments constrain the reference
family rather than the models.

**Open.**

1. Is there a syntactic characterisation of `≡_{I,D}` — a normal form on `step_A` such that
   equivalence is decidable without simulation?
2. Lower bounds on `|I|`. Empirically 2 interventions separate 5 algorithms at depth ≥ 7.
   Is there a family requiring `Ω(log |A|)`, or does a constant always suffice?
3. Does `≡_{I,D}` coarsen monotonically in the instance depth of `D`? A3 is consistent with
   this and it is unproven.
4. Sensitivity normalisation. Responses should be reported relative to a null-control
   intervention that no correct algorithm responds to, so that a globally sensitive model
   does not appear "different" for free. Currently unimplemented, and Proposition 5's
   diagnosis suggests it may absorb a substantial constant offset.
