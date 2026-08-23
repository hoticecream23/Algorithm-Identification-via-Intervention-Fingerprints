# Behavioural Equivalence Under an Intervention Class

*Working draft, started 2026-08-23. Target: ICLR 2027 main track (abstract 18 Sept, paper
25 Sept 2026), 9 pages + references + appendix. Fallback: TMLR (rolling).*

> **Draft conventions.**
> `[CITE: …]` marks a citation that must be filled from the literature. **None are invented**
> — the repository carries no bibliography, and a placeholder is safer than a plausible-looking
> fabrication. `[TODO: …]` marks work still owed. `[FIG n]` marks a figure to be produced.
> Every number in this draft is traceable to a `FINDINGS_*.md` file and its reproducing command;
> §Provenance at the end maps each claim to its source.

---

## Abstract

*(Target 150–200 words. Draft.)*

Two programs that agree on every output of a test set may still be running different
algorithms, and the difference can be exactly what determines whether they generalise. We
formalise **behavioural equivalence relative to an intervention class `I` and an input
distribution `D`**: two executors are `≡_{I,D}`-equivalent when no stable predicate of their
*causal response* to interventions in `I`, over inputs from `D`, distinguishes them. This makes
the natural object of study not a similarity score but the **minimal separating set** — the
smallest set of interventions under which a family of algorithms ceases to be equivalent. On
five classical shortest-path algorithms we show this set has size two on graphs of propagation
depth ≥ 9, size three at depth 6, and **does not exist** below depth ≈ 5, so identity is
distribution-relative as well as intervention-relative. We give a case where behaviour and
accuracy provably come apart: a Bellman-Ford truncated at `k` rounds is output-identical to
full Bellman-Ford on *every* instance of propagation depth at most `k` — by construction, not
approximately — is separated from it by intervention response anyway, and is correct on none
of the instances past that bound. Applying the same instrument to trained
message-passing networks yields a **negative result we characterise precisely**: the networks
are further from every reference algorithm than the references are from each other, and are not
separable from one another either.

---

## 1. Introduction

*(Target ~1.25 pages.)*

Suppose two systems produce identical outputs on every instance you can test. Are they running
the same algorithm?

For most purposes the question sounds idle — until the distribution shifts. Consider
Bellman-Ford, and a truncated variant that stops relaxing after `k` rounds. On any instance
whose propagation depth is at most `k`, these two return **identical answers — on every such
instance, by construction**, since the truncation discards only rounds in which nothing would
have changed. Restricted to that subpopulation they are literally the same function, so no
accuracy metric, no validation loss, and no output-space probe can separate them. Deepen the
distribution past `k` and the truncated variant is correct on **none** of the instances that
cross the bound. The difference was always there; output agreement could not see it.

This is not a contrived example. It is the structure of a real and reproducible problem in
neural algorithmic reasoning `[CITE: CLRS benchmark; NAR position papers]`: models reach
near-ceiling validation accuracy on the training distribution and then behave very differently
out of distribution, which makes validation accuracy nearly useless for model selection
`[CITE: OOD generalisation in algorithmic reasoning; size-generalisation results]`. If two
models agree in-distribution and diverge out of it, something distinguishes them that
in-distribution *outputs* do not expose.

**Our proposal is to define identity by causal response rather than by output.** Interrupt an
execution mid-run — delete an edge, corrupt a node's estimate, inject a value at a node the
algorithm has not reached — and observe how the disturbance propagates. Bellman-Ford re-relaxes
every edge every round and repairs an optimistic corruption; Dijkstra, having committed a
settled node, cannot. A truncated algorithm that has already halted cannot respond to anything.
These are differences in *execution structure*, and they are visible in-distribution.

The central conceptual point is that this only defines identity **relative to two choices**.
Bellman-Ford and Dijkstra are indistinguishable under the empty intervention class on
non-negative weights — they compute the same function. They are also indistinguishable on
graphs of diameter 2 under our full nine-probe suite, because the probe that separates them
needs an unreached node to exist and on a diameter-2 graph nothing stays unreached. Identity is
therefore a property of the triple *(executor, intervention class, input distribution)*, and the
research object we propose is the **minimal separating set**: the smallest `I` under which a
given family stops collapsing.

### Contributions

1. **A formalism.** `≡_{I,D}`-equivalence, the minimal separating set, and five propositions
   with constructive witnesses (§3). Proposition 5 gives a *vacuity condition* under which a
   nearest-reference label carries no information — a diagnostic we then apply to our own
   results.
2. **The identifiability boundary, measured.** Two interventions separate five classical
   algorithms at depth ≥ 9, three at depth 6, and **no subset of the suite suffices below depth
   ≈ 5** (§5). Equivalence coarsens as instances get shallower.
3. **A case where behaviour and accuracy provably diverge**, with the mechanism (§5.2): the
   truncation result above, plus the finding that limited *runtime* is detectable at any depth
   even when limited *propagation depth* is not.
4. **A characterised negative on learned executors** (§6). Applied to message-passing networks
   trained to execute shortest paths — including architectures *built* to differ in procedure —
   the instrument does not resolve them: every network is further from its nearest reference
   algorithm than two references are from each other, and networks are not separable from one
   another either. We report the boundary of the method's applicability rather than a
   qualified success.
5. **A reproducibility artefact.** All experiments are pre-registered with frozen decision
   rules; the repository records five leads that failed replication, alongside the results that
   survived (§7.3).

---

## 2. Related work

*(Target ~0.75 pages. Every citation below is a placeholder — see draft conventions.)*

**Neural algorithmic reasoning.** `[CITE: CLRS-30 benchmark; neural execution of graph
algorithms; hint-supervision results]` Work here trains networks to imitate classical algorithms
step by step, typically supervising on intermediate "hints". Evaluation is overwhelmingly by
output accuracy, in and out of distribution. Our contribution is orthogonal to architecture: we
ask what it would *mean* for a learned executor to be running a given algorithm, and give a
measurement whose answer is not a function of accuracy.
`[TODO: position relative to work that inspects hint-prediction fidelity as a proxy for
procedural fidelity — that is the closest existing idea and must be distinguished carefully.]`

**Causal probing and interchange interventions.** `[CITE: causal abstraction of neural networks;
interchange intervention accuracy; distributed alignment search]` This line establishes when a
neural network *implements* a causal model by intervening on internal activations and checking
downstream effects. We share the interventionist stance but differ in an important way: our
probes act on the **decoded algorithmic state** `(d, π)` and the input graph, never on hidden
units, so the same code runs unchanged against a symbolic executor and a network, and no
neuron-to-variable alignment is assumed or required.
`[TODO: this is the most important related-work boundary in the paper; a reviewer will ask why
not just do causal abstraction. Answer: alignment is the thing we are trying to avoid needing.]`

**Program equivalence and observational equivalence.** `[CITE: observational/contextual
equivalence; bisimulation]` Our `≡_{I,D}` is a coarsened observational equivalence in which the
observation context is a fixed class of runtime perturbations rather than arbitrary program
contexts, and is indexed by an input distribution. The minimal separating set is then a
distinguishing-context question of the kind studied in that literature, restricted to a
semantically meaningful and *physically executable* class of contexts.

**Mechanistic interpretability.** `[CITE: circuits; activation patching; ablation methodology]`
Activation patching is methodologically close to our probes. The distinction is again that we
require no privileged internal decomposition: our observable is defined on the algorithm's
externally decoded state, which is what makes symbolic and neural executors directly comparable.

**Negative results and pre-registration in ML.** `[CITE: pre-registration proposals for ML;
reproducibility/replication studies; the ICBINB line of work]` §7.3 reports our
exploratory/confirmatory protocol and the leads it killed.

---

## 3. Behavioural equivalence under an intervention class

*(Target ~1.75 pages. Source: `THEORY.md`, which carries the full statements and witnesses.)*

### 3.1 Executions and observables

An **executor** `A` on an instance `G` with source `s` produces a trajectory of states
`σ_A(G,s) = (x_0, x_1, …, x_T)`. For single-source shortest paths the observable state is
`x_t = (d_t, π_t)` — the current distance estimates and predecessor pointers. This state is
**decoded**, not internal: for a classical algorithm it is the natural bookkeeping, and for a
neural executor it is the decoder output. Nothing below inspects internals.

Fixing a shared observable across the family is what makes the comparison well-posed, and it
carries a real constraint: any predicate phrased in terms of one algorithm's private bookkeeping
is ill-posed across the family. "Has this node been settled?" is a question about Dijkstra's
priority queue that Bellman-Ford cannot answer. We therefore define settledness
**observationally** — an estimate that has stopped changing — which is meaningful for every
executor including a network. `[TODO: this definition is load-bearing and got a detail wrong;
see §7.2 and `FINDINGS_H.md`.]`

### 3.2 Interventions and the firing rule

An **intervention** `ι` is a map that, at a chosen round, mutates the instance or the executor's
observable state. Two design rules are load-bearing.

**Targets are algorithm-independent.** The node or edge an intervention acts on is chosen from
the graph and its ground truth only, never from the running executor's own state. If
Bellman-Ford deleted an edge chosen by its own pointers and Dijkstra deleted a different one,
their responses would differ for a reason having nothing to do with procedure.

**Firing time is progress-relative, never absolute.** Round 5 is five hops of propagation for
Bellman-Ford and five settled nodes for Dijkstra; these are not comparable moments. We fire at
the midpoint of each executor's own **active life** — the last round at which its estimates
still moved — which guarantees that half the run remains for a response to develop. Firing after
termination measures nothing.

### 3.3 Predicates, fingerprints, and equivalence

For each `(executor, intervention, instance)` we compare the **intervened** trajectory against
a **control** trajectory of the same executor on the same instance. We deliberately do *not*
compare against a fresh reference run on the mutated graph: that would measure whether the poke
moved the *correct answer*, which is a fact about the graph and identical for every executor.

We read off seven qualitative predicates — recovery (`exact`/`partial`/`none`), latency, first-
round spread, propagation depth, termination shift, pointer churn, and whether the poked node
had already settled. Predicates are **categorical by design**: a real-valued response metric has
free parameters (which norm, which weighting, which normalisation) and any separation it reports
can be tuned into existence. "Does the corruption ever wash out" cannot.

A **fingerprint** aggregates modal predicate values across instances drawn from `D`, together
with the *stability* of each mode. Unstable predicates carry no identity information and are
excluded pairwise.

> **Definition (`≡_{I,D}`).** Executors `A` and `B` are **behaviourally equivalent under
> intervention class `I` and distribution `D`** when no jointly stable predicate under any
> `ι ∈ I`, over instances from `D`, takes different modal values for `A` and `B`.

> **Definition (minimal separating set).** For a family `A`, a **separating set** is any
> `S ⊆ I` such that no two distinct members of `A` are `≡_{S,D}`-equivalent. The **minimal
> separating set** is a smallest such `S`, and `|S_min|` measures how much intervention power
> the family's distinctions actually require.

### 3.4 Propositions

Stated informally here; `THEORY.md` carries the formal versions and explicit witnesses.

- **P1 (relativity).** `≡_{I,D}` is a genuine equivalence relation, and coarsens monotonically
  as `I` shrinks. Witness: Bellman-Ford `≡_{∅,D}` Dijkstra on non-negative weights.
- **P2 (behaviour ⊀ accuracy).** There exist `A, B` and `D` with identical output on
  `D`-almost-every instance that are separated by `≡_{I,D}`. Witness: §5.2.
- **P3 (distribution-relativity).** `≡_{I,D}` depends on `D` with `I` fixed. Witness:
  Dijkstra and SPFA separate at `depth_w ≥ 6` and merge below it.
- **P4 (runtime vs depth).** Limited runtime and limited propagation depth are distinct
  observables; the former is detectable at any instance depth, the latter only on instances
  deeper than the truncation bound.
- **P5 (vacuity of nearest-reference identification).** If `δ(M, Â) ≳ median_{A≠B} δ(A,B)`,
  the nearest-reference label carries no information: `M` is as far from its assigned reference
  as two genuinely different references are from each other. **The label alone is never a
  result**; it must be quoted with the nearest distance, the margin, and the
  reference-to-reference scale.

P5 is stated as a contribution rather than a caveat because it is what our own neural results
turn out to satisfy (§6), and because nearest-reference classification against a library of
reference behaviours is a natural thing to do that we believe is under-guarded in practice.

---

## 4. Instantiation

*(Target ~1.25 pages.)*

**Family.** Five classical SSSP executors sharing the `(d, π)` observable: Bellman-Ford,
truncated Bellman-Ford (`k = 3`), SPFA, Dijkstra, and Prim (which keys on edge weight rather
than path length, giving a wrong-but-well-defined SSSP procedure and a useful near-neighbour of
Dijkstra).

**Intervention suite.** Nine probes: two shortest-path-tree edge deletions, a non-tree deletion
serving as a **null control**, an improving-edge insertion at a settled node, a shortcut forcing
propagation from a distant node, a tree-edge weight reduction, optimistic and pessimistic node
corruptions, and an estimate injected at an unreached node. `[TODO: table in appendix.]`

Two probes deserve comment. `insert_improving_edge` presents a late improvement to a node that
already looked finished — Bellman-Ford keeps relaxing and picks it up; Dijkstra's settled set
makes it unreachable. `inject_unreached_node` perturbs what is *eligible* rather than what is
*valued*, which is the only thing that separates SPFA from Dijkstra: they agree on every probe
that perturbs values and differ on one that perturbs the work queue.

**Monotonicity, and what it costs.** Relaxation only ever *lowers* `d`. Repairing a deletion
would require raising an estimate, so **no correct relaxation algorithm responds to edge
deletion at all**, and the deletion probes are near-silent against the symbolic family. We
retain them deliberately: a learned executor has no monotonicity constraint and *can* raise an
estimate, so "responds to a deletion" is itself a sharp signal that a model is not doing
relaxation. §5.4 reports the same argument applied to weight *increases*, which we established
are equally invisible — a constraint on the intervention class, not a fixable gap.

**Distance.** Hamming over jointly stable predicates. Categorical predicates admit no norm, so a
disagreement count is the only distance that does not require inventing a scale per predicate.
Stability is applied **pairwise**: intersecting stability across the whole family discards
precisely the probe that separates a single pair.

---

## 5. The identifiability boundary on classical algorithms

*(Target ~1.75 pages. Sources: `FINDINGS_PHASE_A.md`, `FINDINGS_IDENTIFIABILITY.md`,
`FINDINGS_NOISE.md`.)*

### 5.1 Two interventions separate five algorithms

On 25 high-diameter graphs (`n = 24`), all ten pairs are separated, and the minimal separating
set has **size two**: `{insert_improving_edge, inject_unreached_node}`. The remaining seven
probes are redundant once those two are held. The result is stable across seeds.

Two pairs hang on a single probe each — Dijkstra/Prim on `insert_improving_edge`, Dijkstra/SPFA
on `inject_unreached_node`. These are the fragile edges of the family, and we report them as
such (§7.2).

### 5.2 Behaviour and accuracy come apart — the central positive result

**The right notion of depth.** Truncation cuts *rounds of relaxation*, so the depth that
governs it is the number of edges on a shortest-**weight** path — equivalently, the round at
which synchronous Bellman-Ford settles. This is not the graph's hop eccentricity: relaxation
follows minimum-weight paths, which may use strictly more edges than minimum-hop ones, and on
our generators the two measures differ by one to two rounds. We write `depth_w(G,s)` for the
weighted quantity and use it throughout. (Getting this wrong is not a technicality; §7.3
records the measurement error it caused in an earlier version of this result.)

**The witness is exact, not empirical.** When `depth_w ≤ k`, the rounds `bf_truncated_k`
discards are exactly those in which no estimate would have changed, so it is output-identical
to full Bellman-Ford on *every* such instance and both are exactly correct. When `depth_w > k`
it halts strictly before reaching the fixed point, so it is *never* identical and never
correct. The population therefore splits deterministically at `k` in both directions, with no
instance able to land in between. Over 600 instances spanning five densities and two seeds:

| `depth_w` | n | BF ≡ truncated | truncated correct |
|---|---|---|---|
| ≤ k | 137 | **137 / 137** | **137 / 137** |
| > k | 463 | **0 / 463** | **0 / 463** |

No intermediate cases occur. Restricted to `depth_w ≤ k` the two executors are the same
function, so no output-space test — accuracy, loss, or any probe of the returned answer —
can distinguish them even in principle.

**The fingerprint separates them anyway.** Aggregating over 120 instances drawn *only* from
the `depth_w ≤ k` bucket, where output agreement is total, `insert_shortcut_from_source`
separates them on two stable predicates:

| predicate | `bellman_ford` | `bf_truncated_k3` | stability |
|---|---|---|---|
| recovery | exact | partial | 1.00 / 0.85 |
| term_delta | + | 0 | 0.85 / 0.99 |

Because every instance in this sample has identical output, the separation cannot be an
accuracy difference in disguise. Two further predicates (`latency`, `depth`) show the
expected direction but fail the pairwise stability gate and are excluded — a case of the
rule in §4 doing its job.

And the difference is exactly the one that matters under shift: truncated-BF's accuracy is
precisely the probability mass the truncation budget covers, `P(depth_w ≤ k)`, which falls
from ~73% at density 0.8 to 0% at density 0.05. The two quantities agree instance for
instance at every density we measured. So the intervention response detected, on a
subpopulation where accuracy was *exactly* perfect and *exactly* uninformative, the defect
that accuracy could only reveal after the shift.

**A prediction we made and falsified, which improved the result.** We pre-registered the
expectation that BF and truncated-BF would be *inseparable* below depth 3, on the reasoning that
an early-halting algorithm is only detectably early on instances deep enough to halt within.
They were separable at **every** depth tested, down to `depth_w` 3.2. The error was conflating two
consequences of truncation: limited *propagation depth*, which is indeed invisible on graphs
shallower than `k`, and limited *total runtime*, which is visible at any depth because a probe
fired mid-run gets no response from an executor that has already stopped. This is P4, and it is
a stronger result than the prediction: **a shortcut model can be caught on the easy distribution
it was trained on**, without constructing deep test instances.

### 5.3 Identifiability degrades with instance depth

Sweeping density with `n` fixed, on the weighted depth of §5.2:

| `depth_w` | `|S_min|` |
|---|---|
| 17.1, 8.9 | **2** |
| 6.0 | **3** |
| ≤ 4.7 | **none exists** |

Below `depth_w ≈ 5` no subset of the nine-probe suite separates all pairs: Dijkstra and SPFA
collapse first, below 6.0, and Dijkstra and Prim follow below 4.4. This is P3 with a measured
boundary, and it has a practical consequence — a protocol run on a dense, shallow default
graph distribution lands where the method provably cannot work and returns a null about
nothing. `[FIG 1: |S_min| vs depth_w, with the pairs that merge annotated.]`

### 5.4 A limitation of the intervention class, not a gap in the results

The Dijkstra/Prim boundary rests on one probe, so we attempted a sharper one keyed directly on
the quantity that distinguishes them: Dijkstra selects on `d[u] + w`, Prim on `w` alone. Raising
`w(u,v)` by `δ` while lowering `d[u]` by the same `δ` leaves `d[u]+w` exactly invariant while
`w` strictly moves.

Its manipulation check **failed**, and the reason generalises: by the monotonicity argument of
§4, a weight *increase* is as unobservable as a deletion for **both** algorithms, so the probe's
only live component was a state corruption affecting them symmetrically. We report this as a
constraint on what the intervention class can express — any sharper probe for this pair must
dissociate the two keys through relaxation's one-directional response rather than through a
raise.

### 5.5 Robustness

Identification survives input-dependent observation noise to `σ = 2` against an edge-weight
scale of 1–10, at 100% accuracy, with the margin falling from 0.146 to 0.077. The reason is
structural: a fingerprint is a **differential** measurement, so any error common to the control
and intervened rollouts cancels. This is also precisely why identification is near-orthogonal to
accuracy *by construction* — and why a fingerprint must never be reported as a quality metric.

---

## 6. Application to learned executors: a characterised negative

*(Target ~1.5 pages. Sources: `FINDINGS_E3.md`, `FINDINGS_FAMILY.md`, `FINDINGS_G.md`,
`FINDINGS_H.md`, `FINDINGS_E5.md`.)*

We applied the identical instrument — same probes, same predicates, same distance, same code —
to message-passing networks trained to execute shortest paths, with the algorithmic state
carried through the round loop and re-encoded so that state interventions are meaningful.

### 6.1 The instrument does not resolve these objects

Three architectures were built to differ in *procedure*, not in accuracy: a plain MPNN
(relaxation-like), a gated model that freezes a node's estimate once it stops moving
(Dijkstra-like commitment from a purely local rule), and a model hard-truncated at `k = 8`. All
were trained hint-free on identical data and matched for accuracy, and a mechanism check
confirmed each architectural difference was active. **All 17 admitted models were labelled
`bellman_ford`**, at a family-level Fisher `p = 1.000`.

The label was never supported. Every network sits at `d_near ≈ 0.29` from its nearest reference
against a reference-to-reference median of `0.216` — a ratio of **1.34**, i.e. *further from its
assigned reference than two genuinely different algorithms are from each other*. This is exactly
the hypothesis of P5, so the correct reading is not "these models are Bellman-Ford" but **"these
models execute a procedure outside the reference family."**

### 6.2 They are not separable from each other either

P5 constrains comparison *to a reference family* and is silent on whether two unknowns are
distinguishable from one another — and the discrimination question never needed references at
all. Asked directly, with pre-registered thresholds and an exact permutation null: **1-NN
architecture purity is 3/12 on twelve freshly trained checkpoints, against a chance level of
3/11**, and **0/8** on the pair whose procedural difference is *learned* rather than imposed by
the deployment harness. **Eighteen of sixty-six model pairs have bit-identical fingerprints,
eleven of them across architectures.**

So the failure is not that the reference family is incomplete. On this model class the
observable does not resolve the objects at all: under this `I` and `D`, they are largely a
single equivalence class.

### 6.3 The natural continuous relaxation does not contain them either

If reading a discrete label off a continuous system is the wrong question, the obvious repair is
estimation over a parametric family `(α, τ, γ, c, k, selection)` — damping, aggregation
softness, path discount, commitment, truncation, selection order — containing the discrete
algorithms as limit points. That family contains all five references **bit-identically**
(max deviation 0.0), and damping and path discount have closed-form estimators accurate to four
decimals. It does **not** contain the networks: six of nine checkpoints fit at `χ² ~ 10⁴`,
against an in-family holdout at `0.0000` and a deliberately out-of-family control at `15.0`.
Seven of nine sit further from the family than the negative control does.

### 6.4 Why we believe this is a real boundary and not a measurement artefact

The vacuity gap survived three independent attacks on the measurement:

1. **Sensitivity normalisation.** Responses were never normalised against the null control, so a
   globally jumpy model could look "different" for free. Implementing that normalisation leaves
   the gap intact.
2. **The reference-free reformulation** of §6.2, which P5 does not apply to.
3. **A repaired firing rule.** Our progress-relative rule was satisfied in code and *not in
   effect* for continuous-output executors: its activity threshold is anchored on the largest
   single-round change a run makes, which for a network is a round-1 initialisation transient,
   so networks were probed at **~11% of their active life** against ~45% for the references.
   Repaired, all nine networks fire at 0.44–0.50 — and the gap is **unchanged**, both terms
   falling ~29% together with the ratio moving 1.35 → 1.34. The vacuity is scale-invariant.

We regard (3) as an independently useful finding: **a progress-relative timing rule validated on
discrete frontier-propagating algorithms silently mis-times every continuous-output executor it
is applied to**, because the relative threshold is anchored on a transient the continuous
executor has and the discrete one does not. Anyone building progress-relative probes for learned
models will hit it.

### 6.5 What we do *not* claim

We do not claim intervention fingerprints cannot work on learned executors in general. We claim
that on *this* model class, under *this* intervention class and *this* distribution, they do not
resolve — and that P5's vacuity condition is the right diagnostic for saying so. The method
detects differences in execution *structure*; the networks studied here appear to differ in
*update-rule generalisation* while running the same control flow, which is a difference this
instrument is blind to by construction. `[TODO: soften or sharpen once §7.2's reopened question
is resolved.]`

---

## 7. Discussion

### 7.1 What the framework buys

Three relativities, not one. Identity depends on the intervention class (P1), on the input
distribution (P3), and — through P5 — on the **reference family** against which an unknown is
compared. Each is measurable, and the third is the one most often left implicit.

### 7.2 Limitations

- **Single domain.** Everything here is single-source shortest paths. Whether `|S_min|` behaves
  similarly for sorting, matching, or dynamic programming is open. `[TODO: §5.3 of
  `SCOPE_ENLARGED_FAMILY.md` scopes an enlarged family; a second domain is the stronger move.]`
- **Small family.** Five algorithms, ten pairs. Whether `|S_min|` stays constant as `|A|` grows
  is unresolved, and two data points cannot distinguish a constant from a logarithm.
- **The Dijkstra/Prim edge** rests on one probe (§5.4).
- **One mechanistic explanation was retracted.** We previously reported that the fingerprint's
  discriminative axes and its neural-disagreement axes are orthogonal (`r = −0.233`) and
  concluded no probe redesign could help. On re-examination that correlation is over nine
  points, with an exact permutation `p = 0.5445` and a 95% interval of `[−0.777, +0.510]`, and
  under a separating-power score it is `−0.004`. **The claim is withdrawn**, and with it the
  argument that probe redesign is excluded. `[TODO: decide whether this belongs in the paper at
  all, or only in the artefact. It is honest and it is also a reviewer hook. Recommend keeping
  — a retraction we found ourselves is evidence the protocol works.]`

### 7.3 Methodology, and five leads that died

Every experiment from a fixed point onward was pre-registered: candidate list, statistic,
decision rule and outcome ladder frozen before the run, with both seed sets generated up front.
Two rules earned their place.

**Verify the manipulation before reading the measurement.** One experiment would have been
written up as a null about fingerprints; a cross-fit trajectory check showed the training
difference had barely been installed, making it a null about model capacity instead — opposite
next steps.

**Bucket on the axis the mechanism acts on, not the one that is easy to compute.** Our
strongest positive result was, for a time, wrong for this reason. Truncation cuts rounds of
relaxation, but we bucketed instances by hop eccentricity, which is cheap and intuitive and
*not* the same quantity: relaxation follows minimum-weight paths, so a graph whose every node
is two hops away can still need five rounds. The dense bucket looked homogeneous and was a
mixture of two deterministic regimes, and the headline agreement rate was a draw from that
mixture that did not survive a tenfold increase in sample size. Re-bucketing on the correct
axis turned an empirical rate into an exact claim. The general form: when a claim is
conditioned on a covariate, check that the covariate is the one the mechanism is a function
of — a monotone proxy is not enough, because it makes homogeneous-looking buckets that are
not.

**Leave no intermediate rung in the outcome ladder.** *Direction replicating while resolution
does not* is the characteristic signature of a dead lead: it is what two of our retracted
results looked like at the moment they seemed most promising, one of them with a sound
invariant-based argument behind it. Ladders with a middle rung invite that state to be written
up as partial success.

Five candidate results in this project failed replication and are recorded alongside those that
survived. The recurring cause was selecting the best of several candidate statistics and
reporting before a held-out check; at `n ≈ 10` with four candidates, the best looks strong by
chance often enough to be the default outcome, and a mechanistic story for *why* it should work
offers no protection. `[TODO: consider a short table of the five, with what each looked like
before it died. This may be the most transferable content in the paper.]`

### 7.4 Future work

`[TODO: keep short. Candidates: second domain; enlarged reference family with a
predicted-inseparable positive control; second-order/compositional response, which escapes the
one-round-Jacobian confounds we measured; a family with a state-dependent path gain, which the
residual structure of §6.3 suggests is the missing ingredient.]`

---

## Provenance

Every quantitative claim maps to a findings document and its reproducing command.

| § | Claim | Source |
|---|---|---|
| 5.1 | `|S_min| = 2`; the two probes; seed stability | `FINDINGS_PHASE_A.md` |
| 5.2 | `depth_w` definition; 137/137 vs 0/463 step; predicate table; accuracy = `P(depth_w ≤ k)` | `FINDINGS_P2_WITNESS.md` |
| 5.2 | falsified depth-3 prediction; runtime vs depth | `FINDINGS_IDENTIFIABILITY.md` §2 |
| 5.3 | `|S_min|` vs `depth_w`; impossibility below ≈ 5 | `FINDINGS_IDENTIFIABILITY.md` §3 |
| 5.4 | key-dissociation probe; manipulation failure | `FINDINGS_E6.md` |
| 5.5 | σ = 2; margin 0.146 → 0.077 | `FINDINGS_NOISE.md` |
| 6.1 | 17/17; `p = 1.000`; mechanism check | `FINDINGS_E3.md` |
| 6.1 | `d_near` 0.290 vs median 0.216; ratio 1.34 | `FINDINGS_H.md` (H2-repaired) |
| 6.2 | 3/12; 0/8; 18/66 identical, 11 cross-arm | `FINDINGS_G.md` |
| 6.3 | bit-identical embedding; `χ² ~ 10⁴`; controls | `FINDINGS_FAMILY.md` |
| 6.4 | normalisation; firing repair; 1.35 → 1.34 | `FINDINGS_E5.md`, `FINDINGS_H.md` |
| 7.2 | `r = −0.233` retraction; `p = 0.5445` | `FINDINGS_H.md` (H1) |

**Withdrawn numbers.** An earlier version of §5.2 reported that truncated-BF is
output-identical to Bellman-Ford on **11 of 12** graphs at density 0.8 and correct on **0 of
12** at density 0.2. Both are withdrawn. They were measured on hop eccentricity, where
density 0.8 looks like a homogeneous depth-2 population but is in fact a mixture of instances
needing three to five rounds; at `n = 200` the agreement rate is 70–76%, where truncated-BF is
also ~30 points less accurate and the "no accuracy metric separates them" claim fails. The
`0 of 12` does not reproduce at all (3/12 on the same draw, ~12% at `n = 200`). Keyed on
`depth_w` the claim becomes exact and is reported that way in §5.2.

**Numbers deliberately not used.** §6.1 quotes the H2-repaired distances (0.290 / 0.216) rather
than the originals (0.409 / 0.303); the ratio is 1.34 either way, and the repaired measurement
is the defensible one. `FINDINGS_E5.md` records a normalised-plus-repaired variant reading
"8/9 closed" that is **an artefact** — the mask leaves as few as six comparable slots, inflating
the reference-to-reference denominator — and it is not used anywhere in this paper.

---

## Draft status

**Written:** skeleton complete, §3–§6 substantially drafted from existing results.
**Owed:** all citations (§2 especially); Figure 1; appendix tables for the probe suite and full
separation matrix; abstract tightening; §7.4.
**Open decision:** whether §7.2's retraction stays in the paper (recommend yes) and how much of
§7.3 survives the page limit.
