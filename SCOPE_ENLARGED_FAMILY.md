# Scope: enlarging the reference family (ROUTES §5.3 + §7.2)

**Status: a scope, not a pre-registration and not a result.** It fixes the design, the costs and
the honest limits so a decision can be made before any code exists. If it is approved, the
frozen-decision-rule sections below become `PREREGISTRATION_I.md` unchanged, and nothing is
re-decided after seeing output.

Written 2026-08-23, after G and H closed the top two items of `ROUTES_TO_POSITIVE.md`.

---

## Why this, and why now

Two motivations, and only the first is about the science.

**1. It is the question a reviewer asks first.** The paper's headline symbolic claim is *"two
interventions separate all five algorithms at depth ≥ 7."* Five is a small family, and the
obvious objection is that two probes suffice because there is barely anything to separate.
Right now that objection cannot be answered — no experiment in the repo varies `|A|`.

**2. It is a third relativity, and it is on-message.** The paper argues identity is well-posed
only relative to an intervention class `I` and an input distribution `D`. Enlarging `A` shows the
*reference family* is a third parameter of the same kind: the same network, the same probes, the
same graphs, and a different answer because the comparison set changed. `THEORY.md` Proposition 5
already prescribes exactly this move — *"enlarge `A` until some member is close, or accept that
the family is incomplete"* — and it has never been done.

Cost is low: numpy only, no CUDA, no training, seconds to minutes per run. The expensive part is
implementing five executors correctly, not running them.

---

## §5.3 — the five additions

Requirements each candidate must meet: steppable in the existing round-based harness, exposing
the same `(d, pi)` observable, exactly specifiable (deterministic given a seed), and
*procedurally* distinct rather than a cosmetic variant.

| # | name | what makes it procedurally new | risk |
|---|---|---|---|
| 1 | `delta_stepping` | Bucket-ordered relaxation with light/heavy edge phases. Sits strictly between Bellman-Ford (no ordering) and Dijkstra (total ordering) — a *partial* order over the frontier, which no current reference has. | Δ choice is a free parameter; fix Δ = mean edge weight from the graph, computed from ground truth so every algorithm still receives an identical instance. |
| 2 | `bf_async` | Gauss–Seidel Bellman-Ford: relaxes in place, so a value can travel **more than one hop per round**. Plain BF's one-hop-per-round property is load-bearing for the `depth` predicate; this is the direct test of whether the suite observes update *order*. | Edge order must be fixed per seed and recorded, or the executor is not deterministic. |
| 3 | `floyd_warshall_row` | Orders updates by *intermediate vertex* `k`, not by frontier. Its source row converges in an order unrelated to hop distance. | `pi` semantics differ from an SSSP predecessor; define it explicitly as the last `k` that improved the entry, and document that `pi_churn` is therefore not comparable in the same way. |
| 4 | `bf_early_stop` | Identical to `bellman_ford` except it halts when a round makes no change, instead of running `n-1` rounds. Differs **only** in termination. | None. This is the cheapest and most surgical of the five. |
| 5 | `johnson_sssp` | Reweighting pass, then Dijkstra. **On non-negative weights the reweighting is a no-op, so this should be behaviourally identical to `dijkstra`.** | None — and see below, its *predicted non-separation* is the point. |

### `johnson_sssp` is the control, and it is the most important of the five

Every separation experiment so far has only ever been able to fail by finding *too little*
structure. None has had a pair that **must come back merged**. Johnson supplies one for free:
under `I` and any `D` with non-negative weights it executes Dijkstra's schedule exactly, so

> **NC-J.** `johnson_sssp` and `dijkstra` must be **unseparated** at stability 0.8.

If the suite separates them, it is manufacturing distinctions from implementation noise, and
every separation result in the project — including the size-2 minimal set — needs re-examining
before anything else here is read. This gate runs first and blocks the rest.

It also has a pleasant secondary use: it is a witness for `≡_{I,D}` being a non-trivial
equivalence relation on a family where the two members are *written differently*, which
`THEORY.md` currently demonstrates only for Dijkstra/SPFA on shallow graphs.

### Predictions, recorded now

Written before implementation so they can be wrong in public:

- `bf_early_stop` vs `bellman_ford` — separated by `term_delta` **and nothing else**. If any
  other predicate separates them, that predicate is contaminated by runtime.
- `bf_async` vs `bellman_ford` — separated by `depth`, because multi-hop-per-round propagation
  is exactly what `depth` measures.
- `delta_stepping` — separable from both Dijkstra and Bellman-Ford, and *closer to each of them
  than they are to each other*. It is an interpolant, and the distance matrix should show it.
- `floyd_warshall_row` — separated from everything, on `depth` and `latency`. Least confident of
  the five.

---

## §7.2 — what the enlarged family can and cannot say about `|I_min|`

`THEORY.md` open question 2: *empirically 2 interventions separate 5 algorithms at depth ≥ 7. Is
there a family requiring `Ω(log |A|)`, or does a constant always suffice?*

### The honest limitation, stated first

**Two data points cannot distinguish a constant from a logarithm.** `log₂ 5 ≈ 2.3` and
`log₂ 10 ≈ 3.3`; an observed move from `|I_min| = 2` to `3` is equally consistent with
"logarithmic" and with "constant plus one". Resolving the growth *rate* would need `|A|` ≈ 5, 10,
20, 40 — forty exactly-specified, procedurally-distinct SSSP algorithms, which do not exist and
are not worth manufacturing.

So this experiment **must not be written up as answering open question 2.** What it can do:

1. **Falsify "a constant always suffices"** — a sharp result if `|I_min|` jumps to 5+ at
   `|A| = 10`, and a real constraint on the theory.
2. **Deliver the enlarged separation matrix**, which is the actual product: 45 pairs instead of
   10, showing which probes are load-bearing and which are redundant across a family twice the
   size.
3. **Test whether the existing 9-probe suite has a blind spot.** If some new pair is
   *unseparated*, the suite is incomplete for the enlarged family — a genuine finding, and one
   that bears directly on the paper's central claim, since it would show the minimal separating
   set is a property of `(I, D, A)` jointly.

### Frozen decision rule

Run at depth ≥ 7 (where the size-2 result was established) and at depth 5, on 25 graphs, at
stability 0.8, seed 11 for the report and seed 0 for development.

| outcome | criterion | reading |
|---|---|---|
| **STABLE** | `|I_min|` ≤ 3 at `|A| = 10`, depth ≥ 7 | Consistent with a constant. Does not prove one. Report as a second data point and say so. |
| **GROWS** | `|I_min|` ≥ 5 | "A constant suffices" is falsified for this suite. Report the growth and the pairs driving it. |
| **INCOMPLETE** | any pair unseparated by the full 9-probe suite | The suite has a blind spot for the enlarged family. This outranks the other two: report it as the headline and name the pair. |

`|I_min|` = 4 is the deliberate gap and is reported as **INDETERMINATE** — a two-rung ladder with
nothing in the middle, per this project's standing rule.

### Mandatory regression

Adding algorithms changes a number quoted in three documents. Re-run and re-quote:

- `run_phase_a.py` on the original five — the size-2 set must reproduce exactly, or the
  enlargement has broken something unrelated.
- `run_identifiability.py` — the depth boundary (2 at depth ≥ 7, 3 at depth 5, impossible below
  ~4) must be re-measured on the enlarged family and reported separately from the original.
- Update `FINDINGS_PHASE_A.md`, `FINDINGS_IDENTIFIABILITY.md` and `THEORY.md` §7 with **both**
  numbers, framed as "for this family" rather than replacing the old value. The family-relativity
  is the point, not an inconvenience.

---

## Cost

| item | estimate |
|---|---|
| Implement 5 executors + unit tests against ground-truth SSSP | 4–6 h |
| `run_enlarged.py` (NC-J gate, separation matrix, `|I_min|` sweep, regression) | 1–2 h |
| Runtime | seconds to a few minutes, numpy only, no CUDA |
| Doc updates across 3 findings files + THEORY | 1 h |

The correctness tests are not optional and are most of the cost: an executor that computes
slightly wrong distances produces a confident, wrong separation matrix, and nothing downstream
would catch it. Each new executor must reproduce `true_sssp` exactly on 100 random graphs before
it is allowed into `ALGORITHMS`.

---

## What would make this not worth doing

Stated so the decision is real:

- If the paper is going out on a short deadline, this is the wrong use of the time. The symbolic
  results already stand on their own, and §8.2 (closed-form equivalence classes, algebra only)
  buys more per hour for a theory section.
- If `johnson_sssp` fails NC-J, everything else here stops and the finding is about the existing
  method, not about family size.
- It does **not** touch the neural side. Proposition 5 says enlarging `A` is the prescribed
  repair for the vacuity gap — but these five additions are all *hard-relaxation* algorithms, and
  the networks are not near any hard-relaxation reference. Expect the vacuity gap to be
  unchanged; ROUTES §5.2 (damped/softmin/discounted references) is the enlargement aimed at that,
  and it is a different experiment.
