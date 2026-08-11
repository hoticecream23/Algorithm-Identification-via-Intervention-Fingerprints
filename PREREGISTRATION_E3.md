# Pre-registration — E3: models that genuinely differ in procedure

**Written before the run. Nothing below may be changed after seeing fingerprints.**

Date: 2026-08-10. Implemented by `run_e3.py`. Results go in `FINDINGS_E3.md`.

Follows `PREREGISTRATION_E1E2.md` and reuses its thresholds, gates and outcome ladder
verbatim wherever they apply, so the two runs are directly comparable.

---

## Question

This is the test the method was designed for and has never received.

B1 was negative because shallow- and deep-trained networks differ in *update-rule
generalization*, not in procedure — they run the same control flow with a rule that
degrades off distribution, and intervention fingerprints probe execution structure, so
they are blind to it by construction.

E1 then tried to install a procedural difference by supervision and could not: a
min-aggregating MPNN has no global readout, cannot learn Dijkstra's argmin over unsettled
nodes, and never reached admissible accuracy (0/6 admitted, loss plateaued at 1.64–2.00
across all six seeds).

So install the difference **architecturally**, where no global comparison is required.

**Does the fingerprint separate three models built to execute differently?**

---

## Arms

All three trained **hint-free on identical data**, so supervision is held constant and
architecture is the only manipulated variable. E2 justifies hint-free: those models reach
hint-supervised accuracy and hold the only margins in the project that clear their own
noise floor, whereas every hint-supervised model's margin sits below it.

| Arm | Architecture | Train rounds | Deploy `max_rounds` | Seeds | Predicted |
|---|---|---|---|---|---|
| `plain` | `MPNN` | 24 | `None` | 0–5 | `relaxation` / `bellman_ford` |
| `gated` | `GatedMPNN(eps=0.002)` | 24 | `None` | 300–305 | `settling` / `dijkstra` |
| `halt` | `MPNN` | 8 | 8 | 400–405 | `relaxation` / `bf_truncated_k3` |

`gated` freezes a node's estimate permanently once it is *reached* (below the sentinel)
and has moved less than `eps` in a round. This is Dijkstra's defining property —
commitment — obtained from a purely local rule, which is precisely why it sidesteps E1's
capacity block. The reached condition is required: without it every node still sitting at
`UNREACHED` freezes on round 1, having not changed because nothing has arrived yet.

`halt` is trained and deployed for `k = 8` rounds. Since probe graphs have depth 5–6 it
still gets the right answer, reproducing A3's situation exactly — truncation that is
correct in-distribution and detectable anyway, because limited *runtime* is observable
even when limited *propagation depth* is not.

Everything else — n=24, density 0.05, depth >= 5 filter, 24 train graphs, 8 probe graphs,
1500 epochs, budget 48, `tol` 2.5, `stability` 0.8, `UNREACHED` 80, references =
5 `ALGORITHMS` under `finite_init` — is unchanged from E1/E2.

### Calibration of `eps` (done before this document was finalised, blind to fingerprints)

`eps` was chosen on **commitment timing and accuracy only**. No fingerprint was computed
at any point during calibration. Single seed, 1500 epochs:

| eps | MAE | 50% committed | 90% committed |
|---|---|---|---|
| 0.002 | **0.708** | 14 | 19 |
| 0.005 | 1.261 | 11 | 16 |
| 0.01 | 1.326 | 14 | 19 |
| 0.02 | 1.420 | 11 | 14 |
| 0.05 | 1.996 | **2** | — |

Chose **0.002** on two criteria fixed in advance of looking:

1. **Accuracy must not be confounded with architecture.** 0.708 sits inside the plain
   hint-free range (0.38–0.92) measured in E2, so `gated` and `plain` are accuracy-matched
   and no fingerprint difference can be attributed to one arm simply being worse. (A2 and
   B0 both say identity is decoupled from accuracy, but matching them removes the question.)
2. **Commitment must follow propagation, not precede it.** At `eps = 0.05` the model
   commits at round 2, before information has travelled the graph's depth of 5–6 — it
   freezes on garbage, and is a global halt wearing a gate's clothes, i.e. a second copy
   of the `halt` arm. At 0.002 commitment runs from round 14 to 19, well after arrival.

---

## Gates, in order. A failed gate stops the interpretation.

### Gate 0 — admission

Held-out MAE against true SSSP < 2.5 (constraint 9), graded at each arm's own training
horizon. Fewer than 5/6 admitted in any arm ⇒ **INCONCLUSIVE (training)**.

Note the halt arm is graded at row `k`, not the last row: `evaluate` compares against row
`k` of the trajectory, so writing the true answer only into the final row would silently
grade an 8-round model against Bellman-Ford's round-8 state. Those nearly agree here,
which is exactly why it would go unnoticed.

### Gate 1 — mechanism check

E1's lesson: verify the manipulation was installed **before** reading any fingerprint,
because "the difference was never there" and "the difference was there and the fingerprint
missed it" have opposite next steps. For E3 the manipulation is architectural, so:

- `gated`: final frozen fraction > 0.50 for **every** model; commitment spans **>= 3
  rounds** between 25% and 90%; and commitment **starts after round 5**. The span
  requirement is what distinguishes per-node settling from a synchronised global halt —
  without it the arm is the `halt` arm in disguise.
- `halt`: median firing round **< k = 8** on probe graphs, so an intervention lands with
  response room left. Firing after termination measures nothing (constraint 2).
- `plain`: no gate, no halt; frozen fraction reported as 0 by construction.

Failure ⇒ **INCONCLUSIVE (mechanism)**.

### Gate 2 — discrimination

Classification against the 5 symbolic references at `stability = 0.8`.

**PRIMARY: `gated` vs `plain`, family level.** Families as in E1: `settling` =
{dijkstra, prim}, `relaxation` = {bellman_ford, bf_truncated_k3, spfa}. Fisher exact,
two-sided. Family level is primary for the same reason as in E1 — the dijkstra/prim
distance is ~0.02 against a 0.094 noise floor, so an exact-label primary would demand
resolution the instrument does not have.

This contrast is primary because it is the one E1 was built to deliver and could not.

**SECONDARY, all fixed here:**

1. `halt` vs `plain` on the exact label `bf_truncated_k3`, Fisher exact two-sided.
2. Three-way exact-label permutation test: count models whose label equals their arm's
   predicted reference, permute arm labels across all admitted models, **20 000
   permutations, `default_rng(12345)`**, p = (hits + 1)/(perms + 1).
3. Full 3 x 5 confusion matrix.
4. Per-model margins, with margin < 0.094 flagged `unresolved`. The primary statistic is
   computed over all admitted models regardless of margin, so no selection occurs; a
   resolved-only sensitivity analysis is reported as secondary.
5. `term_delta`-dropped ablation of the primary contrast. Runtime is a legitimate
   observable (A3), so this decomposes rather than discounts.
6. Per-arm MAE reported next to every label, so any within-arm correlation between
   accuracy and label is visible rather than assumed absent.

### Outcome ladder (identical to E1's)

| Verdict | Condition |
|---|---|
| **STRONG** | primary Fisher p < 0.05, direction as predicted, **and** every correctly-labelled model has margin > 0.094 |
| **WEAK** | p < 0.20 in the predicted direction, or p < 0.05 with margins below the floor. Suggestive, **not established**; requires an independent seed set before any claim |
| **NULL** | anything else |

With 6v6 admitted, p < 0.05 requires >= 11/12 correct.

---

## What each outcome means

- **STRONG.** The method works on its actual target: models that genuinely differ in
  procedure are separated by causal response alone, with no weight inspection. B1's
  negative is then specifically about the proxy it chose, not about the method, and the
  neural side of the project has a positive result to stand on.
- **NULL, Gate 1 passed.** Fingerprints fail to separate architectures *built* to differ in
  procedure. That is the method's own target, so the neural side is finished and the
  write-up is the symbolic contribution plus a well-characterised negative.
- **INCONCLUSIVE (mechanism).** The gate did not install what it was meant to. Says nothing
  about the method; fix the mechanism and re-run this same rule.

A WEAK verdict is not written up as a positive result. The only registered response to
WEAK is replication on an independent seed set.
