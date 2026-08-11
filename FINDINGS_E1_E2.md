# FINDINGS — E1 (Dijkstra-trained discrimination) + E2 (hint-free regime)

Run: 2026-08-10, `python run_e1_e2.py`. Log `e1_e2_log.txt`, per-model records
`e1_e2_results.json`. Decision rule fixed in advance in `PREREGISTRATION_E1E2.md`; nothing
below deviates from it.

**E1: INCONCLUSIVE (training).** Not a null. No Dijkstra-trained model reached admissible
accuracy, so no fingerprint of one was ever computed.

**E2: POSITIVE, and the least tautological evidence in the project.** Hint-free models train
to hint-supervised accuracy, do *not* reproduce Bellman-Ford's trajectory, and still
fingerprint as `bellman_ford` — with **larger** margins than the hint-supervised models.

---

## E1 — the architecture cannot execute Dijkstra

### Gate 0, admission: FAIL (0/6 Dijkstra-trained models admitted)

| Group | MAE vs true SSSP | Final training loss | Admitted |
|---|---|---|---|
| `bf_hint` | 0.18 – 1.45 | 0.0023 – 0.0376 | 6/6 |
| `dj_hint` | **2.78 – 3.90** | **1.64 – 2.00** | **0/6** |
| `nohint` | 0.38 – 0.92 | 0.0001 – 0.0005 | 6/6 |

Admission threshold is 2.5 (constraint 9: tolerance must exceed the model's error floor and
stay below the edge-weight scale ~5). Every Dijkstra-trained model sits above it.

**The threshold was not relaxed to admit them, and must not be.** A model at MAE 3.5 with
`tol = 2.5` has an error floor above the measurement tolerance, so its predicates read model
noise rather than algorithmic response. Admitting it would produce a fingerprint that looks
like data and is not. This is the same class of error as `tol = 4 sigma` (constraint 6).

### Gate 1, manipulation check: the difference *was* installed, partially

The gate could not formally evaluate — it requires admitted models in both groups — but the
diagnostic is unambiguous over all 18 models. `delta = traj_MAE_bf − traj_MAE_dj`, positive
meaning the model tracks Dijkstra's trajectory:

| Group | mean `delta` | range |
|---|---|---|
| `bf_hint` | **−10.92** | [−11.11, −10.46] |
| `dj_hint` | **+3.96** | [+2.60, +5.01] |
| `nohint` | −4.84 | [−5.58, −3.00] |

Correct sign in both groups, ranges disjoint by a wide margin. Dijkstra-supervised models
genuinely follow Dijkstra's round-by-round schedule more closely than Bellman-Ford's. The
manipulation took; the accuracy did not follow.

### Diagnosis: capacity, not optimisation

Dijkstra selects the globally cheapest unsettled node each round. A dense-but-sparse
(density 0.05) min-aggregating MPNN has no global readout, so a node cannot observe whether
it is the global minimum — only whether it beats its own neighbours. The evidence that this
is a representational floor rather than undertraining:

- Training loss plateaus at **1.64–2.00 across all six seeds** — a spread of 0.36 against a
  Bellman-Ford spread of 0.035 on the same architecture. Optimisation failures are seed-
  dependent; capacity limits are not.
- The loss gap is **~100x**, not a matter of a few hundred more epochs.
- Trajectory MAE against Dijkstra is 7.6–8.4 for Dijkstra-trained models. They have learned
  *something* of the schedule (that is Gate 1) and cannot execute it accurately.

More epochs are unlikely to fix this. The fix is architectural.

### What E1 does and does not license

**Does:** the core identification claim on neural models remains **untested**. B0's "all six
models identify as `bellman_ford`" is still not a discrimination test, and this run did not
turn it into one.

**Does not:** this is not evidence against the fingerprint. No Dijkstra fingerprint was
measured. Anyone reading a null here is reading a claim the data cannot make.

**Correct next step** (already named in the pre-registration): **E3**. Install the procedural
difference *by construction* rather than by supervision — a gated model that freezes a
node's estimate once it stops changing gives Dijkstra-like commitment without requiring the
network to learn a global argmin, and `NeuralExecutor.max_rounds` already gives hard
halting. E3 tests the same claim with the capacity obstacle removed.

---

## E2 — hint-free models: the strongest neural evidence so far

Hint-free training had never been run. A 24-step unroll with loss on the final step only was
a plausible optimisation failure; it is not one.

**All 6 admitted, MAE 0.38–0.92** — squarely within the hint-supervised range (0.18–1.45).
Final-answer supervision alone suffices at this depth.

### The result

| | hint-supervised (`bf_hint`) | hint-free (`nohint`) |
|---|---|---|
| Label | 6/6 `bellman_ford` | 6/6 `bellman_ford` |
| Margin | 0.004 – 0.076 | 0.065 – 0.129 |
| Mean margin | 0.028 | **0.106** |
| Resolved (margin > 0.094) | **0 / 6** | 4 / 6 |
| Trajectory MAE vs BF | 0.27 – 1.33 | **7.89 – 10.71** |

The last two rows are the finding. Hint-free models were never shown Bellman-Ford's
trajectory and **do not reproduce it** — their round-by-round states are an order of
magnitude further from it than a hint-trained model's — yet they respond to interventions
like Bellman-Ford, and do so *more* cleanly than models explicitly supervised on it.

That is the first result here where a fingerprint match is not near-tautological. A
hint-supervised model is told the trajectory, so matching it is close to assumed. A
hint-free model was told only the answer, chose its own procedure, and the fingerprint still
identified relaxation.

### Runtime ablation sharpens this

Dropping the `term_delta` predicate and reclassifying:

- `bf_hint`: **4 of 6 flip to `bf_truncated_k3`** (margins 0.004–0.041). Their
  `bellman_ford` label leans on the termination predicate alone.
- `nohint`: **6/6 stay `bellman_ford`**, and margins *rise* to 0.081–0.157.

The hint-free identification is robust to removing a whole predicate; the hint-supervised
one is not. Consistent with hint supervision teaching the model *when to stop* more reliably
than *how to propagate*.

### Negative control: clean, and weak by construction

At `rounds = n` both reference algorithms have converged, so the hint target cannot reach a
final-answer-only loss and all six hint-free models were trained on identical data. Their
label spread is therefore pure instrument noise. Observed: **1 distinct label, 0/20 balanced
splits achieve family-perfect separation** — an empirical false-positive rate of 0.000 for
the E1 statistic under a true null.

Clean, but note what it can and cannot show: with all six models on one label, a false
split was not *available*. This bounds the false-positive rate from above; it does not
demonstrate the statistic has power. Only E1, run on models that genuinely differ, can do
that.

### Caveats, stated plainly

1. **Still not a discrimination test.** Six models all labelling `bellman_ford` is consistent
   with a fingerprint that always says `bellman_ford`. E2 raises the evidential value of the
   match; it does not establish discrimination. That is E1/E3's job and it remains open.
2. **Margins are marginal.** 0.106 mean against a 0.094 noise floor; 2 of 6 below it. Above
   the floor, not comfortably.
3. **One architecture, one graph family.**

---

## Incidental result: B0's identification is below the resolution floor

Re-running the hint-supervised condition at `rounds = 24` gives margins of **0.004–0.076 for
all six seeds — every one below the 0.094 cross-seed noise floor.** Under the pre-registered
margin rule, none of those labels is established.

B0 already carried this as a caveat (thinnest margin 0.061 vs worst cross-seed noise 0.094);
this run says the caveat is the main story for hint-supervised models, not a footnote. The
hint-free models are the only neural models in the project whose labels clear their own
noise floor.

---

## Summary

| Claim | Status after this run |
|---|---|
| Fingerprint discriminates training provenance | **Untested.** Blocked on MPNN capacity, not on the method |
| Hint-free models train at all | **Yes**, to hint-supervised accuracy |
| Hint-free models fingerprint as relaxation | **Yes**, without reproducing BF's trajectory, at margins above the noise floor |
| Hint-supervised labels are resolved | **No** — 0/6 clear the noise floor; ablation flips 4/6 |
| False-discrimination rate under a true null | 0.000, bounded above only |
