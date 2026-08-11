# FINDINGS — E3: models that genuinely differ in procedure

Run: 2026-08-10, `python run_e3.py`. Log `e3_log.txt`, records `e3_results.json`.
Decision rule fixed in advance in `PREREGISTRATION_E3.md`.

**Verdict: NULL. Both gates passed, so this one counts.**

Three architectures built to execute differently — plain relaxation, Dijkstra-like
commitment, and hard truncation — produce **17/17 models labelled `bellman_ford`**. The
mechanism check confirms the architectural differences were real and active. This is the
method's own target, and it failed on it.

**The more important finding is why.** Every neural model sits **further from its nearest
symbolic reference than two different symbolic algorithms sit from each other.** The
labels were never supported by the distance profile — in this run or in any earlier one.

---

## Gate 0 — admission: PASS

| Arm | MAE | Admitted |
|---|---|---|
| `plain` | 0.54 – 2.69 | 5/6 (seed 0 excluded at 2.69) |
| `gated` | 0.34 – 1.35 | 6/6 |
| `halt` | 0.53 – 0.96 | 6/6 |

Arms are accuracy-matched (mean MAE 0.67 / 0.75 / 0.73), as the `eps` calibration was
designed to ensure. No result here can be attributed to one arm simply being worse.

## Gate 1 — mechanism check: PASS

The architectural differences were genuinely installed and active:

- `gated`: final frozen fraction 0.66–1.00 on every model. Commitment spans **6–9 rounds**
  between 25% and 90%, starting at rounds 8–12 — i.e. per-node settling spread over time,
  well after propagation depth 5–6 has delivered values, not a synchronised global halt.
- `halt`: median firing round 2–3 against k=8, leaving 5–6 rounds of response room.
- `plain`: frozen fraction 0.00, no halt.

This is what E1 lacked and it is what makes the null interpretable. The difference was
there; the fingerprint did not see it.

## Gate 2 — discrimination: NULL

| Arm | bellman_ford | bf_truncated | dijkstra | prim | spfa |
|---|---|---|---|---|---|
| `plain` | **5** | 0 | 0 | 0 | 0 |
| `gated` | **6** | 0 | 0 | 0 | 0 |
| `halt` | **6** | 0 | 0 | 0 | 0 |

- **Primary** (`gated` vs `plain`, family level): 5/11 matching, Fisher **p = 1.000**.
- `halt` vs `plain` on `bf_truncated_k3`: 0/6 vs 0/5, **p = 1.000**.
- Three-way permutation test: 5/17 matching predicted reference, **p = 1.000**.
- `term_delta` ablation: **p = 1.000**.

Margins: `plain` 0.095–0.122, `gated` 0.038–0.125, `halt` 0.013–0.073. Eleven of
seventeen fall below the 0.094 noise floor and are flagged unresolved.

---

## The structural finding: nothing was ever identified

Distances from each arm to every symbolic reference, against the distances between the
symbolic references themselves — measured on the same probe graphs at the same tolerance.

**Neural model → nearest reference:**

| Arm | nearest | 2nd | farthest |
|---|---|---|---|
| `plain` | 0.402 | 0.507 | 0.671 |
| `gated` | 0.388 | 0.477 | 0.674 |
| `halt` | 0.371 | 0.406 | 0.716 |

**Symbolic reference → symbolic reference:**

| Pair | Distance |
|---|---|
| dijkstra ↔ spfa | 0.019 |
| dijkstra ↔ prim | 0.031 |
| prim ↔ spfa | 0.031 |
| bellman_ford ↔ bf_truncated_k3 | 0.235 |
| bellman_ford ↔ prim | 0.300 |
| **bellman_ford ↔ dijkstra** | **0.306** |
| bellman_ford ↔ spfa | 0.343 |
| bf_truncated ↔ prim | 0.367 |
| bf_truncated ↔ dijkstra | 0.417 |
| bf_truncated ↔ spfa | 0.435 |
| **mean / median** | **0.248 / 0.303** |

**Every neural model is further from Bellman-Ford (0.37–0.40) than Bellman-Ford is from
Dijkstra (0.306).** The nearest-reference classifier is reporting the least-bad of five
poor matches. `fpid/identify.py`'s own docstring names this failure exactly: *"a model
roughly equidistant from everything is not a Bellman-Ford, it is an unidentified
algorithm, and the margin between first and second place is what tells the two cases
apart."*

This reframes the whole neural line, not just E3:

- **B0's** "six seeds all identify as `bellman_ford`" — margins 0.004–0.076.
- **B1's** "all 12 label `bellman_ford`".
- **E2's** hint-free result — mean nearest distance 0.407, the best margins in the project
  at 0.106, still barely over the floor.
- **E3's** 17/17.

Every one of these is a nearest-reference label on a fingerprint that is not near the
reference. A2 already said the right thing and it was not followed through: *"Report the
**margin**, not just the label."* The margins have been 0.01–0.13 the entire time against
a between-algorithm scale of 0.25–0.44. The correct reading of the whole neural programme
is that **these MPNNs execute something that is not in the reference set**, and no amount
of nearest-reference classification will name it.

That is a real result about neural algorithmic reasoning, and it is more interesting than
the label ever was. It is also consistent with E2's other observation: hint-free models do
not reproduce Bellman-Ford's trajectory (trajectory MAE 7.9–10.7) yet were labelled
Bellman-Ford anyway.

### Confirmed on a fresh probe set, with one qualification

E3b regenerated the probe graphs from a different RNG stream. Reference-to-reference
distance there: mean 0.234, median 0.246. Neural nearest distances: `plain` **0.390**,
`halt` **0.296**.

So the effect reproduces off the probe set it was observed on, but state it at the right
strength: `plain` sits well beyond the typical distance between two *distinct* symbolic
algorithms; `halt` sits at 0.296 against a 0.246 median, which is comparable rather than
clearly beyond. The defensible claim is **"a neural model's distance to its nearest
reference is comparable to or greater than the distance between two genuinely different
algorithms"** — not the stronger form for every arm on every probe set. Note also that the
dijkstra/spfa/prim cluster sits 0.019–0.031 apart, so "further than the references are from
each other" is false for those pairs and the comparison must be made against the median or
against a named pair such as bellman_ford ↔ dijkstra (0.306 on E3's set).

---

## Candidate lead — NOT a result

`halt` models are relatively closer to the truncation reference than `plain` models are.
Using the signed quantity `d(bellman_ford) − d(bf_truncated_k3)`, more negative meaning
more Bellman-Ford-like:

| Arm | per-model values | mean |
|---|---|---|
| `plain` | −0.099, −0.099, −0.095, −0.122, −0.111 | −0.105 |
| `gated` | −0.125, −0.038, −0.113, −0.082, −0.114, −0.065 | −0.089 |
| `halt` | −0.014, −0.073, −0.044, −0.048, −0.017, −0.013 | −0.035 |

`plain` ∈ [−0.122, −0.095] and `halt` ∈ [−0.073, −0.013] are **disjoint**, separated
cleanly by a threshold near −0.084. So the fingerprint does move truncated models toward
the truncation reference — directionally right, and never enough to change the label.

**This is exactly what `overshoot` looked like before it died.** B2's `overshoot` was
perfectly monotone within a group on the first seed set, had a sound invariant argument
behind it, and collapsed on held-out seeds with the effect size sign-flipped. This
quantity was chosen *after* seeing the data, on n=11, from a set of comparisons that
includes every reference pair.

Per this project's own rule — *never report a best-of-N signal without held-out
replication; four results died this way* — it was recorded as a hypothesis with a frozen
decision rule (`PREREGISTRATION_E3B.md`) and tested rather than reported.

### E3b — the lead FAILED to replicate. That is now five.

`run_e3b.py`, 2026-08-11, log `e3b_log.txt`. Fresh seeds (500–505, 600–605) and a fresh
probe stream (`default_rng(7)`), threshold frozen at τ = −0.084 and not refitted.

| | E3 (fitted) | E3b (held out) |
|---|---|---|
| `plain` range | [−0.122, −0.095] | [−0.179, −0.062] |
| `halt` range | [−0.073, −0.013] | [−0.129, +0.019] |
| Separation | **disjoint** | **overlapping by 0.067** |

Frozen rule: **9/11 correct, Fisher p = 0.0801.** Above the pre-registered 0.05, ranges
overlapping. Gate 0 and Gate 1 both passed (5/6 and 6/6 admitted; halt firing at rounds
2–3 against k=8), so the failure is the statistic's, not the run's.

The direction survived and the resolution did not — which is the same half-alive state
`overshoot` reached in B2 before it was written up and retracted. The pre-registration
anticipated exactly this and left no intermediate rung for it to land on. **FAILED.**

One incidental: `halt` seed 601 is the **first non-`bellman_ford` label produced by any
neural model in this project** — `bf_truncated_k3`, at margin 0.019. That is a fifth of
the noise floor, so it is unresolved and reported as noise, not as a discovery.

---

## Where this leaves the project

**The neural side is finished.** E3b did not replicate.

- The method works symbolically, decisively, with a clean theory (Phase A, A2, A3).
- On neural models it has now failed its designed target under a passing mechanism check,
  and the diagnostic says the failure is not marginal: the models are not close to any
  reference in the suite.
- The honest write-up is the symbolic contribution — intervention-relative algorithm
  identity, the minimal separating set, depth-dependent identifiability, truncated-BF
  separable from BF on instances where both return identical answers — plus a
  well-characterised negative on neural transfer with a *mechanistic* account of why:
  fingerprints probe execution structure, and these networks' execution structure does
  not resemble any of the reference procedures closely enough to be matched, even when
  built to differ from each other.
