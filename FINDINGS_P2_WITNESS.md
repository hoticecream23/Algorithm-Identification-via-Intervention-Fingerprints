# P2 witness — output-identical executors, separated by causal response

Reproduce:

```bash
python run_p2_witness.py                 # ~10 min, numpy only
python run_p2_witness.py --graphs 400    # tighter intervals
```

This supersedes §1 of `FINDINGS_IDENTIFIABILITY.md`, which stated the same result on the
wrong axis and at a sample size too small to expose the error. The claim survives in a
**stronger** form: what was an empirical agreement rate is now a guarantee.

---

## 1. What was wrong

The original measurement bucketed graphs by **hop depth** (`hop_distances().max()`) and
reported that at density 0.8 (hop depth 2.0) truncated-BF with `k = 3` returns output
identical to full Bellman-Ford on **11 of 12** graphs.

Hop depth is the wrong ruler. Relaxation propagates along minimum-**weight** paths, which
may use strictly more edges than the minimum-**hop** path. At density 0.8 essentially every
node is two hops from the source, yet full Bellman-Ford needs **three to five rounds** to
settle. The density-0.8 sample was therefore not a homogeneous population of shallow graphs
— it was a mixture of two deterministic regimes, and `11/12` was a lucky draw from that
mixture.

At `n = 200` the same measurement gives:

| seed | density | BF == truncated |
|---|---|---|
| 0 | 0.8 | 139/200 (70%) |
| 7 | 0.8 | 152/200 (76%) |

At a ~72% agreement rate the truncated executor is also ~30 points less accurate, so the
load-bearing sentence — *no accuracy metric can tell them apart* — is **false** as the
result was originally stated. This was not a small correction to a real finding; at the
sample size a reviewer would use, the witness failed.

A second number does not reproduce either. The original doc reports truncated-BF correct on
**0 of 12** at density 0.2; the same draw gives 3/12, and `n = 200` gives ~12%. The collapse
is real, but it is not to zero.

## 2. The right axis

`fpid.graphs.weighted_depth(graph, source)` — the number of rounds of synchronous relaxation
needed to reach ground truth, equivalently the largest number of edges on any shortest-weight
path from the source. Because `BellmanFord` here is synchronous (it reads `d` from the
previous round, so information travels exactly one hop per round), this is *exactly* the
round at which full BF settles, and `bf_truncated_k` is correct **iff** `weighted_depth <= k`.

The two measures diverge substantially on these generators — weighted depth runs 1–2 rounds
above hop eccentricity, and exceeds it on 57–100% of instances depending on density.

## 3. The witness, restated exactly

Pooled over five densities and 300 graphs per seed, bucketed by weighted depth (`k = 3`):

| weighted depth | n | BF == truncated | truncated correct |
|---|---|---|---|
| ≤ 3 (= k) | 137 | **137/137** | **137/137** |
| ≥ 4 | 463 | **0/463** | **0/463** |

A step function at exactly `k`, on both seeds, with no intermediate cases. Full Bellman-Ford
is correct on 600/600 throughout.

So the claim no longer rests on a sampled rate. **On instances of weighted depth ≤ k, the
two executors are output-identical with probability 1** — not on 11 of 12, but on every
instance, and provably so from the definition of `weighted_depth`. No output-space test can
distinguish them on that subpopulation, because there is nothing to distinguish.

## 4. And the fingerprint separates them anyway

Fingerprints computed over the `≤ k` bucket **only** — 120 graphs per seed, every one of
them output-identical — so any separation found here cannot be an accuracy difference
wearing a disguise.

Both seeds separate, on the same single probe, `insert_shortcut_from_source`:

| predicate | `bellman_ford` | `bf_truncated_k3` | stability (BF / trunc) | separates? |
|---|---|---|---|---|
| recovery | exact | partial | 1.00 / 0.85 | **yes** |
| term_delta | + | 0 | 0.85 / 0.99 | **yes** |
| latency | 2 | never | 0.56 / 0.85 | no — BF's mode is unstable |
| depth | 2 | 1 | 0.54 / 1.00 | no — BF's mode is unstable |
| spread1 | 1 | 1 | 1.00 / 1.00 | no — agree |
| pi_churn | many | many | 0.96 / 0.93 | no — agree |
| poked_settled | no | no | 1.00 / 1.00 | no — agree |

**The predicate table in the superseded version had three rows and should have had two.**
`latency` (2 vs never) was quoted as separating, but Bellman-Ford's modal latency holds on
only 56% of instances — below the 0.8 stability threshold — so it is excluded pairwise and
carries no identity information. `depth` fails the same way. The separation rests on
`recovery` and `term_delta`.

This is the P2 witness in the form the paper should state it: identical outputs by
construction, separated by causal response.

### Sample-size sensitivity

Does `insert_shortcut_from_source` separate, as a function of how many `≤ k` instances the
fingerprint aggregates over?

| n instances | seed 0 | seed 7 |
|---|---|---|
| 10 | separates | separates |
| 20 | **NONE** | separates |
| 40, 60, 80, 100, 120, 160 | separates | separates |

Separation is found at 15 of 16 (sample size × seed) cells. The single failure — `n = 20`,
seed 0 — is a stability flake rather than a power threshold: it separates at `n = 10` and at
every size from 40 up, so there is no monotone sample-size boundary being crossed. It is a
reminder that at small `n` the 0.8 stability gate is itself noisy, in both directions.

Quote the result at `n ≥ 40`. Do not quote it at `n = 12`, which is where the superseded
version lived — not because the separation is absent there, but because at that size a
single instance moves a modal value across the gate.

## 5. The OOD half

Truncated-BF's correctness tracks the fraction of the distribution within its budget, exactly
as the mechanism predicts:

| density | mean weighted depth | frac ≤ k | truncated correct |
|---|---|---|---|
| 0.8 | 3.20–3.33 | 68–78% | 68–78% |
| 0.4 | 3.73–3.83 | 28–30% | 28–30% |
| 0.2 | 4.23–4.43 | 10% | 10% |
| 0.1 | 4.97–5.03 | 2% | 2% |
| 0.05 | 5.65–5.78 | 0% | 0% |

`frac ≤ k` and `truncated correct` agree to the instance at every density on both seeds,
which is the same step function seen from the distribution's side. The OOD story is intact
and now has an exact mechanism: **accuracy under shift is precisely the probability mass the
truncation budget covers.**

## 6. What this changes elsewhere

- `PAPER.md` §5.2 — restated; the `11/12` and `0/12` figures are withdrawn.
- `PAPER.md` §5.3 and Figure 1 — the depth axis is hop depth and should be re-cut on
  weighted depth. The `|S_min|` result itself replicates (see `FINDINGS_IDENTIFIABILITY.md`).
- `CLAUDE.md` constraint 1 ("propagation depth ≥ 5") — the threshold is stated on the hop
  axis and needs restating on the weighted one.
- Nothing in the neural results depends on either figure.

## 7. Status

Confirmatory re-measurement with a fixed statistic, not a search: the quantities were fixed
by the claim being checked, both seed sets were generated up front, and both are reported.
No candidate selection occurred at any point.
