# Experiment S — the machinery transfers to a second domain. Its calibration does not.

Reproduce:

```bash
python run_sorting.py
```

Pre-registered in `PREREGISTRATION_SORT.md`, with amendments §11 and §12, all frozen before
the REPORT seed was run. Ladder: TRANSFERS / DOES NOT TRANSFER, no middle rung.
**Verdict: DOES NOT TRANSFER**, at both sample sizes, with all five gates passing.

---

## 1. The question

Every experiment in this project was single-source shortest paths, so `≡_{I,D}` had been
instantiated exactly once. The standing objection: the nine probes were reverse-engineered
from relaxation semantics, and the framework is one engineered notion of distinguishability
rather than a reusable one.

Experiment S instantiates the framework on comparison sorting — new executors, new targets, a
new ten-probe intervention class, a new depth measure — and asks whether everything downstream
of the intervention class still works.

## 2. The result in one line

The **machinery** transfers completely and is proven to. The **calibration** does not, and the
pre-registered ladder fails on one pair out of ten because the probe distribution was chosen by
importing an intuition from SSSP that turns out to invert.

## 3. What transferred, mechanically

`fingerprint.probe` was given a `Domain` seam carrying the three domain-coupled pieces:
targets, the metric the `depth` predicate reports in, and the at-risk rule. Everything else is
shared source: `aggregate`, the stability rule, all of `separation.py` and `identify.py`, and
the whole timing layer (`settle_round`, `settle_round_absolute`, `firing_round`,
`converged_mask`, `was_converged`).

**G5 makes this auditable rather than asserted.** `run_phase_a.py` and `run_p2_witness.py` were
run against a pre-refactor tree exported from `HEAD` and against the refactored tree, and both
produce **byte-identical output**. The seam is inert; no SSSP number moved.

The state abstraction maps `d` -> the working array and `pi` -> provenance (which original
index sits in each slot). That is a reuse of the abstraction, not the name: `pi` in SSSP is a
record of where the value in this slot came from, and so is this.

## 4. The witnesses — the strongest in the project

### 4.1 Unconditional

`insertion_sort`, `selection_sort`, `bubble_sort` and `cocktail_sort` produce the identical
output array on **400/400** instances, at both `n = 16` and `n = 24`.

This is not a rate. Comparison sorts agree in output on *every* input by definition. §5.2's SSSP
witness needs the qualifier "on instances shallow enough"; this one needs no qualifier at all.
Four algorithms, provably one function in output space, and the fingerprint separates five of
their six pairs.

### 4.2 Conditional — an exact step function at k

A bubble pass moves any element at most one position *left* but can carry it arbitrarily far
*right*, so the passes needed equal the maximum leftward displacement:

```
leftward_depth(a) = max_i max(0, i - rank(a[i]))
```

Verified on 20 000 random arrays with duplicates, 0 mismatches. Hence `bubble_truncated_k` is
correct **iff** `leftward_depth(a) <= k`:

| pass | `depth <= 3` agree with full bubble | `depth > 3` agree |
|---|---|---|
| n = 16 | **139 / 139** | **0 / 261** |
| n = 24 | **132 / 132** | **0 / 268** |

The same exact step `FINDINGS_P2_WITNESS.md` established for SSSP, reproduced in a second
domain with a different mechanism.

### 4.3 The wrong ruler exists here too, with the opposite sign

The naive depth measure is maximum *absolute* displacement. It overstates `leftward_depth` on
**60/120** instances at `n = 16` and **53/120** at `n = 24`.

Note the sign. In SSSP the naive measure (hop eccentricity) *understated* the depth relaxation
sees, so gating on it was conservative and cost only headroom — nothing was invalidated. Here
the naive measure *overstates* depth, so gating on it would admit arrays too shallow to
separate, producing false positives rather than weak negatives. Same class of error, worse
consequence. This was recorded in the pre-registration before the run.

## 5. The primary ladder — DOES NOT TRANSFER

Nine of ten pairs separate. One does not, at both sample sizes and on both distributions:

| pass | distribution | unseparated |
|---|---|---|
| n = 16 | uniform (pre-registered) | `bubble_sort` / `cocktail_sort` |
| n = 16 | mixed (§12.2 sensitivity arm) | `bubble_sort` / `cocktail_sort` |
| n = 24 | uniform | `bubble_sort` / `cocktail_sort` |
| n = 24 | mixed | `bubble_sort` / `cocktail_sort` |

The pre-registered ladder has no middle rung, so this is DOES NOT TRANSFER, not "nearly
transferred". That rung was deliberately omitted, and §6 is why.

## 6. Why the pair fails, and why the ladder is right to call it a failure

At the pre-registered distribution (`n = 16`, mean `leftward_depth` 10.4), `bubble_sort` and
`cocktail_sort` differ on **17 of 70 predicate slots — and all 17 are dropped for
instability**:

| slot | bubble | cocktail |
|---|---|---|
| `corrupt_up_early` / `spread1` | `4+` (0.87) | `1` (**0.43**) |
| `corrupt_down_early` / `latency` | `3` (**0.50**) | `1` (**0.41**) |
| `corrupt_down_late` / `poked_settled` | `yes` (0.83) | `no` (**0.60**) |
| `corrupt_up_early` / `term_delta` | `0` (**0.74**) | `+` (**0.55**) |

The direction is there. The resolution is not. That is verbatim the signature this project's
own methodology file names as **the characteristic mark of a dead lead** — the state both
`overshoot` and E3b's truncation statistic were in when they evaporated. The two-rung ladder
exists precisely so that this state has nowhere to land, and it is being honoured here rather
than argued around.

## 7. The diagnosis — the identifiability boundary runs the other way

The pair is not genuinely equivalent. It separates readily at moderate disorder and stops
separating as disorder grows:

| band (`n = 16`) | mean depth | `\|S_min\|` | unseparated |
|---|---|---|---|
| [0, 1] | 0.81 | 2 | none |
| [2, 3] | 2.42 | 2 | none |
| [4, 5] | 4.38 | **1** | none |
| [6, 8] | 7.15 | — | 1 pair |
| [9, 12] | 10.59 | — | 1 pair |
| [13, 15] | 13.47 | — | 1 pair |

At the [4, 5] band all ten pairs separate and `bubble`/`cocktail` separates on three probes.

**Mechanism.** At high disorder every sort does large, sprawling work everywhere, so each
probe's response varies from instance to instance and no slot survives the 0.8 stability gate.
At moderate disorder the responses are localised and algorithm-characteristic. Resolving power
therefore *decreases* with depth here — the exact inverse of SSSP, where deeper instances give
relaxation more rounds in which to reveal itself (`FINDINGS_DEPTH.md`).

**This is the real finding.** The DEEP distribution was pre-registered at `leftward_depth >= 6`
because in SSSP more depth means more resolving power. That intuition is domain-specific, and
importing it chose a probe distribution near this domain's *upper* resolution limit. The
framework transferred; the calibration did not.

**Do not over-read the sweep.** It is non-monotone at both sizes and the clean `n = 16` picture
does not replicate at `n = 24`, where [0,1] leaves 8 pairs unseparated (nearly-sorted arrays
give every algorithm almost no work to do), [4,8] separates, [9,12] does not, and [13,23] does.
What replicates is narrow and is all that should be quoted: **the pre-registered deep band
fails at both sizes, and the mid-range band [4, 8] separates at both.**

## 8. Pre-registered predictions that failed

Both recorded before the run, both reported as negatives.

- **§8 prediction 1 — `|S_min|` non-increasing in `leftward_depth`: FALSIFIED.** It is
  non-monotone at both sample sizes, and separation degrades at the deep end rather than
  improving.
- **§11 secondary — linear vs binary insertion sort: NOT SEPARATED.** The prediction was that
  the intervention class would separate two algorithms equivalent under the *empty* class
  (identical output on every input, and identical round-by-round state on every unpoked input).
  It does not, at either sample size. An earlier 25-instance check appeared to separate them;
  at 120 it does not, which is a textbook instance of this project's own rule against reporting
  a best-of-N signal. The negative is the result.

## 9. Gates

All five pass at both sample sizes.

- **G1 depth.** Mean `leftward_depth` 10.4 (`n=16`) and 17.5 (`n=24`), minimum 6 and 10.
- **G2 firing.** 0.47–0.49 of active life for every executor. `bubble_truncated_k3` reports
  0.333 and is exempt on the arithmetic ground declared in §12.4: its active life is 3 rounds
  and `firing_round` returns an integer, so no round exists at 0.5.
- **G3 witnesses.** §4 above.
- **G4 clone control.** `insertion_sort` and `insertion_clone` — the same state update by a
  different code path — come back **unseparated**, at both sizes. The instrument is not
  manufacturing signal from noise.
- **G-null.** See §10.

## 10. The null control, and a defect in its pre-registered criterion

`swap_duplicate_value` swaps two slots holding the same value, so the array is provably
unchanged elementwise and the `d` trajectory must be bit-identical to the control.

**Trajectory level: 120/120 identical for all seven executors, at both sample sizes.** This is
the direct check and it passes universally.

**Predicate level: fails for some executors**, and the pre-registered criterion is at fault
rather than the instrument. Two mechanisms, both identified and neither a response to the poke:

1. `probe`'s reference arm restarts the executor on the post-poke array. For a *budget-limited*
   executor that hands `bubble_truncated_k3` a fresh allowance of `k` passes which the
   intervened run has already spent, so the reference differs from the control on 40/40
   instances while the intervened trajectory is identical on all of them.
2. `latency` is defined as rounds until the intervened run matches the reference's *final*
   state. Under a null poke fired mid-run it therefore measures how long the algorithm still
   needs to finalise that slot — a property of the algorithm and the firing round, not of the
   edit. It reads `0` at `n = 16` and `4+` at `n = 24` for the insertion family for exactly
   this reason.

The criterion has not been quietly relaxed. It is reported as failing, with both mechanisms
named, and the trajectory-level check is offered as the sound version. A future revision should
define the null criterion as "equal to this executor's own control value", not as a fixed
per-predicate constant.

## 11. Two bugs found before the REPORT run, both recorded in the pre-registration

Kept here because both are transfer lessons rather than typos.

**Positions are not stable handles.** The null control originally swapped two *positions* that
held equal values in the input. A sorting algorithm permutes positions, so by firing time those
slots held different values and the "null" control was making a real edit — `spread1` and
`depth` were non-null for all seven executors. In SSSP a node keeps its identity under
relaxation, so a position is a stable handle there and this failure mode cannot arise. **When
transferring an intervention class, targets must name what the algorithm cannot permute** — the
value, not the slot.

**G4's original premise was false.** The gate first proposed `binary_insertion_sort` versus
`insertion_sort` as the trivial-equivalence control, on the grounds that they are identical by
construction. They are identical only while the sorted prefix really is sorted, and four of the
ten probes scramble that prefix deliberately: 0/20000 insertion rounds differ on a sorted
prefix, 1654/20000 on a scrambled one. The gate was replaced by a genuine clone and the pair
was promoted to a declared secondary — the one that then failed (§8).

## 12. What this does not settle

- **Symbolic only.** No sorting network was trained, so this says nothing about whether §6's
  vacuity result is domain-specific. That would be a third experiment.
- **Two domains is not "the framework generalises".** It is one instantiation more than one,
  and it converts `PAPER.md` §7.1 from a promise into a data point — a data point with a
  qualification attached.
- The five sorting algorithms share more structure with each other than Bellman-Ford and
  Dijkstra do. That makes this a harder case on family similarity and an easier one on
  algorithmic diversity; neither direction should be over-read.

## 13. Consequences

- `PAPER.md` §7.1's promised second domain is discharged, with the honest result: the machinery
  is domain-general and mechanically shown to be; the probe distribution must be recalibrated
  per domain, and the SSSP heuristic "deeper is better" inverts here.
- A new methodological rule for §7.3: **a probe distribution chosen by analogy to another
  domain is an untested assumption, not an inherited one.** It should be swept, not imported.
- `run_sorting.py` should be re-run with the DEEP band set to [4, 8] as an *exploratory*
  follow-up. That number was chosen after seeing this data and carries no pre-registered
  status; any claim resting on it needs its own frozen design and a fresh seed.
