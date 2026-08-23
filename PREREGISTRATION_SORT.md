# Pre-registration — Experiment S: does the framework transfer to a second domain?

Written 2026-08-23, **before** any domain code exists. Frozen at commit time.

Reproduce:

```bash
python run_sorting.py
```

---

## 1. The question

Every experiment in this project is single-source shortest paths. The framework's central
object — behavioural equivalence `≡_{I,D}` relative to an intervention class `I` and input
distribution `D` — is stated generally, but has only ever been instantiated once. A reader can
grant every result and still hold that the nine probes were reverse-engineered from relaxation
semantics and that `≡_{I,D}` is one engineered notion of distinguishability rather than a
reusable one.

**Question.** Instantiate the framework on comparison sorting. Does the machinery downstream of
the intervention class — predicate extraction, pairwise stability gating, Hamming separation,
minimal separating sets — work unchanged, and does it separate a family of sorting algorithms
that are output-identical?

This is the experiment `PAPER.md` §7.1 promises and does not run.

## 2. What is and is not being changed

**Changed:** the domain. New executors, a new target-selection rule, a new intervention class,
and a new input distribution.

**Not changed, and this is the point:** the stability threshold (0.8), the predicate set (the
same seven names), the separation and minimal-separating-set code, the aggregation rule, the
progress-relative firing rule, and the two-rung reporting discipline. If any of these needs
tuning to make sorting work, that is itself a finding and will be reported as one, not
absorbed silently.

**Structural claim under test.** The following are asserted to be domain-agnostic and will be
reused *without modification*: `State`, `d_equal`, `converged_mask`, `settle_round`,
`settle_round_absolute`, `firing_round`, `was_converged`, `Fingerprint`, `aggregate`, and all
of `separation.py` and `identify.py`. Exactly one line of `fingerprint.probe` is domain-coupled
— the `depth` predicate's `graph.hop_distances(site)` — and it will be abstracted behind a
`Domain` seam with the SSSP behaviour as the default, so that every existing runner's output
stays byte-identical. **A byte-identical re-run of `run_p2_witness.py` and `run_phase_a.py`
after the refactor is a release condition**, checked before any sorting number is read.

## 3. The domain

### 3.1 Observable state

`State(d, pi, round)` is reused verbatim, with:

| field | SSSP | sorting |
|---|---|---|
| `d` | distance estimate per node | the working array |
| `pi` | predecessor: where this value came from | provenance: the original index of the element now in this slot |

This is a reuse of the abstraction, not of the name. `pi` in SSSP is a provenance pointer, and
the sorting instantiation is a provenance pointer. Stating it explicitly because the reuse is
what makes the timing and convergence code transfer for free, and a reader is entitled to check
that it is not a coincidence of variable naming.

**Output** for the purposes of the witness claim is the final `d` (the sorted array) alone.
`pi` is state, not output — an unstable sort producing a different provenance trace on
duplicate keys is not producing a different output.

### 3.2 The family

Five algorithms, matching the size of the SSSP family:

| name | one round = |
|---|---|
| `insertion_sort` | place one element into the sorted prefix |
| `selection_sort` | select the suffix minimum, swap it into place |
| `bubble_sort` | one full left-to-right pass of adjacent swaps |
| `cocktail_sort` | one bidirectional pass |
| `bubble_truncated_k3` | as `bubble_sort`, halting after k = 3 passes |

Ten pairs. `bubble_truncated_k3` is the direct analogue of `bf_truncated_k3` and is included
for the same reason: it is the shortcut an in-distribution-perfect learner would acquire.

A sixth executor, `binary_insertion_sort`, is defined but **excluded from the family** and used
only as gate G4 (§6).

### 3.3 Depth, and the ruler

A bubble pass moves any element at most one position **left**, but can carry it arbitrarily far
**right**. So the number of passes to sortedness is the maximum *leftward* displacement:

```
leftward_depth(a) = max_i max(0, i - rank(a[i]))
```

where `rank` is the 0-indexed position in the stably-sorted array.

**Lemma (verified before writing this document, 20 000 random arrays with duplicates, 0
mismatches):** `bubble_sort` reaches the sorted array after exactly `leftward_depth(a)` passes.
Therefore

> `bubble_truncated_k` produces the correct output **iff** `leftward_depth(a) <= k`.

This is an exact step function, not an empirical rate — the same status `FINDINGS_P2_WITNESS.md`
established for the SSSP witness.

**The wrong ruler, with the opposite sign.** The naive depth measure is maximum *absolute*
displacement `max_i |i - rank(a[i])|`. It is `>= leftward_depth` always, and **strictly greater
in 31% of random arrays** (6178/20000, same check). In SSSP the wrong ruler (hop eccentricity)
*understated* the depth relaxation sees, so gating on it was conservative and cost only
headroom. Here the naive ruler *overstates* depth, so gating on it would admit arrays that are
too shallow — producing false separations rather than weak ones. The measure used throughout is
`leftward_depth`. This is recorded here so it cannot be adopted after seeing results.

### 3.4 Input distribution

Arrays of length `n = 16` (primary) and `n = 24` (secondary, pre-declared here). Values drawn
from `0 .. n//2` so that **duplicates are guaranteed** — required by the null control (§5) and
by the stability observation. Instances are rejection-sampled on `leftward_depth`.

Two distributions, both declared now:

| name | `leftward_depth` | purpose |
|---|---|---|
| DEEP | `>= 6` | primary separation measurement |
| WITNESS | unrestricted, bucketed at `k = 3` | the step-function claim |

Budget: `n + 2` rounds, padded, so every algorithm terminates inside it.

## 4. The intervention class

Ten probes. Both SSSP design rules are carried over: targets are computed from the array and
its sorted ground truth **only**, never from the running executor's state; and firing is at the
midpoint of each algorithm's own active life (`firing_round`, `absolute=True` — the H2 repair,
since these are the conditions under which it was shown correct).

Targets: `early` (a position in the first third), `late` (last third), `min_pos` (position of
the global minimum), `max_left` (the position attaining `leftward_depth`).

| probe | edit | what it is for |
|---|---|---|
| `swap_equal_pair` | swap two positions holding **equal** values | **null control** (§5) |
| `swap_adjacent_early` | swap `(early, early+1)` | permutation poke in a finished region |
| `swap_adjacent_late` | swap `(late, late+1)` | permutation poke in an unfinished region |
| `corrupt_down_early` | set `a[early]` below the global min | finished-region write: does the algorithm revisit? |
| `corrupt_up_early` | set `a[early]` above the global max | as above, opposite sign |
| `corrupt_down_late` | set `a[late]` below the global min | unfinished-region control for the two above |
| `reverse_block_early` | reverse 3 adjacent positions at `early` | larger structural poke |
| `move_min_to_end` | rotate the global minimum to the last slot | **depth probe** — forces `n-1` of leftward travel |
| `relocate_min_within_suffix` | move the suffix minimum to a different suffix position | **key dissociation** (below) |
| `inject_extreme_late` | set `a[late]` to a value outside the input range | eligibility rather than value |

**Key dissociation.** `relocate_min_within_suffix` is the sorting analogue of E6's
`dissociate_keys`. Selection sort's decision is *which value is the suffix minimum*; bubble
sort's decisions are *adjacent pairwise orders*. Moving the minimum to a different position in
the unprocessed suffix leaves the selected **value** invariant while changing its **index** and
changing many adjacent orders. Selection sort's chosen key does not move; bubble sort's entire
pass does.

**Predicted-silent probes, kept deliberately.** `corrupt_down_early` and `corrupt_up_early`
land in a region `selection_sort` and `insertion_sort` never rewrite, so they are expected to
be unrepairable for those two and repairable for `bubble_sort`/`cocktail_sort`, which rescan.
This mirrors the SSSP deletion probes: near-silent for some algorithms by construction, and
sharp for exactly that reason.

## 5. The null control

`swap_equal_pair` swaps two positions holding equal values. The value array is therefore
**elementwise unchanged**, so no comparison outcome anywhere in any of the five algorithms can
differ, and the entire `d` trajectory is provably bit-identical to the control.

Declared in advance: the six `d`-derived predicates (`recovery`, `latency`, `spread1`, `depth`,
`term_delta`, `poked_settled`) **must** take their no-response values for all five algorithms.
Any deviation means the harness is responding to something other than the algorithm and voids
the run.

`pi_churn` is **exempt and reported separately**: swapping two equal elements changes provenance
by construction, so a non-null `pi_churn` here is expected and is a stability observation (does
this algorithm preserve the relative order of equal keys), not a response. Declaring the
exemption now so it cannot be claimed as a result later.

## 6. Gates — checked and reported before the primary statistic is read

- **G1 — depth manipulation landed.** DEEP achieves mean `leftward_depth >= 6.0` with minimum
  `>= 6`, measured with `leftward_depth`, not the naive ruler.
- **G2 — firing lands mid-run.** Every algorithm fires within `[0.35, 0.60]` of its own active
  life, the interval inherited from `PREREGISTRATION_H.md`.
- **G3 — the witness holds.** On WITNESS instances:
  - **G3a (unconditional):** `insertion`, `selection`, `bubble`, `cocktail` produce the
    identical final array on **100%** of instances. Anything below 100% is a bug, not a rate.
  - **G3b (conditional):** `bubble_truncated_k3` matches `bubble` on **100%** of instances with
    `leftward_depth <= 3` and on **0%** of instances with `leftward_depth > 3`.
- **G4 — trivial-equivalence control.** `binary_insertion_sort` and `insertion_sort` have
  identical round-by-round `(d, pi)` trajectories by construction. They **must** come back
  **unseparated** under the full intervention class. If the instrument separates them it is
  manufacturing signal from noise and the run is void.
- **G5 — refactor is inert.** `run_p2_witness.py` and `run_phase_a.py` produce byte-identical
  output before and after the `Domain` seam is added.

All five must pass. A gate failure means fix and re-run on a **fresh** seed set, with the
failure documented in the findings.

## 7. Primary ladder — two rungs, no middle

Statistic: `unseparated_pairs` over the ten pairs, at stability threshold 0.8, on the DEEP
distribution, computed on the REPORT seed set.

- **TRANSFERS** — `unseparated_pairs` is empty, and `minimal_separating_set` returns a subset.
- **DOES NOT TRANSFER** — any pair is unseparated under the full class.

There is deliberately no middle rung. "Eight of ten pairs separated" is DOES NOT TRANSFER. The
project's own methodological rule — *direction replicating while resolution does not is the
signature of a dead lead* — exists because intermediate rungs were where four dead results
landed. `|S_min|` is reported for interest either way, but it is **not** the ladder.

**No post-hoc probe addition.** The intervention class in §4 is fixed. If the primary comes
back DOES NOT TRANSFER, that is the result and it will be reported as one. Adding an eleventh
probe until the pairs separate is precisely the failure mode `PREREGISTRATION_E3.md` was
written to prevent. A genuine *bug* (a probe that never fires, an executor that terminates
early) may be fixed, and the fix must be re-run on a fresh seed set and documented.

## 8. Secondary — the identifiability boundary on the sorting depth axis

Pre-declared, reported regardless of the primary outcome: sweep `leftward_depth` from 1 upward
and report `|S_min|` and the unseparated pairs per bucket. This is the §5.3 analogue.

Two falsifiable predictions, recorded now:

1. `|S_min|` is **non-increasing** in `leftward_depth`. More depth cannot make the instrument
   need more probes.
2. There is a `d* >= 1` below which separation is **impossible** for the full class.

Prediction 1 is the informative one; prediction 2 is nearly guaranteed at depth 0 (a sorted
input means no algorithm does any work) and is included only to locate `d*`.

## 9. Seed sets

Both generated up front, in one pass, as `PREREGISTRATION_E3.md` requires.

- **SMOKE** (`seed = 11`): used only to confirm the code runs and the gates pass. No number
  from this set is reported as a result.
- **REPORT** (`seed = 23`): every reported number.

Sample sizes: 120 instances for fingerprints (matching `run_p2_witness.py --fp-graphs 120`),
400 for the witness step function.

## 10. What this cannot settle

- It is **symbolic only**. No sorting network is trained, so it says nothing about whether the
  vacuity result of §6 is domain-specific. That would be a third experiment.
- Two domains is not "the framework generalises". It is one instantiation more than one, which
  converts §7.1 from a promise into a data point.
- The five sorting algorithms share more structure with each other than Bellman-Ford and
  Dijkstra do (all are in-place, comparison-based, `O(n^2)`). If they separate, that is a
  *harder* case than SSSP on the axis of family similarity and an *easier* one on the axis of
  algorithmic diversity. Neither direction should be over-read.
