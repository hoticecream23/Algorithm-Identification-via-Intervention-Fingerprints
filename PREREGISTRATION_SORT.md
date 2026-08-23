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

---

## 11. Amendment, 2026-08-23 — G4's premise was false

Recorded **before** any sorting executor was written and before any instance was drawn. No
experimental data has been seen at the time of this amendment.

**The error.** §6 G4 asserted that `binary_insertion_sort` and `insertion_sort` "have identical
round-by-round `(d, pi)` trajectories by construction". That is true only while the sorted
prefix really is sorted. Linear insertion scans left and stops at the first element that does
not exceed the key; binary insertion indexes into the prefix assuming it is ordered. Several
probes in §4 (`swap_adjacent_early`, `reverse_block_early`, `corrupt_*_early`) deliberately
scramble that prefix, at which point the two disagree.

Measured, 20 000 random insertion rounds:

| prefix | rounds where the two differ |
|---|---|
| sorted | **0 / 20000** |
| scrambled | **1654 / 20000** |

So G4 as written would have failed for a correct reason and voided a valid run.

**Replacement gate.** G4 becomes a *clone control*: `insertion_sort` versus
`insertion_sort_clone`, a second class computing the identical state update by a different code
path (an explicit slice rotation rather than a shift loop). Being the same function of the
state, it is identical under every intervention, not merely on sorted prefixes. It tests what
the original gate was meant to test — that the harness does not manufacture differences between
two runs of the same procedure — and it tests it unconditionally. **The clone must come back
unseparated. If it does not, the run is void.**

**The bug is promoted to a secondary observation.** `binary_insertion_sort` is now a declared
secondary, reported regardless of the primary outcome and outside the primary ladder:

> Linear and binary insertion sort agree in output on every input, and agree in their entire
> round-by-round state trajectory on every unpoked input. They are equivalent under the empty
> intervention class. **Prediction (recorded now): the intervention class separates them.**

If that prediction holds it is the sharpest witness in the project. The two algorithms are
indistinguishable not merely in output but in their whole observable trajectory on the input
distribution — `≡_{∅,D}` holds in the strongest available sense — and the fingerprint still
tells them apart. That is `≡_{I,D}` doing exactly what the framework claims it does, on a pair
where every weaker notion of behavioural equivalence says "same".

The prediction is falsifiable and may fail: the divergence requires a probe to scramble the
prefix *before* the insertion point of a later round, which not every instance will satisfy.
A negative here is reported as a negative.

---

## 12. Amendment, 2026-08-23 — instance generation, and which distribution the ladder keys on

Recorded after SMOKE-seed diagnostics and **before** any REPORT-seed number was produced.
What was seen on the smoke seed is stated explicitly below, because it is the reason this
amendment exists and the reader should be able to discount it.

### 12.1 Two bugs fixed

**The null control was not null.** `swap_equal_pair` swapped two *positions* holding equal
values **in the input**. By firing time the algorithm has permuted the array, so those slots no
longer hold equal values, and the control was making a real edit: `spread1` and `depth` were
non-null for all seven executors on the smoke seed.

The underlying asymmetry with SSSP is worth recording as a transfer lesson. A node keeps its
identity under relaxation, so a *position* is a stable handle in that domain. A sorting
algorithm permutes positions, so the only handle it does not move is the *value*. The probe is
renamed `swap_duplicate_value` and swaps the first and last slots currently holding a
pre-selected duplicated value. Both slots hold the same value by construction for every
algorithm on every instance, so the array is provably unchanged elementwise.

**Shallow instances were unreachable.** At `n = 16` essentially every uniform random array has
`leftward_depth >= 9`, so G3b had an empty `<= k` bucket (0 instances) and could not be
evaluated, and the §8 sweep had no low rungs. `arrays_in_band` adds a second construction --
disordering a sorted array with adjacent transpositions -- and mixes the two 50/50 before
rejection-sampling on depth, so every band is reachable and no band is served by a single
construction.

### 12.2 Which distribution the primary ladder keys on

§3.4 specifies the DEEP distribution as values "drawn from `0 .. n//2`" and "rejection-sampled
on `leftward_depth`". That is i.i.d. uniform arrays with a depth filter. The transposition
construction is a different process and was introduced by this amendment, not by §3.4.

**The primary ladder therefore keys on the uniform distribution**, which is what §3.4 says. The
mixed distribution is reported alongside it as a declared sensitivity arm.

This is decided on the pre-registered text, not on the outcome — and it is decided *against*
the arm that looked better on the smoke seed. Stated plainly so it can be checked: on the SMOKE
seed the full family separated under the transposition-built distribution and did **not**
separate under the mixed one, with `bubble_sort` / `cocktail_sort` as the single failing pair.
Keying the ladder on the uniform arm is therefore the conservative choice, not the flattering
one.

### 12.3 One amendment round only

This is the last amendment. The REPORT seed is run once after this document is committed, and
whatever it returns is the result. No further generator, band, probe or threshold change will
be made on the basis of a REPORT-seed number.

### 12.4 G2's arithmetic edge case

`firing_round` returns an integer round, so an executor whose active life is 3 rounds can fire
only at 0.33 or 0.67 of it -- no round exists at 0.5. `bubble_truncated_k3` has a mean active
life of 3.0 and therefore cannot satisfy `[0.35, 0.60]` for arithmetic reasons rather than
methodological ones. Executors with mean active life below 5 rounds are reported with their
fraction and exempted from G2 on that ground. Declared here rather than discovered later.

### 12.5 G-null is reported at two levels

The null control provably cannot change `d`, so the direct check is whether the intervened
trajectory is bit-identical to the control. That is reported alongside the pre-registered
predicate-level criterion.

They are expected to disagree for `bubble_truncated_k3`, for a reason that is not a response:
`probe`'s reference arm restarts the executor on the post-poke array, which hands a
budget-limited algorithm a fresh allowance of k passes that the intervened run has already
spent. Measured on the smoke seed: intervened == control on 120/120 for **all seven**
executors, while the reference differs from the control on 40/40 for the truncated one alone.
Both numbers are reported. The predicate-level criterion is not quietly relaxed — it is
reported as failing for that executor, with the mechanism named.
