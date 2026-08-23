"""The sorting domain — a second instantiation of the intervention framework.

Everything in this module is domain-specific by construction: the executors, the
target-selection rule, the intervention class, and the depth measure. Nothing
downstream of it is. `fingerprint.aggregate`, all of `separation.py` and
`identify.py`, and the whole timing layer (`settle_round`, `firing_round`,
`converged_mask`, `was_converged`) are reused unmodified, which is the claim
`PREREGISTRATION_SORT.md` §2 exists to test.

The state mapping is `d` -> the working array, `pi` -> provenance (which original
index sits in each slot). That is a reuse of the abstraction rather than of the
name: `pi` in SSSP is a predecessor pointer, i.e. a record of where the value in
this slot came from, and so is this. It is what makes the timing code transfer
for free.

One asymmetry with SSSP is worth stating because it drove the intervention design.
In SSSP `d` is *derived* -- a corrupted estimate can be recomputed from a node's
neighbours, so a poke can wash out. In sorting the array holds *primitive* data
that is only ever permuted, never recomputed, so a corrupted value can never be
restored. Recovery is therefore defined against a reference run started from the
post-poke array (`fingerprint.probe` already does exactly this), which asks the
right question: did the mid-run poke leave the algorithm where it would have been
had the input always been that way. Permutation probes carry most of the
discriminative load; value probes still separate, but on the spatial predicates
(`depth`, `spread1`) rather than on `recovery`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithms import Executor, d_equal
from .fingerprint import Domain
from .interventions import Intervention

# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


@dataclass
class ArrayInstance:
    """The problem instance. Mirrors `Graph`'s role: the thing `probe` copies."""

    a: np.ndarray

    def copy(self) -> ArrayInstance:
        return ArrayInstance(self.a.copy())

    def __len__(self) -> int:
        return int(self.a.shape[0])


def rank_of(a: np.ndarray) -> np.ndarray:
    """0-indexed position of each element in the stably-sorted array."""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=int)
    r[order] = np.arange(len(a))
    return r


def leftward_depth(a: np.ndarray) -> int:
    """Rounds of bubble-sort passes needed to reach the sorted array.

    A bubble pass moves any element at most one position *left*, but can carry it
    arbitrarily far *right*: once the scan picks an element up it keeps carrying it
    until it meets something larger. So the binding constraint is the largest
    leftward journey any element has to make, and

        bubble_sort settles after exactly leftward_depth(a) passes

    which makes `bubble_truncated_k` correct iff `leftward_depth(a) <= k`. Verified
    on 20 000 random arrays with duplicates, 0 mismatches, before
    `PREREGISTRATION_SORT.md` was written.

    This is the sorting counterpart of `graphs.weighted_depth`, and it exists for
    the same reason: the obvious measure is the wrong one. See `naive_depth`.
    """
    a = np.asarray(a)
    return int(max(0, np.max(np.arange(len(a)) - rank_of(a))))


def naive_depth(a: np.ndarray) -> int:
    """Maximum *absolute* displacement — the wrong ruler, kept so it can be reported.

    Always `>= leftward_depth`, and strictly greater in ~31% of random arrays.
    Note the sign relative to SSSP: there the naive measure (hop eccentricity)
    *understated* the depth relaxation sees, so gating on it was conservative and
    cost only headroom. Here it *overstates* depth, so gating on it would admit
    arrays too shallow to separate — false positives rather than weak negatives.
    """
    a = np.asarray(a)
    return int(np.max(np.abs(np.arange(len(a)) - rank_of(a))))


def random_array(rng: np.random.Generator, n: int) -> np.ndarray:
    """Values from `0 .. n//2`, so duplicates are guaranteed by pigeonhole.

    The null control (`swap_equal_pair`) needs a pair of equal values to exist in
    every instance, and drawing `n` values from fewer than `n` buckets guarantees
    one without rejection sampling.
    """
    return rng.integers(0, max(2, n // 2), size=n).astype(float)


def arrays_in_band(
    rng: np.random.Generator,
    count: int,
    n: int,
    lo: int,
    hi: int,
    max_tries: int = 400000,
    min_count: int | None = None,
) -> list[ArrayInstance]:
    """Instances with `lo <= leftward_depth <= hi`, across the whole depth range.

    Neither construction alone covers the range. Uniform random arrays cannot supply
    the shallow end -- at `n = 16` essentially every draw has `leftward_depth >= 9`,
    which left the truncation witness with no `<= k` bucket to test against and the
    depth sweep with no low rungs. Disordering a sorted array with adjacent
    transpositions raises depth gradually and covers the shallow end, but reaches
    the deep end only rarely.

    So each draw picks one of the two at random and the result is rejection-sampled
    on the measured depth. Every band is then reachable, and every band is filled by
    both constructions rather than each band having its own generator -- which would
    confound depth with construction method.

    `min_count` allows a shortfall for bands near the extremes of the depth range
    (`leftward_depth = n-1` needs the minimum at the last position, which is rare).
    The caller is told the realised count so the sample size can be reported.
    """
    out: list[ArrayInstance] = []
    tries = 0
    while len(out) < count and tries < max_tries:
        tries += 1
        if rng.random() < 0.5:
            a = random_array(rng, n)
        else:
            a = np.sort(random_array(rng, n))
            for _ in range(int(rng.integers(0, 4 * n + 1))):
                i = int(rng.integers(0, n - 1))
                a[i], a[i + 1] = a[i + 1], a[i]
        d = leftward_depth(a)
        if lo <= d <= hi:
            out.append(ArrayInstance(a))
    floor = count if min_count is None else min_count
    if len(out) < floor:
        raise RuntimeError(
            f"only built {len(out)}/{count} arrays with leftward_depth in "
            f"[{lo}, {hi}] at n={n} after {tries} tries (floor {floor})"
        )
    return out


def deep_arrays(
    rng: np.random.Generator,
    count: int,
    n: int,
    min_depth: int,
    max_depth: int | None = None,
    max_tries: int = 200000,
) -> list[ArrayInstance]:
    """Rejection-sample instances whose `leftward_depth` sits in the target band."""
    out: list[ArrayInstance] = []
    tries = 0
    while len(out) < count and tries < max_tries:
        tries += 1
        a = random_array(rng, n)
        d = leftward_depth(a)
        if d < min_depth:
            continue
        if max_depth is not None and d > max_depth:
            continue
        out.append(ArrayInstance(a))
    if len(out) < count:
        raise RuntimeError(
            f"only drew {len(out)}/{count} arrays with leftward_depth in "
            f"[{min_depth}, {max_depth}] at n={n} after {tries} tries"
        )
    return out


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


class SortExecutor(Executor):
    """Base class. `d` is the working array, `pi` the provenance of each slot.

    Deliberately does not call `Executor.__init__`: there is no source node and no
    `d_init` sentinel here. Everything else in the base class -- `snapshot`, `step`,
    `run` and the hook protocol the interventions fire through -- is inherited
    unchanged.
    """

    name = "?"

    def __init__(self, inst: ArrayInstance, source: int = 0):
        self.arr = inst
        self.graph = inst  # alias: `probe` and the base class both say `graph`
        self.source = source
        self.n = len(inst)
        self.d = inst.a.astype(float).copy()
        self.pi = np.arange(self.n)
        self.round = 0
        self.finished = False

    def _swap(self, i: int, j: int) -> None:
        self.d[i], self.d[j] = self.d[j], self.d[i]
        self.pi[i], self.pi[j] = self.pi[j], self.pi[i]


class InsertionSort(SortExecutor):
    """One round places one element into the sorted prefix.

    The prefix is rewritten by the shift, but an element already placed is never
    re-examined against anything to its right, so damage to the prefix is not
    repaired -- the property `corrupt_*_early` and `swap_adjacent_early` probe.
    """

    name = "insertion_sort"

    def _step(self) -> None:
        i = self.round + 1
        if i >= self.n:
            self.finished = True
            return
        key, kp = self.d[i], self.pi[i]
        j = i - 1
        while j >= 0 and self.d[j] > key:
            self.d[j + 1] = self.d[j]
            self.pi[j + 1] = self.pi[j]
            j -= 1
        self.d[j + 1] = key
        self.pi[j + 1] = kp
        if i >= self.n - 1:
            self.finished = True


class InsertionSortClone(SortExecutor):
    """`InsertionSort`'s state update by a different code path — gate G4.

    Finds the same insertion point by counting the contiguous run of larger
    elements and moving the block in one slice assignment, rather than shifting
    element by element. Being the same function of the state, it is identical to
    `InsertionSort` under *every* intervention, not merely on sorted prefixes --
    which is what `binary_insertion_sort` turned out not to be, and why this class
    exists. See `PREREGISTRATION_SORT.md` §11.
    """

    name = "insertion_clone"

    def _step(self) -> None:
        i = self.round + 1
        if i >= self.n:
            self.finished = True
            return
        key, kp = self.d[i], self.pi[i]
        run = 0
        while run < i and self.d[i - 1 - run] > key:
            run += 1
        pos = i - run
        if run:
            self.d[pos + 1 : i + 1] = self.d[pos:i]
            self.pi[pos + 1 : i + 1] = self.pi[pos:i]
        self.d[pos] = key
        self.pi[pos] = kp
        if i >= self.n - 1:
            self.finished = True


class BinaryInsertionSort(SortExecutor):
    """Insertion sort that locates the insertion point by binary search.

    Identical to `InsertionSort` on every unpoked input -- same output, same
    round-by-round state -- because a binary search over a sorted prefix finds the
    same slot a linear scan does. They are equivalent under the empty intervention
    class in the strongest available sense.

    They are *not* identical once a probe scrambles the prefix: the linear scan
    stops at the first blocker it meets, while the binary search indexes into a
    region whose ordering assumption no longer holds. Measured, 1654/20000 rounds
    differ on a scrambled prefix against 0/20000 on a sorted one. That is why this
    is a declared secondary rather than the equivalence control.
    """

    name = "binary_insertion_sort"

    def _step(self) -> None:
        i = self.round + 1
        if i >= self.n:
            self.finished = True
            return
        key, kp = self.d[i], self.pi[i]
        lo, hi = 0, i
        while lo < hi:
            mid = (lo + hi) // 2
            if self.d[mid] > key:
                hi = mid
            else:
                lo = mid + 1
        self.d[lo + 1 : i + 1] = self.d[lo:i]
        self.pi[lo + 1 : i + 1] = self.pi[lo:i]
        self.d[lo] = key
        self.pi[lo] = kp
        if i >= self.n - 1:
            self.finished = True


class SelectionSort(SortExecutor):
    """One round settles one slot: pick the suffix minimum, swap it into place.

    The prefix is *final* -- never read, never written again. This is the sorting
    counterpart of Dijkstra's settled set, and it is what the `_early` probes are
    aimed at: damage behind the frontier is permanent here and repairable for the
    pass-based sorts.
    """

    name = "selection_sort"

    def _step(self) -> None:
        r = self.round
        if r >= self.n - 1:
            self.finished = True
            return
        j = r + int(np.argmin(self.d[r:]))
        if j != r:
            self._swap(r, j)
        if r + 1 >= self.n - 1:
            self.finished = True


class BubbleSort(SortExecutor):
    """One round is one full left-to-right pass of adjacent swaps.

    Rescans the whole array every pass, so unlike the two insertion-style sorts it
    *can* repair damage behind its own frontier -- but only at one position per
    pass, which is what `leftward_depth` counts and what the `depth` predicate
    reads off.
    """

    name = "bubble_sort"
    max_passes: int | None = None

    def _step(self) -> None:
        changed = False
        for i in range(self.n - 1):
            if self.d[i] > self.d[i + 1]:
                self._swap(i, i + 1)
                changed = True
        if not changed:
            self.finished = True  # converged
            return
        if self.max_passes is not None and self.round + 1 >= self.max_passes:
            self.finished = True  # truncated, converged or not


class TruncatedBubbleSort(BubbleSort):
    """Bubble sort halted after k passes, converged or not.

    The sorting analogue of `TruncatedBellmanFord`, and included for the same
    reason: it is the shortcut a learner trained only on shallow instances would
    acquire, and it is indistinguishable from the real thing in output space on
    exactly the instances with `leftward_depth <= k`.
    """

    name = "bubble_truncated"

    def __init__(self, inst: ArrayInstance, source: int = 0, k: int = 3):
        super().__init__(inst, source)
        self.max_passes = k
        self.name = f"bubble_truncated_k{k}"


class CocktailSort(SortExecutor):
    """Bidirectional bubble sort: alternate forward and backward passes.

    Same output as bubble sort always, but a backward pass carries an element
    arbitrarily far *left*, so its depth characteristic differs -- the point of
    including it. `bubble_sort` and `cocktail_sort` are the pair most likely to be
    hard to separate, since they share their comparison rule entirely.
    """

    name = "cocktail_sort"

    def _step(self) -> None:
        changed = False
        sweep = (
            range(self.n - 1) if self.round % 2 == 0 else range(self.n - 2, -1, -1)
        )
        for i in sweep:
            if self.d[i] > self.d[i + 1]:
                self._swap(i, i + 1)
                changed = True
        if not changed:
            self.finished = True


SORT_ALGORITHMS = {
    "insertion_sort": InsertionSort,
    "selection_sort": SelectionSort,
    "bubble_sort": BubbleSort,
    "cocktail_sort": CocktailSort,
    "bubble_truncated_k3": lambda inst, s=0: TruncatedBubbleSort(inst, s, k=3),
}

# Not in the family. `insertion_clone` is gate G4; `binary_insertion_sort` is the
# declared secondary of PREREGISTRATION_SORT.md §11.
SORT_EXTRAS = {
    "insertion_clone": InsertionSortClone,
    "binary_insertion_sort": BinaryInsertionSort,
}

# The output-identical set: these four sort correctly on every input, so any
# difference the fingerprint reports between them is a difference in procedure
# alone. `bubble_truncated_k3` is excluded -- it is output-identical only on the
# shallow bucket, which is the conditional witness rather than the unconditional one.
OUTPUT_IDENTICAL = ("insertion_sort", "selection_sort", "bubble_sort", "cocktail_sort")


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


@dataclass
class ArrayTargets:
    """Landmarks computed from the instance and its sorted ground truth only.

    Never from a running executor's state -- the same rule as `interventions.Targets`,
    and for the same reason: if each algorithm poked a site chosen from its own
    progress they would receive different pokes and their fingerprints would differ
    for the wrong reason.
    """

    sorted_a: np.ndarray
    rank: np.ndarray
    lo: float
    hi: float
    early: int
    late: int
    min_pos: int
    suffix_min_pos: int
    max_left: int
    eq_i: int
    eq_j: int
    dup_value: float | None


def compute_array_targets(inst: ArrayInstance, source: int = 0) -> ArrayTargets:
    a = inst.a
    n = len(a)
    rank = rank_of(a)

    # A quarter of the way in, not the midpoint: the `_early` probes need a slot
    # that every algorithm has finished with well before its own firing round, and
    # the midpoint is exactly where firing happens.
    early = int(np.clip(n // 4, 1, n - 4))
    late = int(np.clip((3 * n) // 4, early + 2, n - 2))

    min_pos = int(np.argmin(a))
    suffix_min_pos = late + int(np.argmin(a[late:]))
    max_left = int(np.argmax(np.arange(n) - rank))

    # The most frequent value, so the null control has the most slots to choose
    # from and is least likely to degenerate. Ties broken by value for determinism.
    vals, counts = np.unique(a, return_counts=True)
    dup_value = float(vals[int(np.argmax(counts))]) if counts.max() >= 2 else None

    eq_i = eq_j = -1
    if dup_value is not None:
        where = np.nonzero(a == dup_value)[0]
        eq_i, eq_j = int(where[0]), int(where[-1])

    return ArrayTargets(
        sorted_a=np.sort(a),
        rank=rank,
        lo=float(a.min()),
        hi=float(a.max()),
        early=early,
        late=late,
        min_pos=min_pos,
        suffix_min_pos=suffix_min_pos,
        max_left=max_left,
        eq_i=eq_i,
        eq_j=eq_j,
        dup_value=dup_value,
    )


# ---------------------------------------------------------------------------
# the intervention class
# ---------------------------------------------------------------------------


def _poke(fn):
    """Wrap an edit so the post-poke array is written back to the instance.

    `probe` builds its reference run from the instance object the executor holds, so
    the instance has to carry the poked array for "where would you be if the input
    had always been this" to mean anything. In SSSP the graph-editing probes mutate
    `ex.graph` directly and get this for free; here every probe edits live state, so
    the sync is explicit.
    """

    def apply(ex: SortExecutor, t: ArrayTargets) -> None:
        fn(ex, t)
        ex.arr.a = ex.d.copy()

    return apply


def _swap_duplicate_value(ex: SortExecutor, t: ArrayTargets) -> None:
    """Null control: swap the first and last slots currently holding `dup_value`.

    Both slots hold the same value by construction, so the array is elementwise
    unchanged, no comparison outcome anywhere in any algorithm can differ, and the
    entire `d` trajectory is provably bit-identical to the control. `pi_churn` is
    exempt by pre-registration -- provenance changes by construction here -- and
    reports whether the algorithm preserves the relative order of equal keys.

    **Targeted by value, not by position, and this is the point.** The first version
    of this probe swapped two *positions* that held equal values in the input. By
    firing time the algorithm has permuted the array, those slots hold different
    values, and the "null" control was measured making a real edit -- `spread1` and
    `depth` were non-null for all seven executors.

    The underlying asymmetry with SSSP: a node keeps its identity under relaxation,
    so a position is a stable handle there. A sorting algorithm permutes positions,
    so the only handle it does not move is the value. Every instance has a duplicated
    value (`random_array` draws `n` values from fewer than `n` buckets), and a
    permutation cannot destroy a duplicate, so this probe is well-defined for every
    algorithm on every instance.
    """
    if t.dup_value is None:
        return
    idx = np.nonzero(ex.d == t.dup_value)[0]
    if idx.size >= 2:
        ex._swap(int(idx[0]), int(idx[-1]))


def _swap_adjacent(where: str):
    def fn(ex: SortExecutor, t: ArrayTargets) -> None:
        i = t.early if where == "early" else t.late
        if i + 1 < ex.n:
            ex._swap(i, i + 1)

    return fn


def _corrupt(where: str, direction: str):
    def fn(ex: SortExecutor, t: ArrayTargets) -> None:
        i = t.early if where == "early" else t.late
        ex.d[i] = t.lo - 1.0 if direction == "down" else t.hi + 1.0

    return fn


def _reverse_block_early(ex: SortExecutor, t: ArrayTargets) -> None:
    i = t.early
    j = min(i + 3, ex.n)
    ex.d[i:j] = ex.d[i:j][::-1]
    ex.pi[i:j] = ex.pi[i:j][::-1]


def _move_min_to_end(ex: SortExecutor, t: ArrayTargets) -> None:
    """Depth probe: force a maximal leftward journey.

    Takes whatever now sits at the instance's original minimum position and rotates
    it to the last slot, shifting the tail left by one. Bubble sort must then carry
    it back one position per pass, so the `depth` predicate reads the truncation off
    directly; selection sort's suffix scan finds it in a single round.

    Whether that slot still holds the minimum at firing time is part of the
    algorithm's response, not part of the poke -- the same stance
    `interventions._dissociate_keys` takes, and required by the targets rule.
    """
    i = t.min_pos
    if i >= ex.n - 1:
        return
    v, p = ex.d[i], ex.pi[i]
    ex.d[i : ex.n - 1] = ex.d[i + 1 : ex.n]
    ex.pi[i : ex.n - 1] = ex.pi[i + 1 : ex.n]
    ex.d[ex.n - 1] = v
    ex.pi[ex.n - 1] = p


def _relocate_min_within_suffix(ex: SortExecutor, t: ArrayTargets) -> None:
    """Key dissociation — the sorting counterpart of E6's `dissociate_keys`.

    Selection sort's decision is *which value* is the suffix minimum; bubble sort's
    decisions are *adjacent pairwise orders*. Swapping the suffix minimum with
    another suffix slot leaves the selected value invariant while moving its index
    and changing many adjacent orders. Selection's key does not move; bubble's whole
    pass does.
    """
    i, j = t.suffix_min_pos, ex.n - 1
    if i != j:
        ex._swap(i, j)


def _inject_extreme_late(ex: SortExecutor, t: ArrayTargets) -> None:
    ex.d[t.late] = t.hi + 10.0


SORT_INTERVENTIONS: list[Intervention] = [
    Intervention(
        "swap_duplicate_value",
        lambda t: max(t.eq_i, 0),
        _poke(_swap_duplicate_value),
        "null control: targeted by value, so the array is elementwise unchanged",
    ),
    Intervention(
        "swap_adjacent_early",
        lambda t: t.early,
        _poke(_swap_adjacent("early")),
        "permutation poke behind the frontier: does the algorithm rescan?",
    ),
    Intervention(
        "swap_adjacent_late",
        lambda t: t.late,
        _poke(_swap_adjacent("late")),
        "permutation poke ahead of the frontier: everyone should absorb it",
    ),
    Intervention(
        "corrupt_down_early",
        lambda t: t.early,
        _poke(_corrupt("early", "down")),
        "value below the range, behind the frontier",
    ),
    Intervention(
        "corrupt_up_early",
        lambda t: t.early,
        _poke(_corrupt("early", "up")),
        "value above the range, behind the frontier",
    ),
    Intervention(
        "corrupt_down_late",
        lambda t: t.late,
        _poke(_corrupt("late", "down")),
        "unfinished-region control for the two `_early` corruptions",
    ),
    Intervention(
        "reverse_block_early",
        lambda t: t.early,
        _poke(_reverse_block_early),
        "larger structural poke behind the frontier",
    ),
    Intervention(
        "move_min_to_end",
        lambda t: t.min_pos,
        _poke(_move_min_to_end),
        "depth probe: forces a maximal leftward journey",
    ),
    Intervention(
        "relocate_min_within_suffix",
        lambda t: t.suffix_min_pos,
        _poke(_relocate_min_within_suffix),
        "key dissociation: selection's chosen value is invariant, bubble's pass is not",
    ),
    Intervention(
        "inject_extreme_late",
        lambda t: t.late,
        _poke(_inject_extreme_late),
        "out-of-range value ahead of the frontier",
    ),
]

SORT_INTERVENTION_NAMES = [iv.name for iv in SORT_INTERVENTIONS]


def sort_site_distance(inst: ArrayInstance, site: int) -> np.ndarray:
    """Index distance — the `depth` predicate's metric in this domain.

    Stands where `graph.hop_distances(site)` stands in SSSP, and means the same
    thing: how far from the poked site the response was ever visible. Adjacency in
    an array is index adjacency, so this is the metric the algorithms' own locality
    is defined in.
    """
    return np.abs(np.arange(len(inst)) - site).astype(float)


def sort_at_risk(pre: np.ndarray, post: np.ndarray, tol: float) -> np.ndarray:
    """Slots the poke itself moved.

    SSSP probes touch a single node, so `probe` marking the target site is enough.
    Sorting probes are permutations: a swap moves two slots and `move_min_to_end`
    moves the whole tail. Restricting the recovery predicates to one nominal site
    would score most of the damage as untouched.
    """
    return ~d_equal(pre, post, tol)


SORT_DOMAIN = Domain(
    name="sorting",
    make_targets=compute_array_targets,
    site_distance=sort_site_distance,
    at_risk=sort_at_risk,
)
