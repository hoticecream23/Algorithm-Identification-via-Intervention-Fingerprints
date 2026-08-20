"""Nearest-reference identification.

Given an unlabelled model's fingerprint, name its algorithm. This is the Phase C
classifier; it is built and calibrated here against noisy symbolic algorithms, where
ground truth is known exactly.

Distance is Hamming over predicates, not a norm over response vectors. The predicates
are categorical, so a Hamming count is the only distance that does not require
inventing a scale for each one.

Stability is applied *pairwise*. Intersecting stability across every reference throws
away precisely the predicates that discriminate a single pair -- under that rule
Dijkstra and SPFA lose the one probe that separates them, collapse to identical
vectors, and get classified by tie-break.
"""

from __future__ import annotations

from .fingerprint import Fingerprint

NULL_CONTROL = "delete_nontree_edge"


def shared_keys(a: Fingerprint, b: Fingerprint, stability: float) -> list:
    """Predicates stable in both fingerprints, so comparable between them."""
    return sorted(a.stable_keys(stability) & b.stable_keys(stability))


def informative(fp: Fingerprint, key: tuple[str, str]) -> bool:
    """Whether `fp`'s response at `key` differs from its own response to the null control.

    A response that matches what the same executor does when nothing informative
    happened (`delete_nontree_edge`, an edge no shortest path uses) is not a response
    to the poke -- it is the executor's baseline jumpiness. `key` for the null control
    itself is always informative, since it has nothing to be compared against.
    """
    intervention, predicate = key
    if intervention == NULL_CONTROL:
        return True
    baseline = (NULL_CONTROL, predicate)
    if baseline not in fp.value:
        return True
    return fp.value[key] != fp.value[baseline]


def keys_normalised(a: Fingerprint, b: Fingerprint, stability: float) -> list:
    """Jointly-stable keys, with slots uninformative for *both* fingerprints dropped.

    Masking must be pairwise, not per-fingerprint. Under `corrupt_node_up`,
    Bellman-Ford's own response equals its null-control response (`recovery=exact`
    either way), so a per-fingerprint mask would drop that slot and destroy the
    BF/Dijkstra separation Phase A rests on -- Dijkstra is informative there
    (`recovery=none`), so the pairwise rule keeps it. The null control's own slots
    are excluded: it is the reference, not a measurement.
    """
    return [
        k
        for k in shared_keys(a, b, stability)
        if k[0] != NULL_CONTROL and (informative(a, k) or informative(b, k))
    ]


def distance(
    a: Fingerprint,
    b: Fingerprint,
    stability: float = 0.8,
    normalise: bool = False,
) -> float:
    """Fraction of jointly-stable predicates on which two fingerprints disagree.

    `normalise=True` additionally drops slots uninformative for both fingerprints
    (see `keys_normalised`). Off by default so every existing reported number stays
    reproducible by the command at the top of its findings document.
    """
    keys = keys_normalised(a, b, stability) if normalise else shared_keys(a, b, stability)
    if not keys:
        return 1.0
    return sum(1 for k in keys if a.value.get(k) != b.value.get(k)) / len(keys)


def classify(
    fp: Fingerprint,
    refs: dict[str, Fingerprint],
    stability: float = 0.8,
    normalise: bool = False,
) -> tuple[str, dict[str, float]]:
    """Return the nearest reference algorithm and the full distance profile.

    The profile matters as much as the label: a model roughly equidistant from
    everything is not a Bellman-Ford, it is an unidentified algorithm, and the margin
    between first and second place is what tells the two cases apart.
    """
    dists = {
        name: distance(fp, ref, stability, normalise) for name, ref in refs.items()
    }
    return min(dists, key=lambda k: dists[k]), dists


def margin(dists: dict[str, float]) -> float:
    """Gap between nearest and second-nearest reference. 0 means ambiguous."""
    ordered = sorted(dists.values())
    return (ordered[1] - ordered[0]) if len(ordered) > 1 else 1.0
