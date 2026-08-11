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


def shared_keys(a: Fingerprint, b: Fingerprint, stability: float) -> list:
    """Predicates stable in both fingerprints, so comparable between them."""
    return sorted(a.stable_keys(stability) & b.stable_keys(stability))


def distance(a: Fingerprint, b: Fingerprint, stability: float = 0.8) -> float:
    """Fraction of jointly-stable predicates on which two fingerprints disagree."""
    keys = shared_keys(a, b, stability)
    if not keys:
        return 1.0
    return sum(1 for k in keys if a.value.get(k) != b.value.get(k)) / len(keys)


def classify(
    fp: Fingerprint,
    refs: dict[str, Fingerprint],
    stability: float = 0.8,
) -> tuple[str, dict[str, float]]:
    """Return the nearest reference algorithm and the full distance profile.

    The profile matters as much as the label: a model roughly equidistant from
    everything is not a Bellman-Ford, it is an unidentified algorithm, and the margin
    between first and second place is what tells the two cases apart.
    """
    dists = {name: distance(fp, ref, stability) for name, ref in refs.items()}
    return min(dists, key=lambda k: dists[k]), dists


def margin(dists: dict[str, float]) -> float:
    """Gap between nearest and second-nearest reference. 0 means ambiguous."""
    ordered = sorted(dists.values())
    return (ordered[1] - ordered[0]) if len(ordered) > 1 else 1.0
