"""Separation analysis and minimal separating sets.

Formalises the claim that algorithm identity is only defined *relative to an
intervention class*: two algorithms are behaviourally equivalent under a set S of
interventions when no stable predicate under S tells them apart. Bellman-Ford and
Dijkstra are equivalent under the empty set on positive weights -- they return the
same answer -- and separate only once S is rich enough.

The object of interest is then the smallest S that separates a whole family.
"""

from __future__ import annotations

from itertools import combinations

from .fingerprint import PREDICATES, Fingerprint


def separating_predicates(
    a: Fingerprint,
    b: Fingerprint,
    intervention: str,
    stability: float = 0.8,
) -> list[str]:
    """Predicates under one intervention on which a and b stably disagree."""
    out = []
    for p in PREDICATES:
        k = (intervention, p)
        if k not in a.value or k not in b.value:
            continue
        if a.stability[k] < stability or b.stability[k] < stability:
            continue
        if a.value[k] != b.value[k]:
            out.append(p)
    return out


def separates(a, b, subset, stability: float = 0.8) -> bool:
    return any(separating_predicates(a, b, i, stability) for i in subset)


def separation_matrix(
    fps: dict[str, Fingerprint],
    interventions: list[str],
    stability: float = 0.8,
) -> dict[tuple[str, str], list[str]]:
    """For each pair of algorithms, which single interventions separate them."""
    out = {}
    for a, b in combinations(sorted(fps), 2):
        out[(a, b)] = [
            i
            for i in interventions
            if separating_predicates(fps[a], fps[b], i, stability)
        ]
    return out


def minimal_separating_set(
    fps: dict[str, Fingerprint],
    interventions: list[str],
    stability: float = 0.8,
) -> list[str] | None:
    """Smallest intervention subset that separates every pair. None if impossible."""
    pairs = list(combinations(sorted(fps), 2))
    for size in range(1, len(interventions) + 1):
        for subset in combinations(interventions, size):
            if all(separates(fps[a], fps[b], subset, stability) for a, b in pairs):
                return list(subset)
    return None


def unseparated_pairs(
    fps: dict[str, Fingerprint],
    interventions: list[str],
    stability: float = 0.8,
) -> list[tuple[str, str]]:
    return [
        (a, b)
        for a, b in combinations(sorted(fps), 2)
        if not separates(fps[a], fps[b], interventions, stability)
    ]
