"""Predicate fingerprints.

Everything here reads only `d` and `pi` -- the decoded state. Nothing touches an
algorithm's internals, so the identical code runs against a neural model's decoder
output in Phase C.

Predicates are qualitative buckets rather than a continuous distance. A real-valued
metric over response vectors has many free parameters (which norm, which weighting,
which normalisation) and any separation it reports can be tuned into existence.
"does the corruption ever wash out: exact / partial / never" cannot.

`tol` is the one continuous knob, and it exists only because a neural decoder emits
floats. It should be set from the noise floor measured on control runs, not by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithms import State, d_equal
from .graphs import Graph
from .interventions import (
    Intervention,
    compute_targets,
    firing_round,
    was_converged,
)

TOL = 1e-6


_eq = d_equal  # one definition of "these estimates agree", shared with the timing code


def _bucket_count(k: int) -> str:
    if k == 0:
        return "0"
    if k == 1:
        return "1"
    if k <= 3:
        return "2-3"
    return "4+"


def _bucket_depth(x: float) -> str:
    if not np.isfinite(x):
        return "none"
    if x >= 4:
        return "4+"
    return str(int(x))


def _settle_round(traj: list[State], tol: float = TOL) -> int:
    """Last round at which `d` changed, i.e. when the run went quiet."""
    last = 0
    for r in range(1, len(traj)):
        if not _eq(traj[r].d, traj[r - 1].d, tol).all():
            last = r
    return last


def probe(
    alg_ctor,
    graph: Graph,
    source: int,
    intervention: Intervention,
    budget: int,
    tol: float = TOL,
    absolute_firing: bool = False,
) -> dict[str, str]:
    """Run control / intervened / reference rollouts and read off the predicates.

    The reference is a fresh run of the *same* algorithm on the post-intervention
    graph. That makes "recovered" mean something precise: the mid-run poke left the
    algorithm in the state it would have reached had the graph always been that way.
    """
    targets = compute_targets(graph, source)
    site = intervention.target_node(targets)

    control = alg_ctor(graph.copy(), source).run(budget)
    t = firing_round(control, budget, tol, absolute=absolute_firing)

    g_live = graph.copy()
    ex = alg_ctor(g_live, source)
    fired = {"done": False}

    def hook(executor, r):
        if r == t and not fired["done"]:
            intervention.apply(executor, targets)
            fired["done"] = True

    intervened = ex.run(budget, hook=hook)
    reference = alg_ctor(g_live.copy(), source).run(budget)

    ctrl_f, iv_f, ref_f = control[-1].d, intervened[-1].d, reference[-1].d

    # Nodes whose correct answer the poke actually moved, plus the node poked
    # directly. Predicates restricted to this set ignore the untouched majority,
    # which would otherwise wash every difference out.
    at_risk = ~_eq(ctrl_f, ref_f, tol)
    at_risk[site] = True

    correct = _eq(iv_f, ref_f, tol)[at_risk]
    frac = float(correct.mean()) if correct.size else 1.0
    if frac >= 1.0 - 1e-9:
        recovery = "exact"
    elif frac > 0.0:
        recovery = "partial"
    else:
        recovery = "none"

    latency = "never"
    for k in range(0, budget - t):
        r = t + 1 + k
        if r >= len(intervened):
            break
        if _eq(intervened[r].d, ref_f, tol)[at_risk].all():
            latency = str(k) if k < 4 else "4+"
            break

    r1 = min(t + 1, len(intervened) - 1)
    spread1 = _bucket_count(int((~_eq(intervened[r1].d, control[r1].d, tol)).sum()))

    hop = graph.hop_distances(site)
    ever = np.zeros(graph.n, dtype=bool)
    for r in range(t + 1, len(intervened)):
        ever |= ~_eq(intervened[r].d, control[r].d, tol)
    depth = _bucket_depth(hop[ever].max() if ever.any() else np.inf)

    dt = _settle_round(intervened, tol) - _settle_round(control, tol)
    term_delta = "0" if dt == 0 else ("+" if dt > 0 else "-")

    # Against the control, not the reference. Comparing to the reference would
    # measure whether the poke moved the *right answer*, which is a fact about the
    # graph and identical for every algorithm -- not a response at all.
    pi_diff = int((intervened[-1].pi != control[-1].pi).sum())
    pi_churn = "none" if pi_diff == 0 else ("some" if pi_diff <= 2 else "many")

    return {
        "recovery": recovery,
        "latency": latency,
        "spread1": spread1,
        "depth": depth,
        "term_delta": term_delta,
        "pi_churn": pi_churn,
        "poked_settled": "yes" if was_converged(control, site, t, tol) else "no",
    }


PREDICATES = (
    "recovery",
    "latency",
    "spread1",
    "depth",
    "term_delta",
    "pi_churn",
    "poked_settled",
)


@dataclass
class Fingerprint:
    """Modal predicate value per (intervention, predicate), with its stability."""

    value: dict[tuple[str, str], str]
    stability: dict[tuple[str, str], float]

    def stable_keys(self, threshold: float) -> set[tuple[str, str]]:
        return {k for k, s in self.stability.items() if s >= threshold}


def aggregate(runs: list[dict[tuple[str, str], str]]) -> Fingerprint:
    """Modal value across graph instances, plus how often that mode held.

    Stability is not decoration. A predicate that flips between instances carries no
    identity information, and including it would manufacture separation out of
    sampling noise.
    """
    value: dict[tuple[str, str], str] = {}
    stability: dict[tuple[str, str], float] = {}
    keys = {k for r in runs for k in r}
    for k in sorted(keys):
        vals = [r[k] for r in runs if k in r]
        uniq, counts = np.unique(np.array(vals), return_counts=True)
        i = int(np.argmax(counts))
        value[k] = str(uniq[i])
        stability[k] = float(counts[i] / len(vals))
    return Fingerprint(value, stability)
