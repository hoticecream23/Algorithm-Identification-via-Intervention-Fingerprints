"""Linear-response fingerprints: the Jacobian of one round of execution.

The categorical predicate suite asks nine questions and buckets each answer, giving 63
slots. That was the right conservative choice -- a real-valued metric over response vectors
has many free parameters and any separation it reports can be tuned into existence -- but it
buys robustness with resolution, and the neural results ran out of resolution.

This measures a different object: the local response operator

    J_t[v, u] = - d(d_v at round t+1) / d(d_u at round t)

estimated by lowering `d_u` a little and stepping once. Downward is the informative
direction, because relaxation only ever lowers `d`: lowering an argmin propagates, raising
it may simply hand the argmin to someone else and read as no response.

The point is that `J` *is* the update rule, and the reference algorithms have sharply
different Jacobian structure that follows from their definitions rather than from tuning.

**The statistic that works is the diagonal fraction** `||diag J|| / ||J||`: the share of the
state that this round maps to itself. A frozen or halted node contributes `d_v(t+1) = d_v(t)`
and so an exact diagonal entry, while a node still receiving propagation contributes
off-diagonal mass. Measured on depth >= 5 instances:

| Algorithm | diagonal fraction, rounds 0-7 |
|---|---|
| Bellman-Ford | 0.92 -> 0.45, and stays low: propagation never stops |
| truncated BF (k=3) | tracks Bellman-Ford, then **exactly 1.00 from round 3** |
| SPFA | 0.88-1.00: one node active per round |
| Dijkstra | 0.88-1.00 |
| Prim | 1.00 |

This is a continuous, per-round, purely observational measure of **commitment** -- the
quantity constraint 4 says cannot be read off an algorithm's internals, since Bellman-Ford
has no settled set and a network has no inspectable state. Here it needs neither.

Prim's 1.00 confirms the structural prediction for the *value* update: Prim keys on `w`
alone, so `d_v` does not depend on `d_u` and the update is diagonal, while Dijkstra keys on
`d_u + w`. The Dijkstra/Prim boundary otherwise rests on a single intervention and is the
weakest claim in the project.

**Two predictions that were wrong, kept because the reasons are load-bearing.**

*Prim's Jacobian was predicted to vanish identically.* It does not: `d` is also Prim's
selection key, so lowering `d_u` can change which node enters the tree, and that reroutes
the whole downstream state. The value update is diagonal; the selection is not. For any
selection-based algorithm, a state perturbation cannot cleanly separate value propagation
from control-flow change, because **the value is the control signal**. The diagonal fraction
survives this because a selection flip shows up as a rare off-diagonal outlier rather than
as a bias.

*Those flips were then predicted to be pervasive.* They are rare -- 0/24 perturbations
changed Dijkstra's choice, 2/24 changed Prim's. So they contaminate `row_sum` and `mass` as
a heavy tail (means of 15-28 against a true value of 1) while leaving rank- and
support-based statistics intact. Report medians, not means, for anything scale-valued here.

Two further quantities matter for learned models specifically:

- **row sum** is the damping coefficient. Exact relaxation passes `d_u + w` through, so
  d(d_v)/d(d_u) = 1. A model doing `d <- d + alpha (agg(d) - d)` reports `alpha`.
- **nonzeros per row** is the aggregation width. A hard `min` depends on one parent, so 1.
  A softmin depends on every neighbour, so ~deg(v). This is the one measurement that can
  tell a hard relaxation from a soft one, and it needs no reference algorithm at all.

Both are *estimates of a procedure's parameters* rather than a nearest-neighbour label,
which is what a family of learned models sitting between references actually needs.
"""

from __future__ import annotations

import copy

import numpy as np

from .graphs import Graph


def jacobian_at(ctor, graph: Graph, source: int, t: int, eps: float = 0.25) -> np.ndarray:
    """Response operator at round `t`, by one-sided finite differences.

    `eps` is a real perturbation, not an infinitesimal: these updates are piecewise
    linear, so any `eps` below the gap to the second-best parent gives the exact local
    derivative, and one above it reports a secant. 0.25 sits well under the 1-10
    edge-weight scale.
    """
    n = graph.n
    base = ctor(graph.copy(), source)
    if t > 0:
        base.run(t)

    ref = copy.deepcopy(base)
    ref.step()
    after = ref.d.copy()

    J = np.zeros((n, n))
    for u in range(n):
        if not np.isfinite(base.d[u]):
            continue
        pert = copy.deepcopy(base)
        pert.d[u] -= eps
        pert.step()
        delta = after - pert.d          # positive: lowering u lowered v
        J[:, u] = np.where(np.isfinite(delta), delta, 0.0) / eps
    return J


def summarise(J: np.ndarray, theta: float = 0.05) -> dict[str, float]:
    """Structural summaries of one Jacobian. `theta` is a support threshold, not a scale.

    Scale-valued statistics use the median across rows: a rare selection flip produces an
    enormous single-row response that drags any mean by an order of magnitude.
    """
    A = np.abs(J)
    active = A > theta
    rows = active.any(axis=1)
    off = A.sum() - np.abs(np.diag(J)).sum()
    return {
        # The one that discriminates: share of the state mapping to itself this round.
        "diag_frac": float(np.abs(np.diag(J)).sum() / A.sum()) if A.sum() > 0 else 1.0,
        "off_mass": float(off),
        "active_cols": float(active.any(axis=0).sum()),
        "active_rows": float(rows.sum()),
        "row_sum": float(np.median(A[rows].sum(axis=1))) if rows.any() else 0.0,
        "width": float(np.median(active[rows].sum(axis=1))) if rows.any() else 0.0,
    }


def response_profile(
    ctor,
    graph: Graph,
    source: int,
    rounds: int,
    eps: float = 0.25,
    theta: float = 0.05,
) -> dict[str, list[float]]:
    """Jacobian summaries across the first `rounds` rounds of execution."""
    out: dict[str, list[float]] = {}
    for t in range(rounds):
        s = summarise(jacobian_at(ctor, graph, source, t, eps), theta)
        for k, v in s.items():
            out.setdefault(k, []).append(v)
    return out


def aggregate_profiles(profiles: list[dict[str, list[float]]]) -> dict[str, np.ndarray]:
    """Mean profile across graph instances, per statistic."""
    keys = profiles[0].keys()
    return {k: np.mean([p[k] for p in profiles], axis=0) for k in keys}
