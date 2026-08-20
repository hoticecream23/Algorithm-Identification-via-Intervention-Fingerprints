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

**Two further predictions that were also wrong, and this pair mattered.** An earlier version
of this docstring claimed that `row_sum` is the damping coefficient and that `width` is the
aggregation width, all five references sitting at exactly 1.00 so that a network measured at
2-3 could not be any of them. Both claims are false, and they failed together:

| statistic | claimed | measured (`--reproduce-response-table`) |
|---|---|---|
| `width`, Bellman-Ford | 1.00 at every round | 1, 1, 1, **2, 2, 2, 2, 2** |
| `row_sum`, Bellman-Ford | equals `alpha` | 1, 1, 1, **2, 2, 2, 2, 2** |

One cause for both: **the one-sided difference double-counts at a hard-min tie.** At a
converged node `d_v == d_u + w` exactly, so lowering *either* argument by `eps` moves the min
by the full `eps` and both `J[v, v]` and `J[v, u]` read 1. Bellman-Ford reaches that state
around round 3 on depth-5 instances and stays there; the selection-based algorithms do not,
because only one node is active per round. So the "2-3 neighbours per update versus exactly
1" separation is substantially an artefact of the estimator, and the `width` statistic cannot
carry a claim about aggregation hardness.

`row_sum` fails for a second, independent reason that is worth stating because it is what
fixes the measurement. For any relaxation of the form `d_v <- (1-a) d_v + a * agg(gamma d_u +
w)` the softmin weights are a partition of unity, so the *total* row sums to `1 - a(1-gamma)`
-- at `gamma = 1` that is identically 1 whatever `alpha` is. The damping coefficient lives on
the **diagonal**, not in the row sum:

- `alpha = 1 - |J[v, v]|` on rows that are reached and still moving. Exact.
- `gamma = (off-diagonal row sum) / alpha`. Exact.
- **`row_sum = 1 - alpha(1-gamma)` is then a membership test with no fitted parameters at
  all**: any object whose row sum disagrees with the value predicted by its own diagonal and
  off-diagonal is not in the family. This is the cheapest available family-membership
  diagnostic, and it is the direct repair of `THEORY.md` Proposition 5 -- a residual with a
  built-in scale, rather than a distance to an arbitrary reference.
- aggregation softness is read from the **off-diagonal participation ratio**,
  `1 / sum_u p_u^2` with `p` the off-diagonal row normalised to sum 1. Normalising divides
  out `alpha * gamma` exactly, so the participation ratio depends on the softmin weight
  distribution and nothing else. It is 1.000 at a hard min and rises toward `deg(v)` as the
  aggregation softens.

The participation ratio has a hard limit that is a property of the operating point rather
than of the estimator, and it must be reported rather than fitted around: softness is only
visible when the aggregation temperature is comparable to the typical best-minus-second-best
message gap. Measured on depth-5 instances at `gamma = 1`, that gap is ~3.9 at `alpha = 1`
but ~21.8 at `alpha = 0.7` and ~29.5 at `alpha = 0.5`, because damping holds the estimate far
above its fixed point where the minimum is unambiguous. So a participation ratio of 1.000
means "**this update behaves as a hard min at this operating point**", which is not the same
claim as "the aggregation is hard", and a damped model's temperature may simply be
unidentifiable. Say so instead of returning a number.

All of the above needs the row mask in `summarise_masked`, not `summarise`: an unreached node
whose whole neighbourhood sits at the sentinel sees gaps of order the edge-weight scale and
so looks soft for reasons that are a fact about the graph. `summarise` is left exactly as it
was so the numbers above stay reproducible as a record of what the unmasked statistics do.
"""

from __future__ import annotations

import copy

import numpy as np

from .algorithms import UNREACHED
from .graphs import Graph

# Statistics reported by `summarise_masked`, in the order `estimate.profile_vector` packs them.
MASKED_STATS = (
    "diag_frac",
    "self_med",
    "off_row_med",
    "off_pr",
    "off_rows",
    "off_width",
    "row_sum_med",
    "n_fit",
)


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


def jacobian_pure(step_fn, d: np.ndarray, eps: float = 0.25) -> np.ndarray:
    """Response operator for an update that is a pure function of `d`.

    Same estimator and sign convention as `jacobian_at`, but `step_fn(d) -> d_next` carries
    the non-`d` state in a closure, so the `n` perturbations need no `deepcopy` of an
    executor. That is the whole cost of `jacobian_at`: a grid search over family parameters
    does ~10^5 profiles, and copying a Graph and its executor `n` times per round dominates
    everything else.

    A perturbation that flips which node a selection rule picks is preserved, not suppressed:
    `step_fn` recomputes the sender from the perturbed `d`. Those flips are rare and show up
    as a heavy tail, so scale-valued statistics here still have to use medians.
    """
    n = d.shape[0]
    after = step_fn(d)
    J = np.zeros((n, n))
    for u in range(n):
        if not np.isfinite(d[u]):
            continue
        pert = d.copy()
        pert[u] -= eps
        delta = after - step_fn(pert)
        J[:, u] = np.where(np.isfinite(delta), delta, 0.0) / eps
    return J


def summarise_masked(
    J: np.ndarray,
    d_now: np.ndarray,
    d_next: np.ndarray,
    theta: float = 0.05,
    eps: float = 0.25,
    kappa: float = 1.0,
    unreached: float = UNREACHED,
    reached_frac: float = 0.9,
) -> dict[str, float]:
    """Parameter-bearing summaries, restricted to rows where they mean anything.

    The mask is two conditions, and neither is optional:

    - **reached**: `d_v < reached_frac * unreached`. A node whose neighbourhood is all at the
      sentinel sees message gaps of order the edge-weight scale, so it looks soft for reasons
      that are a property of the graph's degree and weight range rather than of the update
      rule. Including those rows makes aggregation temperature look observable when it is not.
    - **moving**: `|d_v(t+1) - d_v(t)| > kappa * eps`. A node moving less than the probe
      amplitude sits at a kink, where the one-sided difference is a secant rather than a
      derivative. Tying this to `eps` rather than to a small absolute constant is required,
      not a convenience: at `alpha < 1` the approach to a fixed point is geometric and exact
      standstill never happens, so a `1e-9` threshold would admit every barely-twitching row
      and swamp the medians.

    Medians throughout, for the selection-flip reason in the module docstring. Returns `nan`
    for every median when the mask is empty, with `n_fit = 0` so a caller can tell "no
    information" from "measured zero" -- the distinction the old `summarise` could not make.
    """
    mask = (d_now < reached_frac * unreached) & (np.abs(d_next - d_now) > kappa * eps)
    A = np.abs(J)
    diag = np.abs(np.diag(J))
    row = A.sum(axis=1)
    off_row = row - diag

    out: dict[str, float] = {
        "diag_frac": float(diag.sum() / A.sum()) if A.sum() > 0 else 1.0,
        "n_fit": float(mask.sum()),
    }
    if not mask.any():
        for k in ("self_med", "off_row_med", "off_pr", "off_width", "row_sum_med"):
            out[k] = float("nan")
        out["off_rows"] = 0.0
        return out

    out["self_med"] = float(np.median(diag[mask]))
    out["off_row_med"] = float(np.median(off_row[mask]))
    out["row_sum_med"] = float(np.median(row[mask]))

    live = mask & (off_row > theta)
    out["off_rows"] = float(live.sum())
    if not live.any():
        # Every moving row responds only to itself: a diagonal update. At gamma = 0 this is
        # the whole story and both alpha and the temperature are unobservable -- which is a
        # diagnosis, so it is reported as nan rather than as a participation ratio of 1.
        out["off_pr"] = float("nan")
        out["off_width"] = float("nan")
        return out

    prs, widths = [], []
    for v in np.nonzero(live)[0]:
        o = A[v].copy()
        o[v] = 0.0
        p = o / o.sum()
        prs.append(1.0 / float(np.square(p).sum()))
        widths.append(float((o > theta).sum()))
    out["off_pr"] = float(np.median(prs))
    out["off_width"] = float(np.median(widths))
    return out


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


def response_profile_masked(
    ctor,
    graph: Graph,
    source: int,
    rounds: int,
    eps: float = 0.25,
    theta: float = 0.05,
    kappa: float = 1.0,
    unreached: float = UNREACHED,
) -> dict[str, list[float]]:
    """`summarise_masked` across rounds, carrying the `d` pair each mask needs.

    `response_profile` cannot be reused: the mask is a function of `d(t)` and `d(t+1)`, which
    the unmasked path never exposes.

    An executor advertising `step_fn()` -- a pure `d -> d_next` closure at its current state
    -- takes the `jacobian_pure` path, which skips `n` executor deepcopies per round. The two
    paths are checked against each other in `run_family_spike.py`; they must agree exactly,
    because a closure that forgets any part of the non-`d` state silently measures a
    different operator.
    """
    out: dict[str, list[float]] = {}
    for t in range(rounds):
        ex = ctor(graph.copy(), source)
        if t > 0:
            ex.run(t)
        d_now = ex.d.copy()

        if hasattr(ex, "step_fn"):
            step = ex.step_fn()
            d_next = step(d_now)
            J = jacobian_pure(step, d_now, eps)
        else:
            probe = ctor(graph.copy(), source)
            if t > 0:
                probe.run(t)
            probe.step()
            d_next = probe.d.copy()
            J = jacobian_at(ctor, graph, source, t, eps)

        s = summarise_masked(J, d_now, d_next, theta, eps, kappa, unreached)
        for k, v in s.items():
            out.setdefault(k, []).append(v)
    return out


def aggregate_profiles(profiles: list[dict[str, list[float]]]) -> dict[str, np.ndarray]:
    """Mean profile across graph instances, per statistic."""
    keys = profiles[0].keys()
    return {k: np.mean([p[k] for p in profiles], axis=0) for k in keys}
