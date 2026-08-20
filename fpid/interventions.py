"""The intervention suite.

Two design rules, both load-bearing:

1. Targets are chosen from the *graph and its ground truth only*, never from the
   running algorithm's own state. If Bellman-Ford deleted an edge its own pointers
   picked and Dijkstra deleted a different one, their fingerprints would differ for
   the wrong reason. Every algorithm receives the identical poke.

2. The firing round is observational: one round after the target node's estimate
   stops changing. "Round 5" means different things to Bellman-Ford (5 hops of
   propagation) and Dijkstra (5 nodes settled), so a fixed round number is not a
   comparable moment. "Just after this node looked done" is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .algorithms import Executor, State, converged_mask, d_equal
from .graphs import EPS, INF, Graph, true_sssp


@dataclass
class Targets:
    """Algorithm-independent landmarks in the graph."""

    hop: np.ndarray
    d_true: np.ndarray
    pi_true: np.ndarray
    far: int
    near: int


def compute_targets(graph: Graph, source: int) -> Targets:
    hop = graph.hop_distances(source)
    d = np.full(graph.n, INF)
    pi = np.full(graph.n, -1, dtype=int)
    d[source] = 0.0
    for _ in range(graph.n):
        changed = False
        for u, v, w in graph.arcs():
            if d[u] + w < d[v] - EPS:
                d[v] = d[u] + w
                pi[v] = u
                changed = True
        if not changed:
            break

    reachable = np.nonzero(np.isfinite(hop))[0]
    far = int(reachable[np.argmax(hop[reachable])])

    # A node one third of the way out, not the median. The state probes need a node
    # that converges early enough to leave the algorithm plenty of run left to
    # respond in -- poking a node that settles on the last round measures nothing.
    want = hop[far] / 3.0
    order = np.argsort(np.abs(hop[reachable] - want))
    near = int(reachable[order[0]])
    if near == source and reachable.size > 1:
        near = int(reachable[order[1]])
    return Targets(hop, d, pi, far, near)


@dataclass
class Intervention:
    name: str
    target_node: Callable[[Targets], int]
    apply: Callable[[Executor, Targets], None]
    note: str = ""


def _tree_parent(t: Targets, v: int) -> int:
    return int(t.pi_true[v])


def _delete_tree_edge(v_of: Callable[[Targets], int]):
    def apply(ex: Executor, t: Targets) -> None:
        v = v_of(t)
        u = _tree_parent(t, v)
        if u >= 0:
            ex.graph.delete_edge(u, v)

    return apply


def _delete_nontree_edge(ex: Executor, t: Targets) -> None:
    """Null control: remove an edge no shortest path uses.

    A model whose fingerprint responds strongly here is telling us its Lipschitz
    constant, not its algorithm. Every other response gets normalised against this.
    """
    for u, v, _w in ex.graph.arcs():
        if u < v and _tree_parent(t, v) != u and _tree_parent(t, u) != v:
            ex.graph.delete_edge(u, v)
            return


def _insert_improving_edge(ex: Executor, t: Targets) -> None:
    """A late-arriving improvement to a node that already looked finished.

    A genuinely negative weight would be the sharper probe, but on an undirected
    graph any negative edge is a negative cycle, which is degenerate rather than
    informative. A positive edge that strictly improves an already-settled node
    carries the same discriminative content: Bellman-Ford keeps relaxing and picks
    it up, Dijkstra's settled set makes it unreachable.
    """
    u, v = ex.source, t.far
    if np.isfinite(t.d_true[v]) and t.d_true[v] > t.d_true[u] + 2 * EPS:
        w = float((t.d_true[v] - t.d_true[u]) / 2.0)
        ex.graph.set_weight(u, v, w)


def _insert_shortcut(ex: Executor, t: Targets) -> None:
    ex.graph.set_weight(ex.source, t.far, 0.5)


def _lower_tree_edge_weight(ex: Executor, t: Targets) -> None:
    """Separates Dijkstra from Prim: they key on `d[u] + w` versus `w`."""
    v = t.near
    u = _tree_parent(t, v)
    if u >= 0:
        ex.graph.set_weight(u, v, 0.5)


def _tree_edge_delta(ex: Executor, t: Targets, u: int, v: int, cap: float = 2.0):
    """How far `w(u,v)` can rise before `(u,v)` stops being the shortest-path tree
    edge into `v`, halved and capped -- or `None` if the instance is too degenerate.

    `d_alt` is the shortest distance to `v` on the graph with `(u,v)` deleted: the
    best route that does not use this edge. If raising `w` by more than
    `d_alt[v] - d_true[v]` would make that alternative route shorter, so `delta`
    must stay under half of that gap to keep both arms measuring the same tree.
    An infinite gap (no alternative route exists) places no upper bound on delta,
    so only the fixed cap applies. `gap <= 0` is a degenerate instance (already at
    a tie) and is skipped, as is any resulting delta below the measurement floor.
    """
    g_alt = ex.graph.copy()
    g_alt.delete_edge(u, v)
    d_alt = true_sssp(g_alt, ex.source)
    gap = float(d_alt[v] - t.d_true[v])
    if not (gap > 0):
        return None
    delta = min(cap, 0.5 * gap)
    if delta < 0.1:
        return None
    return delta


def _raise_tree_edge_weight(ex: Executor, t: Targets) -> None:
    """Control arm for `dissociate_keys`: raise `w(u,v)` alone, moving both
    Dijkstra's key (`d[u]+w`) and Prim's key (`w`)."""
    v = t.near
    u = _tree_parent(t, v)
    if u < 0:
        return
    delta = _tree_edge_delta(ex, t, u, v)
    if delta is None:
        return
    ex.graph.set_weight(u, v, float(ex.graph.W[u, v]) + delta)


def _dissociate_keys(ex: Executor, t: Targets) -> None:
    """Move Prim's key without moving Dijkstra's.

    Dijkstra selects on `d[u] + w`; Prim selects on `w` alone. Raising `w(u,v)` by
    `delta` and simultaneously lowering `d[u]` by the same `delta` leaves
    `d[u] + w` exactly where it was -- Dijkstra's key is invariant -- while `w`
    itself has strictly risen -- Prim's key has moved. Both edits are computed
    from the graph and its ground truth only (`_tree_edge_delta`, `t.d_true`), so
    every algorithm receives the identical poke; whether a given executor's own
    `ex.d[u]` happened to equal `d_true[u]` at firing time is part of its
    response, not part of the poke.
    """
    v = t.near
    u = _tree_parent(t, v)
    if u < 0:
        return
    delta = _tree_edge_delta(ex, t, u, v)
    if delta is None:
        return
    ex.graph.set_weight(u, v, float(ex.graph.W[u, v]) + delta)
    if np.isfinite(ex.d[u]):
        ex.d[u] = t.d_true[u] - delta


def _inject_unreached(ex: Executor, t: Targets) -> None:
    """Give a node the algorithm has not reached yet a small finite estimate.

    Separates state-driven selection from queue-driven selection. Dijkstra and Prim
    pick the next node by argmin over `d`, so an injected estimate is picked up
    immediately; Bellman-Ford relaxes every edge every round and propagates it too.
    SPFA only ever works on what is in its queue, and the injected node is not in
    it, so the injection is invisible until a neighbour happens to be processed.

    This is what finally tells SPFA and Dijkstra apart: they agree on every probe
    that perturbs *values*, and differ only on one that perturbs what is *eligible*.
    """
    ex.d[t.far] = 1.0


def _corrupt(delta: float, v_of: Callable[[Targets], int]):
    def apply(ex: Executor, t: Targets) -> None:
        v = v_of(t)
        if np.isfinite(ex.d[v]):
            ex.d[v] = ex.d[v] + delta

    return apply


INTERVENTIONS: list[Intervention] = [
    # The two deletion probes are near-silent against symbolic algorithms, because
    # relaxation only ever lowers `d` and repairing a deletion requires raising it.
    # They are kept deliberately: a neural model has no such monotonicity constraint
    # and *can* raise an estimate, so "responds at all to a deletion" is itself a
    # sharp fingerprint dimension separating learned reasoners from real relaxation.
    Intervention(
        "delete_tree_edge_near",
        lambda t: t.near,
        _delete_tree_edge(lambda t: t.near),
        "monotonicity probe: relaxation cannot repair a deletion",
    ),
    Intervention(
        "delete_tree_edge_far",
        lambda t: t.far,
        _delete_tree_edge(lambda t: t.far),
        "monotonicity probe at depth",
    ),
    Intervention(
        "delete_nontree_edge",
        lambda t: t.near,
        _delete_nontree_edge,
        "null control: correct response is no response",
    ),
    Intervention(
        "insert_improving_edge",
        lambda t: t.far,
        _insert_improving_edge,
        "late improvement to a settled node: BF sees it, Dijkstra cannot",
    ),
    # The depth probe. A cheap edge to the far node forces an improvement that must
    # then propagate outward from there, hop by hop. An algorithm that stops after k
    # rounds fixes only the k-hop neighbourhood, which `depth` reads off directly.
    Intervention(
        "insert_shortcut_from_source",
        lambda t: t.far,
        _insert_shortcut,
        "depth probe: improvement must propagate from the far node outward",
    ),
    Intervention(
        "lower_tree_edge_weight",
        lambda t: t.near,
        _lower_tree_edge_weight,
        "Dijkstra keys on d[u]+w, Prim keys on w",
    ),
    Intervention(
        "corrupt_node_up",
        lambda t: t.near,
        _corrupt(+5.0, lambda t: t.near),
        "does the algorithm re-relax a node it already finished?",
    ),
    Intervention(
        "corrupt_node_down",
        lambda t: t.near,
        _corrupt(-5.0, lambda t: t.near),
        "an optimistic error only propagates if the node is still live",
    ),
    Intervention(
        "inject_unreached_node",
        lambda t: t.far,
        _inject_unreached,
        "state-driven vs queue-driven selection",
    ),
    # E6: dissociates Dijkstra's key (d[u]+w) from Prim's (w) directly, rather than
    # relying on `lower_tree_edge_weight` to move both at once and inferring the
    # boundary from the manner of the two algorithms' responses.
    Intervention(
        "raise_tree_edge_weight",
        lambda t: t.near,
        _raise_tree_edge_weight,
        "control arm: raises w(u,v), moving both Dijkstra's and Prim's key",
    ),
    Intervention(
        "dissociate_keys",
        lambda t: t.near,
        _dissociate_keys,
        "raises w(u,v) while lowering d[u] by the same amount: Dijkstra's "
        "key (d[u]+w) is invariant, Prim's (w) is not",
    ),
]


def settle_round(traj: list[State], tol: float = 1e-9, rel: float = 0.05) -> int:
    """Last round at which `d` changed materially: when this run went quiet.

    A change counts if a node became reachable, or if it is large relative to the
    biggest change the run ever made. An absolute threshold is not enough: over
    ~n nodes x ~2n rounds there are hundreds of comparisons, so noise reliably
    produces at least one spurious crossing, and a single false change at a late
    round drags the settle round to the end of the budget.

    Early-terminating algorithms are hit hardest -- truncated Bellman-Ford really
    settles at round k, so one spurious late change moves its firing time by an
    order of magnitude and its fingerprint stops resembling itself. Those are
    precisely the shortcut models the project exists to detect, so this estimator
    has to be robust for them specifically.
    """
    mags, reached = [], []
    for r in range(1, len(traj)):
        a, b = traj[r - 1].d, traj[r].d
        both = np.isfinite(a) & np.isfinite(b)
        mags.append(float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0)
        reached.append(bool((np.isfinite(b) & ~np.isfinite(a)).any()))

    if not mags:
        return 0

    # `tol` bounds a *single* comparison, but this scan makes one per node per round.
    # The relevant statistic is the max over all of them, so the threshold carries a
    # Gumbel-style sqrt(2 ln N) correction. Without it noise crosses `tol` somewhere
    # in almost every run and drags the settle round to the end of the budget.
    #
    # The relative term alone is not enough either: an algorithm whose updates are
    # mostly INF -> finite transitions (truncated Bellman-Ford, which is exactly the
    # shortcut we need to detect) has almost no finite-to-finite deltas, so
    # max(mags) collapses to ~0 and the relative threshold degenerates to `tol`.
    n_comparisons = max(2, len(traj) * traj[0].d.shape[0])
    floor = tol * float(np.sqrt(2.0 * np.log(n_comparisons)))
    threshold = max(floor, rel * max(mags))
    last = 0
    for i, (m, got_reach) in enumerate(zip(mags, reached), start=1):
        if got_reach or m > threshold:
            last = i
    return last


def firing_round(control: list[State], budget: int, tol: float = 1e-9) -> int:
    """Halfway through the algorithm's own active life.

    An absolute round number is not a comparable moment across algorithms -- round 5
    is five hops of propagation for Bellman-Ford and five settled nodes for
    Dijkstra. Relative progress is the only shared clock, and taking the midpoint
    guarantees that half the run remains for the algorithm to respond in. Firing
    after termination measures nothing, which is what an absolute round did.
    """
    return int(np.clip(settle_round(control, tol) // 2, 1, budget - 1))


def was_converged(
    control: list[State], target: int, at: int, tol: float = 1e-9
) -> bool:
    """Whether the poke landed on a node the algorithm had already finished with."""
    if at >= len(control):
        return False
    return bool(converged_mask(control, at, tol=tol)[target])
