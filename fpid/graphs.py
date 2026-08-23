"""Graph container and generators.

The graph is mutable mid-execution: an intervention edits `W` in place while an
executor is still running. That is the "poke it" primitive the project needs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

INF = np.inf
EPS = 1e-9


@dataclass
class Graph:
    """Dense weighted graph. `W[u, v] == INF` means "no arc u->v"."""

    W: np.ndarray
    directed: bool = False

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def copy(self) -> "Graph":
        return Graph(self.W.copy(), self.directed)

    def arcs(self):
        """All present arcs as (u, v, w). Undirected graphs yield both directions."""
        us, vs = np.nonzero(np.isfinite(self.W))
        for u, v in zip(us, vs):
            if u != v:
                yield int(u), int(v), float(self.W[u, v])

    def neighbours(self, u: int):
        for v in np.nonzero(np.isfinite(self.W[u]))[0]:
            if int(v) != u:
                yield int(v), float(self.W[u, v])

    def has_edge(self, u: int, v: int) -> bool:
        return bool(np.isfinite(self.W[u, v]))

    def set_weight(self, u: int, v: int, w: float) -> None:
        self.W[u, v] = w
        if not self.directed:
            self.W[v, u] = w

    def delete_edge(self, u: int, v: int) -> None:
        self.set_weight(u, v, INF)

    def hop_distances(self, source: int) -> np.ndarray:
        """Unweighted BFS distance from `source`, ignoring edge weights.

        This is how far a perturbation's effect *could* travel; comparing it to how
        far the effect actually travelled is the depth-limited-execution probe.
        """
        dist = np.full(self.n, INF)
        dist[source] = 0.0
        q = deque([int(source)])
        while q:
            u = q.popleft()
            for v, _w in self.neighbours(u):
                if np.isinf(dist[v]):
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist


def random_connected_graph(
    n: int,
    rng: np.random.Generator,
    density: float = 0.25,
    w_low: float = 1.0,
    w_high: float = 10.0,
) -> Graph:
    """Erdos-Renyi undirected graph, forced connected by a random spanning path."""
    W = np.full((n, n), INF)
    perm = rng.permutation(n)
    for i in range(n - 1):
        u, v = int(perm[i]), int(perm[i + 1])
        W[u, v] = W[v, u] = float(rng.uniform(w_low, w_high))

    for u in range(n):
        for v in range(u + 1, n):
            if np.isinf(W[u, v]) and rng.random() < density:
                W[u, v] = W[v, u] = float(rng.uniform(w_low, w_high))
    return Graph(W, directed=False)


def high_diameter_graph(
    n: int,
    rng: np.random.Generator,
    w_low: float = 1.0,
    w_high: float = 10.0,
    chord_frac: float = 0.12,
) -> Graph:
    """A long path plus a few chords. High diameter on purpose.

    A dense random graph has diameter ~2, where "propagated 3 hops" and
    "propagated everywhere" are the same observation -- the truncation probe is
    vacuous there. Depth-sensitive predicates need room to travel.
    """
    W = np.full((n, n), INF)
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = float(rng.uniform(w_low, w_high))

    for _ in range(max(1, int(chord_frac * n))):
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if abs(u - v) > 2:
            W[u, v] = W[v, u] = float(rng.uniform(w_low, w_high))
    return Graph(W, directed=False)


def true_sssp(graph: Graph, source: int) -> np.ndarray:
    """Ground-truth shortest path distances, run to a fixed point."""
    d = np.full(graph.n, INF)
    d[source] = 0.0
    for _ in range(graph.n):
        changed = False
        for u, v, w in graph.arcs():
            if d[u] + w < d[v] - EPS:
                d[v] = d[u] + w
                changed = True
        if not changed:
            break
    return d


def d_close(a: np.ndarray, b: np.ndarray) -> bool:
    """Elementwise agreement that counts INF == INF as agreement."""
    both_inf = np.isinf(a) & np.isinf(b)
    return bool(np.all(both_inf | (np.abs(a - b) <= EPS)))


def weighted_depth(graph: Graph, source: int) -> float:
    """Rounds of synchronous relaxation needed to reach ground truth.

    This is the propagation depth that *relaxation* actually sees, and it is not
    `hop_distances(...).max()`. Hop eccentricity counts edges on a minimum-*hop*
    path, while relaxation propagates along minimum-*weight* paths, which may use
    strictly more edges. A graph in which every node is two hops from the source
    can therefore still need four or five rounds to settle. The two measures
    diverge exactly where truncation does its damage, which is why the truncation
    results are keyed on this one and not on hop eccentricity.

    Equivalently: the largest number of edges on any shortest-weight path from
    `source`, taking the fewest-edge path where several are tied.

    Because `BellmanFord` here is synchronous -- it reads `d` from the previous
    round, so information travels exactly one hop per round -- this is also
    exactly the round at which full Bellman-Ford settles, and `bf_truncated_k`
    returns the correct answer iff `weighted_depth <= k`.
    """
    truth = true_sssp(graph, source)
    reach = np.isfinite(truth)
    d = np.full(graph.n, INF)
    d[source] = 0.0
    for k in range(graph.n + 1):
        if d_close(d[reach], truth[reach]):
            return float(k)
        new_d = d.copy()
        for u, v, w in graph.arcs():
            if d[u] + w < new_d[v] - EPS:
                new_d[v] = d[u] + w
        d = new_d
    return INF
