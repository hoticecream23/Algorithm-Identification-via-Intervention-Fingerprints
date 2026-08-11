"""Steppable reference algorithms over a shared observable state.

Every algorithm exposes the same state -- `d` (distance/key estimate) and `pi`
(predecessor) -- so trajectories are comparable across algorithms. This is the
same state a CLRS decoder emits, so fingerprints computed here transfer to a
neural model unchanged.

Executors advance one round at a time rather than running to completion, because
an intervention has to fire mid-execution and mutate the graph or the state while
the algorithm is still running.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .graphs import EPS, INF, Graph


@dataclass
class State:
    d: np.ndarray
    pi: np.ndarray
    round: int


class Executor:
    """Base class. Subclasses implement one round of work in `_step`."""

    name = "?"

    def __init__(self, graph: Graph, source: int, d_init: float = INF):
        self.graph = graph
        self.source = source
        self.d = np.full(graph.n, d_init)
        self.d[source] = 0.0
        self.pi = np.full(graph.n, -1, dtype=int)
        self.round = 0
        self.finished = False

    def snapshot(self) -> State:
        return State(self.d.copy(), self.pi.copy(), self.round)

    def step(self) -> bool:
        """Advance one round. Returns False once the algorithm has terminated."""
        if self.finished:
            return False
        self._step()
        self.round += 1
        return True

    def _step(self) -> None:
        raise NotImplementedError

    def run(self, budget: int, hook=None) -> list[State]:
        """Roll out for exactly `budget` rounds, freezing after termination.

        `hook(executor, round)` fires *before* each round, which is where
        interventions mutate the graph or the state. Trajectories are padded to a
        fixed length so runs of different algorithms stay index-comparable.
        """
        traj = [self.snapshot()]
        for r in range(budget):
            if hook is not None:
                hook(self, r)
            self.step()
            traj.append(self.snapshot())
        return traj


class BellmanFord(Executor):
    """Synchronous Bellman-Ford: relax every edge once per round.

    Reads `d` from the previous round, so information travels exactly one hop per
    round. No settled set -- every node stays eligible for relaxation forever,
    which is the property that makes it self-correcting.
    """

    name = "bellman_ford"

    def __init__(
        self,
        graph: Graph,
        source: int,
        max_rounds: int | None = None,
        d_init: float = INF,
    ):
        super().__init__(graph, source, d_init)
        self.max_rounds = max_rounds if max_rounds is not None else graph.n - 1

    def _step(self) -> None:
        d = self.d
        new_d = d.copy()
        new_pi = self.pi.copy()
        for u, v, w in self.graph.arcs():
            if d[u] + w < new_d[v] - EPS:
                new_d[v] = d[u] + w
                new_pi[v] = u
        self.d = new_d
        self.pi = new_pi
        if self.round + 1 >= self.max_rounds:
            self.finished = True


class TruncatedBellmanFord(BellmanFord):
    """Bellman-Ford that halts after k rounds whether or not it has converged.

    The shortcut hypothesis: a model trained on graphs of diameter <= k learns
    this, not Bellman-Ford, and looks perfect in-distribution while failing on any
    instance needing deeper propagation.
    """

    name = "bf_truncated"

    def __init__(self, graph: Graph, source: int, k: int = 3, d_init: float = INF):
        super().__init__(graph, source, max_rounds=k, d_init=d_init)
        self.name = f"bf_truncated_k{k}"


class SPFA(Executor):
    """Queue-based asynchronous Bellman-Ford. One node relaxed per round.

    Same relaxation rule as Bellman-Ford but work is queue-ordered, so a
    correction propagates only where the queue carries it.
    """

    name = "spfa"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source)
        self.queue = deque([source])
        self.in_queue = {source}

    def _step(self) -> None:
        if not self.queue:
            self.finished = True
            return
        u = self.queue.popleft()
        self.in_queue.discard(u)
        for v, w in self.graph.neighbours(u):
            if self.d[u] + w < self.d[v] - EPS:
                self.d[v] = self.d[u] + w
                self.pi[v] = u
                if v not in self.in_queue:
                    self.queue.append(v)
                    self.in_queue.add(v)


class Dijkstra(Executor):
    """Settle the cheapest unsettled node per round, then relax its out-edges.

    The settled set is final: a settled node is never updated again. That single
    line is what a corrupted-node probe is designed to detect.
    """

    name = "dijkstra"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source)
        self.settled = np.zeros(graph.n, dtype=bool)

    def _step(self) -> None:
        cand = np.nonzero(~self.settled & np.isfinite(self.d))[0]
        if cand.size == 0:
            self.finished = True
            return
        u = int(cand[np.argmin(self.d[cand])])
        self.settled[u] = True
        for v, w in self.graph.neighbours(u):
            if not self.settled[v] and self.d[u] + w < self.d[v] - EPS:
                self.d[v] = self.d[u] + w
                self.pi[v] = u


class Prim(Executor):
    """Prim's MST. Structurally identical to Dijkstra, different key.

    Dijkstra keys on `d[u] + w`, Prim keys on `w` alone. Lowering the weight of a
    frontier edge therefore moves their selection order differently -- that is the
    cleanest available separation between two algorithms with the same control
    flow.
    """

    name = "prim"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source)
        self.in_tree = np.zeros(graph.n, dtype=bool)

    def _step(self) -> None:
        cand = np.nonzero(~self.in_tree & np.isfinite(self.d))[0]
        if cand.size == 0:
            self.finished = True
            return
        u = int(cand[np.argmin(self.d[cand])])
        self.in_tree[u] = True
        for v, w in self.graph.neighbours(u):
            if not self.in_tree[v] and w < self.d[v] - EPS:
                self.d[v] = w
                self.pi[v] = u


def d_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Elementwise equality of estimates, treating INF == INF as True.

    The single definition of "these two estimates agree". It lives here because
    both the predicate extractor and the intervention timing need it, and having two
    copies is how one of them silently kept using exact equality: any noise at all
    then made every round look like a change, which pinned the firing round to the
    midpoint of the budget regardless of the noise scale.
    """
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    close = np.isclose(a, b, rtol=0, atol=tol, equal_nan=True)
    return both_inf | (np.isfinite(a) & np.isfinite(b) & close)


def converged_mask(
    traj: list[State], at: int, patience: int = 1, tol: float = 1e-9
) -> np.ndarray:
    """Nodes whose `d` is finite and has not changed for `patience` rounds.

    Dijkstra has an explicit settled set and Bellman-Ford does not, so "settled"
    cannot be read off the algorithm's internals if the predicate is to apply to
    every algorithm -- and to a neural model, which has no internals we are allowed
    to look at. Defining it observationally keeps the probe well-posed everywhere.
    """
    if at <= 0:
        return np.isfinite(traj[0].d) & (traj[0].d == 0.0)
    lo = max(0, at - patience)
    stable = np.ones_like(traj[at].d, dtype=bool)
    for r in range(lo, at):
        stable &= d_equal(traj[r].d, traj[at].d, tol)
    return stable & np.isfinite(traj[at].d)


ALGORITHMS = {
    "bellman_ford": BellmanFord,
    "bf_truncated_k3": lambda g, s: TruncatedBellmanFord(g, s, k=3),
    "spfa": SPFA,
    "dijkstra": Dijkstra,
    "prim": Prim,
}

# Networks cannot emit infinity, so comparing a model against a symbolic reference
# needs one shared representation of "not reached yet". Thresholding a model's output
# back to INF would hide a discretisation the model could sit right on top of, so the
# symbolic side takes a finite sentinel instead.
#
# This is sound rather than a fudge: relaxation from an unreached node yields
# BIG + w, which exceeds BIG whenever BIG is larger than any real path length, so an
# unreached node still cannot propagate. Behaviour is identical to INF.
#
# The magnitude matters for training, not just for correctness. The sentinel only has
# to exceed the longest achievable path so that BIG + w still fails to improve a
# reached node. On the graphs used here (n=24, density 0.05) the longest shortest path
# is ~39, so 80 gives 2x margin.
#
# Over-provisioning costs real accuracy: at 1e3 the normalised target was 10.0 while
# true distances sat near 0.3, so squared error was dominated by reproducing the
# sentinel and errors on actual distances barely registered in the gradient. Same
# class of mistake as tying a test tolerance to the noise amplitude -- a constant
# picked for safety on one axis silently destroying sensitivity on another.
#
# NOTE: this bound is graph-family dependent. Depth-17 path graphs reach ~126, so
# anything fingerprinted on sparser families than density 0.05 needs it raised.
UNREACHED = 80.0


def finite_init(ctor, d_init: float = UNREACHED):
    """Wrap any algorithm constructor to start from a finite sentinel.

    Patching after construction rather than threading a parameter through every
    subclass: the executors only ever *compare* estimates, never test for infinity,
    so swapping the initial value is enough and works for the lambdas in ALGORITHMS
    too.
    """

    def make(graph, source):
        ex = ctor(graph, source)
        ex.d = np.where(np.isinf(ex.d), d_init, ex.d)
        return ex

    return make
