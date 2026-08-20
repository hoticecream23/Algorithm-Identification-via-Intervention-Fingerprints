"""A parameterised relaxation family containing the reference algorithms and the networks.

The nearest-reference method asks which of five discrete algorithms a trained MPNN *is*.
`FINDINGS_DECOMPOSITION.md` measured the answer to be "none of them, and not close": every
neural model sits 0.37-0.40 from its nearest reference while the reference-to-reference
median is 0.303. `THEORY.md` Proposition 5 formalises that as vacuity. The hypothesis space
does not contain the measured object.

This module is the repair. One update rule with six parameters, in which the five reference
algorithms are *limit points* and a gradient-trained network can be an interior point:

    m_uv(t)  = gamma * d_u(t) + w_uv                for u in N(v) & senders(t)
    S_v(t)   = softmin_tau({ m_uv })                # tau = 0 is the hard-min limit
    G_v(t)   = min( d_v(t), S_v(t) )
    a_v(t)   = alpha * (1 - m_v(t))
    d_v(t+1) = (1 - a_v) * d_v(t) + a_v * G_v(t)

Identification then becomes estimation -- "this model sits at alpha=0.7, tau=0.3, gamma=0.9"
-- rather than a distance to an arbitrary reference.

**`d_v` sits outside the softmin, and this is not a stylistic choice.** Putting it inside
gives `softmin_tau({d_u + w} U {d_v}) < d_v` strictly, at every finite tau, even at
convergence -- the correction term is `-tau * log(1 + sum exp(-delta/tau))`, which is
negative whatever the gaps are. So the estimate sinks forever and the rule has no fixed
point: measured at alpha=1, gamma=1, tau=0.5 over 30 rounds, the median `d - d_true` was
-4.16 against -0.00 for the form used here. Outside the softmin it is also a genuine
relaxation at every parameter setting (monotone non-increasing, HANDOFF constraint 3), and
the diagonal of the Jacobian becomes `(1-alpha) + alpha*1[d_v <= S_v]` with no softmin
weight in it, which is what decouples alpha from tau.

**Parameterised by `tau = 1/beta`, not beta.** The hard min is then the interior grid point
`tau = 0` rather than an unreachable `beta = inf`, so every grid and line search is finite.
`tau` also has units of distance, directly comparable to the typical best-minus-second-best
message gap on the graph family -- which is the quantity that decides whether tau is
observable at all.

**gamma, the path-accumulation discount, is the sixth axis and the reason the family is not
just the old five points smeared.** A network is free to learn `d_v <- min(d_v, gamma d_u +
w)` with gamma < 1: correct on short paths, error compounding as gamma^depth on long ones.
No (alpha, tau, c, k, selection) setting can express that. gamma also subsumes Prim at
gamma=0, which keys on `w` alone -- strictly better than a discrete `key in {sum, weight}`
axis, which could not represent a fractional discount.

**Two commitment channels, deliberately kept apart.**

- *Selection commitment* is implied by `selection="argmin"`: a node is selected as sender at
  most once, and once selected its value freezes. Dijkstra's settled set does exactly both
  things (`if not self.settled[v]` guards the update, `~self.settled` guards the selection),
  so for Dijkstra and Prim this channel alone is the settled set.
- *Standstill commitment* `c` is the continuous axis: a reached node that stops moving
  freezes, at rate `c`. This is E3's `gated` arm as a parameter, and it applies under any
  selection mode.

Folding the first into the second would make `c=1` mean "Dijkstra" and leave no independent
commitment axis to estimate. So the reference table below has `c=0` everywhere: none of the
five references commits on standstill. `c > 0` is uninhabited by the reference set, which is
the point -- it is where the learned models may live.

**The standstill tolerance is a constant, and it has to be a loose one.** At alpha < 1 the
approach to a fixed point is geometric and exact standstill never happens, so a 1e-9
tolerance would mean "never freeze" for every alpha < 1 and the `c` axis would silently die
across the entire interior of the family -- the same class of failure as tying `GatedMPNN.eps`
to a scale it could not reach (E3). It is therefore fixed at `kappa * eps` with `eps` the
Jacobian probe amplitude, pre-registered, and swept for sensitivity rather than tuned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .algorithms import UNREACHED, Executor
from .graphs import EPS, INF, Graph

# Probe amplitude the standstill tolerance is tied to. Matches `response.jacobian_at`'s
# default: a node moving less than the probe amplitude per round sits at a kink, where the
# finite difference across it is a secant rather than a derivative.
PROBE_EPS = 0.25
KAPPA = 1.0
REACHED_FRAC = 0.9  # d below REACHED_FRAC * UNREACHED counts as reached

SELECTIONS = ("sync", "argmin", "fifo")


@dataclass(frozen=True)
class FamilyParams:
    """A point in the family. `k=None` means no round cap."""

    alpha: float = 1.0
    tau: float = 0.0
    gamma: float = 1.0
    c: float = 0.0
    k: int | None = None
    selection: str = "sync"

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")

    @property
    def beta(self) -> float:
        """Aggregation sharpness. Reported in prose; never fitted in this parameterisation."""
        return INF if self.tau <= 0.0 else 1.0 / self.tau

    def key(self) -> tuple:
        """Hashable identity, for profile caching."""
        return (self.alpha, self.tau, self.gamma, self.c, self.k, self.selection)


# Synchronous exact relaxation: the family's reference point, and Bellman-Ford apart from its
# round cap. Safe as a default argument because `FamilyParams` is frozen.
DEFAULT_PARAMS = FamilyParams()


def softmin(x: np.ndarray, tau: float, axis: int = 0) -> np.ndarray:
    """Soft minimum with the min-shift, exact at `tau = 0`.

    `softmin_tau(x) = min(x) - tau * log sum exp(-(x - min(x)) / tau)`

    Shifting by the min rather than by an arbitrary constant does two jobs: every exponent
    argument is <= 0 so nothing overflows at small tau, and `tau -> 0` reaches the hard min
    by construction rather than by an underflow accident.

    Translation equivariance, `softmin(x + c) = softmin(x) + c`, is why the magnitude of
    `UNREACHED` cannot affect any aggregate here. Only *gaps* matter. A reached node's gap to
    a sentinel neighbour is ~75 against an edge-weight scale of 1-10, so the sentinel is
    invisible to the aggregate for any tau below ~10.
    """
    mn = np.min(x, axis=axis)
    if tau <= 0.0:
        return mn
    with np.errstate(invalid="ignore", divide="ignore"):
        z = x - np.expand_dims(mn, axis)
        z = np.where(np.isfinite(z), z, INF)  # INF - INF is nan; treat as no contribution
        s = np.exp(-z / tau).sum(axis=axis)
        out = mn - tau * np.log(s)
    return np.where(np.isfinite(mn), out, mn)


def family_step(
    d: np.ndarray,
    W: np.ndarray,
    params: FamilyParams,
    m: np.ndarray | None = None,
    senders: np.ndarray | None = None,
    delta_move: float = KAPPA * PROBE_EPS,
    reached: float = REACHED_FRAC * UNREACHED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One round of the family update. Pure function of `d` given the rest.

    `W[u, v]` is the weight of arc u->v, `INF` for no arc. `senders` is a boolean mask over
    sender nodes (`None` means all). `m` is the standstill commitment level in [0, 1].

    Returns `(d_next, m_next, parent)`. `parent[v]` is the hard-argmin sender for v, or -1,
    and is only used to decode `pi` -- at tau > 0 there is no single parent, so the argmin is
    reported as the modal one. Purity matters: `jacobian_pure` batches `n` perturbations of
    `d` through this in one call, which is what makes the grid searches affordable.
    """
    n = d.shape[0]
    if m is None:
        m = np.zeros(n)

    # M[u, v] = gamma * d_u + w_uv. INF + finite stays INF, so absent arcs and the diagonal
    # (which graphs.Graph leaves at INF) drop out with no masking.
    msg = params.gamma * d[:, None] + W
    if senders is not None and not senders.all():
        msg = np.where(senders[:, None], msg, INF)

    soft = softmin(msg, params.tau, axis=0)
    parent = np.where(np.isfinite(soft), np.argmin(msg, axis=0), -1)

    gate = np.minimum(d, soft)  # d_v OUTSIDE the softmin: keeps the fixed point
    a = params.alpha * (1.0 - m)
    d_next = (1.0 - a) * d + a * gate

    moved = np.abs(d_next - d) > delta_move
    still = (d_next < reached) & ~moved
    m_next = np.minimum(1.0, m + params.c * still)

    parent = np.where(d_next < d - EPS, parent, -1)
    return d_next, m_next, parent


class FamilyExecutor(Executor):
    """The family as a steppable executor, so the whole existing harness applies unchanged.

    Interventions mutate `self.d` and `self.graph.W` mid-rollout exactly as they do for a
    symbolic algorithm, and `fpid.response.jacobian_at` works on it without modification.
    """

    def __init__(
        self,
        graph: Graph,
        source: int,
        params: FamilyParams = DEFAULT_PARAMS,
        d_init: float = UNREACHED,
        delta_move: float = KAPPA * PROBE_EPS,
        reached: float = REACHED_FRAC * UNREACHED,
    ):
        super().__init__(graph, source, d_init)
        self.params = params
        self.name = f"family_{params.key()}"
        self.delta_move = delta_move
        self.reached = reached
        self.m = np.zeros(graph.n)
        # Selection commitment: a node selected as sender under `argmin` is never selected
        # again and its value freezes. This is Dijkstra's settled set, and it is a separate
        # channel from the standstill accumulator `self.m`.
        self.selected = np.zeros(graph.n, dtype=bool)
        self.queue: deque[int] = deque([source])
        self.in_queue: set[int] = {source}

    def _senders(self) -> np.ndarray | None:
        """The sender set for this round, or None for "all"; sets `finished` when exhausted.

        Aggregating over *senders* rather than picking a node to update is what makes
        Dijkstra exact. Dijkstra relaxes only the out-edges of the settled node; a rule that
        instead let every node aggregate over its whole neighbourhood would let an unsettled
        neighbour carrying a finite estimate propagate, which Dijkstra never does.
        """
        n = self.graph.n
        if self.params.selection == "sync":
            return None

        if self.params.selection == "argmin":
            cand = np.nonzero(~self.selected & np.isfinite(self.d))[0]
            if cand.size == 0:
                self.finished = True
                return None
            u = int(cand[np.argmin(self.d[cand])])
            self.selected[u] = True
            self.m[u] = 1.0  # settled: value frozen as well as unselectable
            mask = np.zeros(n, dtype=bool)
            mask[u] = True
            return mask

        # fifo: SPFA's rule. One node dequeued per round, pushed back on improvement.
        if not self.queue:
            self.finished = True
            return None
        u = self.queue.popleft()
        self.in_queue.discard(u)
        mask = np.zeros(n, dtype=bool)
        mask[u] = True
        return mask

    def _step(self) -> None:
        senders = self._senders()
        if self.finished:
            return

        d_prev = self.d
        d_next, m_next, parent = family_step(
            d_prev,
            self.graph.W,
            self.params,
            self.m,
            senders,
            self.delta_move,
            self.reached,
        )
        improved = d_next < d_prev - EPS
        self.d = d_next
        self.m = m_next
        self.pi = np.where(improved, parent, self.pi)

        if self.params.selection == "fifo":
            for v in np.nonzero(improved)[0]:
                if int(v) not in self.in_queue:
                    self.queue.append(int(v))
                    self.in_queue.add(int(v))
            # Termination is detected on the *next* round's `_senders`, matching SPFA, which
            # tests the queue at the top of its step. Testing it here instead would set
            # `finished` a round early -- harmless for `d`, but it would silently desynchronise
            # the round index that `firing_round` derives an intervention time from.

        if self.params.k is not None and self.round + 1 >= self.params.k:
            self.finished = True

    def step_fn(self):
        """A pure `d -> d_next` closure at the executor's current non-`d` state.

        For the Jacobian: the sender set, commitment and queue are held fixed while `d` is
        perturbed, so `n` perturbations batch into `n` calls with no deepcopy. A selection
        flip -- the perturbation changing which node is the argmin -- is a real effect and is
        preserved here, because the sender is recomputed from the perturbed `d`.
        """
        W = self.graph.W.copy()
        m = self.m.copy()
        selected = self.selected.copy()
        params = self.params
        head = self.queue[0] if self.queue else None
        n = self.graph.n
        # A halted executor maps `d` to itself, and the closure has to know that: the round
        # budget `k` is estimated from the round at which off-diagonal response disappears,
        # so a closure that kept updating past the halt would erase the only signal `k` has.
        halted = self.finished

        def run(d: np.ndarray) -> np.ndarray:
            if halted:
                return d.copy()
            m_now = m
            if params.selection == "sync":
                senders = None
            elif params.selection == "argmin":
                cand = np.nonzero(~selected & np.isfinite(d))[0]
                if cand.size == 0:
                    return d.copy()
                u = int(cand[np.argmin(d[cand])])
                senders = np.zeros(n, dtype=bool)
                senders[u] = True
                # `_senders` commits the selected node *before* the update, so its own value
                # is already frozen on the round it settles. Omitting this here made the pure
                # closure disagree with `_step` on exactly one row per round -- the row whose
                # diagonal the alpha estimator reads.
                m_now = m.copy()
                m_now[u] = 1.0
            else:
                if head is None:
                    return d.copy()
                senders = np.zeros(n, dtype=bool)
                senders[head] = True
            return family_step(
                d, W, params, m_now, senders, self.delta_move, self.reached
            )[0]

        return run


def family_ctor(params: FamilyParams, **kwargs):
    """Constructor with the `(graph, source)` signature the harness uses."""

    def make(graph: Graph, source: int) -> FamilyExecutor:
        return FamilyExecutor(graph, source, params, **kwargs)

    return make


# The five references as points in the family. `c=0` throughout: none of them commits on
# standstill -- Dijkstra's and Prim's settling is selection commitment, which `argmin`
# supplies. Bellman-Ford's `k` is graph-size dependent; use `reference_params`.
REFERENCE_LIMITS: dict[str, FamilyParams] = {
    "bellman_ford": FamilyParams(alpha=1.0, tau=0.0, gamma=1.0, c=0.0, k=None, selection="sync"),
    "bf_truncated_k3": FamilyParams(alpha=1.0, tau=0.0, gamma=1.0, c=0.0, k=3, selection="sync"),
    "spfa": FamilyParams(alpha=1.0, tau=0.0, gamma=1.0, c=0.0, k=None, selection="fifo"),
    "dijkstra": FamilyParams(alpha=1.0, tau=0.0, gamma=1.0, c=0.0, k=None, selection="argmin"),
    "prim": FamilyParams(alpha=1.0, tau=0.0, gamma=0.0, c=0.0, k=None, selection="argmin"),
}


def reference_params(name: str, n: int) -> FamilyParams:
    """`REFERENCE_LIMITS[name]` with the graph-size-dependent round cap resolved.

    Bellman-Ford halts at `n-1` rounds, which is a property of the graph rather than of the
    algorithm, so it cannot live in a static table.
    """
    p = REFERENCE_LIMITS[name]
    return replace(p, k=n - 1) if name == "bellman_ford" else p
