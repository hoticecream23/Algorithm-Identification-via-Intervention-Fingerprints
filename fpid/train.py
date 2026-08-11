"""Training for the neural reasoners.

Two supervision regimes, because they are expected to fingerprint differently:

- **full-hint** supervises the estimate at every round against a reference algorithm's
  own trajectory. The model is told the algorithm, so a matching fingerprint is close to
  tautological -- which is exactly why the hint target has to be selectable: training
  half the models on Dijkstra hints is the only way to check the fingerprint discriminates
  at all, rather than always answering Bellman-Ford.
- **no-hint** supervises only the final answer. The intermediate procedure is
  unconstrained, so whatever algorithm the model settles on is genuinely its own. This
  is the regime where the OOD hypothesis has any content.

Note that with `rounds >= n` both reference algorithms have converged, so their final
estimates are identical and the no-hint regime is a *single* condition -- the choice of
hint algorithm cannot reach the loss. That makes a no-hint group trained under two
nominal labels a negative control rather than a comparison.
"""

from __future__ import annotations

import numpy as np
import torch

from .algorithms import UNREACHED, BellmanFord, Dijkstra, finite_init
from .graphs import Graph, random_connected_graph
from .neural import SCALE, MPNN, graph_tensors


def bf_trajectory(graph: Graph, source: int, rounds: int) -> np.ndarray:
    """Bellman-Ford's estimate at each round, from the finite sentinel."""
    ex = BellmanFord(graph, source, max_rounds=rounds, d_init=UNREACHED)
    traj = ex.run(rounds)
    return np.stack([s.d for s in traj])


def dijkstra_trajectory(graph: Graph, source: int, rounds: int) -> np.ndarray:
    """Dijkstra's estimate at each round, from the same finite sentinel.

    Same `(d, pi)` state as Bellman-Ford, so the training code needs no other change.
    Two things differ and both matter for how this may be used as a target:

    - One node settles per round, so convergence takes `n` rounds rather than ~depth.
      Supervising with `rounds < n` trains on a half-finished run, and the resulting
      MAE is not comparable with a Bellman-Ford model's. Use `rounds >= n`.
    - `Dijkstra` selects candidates by `isfinite(d)`, so it needs `finite_init` rather
      than a `d_init` argument. The sentinel is never selected before a real estimate
      because 80 exceeds every achievable path length here (~39).
    """
    ex = finite_init(Dijkstra)(graph, source)
    traj = ex.run(rounds)
    return np.stack([s.d for s in traj])


TRAJECTORIES = {
    "bellman_ford": bf_trajectory,
    "dijkstra": dijkstra_trajectory,
}


def make_dataset(
    n_graphs: int,
    n_nodes: int,
    rounds: int,
    rng: np.random.Generator,
    density: float = 0.05,
    algorithm: str = "bellman_ford",
    min_depth: int = 0,
):
    """Graphs deep enough to fingerprint on (density 0.05 => depth ~5 at n=24).

    `min_depth` rejection-samples on max hop distance from the source. Density 0.05 at
    n=24 actually yields depth 3-7, and constraint 1 says nothing separates below ~4, so
    an unfiltered sample puts a fifth of the graphs where the method cannot work. It
    defaults off so that earlier runs stay reproducible.
    """
    traj_fn = TRAJECTORIES[algorithm]
    out = []
    while len(out) < n_graphs:
        g = random_connected_graph(n_nodes, rng, density=density)
        if min_depth:
            hop = g.hop_distances(0)
            if hop[np.isfinite(hop)].max() < min_depth:
                continue
        out.append((g, 0, traj_fn(g, 0, rounds)))
    return out


def pack(dataset, device):
    """Stack a dataset into one batch. Every graph shares n, so this is free."""
    adj = torch.from_numpy(
        np.stack([np.isfinite(g.W) for g, _, _ in dataset])
    ).to(device)
    w = torch.from_numpy(
        np.stack([np.where(np.isfinite(g.W), g.W, 0.0) for g, _, _ in dataset])
    ).float().div(SCALE).to(device)
    is_src = torch.zeros(len(dataset), dataset[0][0].n, device=device)
    for i, (_g, src, _t) in enumerate(dataset):
        is_src[i, src] = 1.0
    # (rounds+1, B, n)
    target = torch.from_numpy(
        np.stack([traj for _, _, traj in dataset], axis=1)
    ).float().div(SCALE).to(device)
    return adj, w, is_src, target


def train_model(
    dataset,
    rounds: int,
    seed: int,
    hints: bool = True,
    hidden: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cuda",
    agg: str = "min",
    model_fn=None,
) -> tuple[MPNN, float]:
    """`model_fn(hidden, agg)` overrides the architecture, for E3's gated variant."""
    torch.manual_seed(seed)
    model = (model_fn(hidden, agg) if model_fn else MPNN(hidden, agg)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    adj, w, is_src, target = pack(dataset, device)

    last = 0.0
    for _ in range(epochs):
        # Commitment is per-rollout state, not per-model state. An epoch that inherited
        # the previous epoch's frozen set would train a model that is already committed
        # everywhere before it starts.
        if hasattr(model, "reset"):
            model.reset()
        d = target[0].clone()
        loss = torch.zeros((), device=device)
        for t in range(rounds):
            d = model(d, is_src, w, adj)
            if hints:
                loss = loss + torch.nn.functional.mse_loss(d, target[t + 1])
        if not hints:
            loss = torch.nn.functional.mse_loss(d, target[rounds])
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return model, last


@torch.no_grad()
def evaluate(model: MPNN, dataset, rounds: int, device: str = "cuda") -> float:
    """Mean absolute error on the final estimate, in original units."""
    if hasattr(model, "reset"):
        model.reset()
    adj, w, is_src, target = pack(dataset, device)
    d = target[0].clone()
    for _ in range(rounds):
        d = model(d, is_src, w, adj)
    return float((d - target[rounds]).abs().mean() * SCALE)


@torch.no_grad()
def trajectory_mae(model: MPNN, dataset, rounds: int, device: str = "cuda") -> float:
    """Mean absolute deviation from the dataset's *whole* hint trajectory.

    The manipulation check for a hint-target experiment. `evaluate` only compares final
    answers, and when both reference algorithms converge within the round budget their
    final answers agree exactly -- so it is blind to which trajectory a model followed,
    which is the entire manipulated variable. Scoring a model against both trajectories
    is what establishes that the training difference was actually installed, before any
    fingerprint is allowed to be read.
    """
    if hasattr(model, "reset"):
        model.reset()
    adj, w, is_src, target = pack(dataset, device)
    d = target[0].clone()
    total = 0.0
    for t in range(rounds):
        d = model(d, is_src, w, adj)
        total += float((d - target[t + 1]).abs().mean())
    return total / rounds * SCALE
