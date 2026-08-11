"""A small message-passing reasoner that the intervention harness can poke.

Architecture constraint, not a style choice: the recurrent state is carried in the
*decoded* space. Each round encodes the current distance estimate, passes messages,
and decodes the next estimate. If the recurrence lived only in a latent vector,
writing to `ex.d` mid-run would change nothing and every state intervention would be
inexpressible -- the harness could only deliver graph edits.

So: fingerprinting state interventions requires state-encoded recurrence. A
latent-only model is fingerprintable on graph interventions alone, which the
identifiability results say is not enough to separate the interesting pairs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .algorithms import UNREACHED, State
from .graphs import Graph

SCALE = 50.0  # rough magnitude of a path length here (~39 max); keeps inputs order-1


class MPNN(nn.Module):
    """Dense message passing. n is small, so an n x n formulation is simplest."""

    def __init__(self, hidden: int = 64, agg: str = "min"):
        super().__init__()
        # Relaxation is a *min* over d_u + w_uv, so min-aggregation matches the task
        # family. This is an architectural prior of the same kind as CLRS's max-
        # aggregation, not the algorithm hard-coded: the messages are still learned,
        # and a model is free to converge on a shortcut instead of Bellman-Ford --
        # which is precisely what the hint-free regime is meant to let happen.
        self.agg = agg
        self.enc = nn.Linear(2, hidden)
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.dec = nn.Linear(hidden, 1)

    def forward(
        self, d: torch.Tensor, is_src: torch.Tensor, w: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """One round. d: (B,n) scaled, w: (B,n,n) scaled, adj: (B,n,n) bool."""
        h = self.enc(torch.stack([d, is_src], dim=-1))
        b, n, hid = h.shape

        src = h.unsqueeze(2).expand(b, n, n, hid)
        dst = h.unsqueeze(1).expand(b, n, n, hid)
        m = self.msg(torch.cat([src, dst, w.unsqueeze(-1)], dim=-1))

        if self.agg == "min":
            m = m.masked_fill(~adj.unsqueeze(-1), float("inf"))
            agg = torch.nan_to_num(m.min(dim=1).values, posinf=0.0)
        else:
            m = m.masked_fill(~adj.unsqueeze(-1), float("-inf"))
            agg = torch.nan_to_num(m.max(dim=1).values, neginf=0.0)

        h2 = self.upd(torch.cat([h, agg], dim=-1))
        # Residual on the estimate: the target update is a small correction to d, and
        # predicting d outright makes the model relearn identity every round.
        return d + self.dec(h2).squeeze(-1)


class GatedMPNN(nn.Module):
    """MPNN with Dijkstra-like commitment: a reached node that stops changing freezes.

    E1 established that this difference cannot be installed by supervision -- the
    architecture has no global readout, so it cannot learn Dijkstra's argmin over
    unsettled nodes and never reaches admissible accuracy. A gate sidesteps that
    entirely: commitment is a *local* rule (this node stopped moving, so it is done)
    that produces the one property a corrupted-node probe is designed to detect, with
    no global comparison anywhere.

    Two conditions, and the second is not optional:

    - the estimate moved by less than `eps` on this round, and
    - the node is *reached*, i.e. its estimate is below the sentinel.

    Without the reached test every node still sitting at UNREACHED freezes on round 1 --
    it has not changed, because nothing has arrived yet -- and the model commits to the
    sentinel everywhere before any information propagates. Dijkstra settles nodes with a
    finite key for the same reason.

    Freezing is sticky and non-differentiable; gradients flow through the unfrozen path,
    and a node's pre-freeze history stays in the graph.
    """

    def __init__(
        self,
        hidden: int = 64,
        agg: str = "min",
        eps: float = 0.002,
        reached: float = 0.9 * UNREACHED / SCALE,
    ):
        super().__init__()
        self.core = MPNN(hidden, agg)
        self.eps = eps
        self.reached = reached
        self.frozen: torch.Tensor | None = None
        self.freeze_events = 0
        self.node_rounds = 0

    def reset(self) -> None:
        """Clear commitment state. Must run at the start of every rollout."""
        self.frozen = None

    def forward(self, d, is_src, w, adj):
        new = self.core(d, is_src, w, adj)
        if self.frozen is None or self.frozen.shape != d.shape:
            self.frozen = torch.zeros_like(d, dtype=torch.bool)
        out = torch.where(self.frozen, d, new)
        with torch.no_grad():
            settled = (out.abs() < self.reached) & ((out - d).abs() < self.eps)
            self.frozen = self.frozen | settled
            self.freeze_events += int(self.frozen.sum())
            self.node_rounds += int(self.frozen.numel())
        return out

    def freeze_rate(self) -> float:
        """Fraction of node-rounds spent frozen. 0 means the gate never fired."""
        return self.freeze_events / max(1, self.node_rounds)


def graph_tensors(graph: Graph, source: int, device) -> tuple:
    adj = torch.from_numpy(np.isfinite(graph.W)).to(device).unsqueeze(0)
    w = torch.from_numpy(np.where(np.isfinite(graph.W), graph.W, 0.0)).float()
    w = (w / SCALE).to(device).unsqueeze(0)
    is_src = torch.zeros(1, graph.n, device=device)
    is_src[0, source] = 1.0
    return adj, w, is_src


class NeuralExecutor:
    """Adapts a trained MPNN to the executor interface the harness expects.

    Exposes `.d`, `.pi`, `.graph` and steps one round at a time, so interventions
    mutate real state mid-rollout exactly as they do for a symbolic algorithm.
    """

    name = "neural"

    def __init__(
        self,
        model: MPNN,
        graph: Graph,
        source: int,
        device="cpu",
        max_rounds: int | None = None,
    ):
        self.model = model
        self.graph = graph
        self.source = source
        self.device = device
        # A model trained for T rounds is deployed for T rounds and then stops, exactly
        # as truncated Bellman-Ford does. The halt is what makes limited round budget
        # observable: an intervention landing after it cannot be responded to at all,
        # which is the signal that separates truncation from real relaxation even on
        # graphs shallow enough that both produce identical answers.
        self.max_rounds = max_rounds
        # A gated model carries commitment state across rounds, and `probe` runs three
        # rollouts against one model object. Without this the intervened rollout would
        # inherit the control rollout's frozen set and every state probe would read a
        # stale mask.
        if hasattr(model, "reset"):
            model.reset()
        self.d = np.full(graph.n, UNREACHED)
        self.d[source] = 0.0
        self.pi = np.full(graph.n, -1, dtype=int)
        self.round = 0
        self.finished = False

    def _decode_pi(self) -> np.ndarray:
        """Predecessor pointers read off the current estimate.

        The model has no pointer head, so pi is decoded the way CLRS decodes one --
        argmin over incoming relaxations. It is a decoded observable either way, which
        is all the predicates require.
        """
        best = np.full(self.graph.n, -1, dtype=int)
        for v in range(self.graph.n):
            cands = [
                (self.d[u] + w, u)
                for u, w in self.graph.neighbours(v)
                if np.isfinite(self.d[u])
            ]
            if cands:
                best[v] = min(cands)[1]
        return best

    def snapshot(self) -> State:
        return State(self.d.copy(), self.pi.copy(), self.round)

    @torch.no_grad()
    def step(self) -> bool:
        if self.finished:
            return False
        if self.max_rounds is not None and self.round >= self.max_rounds:
            self.finished = True
            return False
        adj, w, is_src = graph_tensors(self.graph, self.source, self.device)
        d = torch.from_numpy(self.d / SCALE).float().to(self.device).unsqueeze(0)
        out = self.model(d, is_src, w, adj)
        self.d = (out.squeeze(0).cpu().numpy() * SCALE).astype(float)
        self.pi = self._decode_pi()
        self.round += 1
        return True

    def run(self, budget: int, hook=None) -> list[State]:
        traj = [self.snapshot()]
        for r in range(budget):
            if hook is not None:
                hook(self, r)
            self.step()
            traj.append(self.snapshot())
        return traj


def neural_ctor(model: MPNN, device="cpu", max_rounds: int | None = None):
    """Constructor with the same (graph, source) signature the harness uses."""

    def make(graph, source):
        return NeuralExecutor(model, graph, source, device, max_rounds)

    return make
