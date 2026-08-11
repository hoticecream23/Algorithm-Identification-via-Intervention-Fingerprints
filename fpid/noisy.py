"""Noisy stand-ins for neural reasoners.

Phase A predicates use exact equality, which is free for a symbolic algorithm and
meaningless for a network. Wrapping the reference algorithms in noise tests whether
the predicate machinery survives continuous outputs *before* any model is trained.

Two independent noise axes, because they fail differently:

- **readout** noise perturbs only what an observer sees. The computation stays exact.
  This is decoder imprecision, and it is the axis the predicate tolerances must
  absorb.
- **execution** noise perturbs the actual state, so errors compound through the run.
  This is a genuinely worse model, and the fingerprint *should* eventually stop
  matching. Robustness to readout noise is required; robustness to execution noise
  would mean the fingerprint cannot see model quality at all.
"""

from __future__ import annotations

import numpy as np

from .algorithms import State


class NoisyExecutor:
    """Wraps an executor, delegating state access so interventions still work.

    Interventions mutate `ex.d` and `ex.graph` in place; attribute delegation means
    they reach the wrapped algorithm's real state without knowing about the wrapper.
    """

    def __init__(
        self,
        base,
        sigma_readout: float = 0.0,
        sigma_exec: float = 0.0,
        sigma_state: float = 0.0,
        sigma_input: float = 0.0,
        seed: int = 0,
    ):
        self.base = base
        self.sigma_readout = sigma_readout
        self.sigma_exec = sigma_exec
        self.sigma_state = sigma_state
        self.sigma_input = sigma_input
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def d(self):
        return self.base.d

    @d.setter
    def d(self, value):
        self.base.d = value

    def _perturb(self, d: np.ndarray, sigma: float) -> np.ndarray:
        """Perturb with noise that is a deterministic function of (seed, round, node).

        Always draw a full-width vector and mask, never `normal(size=finite.sum())`.
        A size-dependent draw desynchronises the RNG stream the instant two rollouts
        differ in how many nodes are reachable, which makes the noise *independent*
        between the control and intervened runs -- and then every predicate is
        reading sampling noise rather than the algorithm's response.

        Determinism here is not a convenience, it is fidelity: a trained network
        returns the same output for the same input, so its imprecision cancels
        between two rollouts wherever the computation agrees. Freshly resampled
        noise would model an ensemble, which is not what gets fingerprinted.
        """
        if sigma <= 0:
            return d
        noise = self.rng.normal(0.0, sigma, d.shape[0])
        out = d.copy()
        finite = np.isfinite(out)
        out[finite] += noise[finite]
        return out

    def _state_dependent(self, d: np.ndarray, eps: float) -> np.ndarray:
        """Error that is a deterministic function of the estimate itself.

        This is the only faithful axis, and the one that can actually break the
        fingerprint. `sigma_readout` and `sigma_exec` are common-mode: the same draw
        lands on the control, intervened and reference rollouts, so it cancels in
        every differential comparison and the fingerprint is near-invariant to it by
        construction. A trained network's error is not common-mode -- it is a
        function of the input, so poking the graph changes the input and therefore
        changes the error, and nothing cancels.

        Bounded and oscillating in the value, so it does not simply rescale `d`.
        """
        if eps <= 0:
            return d
        out = d.copy()
        finite = np.isfinite(out)
        x = out[finite]
        out[finite] = x + eps * np.sin(12.9898 * x) * np.cos(78.233 * x)
        return out

    def _input_dependent(self, d: np.ndarray, eps: float) -> np.ndarray:
        """Error that is a deterministic function of the *whole* input.

        The faithful model, and the only axis that can actually degrade the
        fingerprint. `sigma_state` was still common-mode: it makes each node's error
        a function of that node's own estimate, so a node whose value the
        intervention did not change keeps exactly the same error and it cancels in
        the differential comparison. That is why it moved the margin by literally
        zero.

        A network's error is not per-node. It is a function of the entire graph and
        the entire state, so intervening anywhere shifts the error *everywhere*,
        including at nodes whose true value is untouched. Keying the perturbation on
        a checksum of the whole state reproduces that, and breaks cancellation.
        """
        if eps <= 0:
            return d
        finite = np.isfinite(d)
        checksum = float(d[finite].sum()) + float(np.isfinite(self.base.graph.W).sum())
        idx = np.arange(d.shape[0], dtype=float)
        phase = np.sin(12.9898 * idx + 78.233 * self.base.round + 37.719 * checksum)
        out = d.copy()
        out[finite] += eps * phase[finite]
        return out

    def snapshot(self) -> State:
        s = self.base.snapshot()
        d = self._perturb(s.d, self.sigma_readout)
        d = self._state_dependent(d, self.sigma_state)
        return State(self._input_dependent(d, self.sigma_input), s.pi, s.round)

    def step(self) -> bool:
        advanced = self.base.step()
        if advanced and self.sigma_exec > 0:
            self.base.d = self._perturb(self.base.d, self.sigma_exec)
        return advanced

    def run(self, budget: int, hook=None) -> list[State]:
        traj = [self.snapshot()]
        for r in range(budget):
            if hook is not None:
                hook(self, r)
            self.step()
            traj.append(self.snapshot())
        return traj


def noisy_ctor(
    ctor,
    sigma_readout: float = 0.0,
    sigma_exec: float = 0.0,
    sigma_state: float = 0.0,
    sigma_input: float = 0.0,
    seed: int = 0,
):
    """Turn an algorithm constructor into a noisy one with the same signature."""

    def make(graph, source):
        return NoisyExecutor(
            ctor(graph, source),
            sigma_readout,
            sigma_exec,
            sigma_state,
            sigma_input,
            seed,
        )

    return make
