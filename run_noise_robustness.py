"""How much decoder noise can the fingerprint survive?

Phase A predicates use exact equality, which is free for symbolic algorithms and
impossible for a network. If the predicates dissolve under realistic decoder noise,
Phase C fails for a reason that has nothing to do with any model -- so test it now,
with no training, against noisy copies of the reference algorithms where ground truth
is known.

Two sweeps:
  readout   -- observation is noisy, computation is exact. Identification MUST hold.
  execution -- the state itself is corrupted, errors compound. Identification SHOULD
               degrade; a fingerprint blind to this could not detect a bad model.

    python run_noise_robustness.py
"""

from __future__ import annotations

import argparse

import numpy as np

from fpid.algorithms import ALGORITHMS
from fpid.fingerprint import aggregate, probe
from fpid.graphs import high_diameter_graph
from fpid.identify import classify, margin
from fpid.interventions import INTERVENTIONS
from fpid.noisy import noisy_ctor


def make_graphs(n_graphs, n_nodes, seed):
    rng = np.random.default_rng(seed)
    return [(high_diameter_graph(n_nodes, rng), 0) for _ in range(n_graphs)]


def fingerprint_of(ctor, graphs, budget, tol):
    runs = []
    for g, src in graphs:
        row = {}
        for iv in INTERVENTIONS:
            for p, v in probe(ctor, g, src, iv, budget, tol=tol).items():
                row[(iv.name, p)] = v
        runs.append(row)
    return aggregate(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=15)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--tol-mult", type=float, default=4.0)
    ap.add_argument("--axis", default=None, help="run a single noise axis")
    ap.add_argument(
        "--fixed-tol",
        type=float,
        default=None,
        help=(
            "Hold tolerance fixed instead of tracking sigma. Tying tol to sigma "
            "measures nothing when the perturbation is bounded by sigma -- the "
            "tolerance is then larger than the error by construction. In Phase C "
            "tol is an estimate of an unknown noise floor, so the mis-calibrated "
            "case is the one that matters."
        ),
    )
    args = ap.parse_args()

    graphs = make_graphs(args.graphs, args.nodes, args.seed)
    budget = 2 * args.nodes

    # References must be measured at the SAME tolerance as the model being
    # classified. Tolerance is part of the measurement apparatus -- it changes which
    # differences count as differences, and it shifts the firing round via
    # settle_round -- so a noiseless reference read at tol=1e-6 is not comparable to
    # a noisy model read at tol=0.04. Matching them is the honest protocol, and in
    # Phase C means recomputing symbolic references at the model's noise floor.
    _ref_cache: dict[float, dict] = {}

    def refs_at(tol: float) -> dict:
        if tol not in _ref_cache:
            _ref_cache[tol] = {
                name: fingerprint_of(ctor, graphs, budget, tol=tol)
                for name, ctor in ALGORITHMS.items()
            }
        return _ref_cache[tol]

    base = refs_at(1e-6)
    n_stable = {n: len(f.stable_keys(args.stability)) for n, f in base.items()}
    print(f"stable predicates per reference (tol=1e-6): {n_stable}\n")

    sigmas = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    axes = {
        "readout": "sigma_readout",
        "execution": "sigma_exec",
        "state-dependent": "sigma_state",
        "input-dependent": "sigma_input",
    }
    if args.axis:
        axes = {args.axis: axes[args.axis]}
    for axis in axes:
        print(f"=== {axis} noise ===")
        print(f"  {'sigma':<8}{'tol':<8}{'accuracy':<11}{'mean margin':<13}errors")
        for sigma in sigmas:
            # Tolerance tracks the noise floor rather than being hand-set: a
            # predicate must ignore differences the decoder could have invented.
            tol = (
                args.fixed_tol
                if args.fixed_tol is not None
                else max(1e-6, args.tol_mult * sigma)
            )
            refs = refs_at(tol)
            correct, margins, errors = 0, [], []
            for name, ctor in ALGORITHMS.items():
                nctor = noisy_ctor(ctor, seed=args.seed, **{axes[axis]: sigma})
                fp = fingerprint_of(nctor, graphs, budget, tol=tol)
                pred, dists = classify(fp, refs, args.stability)
                if pred == name:
                    correct += 1
                else:
                    errors.append(f"{name}->{pred}")
                margins.append(margin(dists))
            acc = correct / len(ALGORITHMS)
            print(
                f"  {sigma:<8.2f}{tol:<8.3f}{acc:<11.0%}{np.mean(margins):<13.3f}"
                f"{', '.join(errors) if errors else '-'}"
            )
        print()


if __name__ == "__main__":
    main()
