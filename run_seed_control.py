"""Seed control: does the fingerprint measure the algorithm, or the model?

The gate before the full zoo. Train several models on the same algorithm with
different seeds and fingerprint each. If two seeds of the same algorithm are as far
apart as two different algorithms, the fingerprint is reading model idiosyncrasy and
no identification result built on it means anything.

Required: cross-seed distance << between-algorithm distance.

    python run_seed_control.py --seeds 6 --epochs 200
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, UNREACHED, finite_init
from fpid.fingerprint import aggregate, probe
from fpid.graphs import random_connected_graph
from fpid.identify import classify, distance, margin
from fpid.interventions import INTERVENTIONS
from fpid.neural import neural_ctor
from fpid.train import evaluate, make_dataset, train_model


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
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=200)
    # Between the model's error floor (~0.7 MAE) and the scale of real value gaps
    # (~5, the edge-weight scale). Both bounds are required: below the error floor the
    # predicates read model noise, above the value gaps they stop seeing real
    # differences.
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--hints", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    train_set = make_dataset(args.train_graphs, args.nodes, args.rounds, rng)
    test_set = make_dataset(8, args.nodes, args.rounds, rng)
    probe_graphs = [
        (random_connected_graph(args.nodes, rng, density=0.05), 0)
        for _ in range(args.probe_graphs)
    ]
    # Rollouts need only enough rounds to converge (depth ~5) plus room to respond.
    budget = args.budget if args.budget else 2 * args.nodes

    print(f"device={device}  depth-5 graphs  tol={args.tol}")

    # References measured with the same finite sentinel the network uses, and at the
    # same tolerance -- both sides of a comparison must share the measurement setup.
    refs = {
        name: fingerprint_of(finite_init(ctor), probe_graphs, budget, args.tol)
        for name, ctor in ALGORITHMS.items()
    }

    models, fps = {}, {}
    for seed in range(args.seeds):
        model, loss = train_model(
            train_set,
            args.rounds,
            seed=seed,
            hints=args.hints,
            epochs=args.epochs,
            device=device,
        )
        mae = evaluate(model, test_set, args.rounds, device)
        models[seed] = (model, mae)
        fps[seed] = fingerprint_of(
            neural_ctor(model, device), probe_graphs, budget, args.tol
        )
        print(f"  seed {seed}: train loss {loss:.5f}  test MAE {mae:.3f}")

    print("\n=== Cross-seed distances (same algorithm, different seeds) ===")
    cross = [
        distance(fps[a], fps[b], args.stability) for a, b in combinations(fps, 2)
    ]
    for (a, b), dist in zip(combinations(fps, 2), cross):
        print(f"  seed {a} vs seed {b}: {dist:.3f}")
    print(f"  mean {np.mean(cross):.3f}   max {np.max(cross):.3f}")

    print("\n=== Between-algorithm distances (symbolic references) ===")
    between = [
        distance(refs[a], refs[b], args.stability)
        for a, b in combinations(sorted(refs), 2)
    ]
    print(f"  mean {np.mean(between):.3f}   min {np.min(between):.3f}")

    print("\n=== Identification of each seed against symbolic references ===")
    for seed in fps:
        pred, dists = classify(fps[seed], refs, args.stability)
        best = ", ".join(f"{k}={v:.2f}" for k, v in sorted(dists.items(), key=lambda x: x[1])[:3])
        print(
            f"  seed {seed} (MAE {models[seed][1]:.2f}) -> {pred:<16} "
            f"margin {margin(dists):.3f}   [{best}]"
        )

    print("\n=== Gate ===")
    # Three criteria, because the single obvious one is misleading. Benchmarking seed
    # noise against the *closest* reference pair fails on a boundary no model is near:
    # dijkstra/prim sit ~0.02 apart and A3 already showed that pair is degenerate,
    # separated by one intervention. Seed noise exceeding that says nothing about
    # whether a model's own label is stable.
    labels = {seed: classify(fps[seed], refs, args.stability)[0] for seed in fps}
    margins = [
        margin(classify(fps[seed], refs, args.stability)[1]) for seed in fps
    ]
    consistent = len(set(labels.values())) == 1
    clustering = np.mean(between) / max(np.mean(cross), 1e-9)

    print(f"  1. label consistency : {'PASS' if consistent else 'FAIL'} "
          f"({len(set(labels.values()))} distinct label(s) across {len(fps)} seeds)")
    print(f"  2. clustering        : {'PASS' if clustering > 3 else 'FAIL'} "
          f"(between/cross = {clustering:.1f}x)")
    safe = np.max(cross) < np.min(margins)
    print(f"  3. margin headroom   : {'PASS' if safe else 'MARGINAL'} "
          f"(worst cross-seed {np.max(cross):.3f} vs smallest margin "
          f"{np.min(margins):.3f})")

    if consistent and clustering > 3 and not safe:
        print("\n  Identity is seed-stable and the clusters are well separated, but")
        print("  seed noise is comparable to the thinnest classification margin.")
        print("  Single-model labels near a boundary are therefore not reliable:")
        print("  report the margin alongside the label, and prefer an ensemble of")
        print("  seeds when a model sits close to two references.")


if __name__ == "__main__":
    main()
