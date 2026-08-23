"""Which algorithms are distinguishable at which graph diameter?

Phase A found Dijkstra and SPFA behaviourally identical on dense graphs and separable
on sparse high-diameter ones. That makes behavioural equivalence *distribution*-
relative as well as intervention-relative, which is a stronger claim than the project
started with -- and a blocking decision, since Phase C cannot choose its graph
distribution until the boundary is mapped.

Node count is held fixed and density swept, so diameter varies with `n` controlled.

The falsifiable prediction was: truncated Bellman-Ford with k=3 should be
indistinguishable from real Bellman-Ford whenever the diameter is <= 3, and separable
above it. It was wrong -- they separate at *every* depth, because truncation limits
total runtime as well as propagation depth, and a mid-run probe sees the former on any
instance. That is P4, and it is a better result than the prediction.

Depth is reported on two axes. `mean_diameter` is unweighted hop eccentricity, the
original axis; `mean_weighted_depth` is rounds of relaxation needed to settle. They
differ by 1-2 rounds on these generators because relaxation follows minimum-*weight*
paths, which use more edges than minimum-*hop* paths. Quote the weighted one: the
hop axis was what made the density-0.8 sample look homogeneous when it was in fact a
mixture of two regimes (see run_p2_witness.py).

    python run_identifiability.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from fpid.algorithms import ALGORITHMS
from fpid.fingerprint import aggregate, probe
from fpid.graphs import random_connected_graph, weighted_depth
from fpid.interventions import INTERVENTIONS
from fpid.separation import minimal_separating_set, separation_matrix


def mean_diameter(graphs) -> float:
    """Mean unweighted eccentricity from the source.

    Retained for continuity with the original sweep, but it is *not* the depth
    relaxation sees -- see `mean_weighted_depth` and `fpid.graphs.weighted_depth`.
    Every depth quoted in a result should be the weighted one.
    """
    out = []
    for g, src in graphs:
        hop = g.hop_distances(src)
        finite = hop[np.isfinite(hop)]
        out.append(float(finite.max()) if finite.size else 0.0)
    return float(np.mean(out))


def mean_weighted_depth(graphs) -> float:
    """Mean rounds of synchronous relaxation needed to settle.

    This is the axis the identifiability boundary should be read against: it counts
    edges on minimum-*weight* paths, which is what an intervention's effect actually
    has to travel along, and it runs 1-2 rounds above hop eccentricity on these
    generators.
    """
    return float(np.mean([weighted_depth(g, src) for g, src in graphs]))


def fingerprints_at(graphs, budget, stability):
    fps = {}
    for name, ctor in ALGORITHMS.items():
        runs = []
        for g, src in graphs:
            row = {}
            for iv in INTERVENTIONS:
                for p, v in probe(ctor, g, src, iv, budget).items():
                    row[(iv.name, p)] = v
            runs.append(row)
        fps[name] = aggregate(runs)
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability", type=float, default=0.8)
    args = ap.parse_args()

    densities = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    budget = 2 * args.nodes
    names = [iv.name for iv in INTERVENTIONS]
    pairs = list(combinations(sorted(ALGORITHMS), 2))

    rows = []
    for density in densities:
        rng = np.random.default_rng(args.seed)
        graphs = [
            (random_connected_graph(args.nodes, rng, density=density), 0)
            for _ in range(args.graphs)
        ]
        diam = mean_diameter(graphs)
        wdep = mean_weighted_depth(graphs)
        fps = fingerprints_at(graphs, budget, args.stability)
        matrix = separation_matrix(fps, names, args.stability)
        minimal = minimal_separating_set(fps, names, args.stability)
        rows.append((density, diam, wdep, matrix, minimal))
        print(f"  swept density={density} (hop {diam:.1f}, weighted {wdep:.1f})")

    print("\n=== Separated pairs vs graph depth ===")
    header = f"{'density':<10}{'w.depth':<9}{'hop':<7}" + "".join(
        f"{a[:9]}/{b[:9]:<12}" for a, b in pairs
    )
    print(header)
    for density, diam, wdep, matrix, _minimal in rows:
        cells = []
        for pair in pairs:
            ivs = matrix[pair]
            cells.append(f"{('YES(' + str(len(ivs)) + ')') if ivs else 'no':<22}")
        print(f"{density:<10}{wdep:<9.1f}{diam:<7.1f}" + "".join(cells))

    print("\n=== Minimal separating set vs weighted depth ===")
    print("  (weighted depth is the axis to quote; hop depth is shown for continuity)")
    for density, diam, wdep, _matrix, minimal in rows:
        size = len(minimal) if minimal else "impossible"
        print(
            f"  density {density:<6} weighted depth {wdep:<6.1f} (hop {diam:<5.1f}) -> {size}"
        )

    print("\n=== Truncation: bf vs bf_truncated_k3 ===")
    print("  Separable at every depth -- truncation limits total runtime, which a")
    print("  mid-run probe sees regardless of how deep the instance is (see")
    print("  run_p2_witness.py for the exact-output-agreement version of this).")
    key = tuple(sorted(["bellman_ford", "bf_truncated_k3"]))
    for density, diam, wdep, matrix, _m in rows:
        ivs = matrix[key]
        verdict = "separable" if ivs else "INSEPARABLE"
        print(f"  weighted depth {wdep:<6.1f} {verdict:<14} ({len(ivs)} interventions)")


if __name__ == "__main__":
    main()
