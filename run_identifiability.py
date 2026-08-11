"""Which algorithms are distinguishable at which graph diameter?

Phase A found Dijkstra and SPFA behaviourally identical on dense graphs and separable
on sparse high-diameter ones. That makes behavioural equivalence *distribution*-
relative as well as intervention-relative, which is a stronger claim than the project
started with -- and a blocking decision, since Phase C cannot choose its graph
distribution until the boundary is mapped.

Node count is held fixed and density swept, so diameter varies with `n` controlled.

The falsifiable prediction: truncated Bellman-Ford with k=3 should be indistinguishable
from real Bellman-Ford whenever the diameter is <= 3, and separable above it. An
algorithm that halts early is only detectably early on instances deep enough to halt
in. If that threshold does not land near k, the depth-limited-execution story behind
the whole OOD hypothesis is wrong.

    python run_identifiability.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from fpid.algorithms import ALGORITHMS
from fpid.fingerprint import aggregate, probe
from fpid.graphs import random_connected_graph
from fpid.interventions import INTERVENTIONS
from fpid.separation import minimal_separating_set, separation_matrix


def mean_diameter(graphs) -> float:
    """Mean unweighted eccentricity from the source, i.e. usable propagation depth."""
    out = []
    for g, src in graphs:
        hop = g.hop_distances(src)
        finite = hop[np.isfinite(hop)]
        out.append(float(finite.max()) if finite.size else 0.0)
    return float(np.mean(out))


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
        fps = fingerprints_at(graphs, budget, args.stability)
        matrix = separation_matrix(fps, names, args.stability)
        minimal = minimal_separating_set(fps, names, args.stability)
        rows.append((density, diam, matrix, minimal))
        print(f"  swept density={density} (mean depth {diam:.1f})")

    print("\n=== Separated pairs vs graph depth ===")
    header = f"{'density':<10}{'depth':<8}" + "".join(
        f"{a[:9]}/{b[:9]:<12}" for a, b in pairs
    )
    print(header)
    for density, diam, matrix, _minimal in rows:
        cells = []
        for pair in pairs:
            ivs = matrix[pair]
            cells.append(f"{('YES(' + str(len(ivs)) + ')') if ivs else 'no':<22}")
        print(f"{density:<10}{diam:<8.1f}" + "".join(cells))

    print("\n=== Minimal separating set vs depth ===")
    for density, diam, _matrix, minimal in rows:
        size = len(minimal) if minimal else "impossible"
        print(f"  density {density:<6} depth {diam:<6.1f} -> {size}")

    print("\n=== Truncation prediction: bf vs bf_truncated_k3 ===")
    print("  (should be inseparable at depth <= 3, separable above)")
    key = tuple(sorted(["bellman_ford", "bf_truncated_k3"]))
    for density, diam, matrix, _m in rows:
        ivs = matrix[key]
        verdict = "separable" if ivs else "INSEPARABLE"
        print(f"  depth {diam:<6.1f} {verdict:<14} ({len(ivs)} interventions)")


if __name__ == "__main__":
    main()
