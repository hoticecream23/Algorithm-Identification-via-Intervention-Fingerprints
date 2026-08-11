"""Phase A gate: do intervention fingerprints separate the *symbolic* algorithms?

No neural networks. The entire project rests on the assumption that causal
responses distinguish algorithms; that assumption is testable on the reference
implementations alone, and if it fails here no result about learned models can
exist. Cheap to run, and it fails fast.

    python run_phase_a.py
"""

from __future__ import annotations

import argparse

import numpy as np

from fpid.algorithms import ALGORITHMS
from fpid.fingerprint import PREDICATES, aggregate, probe
from fpid.graphs import high_diameter_graph, random_connected_graph
from fpid.interventions import INTERVENTIONS
from fpid.separation import (
    minimal_separating_set,
    separation_matrix,
    unseparated_pairs,
)


def build_fingerprints(n_graphs: int, n_nodes: int, seed: int, family: str):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        if family == "high_diameter":
            g = high_diameter_graph(n_nodes, rng)
        else:
            g = random_connected_graph(n_nodes, rng)
        graphs.append((g, 0))

    budget = 2 * n_nodes
    fps = {}
    for name, ctor in ALGORITHMS.items():
        runs = []
        for g, src in graphs:
            row = {}
            for iv in INTERVENTIONS:
                res = probe(ctor, g, src, iv, budget)
                for p, v in res.items():
                    row[(iv.name, p)] = v
            runs.append(row)
        fps[name] = aggregate(runs)
        print(f"  fingerprinted {name}")
    return fps


def print_fingerprint_table(fps, stability):
    algs = sorted(fps)
    print("\n=== Fingerprints (modal predicate; * = unstable, excluded) ===")
    for iv in INTERVENTIONS:
        print(f"\n{iv.name}  -- {iv.note}")
        header = f"  {'predicate':<12}" + "".join(f"{a:<18}" for a in algs)
        print(header)
        for p in PREDICATES:
            cells = []
            for a in algs:
                k = (iv.name, p)
                v = fps[a].value.get(k, "?")
                mark = "" if fps[a].stability.get(k, 0) >= stability else "*"
                cells.append(f"{v + mark:<18}")
            row = f"  {p:<12}" + "".join(cells)
            distinct = len({fps[a].value.get((iv.name, p)) for a in algs})
            print(row + ("   <- separating" if distinct > 1 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=25)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument(
        "--family", choices=["high_diameter", "random"], default="high_diameter"
    )
    args = ap.parse_args()

    print(
        f"Phase A: {len(ALGORITHMS)} algorithms x {len(INTERVENTIONS)} interventions "
        f"x {args.graphs} graphs (n={args.nodes}, {args.family})"
    )
    fps = build_fingerprints(args.graphs, args.nodes, args.seed, args.family)

    print_fingerprint_table(fps, args.stability)

    names = [iv.name for iv in INTERVENTIONS]
    print("\n=== Pairwise separation ===")
    matrix = separation_matrix(fps, names, args.stability)
    for (a, b), ivs in matrix.items():
        status = ", ".join(ivs) if ivs else "NOT SEPARATED"
        print(f"  {a:<16} vs {b:<16} : {status}")

    bad = unseparated_pairs(fps, names, args.stability)
    minimal = minimal_separating_set(fps, names, args.stability)

    print("\n=== Phase A gate ===")
    if bad:
        print("  FAIL - these pairs are behaviourally indistinguishable:")
        for a, b in bad:
            print(f"    {a} == {b}")
        print("  Either enrich the intervention set or accept them as equivalent")
        print("  under this class -- both are results, but they change the paper.")
    else:
        print("  PASS - every pair is separated by at least one stable predicate.")
    print(f"\n  Minimal separating set: {minimal}")
    if minimal:
        print(f"  {len(minimal)} of {len(names)} interventions suffice.")


if __name__ == "__main__":
    main()
