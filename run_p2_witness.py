"""P2 witness: output-identical executors separated by intervention response.

This is the reproducing command for PAPER.md 5.2, which previously had none -- the
"11 of 12" and "0 of 12" figures were computed ad hoc and are superseded here.

What changed and why
--------------------
The original measurement bucketed graphs by *hop* depth (`hop_distances().max()`),
on the reasoning that a k-round algorithm is harmed only on instances deeper than k.
Hop depth is the wrong ruler. Relaxation propagates along minimum-*weight* paths,
which can use more edges than the minimum-*hop* path, so a density-0.8 graph whose
every node is two hops from the source routinely needs four or five rounds to
settle. The density-0.8 sample was therefore a mixture of two deterministic regimes,
and "identical on 11 of 12" was a small-sample draw from that mixture: at n = 200 the
same measurement gives ~70-76%, at which point truncated-BF is *also* 30 points less
accurate and the "no accuracy metric separates them" claim fails.

Keyed on `weighted_depth` instead, the mixture resolves into a step function at
exactly k, and the P2 claim strengthens from an empirical rate to a guarantee:
where `weighted_depth <= k` the two executors are output-identical on *every*
instance, and the fingerprint separates them anyway.

Both seed sets are generated up front and reported together. Nothing here selects a
statistic -- the quantities are fixed by PAPER.md 5.2 -- so this is a confirmatory
re-measurement, not a search.

    python run_p2_witness.py                 # ~10 min, numpy only
    python run_p2_witness.py --graphs 60     # ~3 min, same verdict
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from fpid.algorithms import ALGORITHMS
from fpid.fingerprint import aggregate, probe
from fpid.graphs import d_close, random_connected_graph, true_sssp, weighted_depth
from fpid.interventions import INTERVENTIONS
from fpid.separation import separating_predicates

PAIR = ("bellman_ford", "bf_truncated_k3")
K = 3  # the truncation bound of `bf_truncated_k3`
DENSITIES = [0.8, 0.4, 0.2, 0.1, 0.05]


def final_d(name: str, graph, source: int, budget: int) -> np.ndarray:
    return np.asarray(ALGORITHMS[name](graph, source).run(budget)[-1].d, dtype=float)


def draw(n_graphs: int, nodes: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    return [(random_connected_graph(nodes, rng, density=density), 0) for _ in range(n_graphs)]


class Draws:
    """Draw each (density, seed) cell once and measure it once.

    All four parts below read the same instances; without this they are redrawn and
    re-measured four times over, which dominates runtime on the cheap parts.
    """

    def __init__(self, n_graphs: int, nodes: int, budget: int):
        self.n_graphs, self.nodes, self.budget = n_graphs, nodes, budget
        self._cache: dict[tuple[float, int], tuple[list, list]] = {}

    def get(self, density: float, seed: int):
        key = (density, seed)
        if key not in self._cache:
            graphs = draw(self.n_graphs, self.nodes, density, seed)
            self._cache[key] = (graphs, measure(graphs, self.budget))
        return self._cache[key]


def measure(graphs, budget: int):
    """Per-instance depths, output agreement, and correctness."""
    rows = []
    for g, s in graphs:
        truth = true_sssp(g, s)
        reach = np.isfinite(truth)
        hop = g.hop_distances(s)
        hop_fin = hop[np.isfinite(hop)]
        bf = final_d(PAIR[0], g, s, budget)
        tr = final_d(PAIR[1], g, s, budget)
        rows.append(
            {
                "hop": float(hop_fin.max()) if hop_fin.size else 0.0,
                "wdepth": weighted_depth(g, s),
                "agree": d_close(bf[reach], tr[reach]),
                "bf_ok": d_close(bf[reach], truth[reach]),
                "tr_ok": d_close(tr[reach], truth[reach]),
            }
        )
    return rows


def fingerprint_pair(graphs, budget: int, stability: float):
    """Fingerprint both members of PAIR over exactly these graphs."""
    fps = {}
    for name in PAIR:
        runs = []
        for g, s in graphs:
            row = {}
            for iv in INTERVENTIONS:
                for p, v in probe(ALGORITHMS[name], g, s, iv, budget).items():
                    row[(iv.name, p)] = v
            runs.append(row)
        fps[name] = aggregate(runs)
    seps = {
        iv.name: separating_predicates(fps[PAIR[0]], fps[PAIR[1]], iv.name, stability)
        for iv in INTERVENTIONS
    }
    return {k: v for k, v in seps.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=200)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 7])
    ap.add_argument(
        "--fp-graphs",
        type=int,
        default=120,
        help="graphs per fingerprint bucket; the quoted result uses 120, and it also "
        "holds at 60 and at every size from 40 up",
    )
    ap.add_argument("--stability", type=float, default=0.8)
    args = ap.parse_args()
    budget = 2 * args.nodes
    draws = Draws(args.graphs, args.nodes, budget)

    print("=" * 74)
    print("PART 1 -- the two depth measures diverge")
    print("=" * 74)
    print("Hop eccentricity is what the superseded version of this result bucketed on.")
    print("It is not the depth relaxation sees.\n")
    print(f"{'density':<10}{'seed':<7}{'mean hop':<11}{'mean weighted':<16}{'weighted > hop'}")
    for density in DENSITIES:
        for seed in args.seeds:
            _, rows = draws.get(density, seed)
            hop = np.mean([r["hop"] for r in rows])
            wd = np.mean([r["wdepth"] for r in rows])
            frac = np.mean([r["wdepth"] > r["hop"] for r in rows])
            print(f"{density:<10}{seed:<7}{hop:<11.2f}{wd:<16.2f}{frac:.0%}")

    print()
    print("=" * 74)
    print(f"PART 2 -- the P2 witness, keyed on weighted depth (k = {K})")
    print("=" * 74)
    print("Prediction: a step function at k. Identical and correct on every instance")
    print("with weighted depth <= k; identical and correct on none above it.\n")

    for seed in args.seeds:
        buckets = defaultdict(lambda: [0, 0, 0, 0])
        for density in DENSITIES:
            for r in draws.get(density, seed)[1]:
                b = buckets[r["wdepth"]]
                b[0] += 1
                b[1] += int(r["agree"])
                b[2] += int(r["bf_ok"])
                b[3] += int(r["tr_ok"])
        print(f"seed={seed}  ({args.graphs} graphs x {len(DENSITIES)} densities)")
        print(f"  {'weighted depth':<18}{'n':<8}{'BF == trunc':<16}{'BF correct':<15}{'trunc correct'}")
        for w in sorted(buckets):
            n, ag, bo, to = buckets[w]
            mark = "  <- <= k" if w <= K else ""
            print(
                f"  {w:<18.0f}{n:<8}{ag}/{n:<13}{bo}/{n:<12}{to}/{n}{mark}"
            )
        print()

    print("=" * 74)
    print("PART 3 -- output-identical, and separated anyway")
    print("=" * 74)
    print("Fingerprints computed over the <= k bucket ONLY, where output agreement is")
    print("total. Any separation here cannot be an accuracy difference in disguise.\n")

    for seed in args.seeds:
        pool = []
        for density in DENSITIES:
            graphs, rows = draws.get(density, seed)
            pool += [g for g, r in zip(graphs, rows) if r["wdepth"] <= K and r["agree"]]
        sample = pool[: args.fp_graphs]
        seps = fingerprint_pair(sample, budget, args.stability)
        print(f"seed={seed}  {len(sample)} graphs, all output-identical")
        if not seps:
            print("  NO SEPARATION -- the P2 witness fails on this sample.")
        for name, preds in sorted(seps.items()):
            print(f"  {name:<34}{', '.join(preds)}")
        print()

    print("=" * 74)
    print("PART 4 -- the OOD half: correctness collapses as the distribution deepens")
    print("=" * 74)
    print(f"{'density':<10}{'seed':<7}{'mean weighted':<16}{'frac <= k':<12}{'trunc correct'}")
    for density in DENSITIES:
        for seed in args.seeds:
            _, rows = draws.get(density, seed)
            wd = np.mean([r["wdepth"] for r in rows])
            frac = np.mean([r["wdepth"] <= K for r in rows])
            ok = np.mean([r["tr_ok"] for r in rows])
            print(f"{density:<10}{seed:<7}{wd:<16.2f}{frac:<12.0%}{ok:.0%}")


if __name__ == "__main__":
    main()
