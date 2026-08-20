"""E6: closing the Dijkstra/Prim boundary with a probe that dissociates their keys.

Dijkstra and Prim are separated by exactly one intervention, `lower_tree_edge_weight`,
which moves both algorithms' selection keys at once (`d[u]+w` for Dijkstra, `w` for
Prim) and infers the boundary from the *manner* of their responses rather than from one
responding and the other not. `dissociate_keys` (`fpid/interventions.py`) raises
`w(u,v)` while simultaneously lowering `d[u]` by the same amount, so Dijkstra's key is
exactly invariant and Prim's has strictly moved -- in principle a sharper, more direct
probe of the same boundary. `raise_tree_edge_weight` is the control arm: it moves both
keys, exactly as `lower_tree_edge_weight` already does.

Decision rule, the mechanism gate and outcome ladder are fixed in
`PREREGISTRATION_E5E6.md`, written before this ran.

    python run_e6.py
    python run_e6.py --smoke
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fpid.algorithms import ALGORITHMS, d_equal
from fpid.fingerprint import PREDICATES, TOL, aggregate, probe
from fpid.graphs import high_diameter_graph, random_connected_graph
from fpid.interventions import INTERVENTIONS, compute_targets, firing_round
from fpid.separation import minimal_separating_set, separation_matrix, unseparated_pairs

IV_BY_NAME = {iv.name: iv for iv in INTERVENTIONS}


def node_diff_count(ctor, g, src, iv, budget, tol=TOL) -> int:
    """Number of nodes whose final `d` differs between the intervened and control runs
    -- the literal reading of "response strength" the mechanism check is stated on."""
    targets = compute_targets(g, src)
    control = ctor(g.copy(), src).run(budget)
    t = firing_round(control, budget, tol)
    g_live = g.copy()
    ex = ctor(g_live, src)
    fired = {"done": False}

    def hook(executor, r):
        if r == t and not fired["done"]:
            iv.apply(executor, targets)
            fired["done"] = True

    intervened = ex.run(budget, hook=hook)
    return int((~d_equal(intervened[-1].d, control[-1].d, tol)).sum())


def build_probe_graphs(n_graphs, n_nodes, seed, density, min_depth):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_graphs:
        g = random_connected_graph(n_nodes, rng, density=density)
        hop = g.hop_distances(0)
        if not np.isfinite(hop).any() or hop[np.isfinite(hop)].max() < min_depth:
            continue
        out.append((g, 0))
    return out


def mechanism_check(args) -> bool:
    print("=" * 70)
    print("MECHANISM CHECK -- did key-dissociation actually install?")
    print("=" * 70)
    graphs = build_probe_graphs(
        args.mc_graphs, args.nodes, args.dev_seed, args.density, args.min_depth
    )
    raise_iv, diss_iv = IV_BY_NAME["raise_tree_edge_weight"], IV_BY_NAME["dissociate_keys"]

    strengths = {"dijkstra": {"raise": [], "diss": []}, "prim": {"raise": [], "diss": []}}
    for g, src in graphs:
        for name in ("dijkstra", "prim"):
            ctor = ALGORITHMS[name]
            strengths[name]["raise"].append(node_diff_count(ctor, g, src, raise_iv, args.budget))
            strengths[name]["diss"].append(node_diff_count(ctor, g, src, diss_iv, args.budget))

    dj_r = np.array(strengths["dijkstra"]["raise"])
    dj_d = np.array(strengths["dijkstra"]["diss"])
    pr_r = np.array(strengths["prim"]["raise"])
    pr_d = np.array(strengths["prim"]["diss"])

    print(f"  dijkstra  raise mean={dj_r.mean():.2f}  dissociate mean={dj_d.mean():.2f}")
    print(f"  prim      raise mean={pr_r.mean():.2f}  dissociate mean={pr_d.mean():.2f}")

    dj_weaker = (dj_d < dj_r).mean()
    pr_comparable = (np.abs(pr_d - pr_r) <= 1).mean()
    print(f"\n  dijkstra weaker-under-dissociate on {dj_weaker:.0%} of instances "
          f"(need >= 80%)")
    print(f"  prim comparable (within 1) on {pr_comparable:.0%} of instances "
          f"(need >= 80%)")

    ok = dj_weaker >= 0.80 and pr_comparable >= 0.80
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  MC FAILED. The dissociation did not install as designed: Dijkstra's")
        print("  response to dissociate_keys is not weaker than to raise_tree_edge_weight.")
        print("  Reporting MANIPULATION FAILED and stopping -- per the pre-registration,")
        print("  a null read past this point would be about the construction, not the")
        print("  Dijkstra/Prim boundary.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--mc-graphs", type=int, default=30)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--dev-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--graphs", type=int, default=25)
    ap.add_argument("--out", default="e6_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.budget:
        args.budget = 2 * args.nodes
    if args.smoke:
        args.mc_graphs, args.graphs = 6, 6

    mc_ok = mechanism_check(args)
    result = {"mechanism_check": mc_ok}
    if not mc_ok:
        result["verdict"] = "MANIPULATION FAILED"
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=float)
        print(f"\nwrote {args.out}")
        return

    # -- primary and regression, only reached if MC passes --
    print("\n" + "=" * 70)
    print(f"PRIMARY -- Dijkstra/Prim separation with the two new interventions, seed {args.seed}")
    print("=" * 70)
    rng = np.random.default_rng(args.seed)
    graphs = []
    for _ in range(args.graphs):
        g = high_diameter_graph(args.nodes, rng)
        graphs.append((g, 0))
    fps = {}
    for name, ctor in ALGORITHMS.items():
        runs = []
        for g, src in graphs:
            row = {}
            for iv in INTERVENTIONS:
                res = probe(ctor, g, src, iv, args.budget)
                for p, v in res.items():
                    row[(iv.name, p)] = v
            runs.append(row)
        fps[name] = aggregate(runs)

    names = [iv.name for iv in INTERVENTIONS]
    matrix = separation_matrix(fps, names, args.stability)
    dp_ivs = matrix[("dijkstra", "prim")]
    print(f"  dijkstra vs prim separated by: {dp_ivs}  ({len(dp_ivs)} interventions)")
    primary_pass = len(dp_ivs) >= 2
    print(f"  {'PASS' if primary_pass else 'FAIL'} -- need >= 2")

    print("\n" + "=" * 70)
    print("REGRESSION -- did adding interventions change the minimal separating set?")
    print("=" * 70)
    bad = unseparated_pairs(fps, names, args.stability)
    minimal = minimal_separating_set(fps, names, args.stability)
    print(f"  unseparated pairs: {bad or 'none'}")
    print(f"  minimal separating set (with E6 interventions): {minimal}")

    result |= {
        "verdict": "PRIMARY PASS" if primary_pass else "PRIMARY FAIL",
        "dijkstra_prim_separating": dp_ivs,
        "unseparated_pairs": bad,
        "minimal_separating_set": minimal,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
