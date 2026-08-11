"""E1 + E2: does the fingerprint discriminate training provenance at all?

E1. Every neural model in this project so far was supervised on Bellman-Ford, so B0's
"all six models identify as bellman_ford" was never a discrimination test -- nothing
was available for them to be confused with, and the result is equally consistent with a
fingerprint that always answers Bellman-Ford. Half the models here are supervised on
Dijkstra trajectories instead.

E2. Everything so far was hint-supervised, so a matching fingerprint is close to
tautological. Hint-free supervises the final answer only.

The two experiments interlock, because at `rounds = n` both reference algorithms have
converged and their final estimates are identical: the hint-free loss cannot see which
algorithm was nominally used. So the hint-free group is six models trained on identical
data, and its label spread is a direct measurement of the false-discrimination rate of
the very statistic E1 relies on.

Decision rule, thresholds and outcome ladder are fixed in `PREREGISTRATION_E1E2.md` and
were written before this ran. Nothing here selects.

    python run_e1_e2.py
    python run_e1_e2.py --smoke        # minutes, checks plumbing only
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import Fingerprint, aggregate, probe
from fpid.graphs import random_connected_graph, true_sssp
from fpid.identify import classify, margin
from fpid.interventions import INTERVENTIONS
from fpid.neural import neural_ctor
from fpid.train import evaluate, make_dataset, train_model, trajectory_mae

SETTLING = {"dijkstra", "prim"}
RELAXATION = {"bellman_ford", "bf_truncated_k3", "spfa"}

# B0's worst cross-seed distance. A margin below this is seed noise, not identity.
NOISE_FLOOR = 0.094


def family(label: str) -> str:
    return "settling" if label in SETTLING else "relaxation"


def fingerprint_of(ctor, graphs, budget, tol) -> Fingerprint:
    runs = []
    for g, src in graphs:
        row = {}
        for iv in INTERVENTIONS:
            for p, v in probe(ctor, g, src, iv, budget, tol=tol).items():
                row[(iv.name, p)] = v
        runs.append(row)
    return aggregate(runs)


def drop_predicate(fp: Fingerprint, predicate: str) -> Fingerprint:
    """Fingerprint with one predicate removed from every intervention.

    Bellman-Ford goes quiet after ~5 rounds and Dijkstra after ~21, so `term_delta`
    alone could carry a discrimination that has nothing to do with how a perturbation
    propagates. Runtime is a legitimate observable (A3), so this decomposes the result
    rather than discounting it.
    """
    keep = {k for k in fp.value if k[1] != predicate}
    return Fingerprint(
        {k: fp.value[k] for k in keep},
        {k: fp.stability[k] for k in keep},
    )


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p. Hand-rolled to avoid a scipy dependency."""
    r1, r2 = a + b, c + d
    c1, n = a + c, a + b + c + d
    if min(r1, r2, c1, b + d) < 0 or n == 0:
        return 1.0
    denom = comb(n, c1)

    def p(k: int) -> float:
        return comb(r1, k) * comb(r2, c1 - k) / denom

    obs = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return float(min(1.0, sum(p(k) for k in range(lo, hi + 1) if p(k) <= obs * 1.000000001)))


def deep_probe_graphs(n_graphs, n_nodes, rng, density, min_depth):
    """Probe graphs rejection-sampled on propagation depth (constraint 1)."""
    out = []
    while len(out) < n_graphs:
        g = random_connected_graph(n_nodes, rng, density=density)
        hop = g.hop_distances(0)
        if hop[np.isfinite(hop)].max() < min_depth:
            continue
        out.append((g, 0))
    return out


def true_distance_dataset(dataset, at: int = -1):
    """Put true SSSP distances at row `at`, which is where `evaluate` will look.

    Admission has to be measured against the right answer, not against whichever
    trajectory a model was trained on -- otherwise a Dijkstra-trained model is graded on
    a different scale from a Bellman-Ford one and constraint 9's threshold means two
    different things.

    `at` matters for any model deployed at fewer rounds than the dataset was built for:
    `evaluate(model, ds, k)` compares against row `k`, so writing the answer only into
    the last row would silently grade a k-round model against Bellman-Ford's round-k
    state instead. Those happen to nearly agree here, which is exactly why it would go
    unnoticed.
    """
    out = []
    for g, src, traj in dataset:
        t = traj.copy()
        t[at] = true_sssp(g, src)
        out.append((g, src, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="models per group")
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--eval-graphs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=None, help="default: n (Dijkstra needs n)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--admit-mae", type=float, default=2.5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out", default="e1_e2_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.epochs, args.probe_graphs = 2, 60, 3
        args.train_graphs, args.eval_graphs = 8, 4

    rounds = args.rounds if args.rounds else args.nodes
    budget = args.budget if args.budget else 2 * args.nodes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    print(f"device={device}  n={args.nodes}  rounds={rounds}  budget={budget}  "
          f"tol={args.tol}  min_depth={args.min_depth}")
    print("pre-registered decision rule: PREREGISTRATION_E1E2.md\n")

    # ---- data ------------------------------------------------------------------
    # Identical graphs for both hint targets: only the trajectory differs, so nothing
    # about the graph distribution can be confounded with the manipulated variable.
    sets = {}
    for alg in ("bellman_ford", "dijkstra"):
        rng_alg = np.random.default_rng(0)  # same graphs, both algorithms
        sets[alg] = {
            "train": make_dataset(args.train_graphs, args.nodes, rounds, rng_alg,
                                  density=args.density, algorithm=alg,
                                  min_depth=args.min_depth),
            "eval": make_dataset(args.eval_graphs, args.nodes, rounds, rng_alg,
                                 density=args.density, algorithm=alg,
                                 min_depth=args.min_depth),
        }
    admit_set = true_distance_dataset(sets["bellman_ford"]["eval"])

    probe_graphs = deep_probe_graphs(
        args.probe_graphs, args.nodes, rng, args.density, args.min_depth
    )
    depths = [int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max())
              for g, _ in probe_graphs]
    print(f"probe-graph depths: {depths}  (median {np.median(depths):.1f})")

    refs = {
        name: fingerprint_of(finite_init(ctor), probe_graphs, budget, args.tol)
        for name, ctor in ALGORITHMS.items()
    }
    refs_no_term = {k: drop_predicate(v, "term_delta") for k, v in refs.items()}

    # ---- train + fingerprint ---------------------------------------------------
    groups = [
        ("bf_hint", "bellman_ford", True, 0),
        ("dj_hint", "dijkstra", True, 100),
        ("nohint", "bellman_ford", False, 200),
    ]
    records = []
    for group, alg, hints, offset in groups:
        print(f"\n=== {group}: hints={hints} target={alg} ===")
        for i in range(args.seeds):
            seed = offset + i
            model, loss = train_model(
                sets[alg]["train"], rounds, seed=seed, hints=hints,
                epochs=args.epochs, device=device,
            )
            mae = evaluate(model, admit_set, rounds, device)
            mae_bf = trajectory_mae(model, sets["bellman_ford"]["eval"], rounds, device)
            mae_dj = trajectory_mae(model, sets["dijkstra"]["eval"], rounds, device)
            admitted = mae < args.admit_mae

            rec = {
                "group": group, "seed": seed, "loss": loss, "mae": mae,
                "traj_mae_bf": mae_bf, "traj_mae_dj": mae_dj,
                "delta": mae_bf - mae_dj, "admitted": admitted,
            }
            if admitted:
                fp = fingerprint_of(
                    neural_ctor(model, device), probe_graphs, budget, args.tol
                )
                label, dists = classify(fp, refs, args.stability)
                label_nt, dists_nt = classify(
                    drop_predicate(fp, "term_delta"), refs_no_term, args.stability
                )
                rec |= {
                    "label": label, "margin": margin(dists),
                    "family": family(label), "dists": dists,
                    "label_no_term": label_nt, "margin_no_term": margin(dists_nt),
                    "family_no_term": family(label_nt),
                }
            records.append(rec)
            tag = (f"-> {rec['label']:<16} margin {rec['margin']:.3f}"
                   if admitted else "-> EXCLUDED (MAE)")
            print(f"  seed {seed}: loss {loss:.5f}  MAE {mae:.3f}  "
                  f"traj bf/dj {mae_bf:.2f}/{mae_dj:.2f}  {tag}")

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1, default=float)

    by_group = {g: [r for r in records if r["group"] == g] for g, _, _, _ in groups}
    adm = {g: [r for r in v if r["admitted"]] for g, v in by_group.items()}

    # ---- Gate 0: admission -----------------------------------------------------
    print("\n\n" + "=" * 70)
    print("GATE 0 -- admission (MAE < %.1f, constraint 9)" % args.admit_mae)
    need = max(1, args.seeds - 1)
    gate0 = True
    for g in by_group:
        ok = len(adm[g]) >= need
        gate0 &= ok or g == "nohint"
        maes = ", ".join(f"{r['mae']:.2f}" for r in by_group[g])
        print(f"  {g:<8} {len(adm[g])}/{len(by_group[g])} admitted  "
              f"{'PASS' if ok else 'FAIL'}  (MAE {maes})")

    # ---- Gate 1: manipulation check -------------------------------------------
    print("\n" + "=" * 70)
    print("GATE 1 -- manipulation check (delta = traj_MAE_bf - traj_MAE_dj)")
    print("          positive = model follows Dijkstra's trajectory")
    # Printed over *all* models, not just admitted ones. When a group fails admission
    # wholesale this is the only evidence distinguishing "the training difference was
    # never installed" from "it was installed but the model cannot execute it
    # accurately" -- and those two have completely different next steps. The gate
    # itself still evaluates on admitted models only.
    for g in by_group:
        dd = [r["delta"] for r in by_group[g]]
        print(f"  [all models] {g:<8} mean {np.mean(dd):+.3f}  "
              f"range [{min(dd):+.3f}, {max(dd):+.3f}]")

    d_bf = [r["delta"] for r in adm["bf_hint"]]
    d_dj = [r["delta"] for r in adm["dj_hint"]]
    gate1 = False
    if not (d_bf and d_dj):
        print("  gate not evaluable: a group has no admitted model")
    if d_bf and d_dj:
        disjoint = max(d_bf) < min(d_dj) or max(d_dj) < min(d_bf)
        gate1 = np.mean(d_bf) < 0 < np.mean(d_dj) and disjoint
        print(f"  bf_hint delta: mean {np.mean(d_bf):+.3f}  "
              f"range [{min(d_bf):+.3f}, {max(d_bf):+.3f}]")
        print(f"  dj_hint delta: mean {np.mean(d_dj):+.3f}  "
              f"range [{min(d_dj):+.3f}, {max(d_dj):+.3f}]")
        print(f"  ranges disjoint: {disjoint}")
    d_nh = [r["delta"] for r in adm["nohint"]]
    if d_nh:
        print(f"  nohint  delta: mean {np.mean(d_nh):+.3f}  "
              f"range [{min(d_nh):+.3f}, {max(d_nh):+.3f}]  (descriptive; gates nothing)")
    print(f"  => {'PASS' if gate1 else 'FAIL -- E1 is INCONCLUSIVE (manipulation)'}")

    # ---- Gate 2: the primary test ---------------------------------------------
    print("\n" + "=" * 70)
    print("GATE 2 -- E1 primary: family-level discrimination")

    names = sorted(refs)
    print("\n  Confusion matrix (secondary, exact label):")
    print(f"    {'trained on':<12} " + " ".join(f"{n[:9]:>10}" for n in names))
    for g, key in (("bf_hint", "bellman_ford"), ("dj_hint", "dijkstra")):
        row = [sum(1 for r in adm[g] if r["label"] == n) for n in names]
        print(f"    {key:<12} " + " ".join(f"{c:>10}" for c in row))

    a = sum(1 for r in adm["bf_hint"] if r["family"] == "relaxation")
    b = len(adm["bf_hint"]) - a
    c = sum(1 for r in adm["dj_hint"] if r["family"] == "relaxation")
    d = len(adm["dj_hint"]) - c
    p = fisher_exact_2x2(a, b, c, d)
    correct = a + d
    total = a + b + c + d
    print(f"\n  Family 2x2   relaxation  settling")
    print(f"    bf_hint    {a:>10}  {b:>8}")
    print(f"    dj_hint    {c:>10}  {d:>8}")
    print(f"    matching training algorithm: {correct}/{total}   Fisher p = {p:.4f}")

    print("\n  Margins (flagged unresolved below the %.3f noise floor):" % NOISE_FLOOR)
    for g in ("bf_hint", "dj_hint", "nohint"):
        for r in adm[g]:
            flag = "  UNRESOLVED" if r["margin"] < NOISE_FLOOR else ""
            print(f"    {g:<8} seed {r['seed']:>3}  {r['label']:<16} "
                  f"margin {r['margin']:.3f}{flag}")

    resolved = [r for r in adm["bf_hint"] + adm["dj_hint"] if r["margin"] >= NOISE_FLOOR]
    ra = sum(1 for r in resolved if r["group"] == "bf_hint" and r["family"] == "relaxation")
    rb = sum(1 for r in resolved if r["group"] == "bf_hint") - ra
    rc = sum(1 for r in resolved if r["group"] == "dj_hint" and r["family"] == "relaxation")
    rd = sum(1 for r in resolved if r["group"] == "dj_hint") - rc
    print(f"\n  Sensitivity, resolved models only (secondary): "
          f"{ra + rd}/{len(resolved)} matching, Fisher p = "
          f"{fisher_exact_2x2(ra, rb, rc, rd):.4f}")

    na = sum(1 for r in adm["bf_hint"] if r["family_no_term"] == "relaxation")
    nb = len(adm["bf_hint"]) - na
    nc = sum(1 for r in adm["dj_hint"] if r["family_no_term"] == "relaxation")
    nd = len(adm["dj_hint"]) - nc
    print(f"  Runtime ablation, term_delta dropped: {na + nd}/{na + nb + nc + nd} "
          f"matching, Fisher p = {fisher_exact_2x2(na, nb, nc, nd):.4f}")

    # ---- E2 --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("E2 -- hint-free regime")
    if not adm["nohint"]:
        print("  No hint-free model was admitted: a 24-step unroll with loss at the")
        print("  final step only did not train. Reported as a training failure, not")
        print("  as a fingerprint result.")
        fdr_labels, split_frac = 0, 0.0
    else:
        labels = [r["label"] for r in adm["nohint"]]
        fams = [r["family"] for r in adm["nohint"]]
        fdr_labels = len(set(labels))
        print(f"  admitted {len(adm['nohint'])}/{args.seeds}   "
              f"MAE {[round(r['mae'], 2) for r in adm['nohint']]}")
        print(f"  labels: {labels}")
        print(f"  distinct labels: {fdr_labels}   "
              f"families: {fams.count('relaxation')} relaxation / "
              f"{fams.count('settling')} settling")
        mean_min = np.mean([min(r["dists"].values()) for r in adm["nohint"]])
        print(f"  mean distance to nearest reference: {mean_min:.3f}   "
              f"mean margin {np.mean([r['margin'] for r in adm['nohint']]):.3f}")

        print("\n  Negative control -- these models were trained on IDENTICAL data")
        print("  (at rounds=n both hint algorithms have converged, so the hint target")
        print("  cannot reach a final-answer-only loss). Any split is noise.")
        k = len(adm["nohint"]) // 2
        splits = list(combinations(range(len(adm["nohint"])), k))
        perfect = sum(
            1 for s in splits
            if len({fams[i] for i in s}) == 1
            and len({fams[i] for i in range(len(fams)) if i not in s}) == 1
            and fams[s[0]] != fams[[i for i in range(len(fams)) if i not in s][0]]
        )
        split_frac = perfect / len(splits) if splits else 0.0
        print(f"  balanced splits achieving family-perfect separation: "
              f"{perfect}/{len(splits)} = {split_frac:.3f}")
        print("  (empirical false-positive rate of the E1 statistic under a true null)")

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT (pre-registered ladder)")
    if not gate0:
        verdict = "INCONCLUSIVE (training)"
    elif not gate1:
        verdict = "INCONCLUSIVE (manipulation)"
    else:
        all_ok = all(
            r["margin"] > NOISE_FLOOR
            for r in adm["bf_hint"] + adm["dj_hint"]
            if (r["group"] == "bf_hint") == (r["family"] == "relaxation")
        )
        # Direction matters: a significant split in the *wrong* direction is not a weak
        # positive, it is a null with an odd shape.
        right_way = correct > total / 2
        if p < 0.05 and right_way and all_ok:
            verdict = "STRONG"
        elif p < 0.20 and right_way:
            verdict = "WEAK"
        else:
            verdict = "NULL"
        if verdict in ("STRONG", "WEAK") and fdr_labels > 1 and split_frac > 0:
            verdict += " (downgraded: hint-free control splits across families)"
    print(f"  E1: {verdict}")
    if verdict.startswith("WEAK"):
        print("  Pre-registered response to WEAK: replicate on an independent seed set")
        print("  before any claim. Not to be written up as a positive result.")


if __name__ == "__main__":
    main()
