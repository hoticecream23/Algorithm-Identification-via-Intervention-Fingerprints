"""Does the fingerprint predict OOD failure better than validation accuracy?

The central claim, tested with ~10 models instead of a full zoo.

Setup. Two configurations with identical architecture and identical round budget (12),
differing only in the *training distribution*: one sees only shallow graphs, the other
sees deep ones. Both are then judged on a shallow validation set, where both do well
and no validation metric usefully separates them.

An earlier version varied the round budget instead (12 vs 4) and failed cleanly: the
4-round models were *better* out-of-distribution, because fewer unrolled rounds means
less accumulated error and an easier optimisation, and those effects dominate the extra
propagation depth. Round budget is a bad proxy for learned truncation -- it confounds
what the model learned with how hard it was to train. Holding rounds fixed and varying
only what the model was exposed to isolates the thing under test, and matches the real
failure mode in neural algorithmic reasoning: train small and shallow, deploy large and
deep.

Fingerprints are taken on the shallow training distribution only -- no OOD data is used
to make the prediction. This is licensed by the identifiability result: truncation is
separable at every depth, including graphs shallow enough that both models produce
identical answers, because a halted model cannot respond to a mid-run intervention at
all.

OOD is then measured on deep graphs, where four rounds cannot reach the far side.

    python run_ood_prediction.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import aggregate, probe
from fpid.identify import classify, distance, margin
from fpid.interventions import INTERVENTIONS
from fpid.neural import neural_ctor
from fpid.train import evaluate, make_dataset


def fingerprint_of(ctor, graphs, budget, tol):
    runs = []
    for g, src in graphs:
        row = {}
        for iv in INTERVENTIONS:
            for p, v in probe(ctor, g, src, iv, budget, tol=tol).items():
                row[(iv.name, p)] = v
        runs.append(row)
    return aggregate(runs)


def spearman(x, y) -> float:
    """Rank correlation without a scipy dependency."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--train-graphs", type=int, default=32)
    ap.add_argument("--probe-graphs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--shallow-density", type=float, default=0.35)
    ap.add_argument("--deep-density", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    from fpid.train import train_model
    from fpid.graphs import random_connected_graph

    ROUNDS = 12  # identical for both configurations; only the training data differs

    shallow_train = make_dataset(
        args.train_graphs, args.nodes, ROUNDS, rng, density=args.shallow_density
    )
    deep_train = make_dataset(
        args.train_graphs, args.nodes, ROUNDS, rng, density=args.deep_density
    )
    # Judged on shallow data, which is where both look fine.
    val_set = make_dataset(12, args.nodes, ROUNDS, rng, density=args.shallow_density)
    # OOD: deep graphs, needing propagation the shallow-trained models never practised.
    ood_set = make_dataset(12, args.nodes, ROUNDS, rng, density=args.deep_density)

    # Fingerprint on BOTH probe distributions. Shallow probes keep the "predict OOD
    # without OOD data" property but sit in the regime the identifiability sweep found
    # weakest; deep probes are discriminative but are drawn from the OOD distribution
    # itself. Running both separates "the method cannot see this difference" from "the
    # probes were too shallow to resolve it", which a single probe set cannot do.
    probe_sets = {
        "shallow": [
            (random_connected_graph(args.nodes, rng, density=args.shallow_density), 0)
            for _ in range(args.probe_graphs)
        ],
        "deep": [
            (random_connected_graph(args.nodes, rng, density=args.deep_density), 0)
            for _ in range(args.probe_graphs)
        ],
    }

    # References must be measured on the same graphs as the models they are compared to.
    refs = {
        pname: {
            name: fingerprint_of(finite_init(ctor), pgraphs, args.budget, args.tol)
            for name, ctor in ALGORITHMS.items()
        }
        for pname, pgraphs in probe_sets.items()
    }

    print(f"device={device}  shallow train / deep OOD  tol={args.tol}\n")
    print(f"{'model':<14}{'val':<9}{'OOD':<9}{'label@shallow':<14}{'margin':<8}"
          f"{'label@deep':<14}margin")

    rows = []
    for tag, tset in (("shallow", shallow_train), ("deep", deep_train)):
        for seed in range(args.seeds):
            model, _loss = train_model(
                tset, ROUNDS, seed=seed, hints=True,
                epochs=args.epochs, device=device,
            )
            val = evaluate(model, val_set, ROUNDS, device)
            ood = evaluate(model, ood_set, ROUNDS, device)
            row = dict(name=f"{tag}-s{seed}", group=tag, val=val, ood=ood)
            for pname, pgraphs in probe_sets.items():
                fp = fingerprint_of(
                    neural_ctor(model, device, max_rounds=ROUNDS),
                    pgraphs, args.budget, args.tol,
                )
                label, dists = classify(fp, refs[pname], args.stability)
                # Signed preference between the two candidate procedures. Distance to
                # Bellman-Ford alone conflates "looks truncated" with "is far from
                # every reference", and those imply opposite things about OOD.
                row[f"label_{pname}"] = label
                row[f"d_bf_{pname}"] = dists["bellman_ford"]
                row[f"rel_{pname}"] = (
                    dists["bellman_ford"] - dists["bf_truncated_k3"]
                )
                row[f"margin_{pname}"] = margin(dists)
            rows.append(row)
            print(f"{row['name']:<14}{val:<9.3f}{ood:<9.3f}"
                  f"{row['label_shallow'][:12]:<14}{row['margin_shallow']:<8.3f}"
                  f"{row['label_deep'][:12]:<14}{row['margin_deep']:<8.3f}")

    val_all = [r["val"] for r in rows]
    ood_all = [r["ood"] for r in rows]

    print("\n=== Group separability ===")
    shal = [r for r in rows if r["group"] == "shallow"]
    deep = [r for r in rows if r["group"] == "deep"]
    for tag, grp in (("shallow-trained", shal), ("deep-trained", deep)):
        print(f"  {tag:<16} val {np.mean([r['val'] for r in grp]):.3f} "
              f"+/- {np.std([r['val'] for r in grp]):.3f}   "
              f"OOD {np.mean([r['ood'] for r in grp]):.3f}   "
              f"d(BF)@shallow {np.mean([r['d_bf_shallow'] for r in grp]):.3f}   "
              f"d(BF)@deep {np.mean([r['d_bf_deep'] for r in grp]):.3f}")

    print("\n=== Which signal predicts OOD failure? ===")
    print(f"  spearman(val MAE, OOD)                = {spearman(val_all, ood_all):+.3f}")
    for pname in probe_sets:
        dbf = [r[f"d_bf_{pname}"] for r in rows]
        rel = [r[f"rel_{pname}"] for r in rows]
        mg = np.mean([r[f"margin_{pname}"] for r in rows])
        print(f"  spearman(d(BF)@{pname:<8}, OOD)        = {spearman(dbf, ood_all):+.3f}"
              f"   (mean margin {mg:.3f})")
        print(f"  spearman(d(BF)-d(tr)@{pname:<8}, OOD)  = {spearman(rel, ood_all):+.3f}")

    print("\n=== Does the fingerprint separate the two training groups at all? ===")
    for pname in probe_sets:
        s = [r[f"d_bf_{pname}"] for r in shal]
        d = [r[f"d_bf_{pname}"] for r in deep]
        pooled = float(np.sqrt((np.var(s) + np.var(d)) / 2))
        # Guard against a vanishing pooled sd, which otherwise reports an enormous
        # effect size from what is actually an absence of variance.
        eff = f"{(np.mean(s) - np.mean(d)) / pooled:+.2f}" if pooled > 1e-6 else "n/a"
        print(f"  {pname:<8} d(BF): shallow {np.mean(s):.3f} vs deep {np.mean(d):.3f}"
              f"   Cohen's d = {eff}")
        labs_s = {r[f"label_{pname}"] for r in shal}
        labs_d = {r[f"label_{pname}"] for r in deep}
        print(f"           labels: shallow {labs_s}  deep {labs_d}")


if __name__ == "__main__":
    main()
