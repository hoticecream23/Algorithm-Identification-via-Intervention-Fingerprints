"""Find an in-distribution signal that separates models which will fail OOD.

B1 showed that shallow-trained and deep-trained models are indistinguishable on shallow
graphs under the existing intervention suite, while differing 8.6x on deep-graph error.
The probes there asked about execution structure, which is identical for both.

The reframe: a shallow graph cannot be made deep, but the model's *state* can be put
into the regime a deep graph would produce. On deep graphs the model meets distance
values around 39; shallow training only ever showed it ~15. If the learned update rule
is simply wrong outside the value range it was trained on, that is measurable without
ever leaving the shallow distribution.

Candidate diagnostics, all computed on shallow graphs only:

  val_mae        baseline: ordinary validation error
  drift          fixed-point stability. A real relaxation operator is idempotent once
                 converged; iterating 3x longer should change nothing. A model that
                 learned "emit the answer after two rounds" drifts. Deep graphs need
                 many stable iterations, so drift should predict deep failure.
  inject_err     state-space extrapolation. Set one node's estimate to a value well
                 above the training range and check the model still relaxes correctly
                 from it.
  overshoot      does the model ever produce estimates below the true distance? A
                 relaxation operator is monotone from above and can never undershoot;
                 a fitted regression can.
  nonmono        relaxation only ever lowers an estimate. Count rounds where the model
                 raises one, which no correct relaxation would do.

    python run_shallow_diagnostics.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fpid.graphs import random_connected_graph, true_sssp
from fpid.neural import SCALE, NeuralExecutor
from fpid.train import evaluate, make_dataset, train_model


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den else 0.0


def rollout(model, graph, source, rounds, device, inject=None):
    ex = NeuralExecutor(model, graph, source, device, max_rounds=rounds)
    traj = [ex.snapshot()]
    for r in range(rounds):
        if inject is not None and r == inject[0]:
            ex.d[inject[1]] = inject[2]
        ex.step()
        traj.append(ex.snapshot())
    return traj


def diagnostics(model, graphs, rounds, device, inject_scale=3.0):
    """All measured on shallow graphs. No OOD data touched."""
    drift, inject_err, overshoot, nonmono = [], [], [], []

    for g, src in graphs:
        truth = true_sssp(g, src)
        finite = np.isfinite(truth)

        base = rollout(model, g, src, rounds, device)
        long = rollout(model, g, src, rounds * 3, device)

        # Idempotence: iterating 3x longer should be a no-op after convergence.
        drift.append(float(np.abs(long[-1].d[finite] - base[-1].d[finite]).mean()))

        # Does the model ever go below the true distance? Relaxation approaches the
        # answer from above and cannot undershoot; a fitted regression can.
        worst_under = 0.0
        raises = 0
        for r in range(1, len(base)):
            under = (truth[finite] - base[r].d[finite]).max()
            worst_under = max(worst_under, float(under))
            # Relaxation never raises an estimate.
            rise = (base[r].d[finite] - base[r - 1].d[finite]).max()
            if rise > 0.5:
                raises += 1
        overshoot.append(worst_under)
        nonmono.append(raises)

        # State-space extrapolation: push one node's estimate above anything the
        # shallow training distribution ever contained, then check the model still
        # relaxes it back to the right answer.
        v = int(np.argmax(np.where(finite, truth, -np.inf)))
        big = float(truth[finite].max() * inject_scale)
        inj = rollout(model, g, src, rounds * 2, device, inject=(1, v, big))
        inject_err.append(float(abs(inj[-1].d[v] - truth[v])))

    return dict(
        drift=float(np.mean(drift)),
        inject_err=float(np.mean(inject_err)),
        overshoot=float(np.mean(overshoot)),
        nonmono=float(np.mean(nonmono)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    # `overshoot` was selected post-hoc from four candidates on seeds 0-4, so its
    # advantage there is partly a selection effect. Re-running on disjoint seeds is
    # the only way to tell a real signal from the best of four noisy ones.
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--train-graphs", type=int, default=32)
    ap.add_argument("--diag-graphs", type=int, default=8)
    ap.add_argument("--shallow-density", type=float, default=0.35)
    ap.add_argument("--deep-density", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    ROUNDS = 12

    shallow_train = make_dataset(
        args.train_graphs, args.nodes, ROUNDS, rng, density=args.shallow_density
    )
    deep_train = make_dataset(
        args.train_graphs, args.nodes, ROUNDS, rng, density=args.deep_density
    )
    val_set = make_dataset(12, args.nodes, ROUNDS, rng, density=args.shallow_density)
    ood_set = make_dataset(12, args.nodes, ROUNDS, rng, density=args.deep_density)
    diag_graphs = [
        (random_connected_graph(args.nodes, rng, density=args.shallow_density), 0)
        for _ in range(args.diag_graphs)
    ]

    keys = ("drift", "inject_err", "overshoot", "nonmono")
    print(f"device={device}   diagnostics on shallow graphs only\n")
    print(f"{'model':<14}{'val':<8}{'OOD':<8}" + "".join(f"{k:<13}" for k in keys))

    rows = []
    for tag, tset in (("shallow", shallow_train), ("deep", deep_train)):
        for i in range(args.seeds):
            seed = args.seed_offset + i
            model, _ = train_model(
                tset, ROUNDS, seed=seed, hints=True, epochs=args.epochs, device=device
            )
            row = dict(
                name=f"{tag}-s{seed}",
                group=tag,
                val=evaluate(model, val_set, ROUNDS, device),
                ood=evaluate(model, ood_set, ROUNDS, device),
            )
            row.update(diagnostics(model, diag_graphs, ROUNDS, device))
            rows.append(row)
            print(f"{row['name']:<14}{row['val']:<8.3f}{row['ood']:<8.3f}"
                  + "".join(f"{row[k]:<13.3f}" for k in keys))

    ood = [r["ood"] for r in rows]
    shal = [r for r in rows if r["group"] == "shallow"]
    deep = [r for r in rows if r["group"] == "deep"]

    print("\n=== Group means ===")
    for tag, grp in (("shallow-trained", shal), ("deep-trained", deep)):
        print(f"  {tag:<16} val {np.mean([r['val'] for r in grp]):.3f}  "
              f"OOD {np.mean([r['ood'] for r in grp]):.3f}  "
              + "  ".join(f"{k} {np.mean([r[k] for r in grp]):.3f}" for k in keys))

    print("\n=== Predicting OOD error from shallow-graph signals ===")
    print(f"  {'signal':<14}{'spearman':<12}{'Cohen d':<10}separation")
    for k in ("val",) + keys:
        s = [r[k] for r in shal]
        d = [r[k] for r in deep]
        pooled = float(np.sqrt((np.var(s) + np.var(d)) / 2))
        eff = (np.mean(s) - np.mean(d)) / pooled if pooled > 1e-9 else float("nan")
        clean = "CLEAN" if pooled > 1e-9 and min(s) > max(d) else (
            "clean(rev)" if pooled > 1e-9 and max(s) < min(d) else "overlap"
        )
        print(f"  {k:<14}{spearman([r[k] for r in rows], ood):<+12.3f}"
              f"{eff:<+10.2f}{clean}")

    print("\n  CLEAN = the two groups do not overlap at all on this signal.")


if __name__ == "__main__":
    main()
