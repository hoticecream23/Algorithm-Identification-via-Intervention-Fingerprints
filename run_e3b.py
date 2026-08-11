"""E3b: held-out replication of E3's truncation-distance lead.

E3 was NULL -- 17/17 models labelled `bellman_ford` whatever their architecture. One
post-hoc quantity separated the plain and halt arms perfectly:

    s = d(bellman_ford) - d(bf_truncated_k3)

It is being tested rather than reported. B2's `overshoot` was also perfectly monotone on
its first seed set, also had a sound mechanistic argument, and sign-flipped on held-out
seeds. Four results in this project died that way.

The threshold is frozen at tau = -0.084 from E3's gap and is **not** refitted here. Probe
graphs are regenerated from a different RNG stream, because the statistic is a distance
measured on those graphs and E3's set is part of what tau was fitted to.

Decision rule: `PREREGISTRATION_E3B.md`. Two outcomes only, REPLICATED or FAILED.

    python run_e3b.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.identify import classify, distance, margin
from fpid.interventions import firing_round
from fpid.neural import neural_ctor
from fpid.train import evaluate, make_dataset, train_model
from run_e1_e2 import (
    deep_probe_graphs,
    fingerprint_of,
    fisher_exact_2x2,
    true_distance_dataset,
)
from run_e3 import HALT_K

TAU = -0.084  # frozen from E3. Never refitted.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--eval-graphs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--admit-mae", type=float, default=2.5)
    ap.add_argument("--out", default="e3b_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.epochs, args.probe_graphs = 2, 60, 3
        args.train_graphs, args.eval_graphs = 8, 4

    rounds, budget = args.nodes, 2 * args.nodes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device}  tau={TAU} (frozen from E3, not refitted)")
    print("pre-registered decision rule: PREREGISTRATION_E3B.md\n")

    rng_d = np.random.default_rng(0)
    train_set = make_dataset(args.train_graphs, args.nodes, rounds, rng_d,
                             density=args.density, min_depth=args.min_depth)
    eval_set = make_dataset(args.eval_graphs, args.nodes, rounds, rng_d,
                            density=args.density, min_depth=args.min_depth)
    admit = {k: true_distance_dataset(eval_set, at=k) for k in (rounds, HALT_K)}

    # Fresh probe stream. E3's probe set is part of what tau was fitted to.
    probe_graphs = deep_probe_graphs(
        args.probe_graphs, args.nodes, np.random.default_rng(7),
        args.density, args.min_depth,
    )
    depths = [int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max())
              for g, _ in probe_graphs]
    print(f"probe-graph depths (rng 7): {depths}  (median {np.median(depths):.1f})")

    refs = {n: fingerprint_of(finite_init(c), probe_graphs, budget, args.tol)
            for n, c in ALGORITHMS.items()}
    ref_pairs = [distance(refs[a], refs[b], args.stability)
                 for i, a in enumerate(sorted(refs)) for b in sorted(refs)[i + 1:]]
    print(f"symbolic reference-to-reference distance: "
          f"mean {np.mean(ref_pairs):.3f}  median {np.median(ref_pairs):.3f}")

    records = []
    for arch, offset, tr_rounds, deploy in (
        ("plain", 500, rounds, None),
        ("halt", 600, HALT_K, HALT_K),
    ):
        print(f"\n=== {arch}: train_rounds={tr_rounds} deploy={deploy} ===")
        for i in range(args.seeds):
            seed = offset + i
            model, loss = train_model(train_set, tr_rounds, seed=seed, hints=False,
                                      epochs=args.epochs, device=device)
            mae = evaluate(model, admit[tr_rounds], tr_rounds, device)
            rec = {"arch": arch, "seed": seed, "mae": mae, "admitted": mae < args.admit_mae}
            if rec["admitted"]:
                ctor = neural_ctor(model, device, max_rounds=deploy)
                fp = fingerprint_of(ctor, probe_graphs, budget, args.tol)
                label, dists = classify(fp, refs, args.stability)
                s = dists["bellman_ford"] - dists["bf_truncated_k3"]
                rec |= {"label": label, "margin": margin(dists), "dists": dists, "s": s,
                        "pred": "halt" if s > TAU else "plain",
                        "d_near": min(dists.values()),
                        "fire": float(np.median([
                            firing_round(ctor(g.copy(), src).run(budget), budget, args.tol)
                            for g, src in probe_graphs]))}
            records.append(rec)
            tag = (f"s={rec['s']:+.3f} -> pred {rec['pred']:<6} "
                   f"[{rec['label']}, margin {rec['margin']:.3f}]"
                   if rec["admitted"] else "EXCLUDED (MAE)")
            print(f"  seed {seed}: MAE {mae:.3f}  {tag}")

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1, default=float)

    by = {a: [r for r in records if r["arch"] == a] for a in ("plain", "halt")}
    adm = {a: [r for r in v if r["admitted"]] for a, v in by.items()}

    print("\n" + "=" * 70)
    print("GATE 0 -- admission")
    need = max(1, args.seeds - 1)
    gate0 = all(len(adm[a]) >= need for a in adm)
    for a in adm:
        print(f"  {a:<6} {len(adm[a])}/{len(by[a])} admitted  "
              f"(MAE {', '.join(f'{r[chr(0x6d)+chr(0x61)+chr(0x65)]:.2f}' for r in by[a])})")

    print("\nGATE 1 -- mechanism")
    fires = [r["fire"] for r in adm["halt"]]
    gate1 = bool(fires) and max(fires) < HALT_K
    print(f"  halt median firing rounds {fires} vs k={HALT_K}  "
          f"{'PASS' if gate1 else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"GATE 2 -- frozen rule (predict halt if s > {TAU})")
    a = sum(1 for r in adm["plain"] if r["pred"] == "plain")
    b = len(adm["plain"]) - a
    c = sum(1 for r in adm["halt"] if r["pred"] == "plain")
    d = len(adm["halt"]) - c
    p = fisher_exact_2x2(a, b, c, d)
    correct, total = a + d, a + b + c + d
    print(f"    {'':<7} pred plain  pred halt")
    print(f"    plain   {a:>10}  {b:>9}")
    print(f"    halt    {c:>10}  {d:>9}")
    print(f"    correct {correct}/{total}   Fisher p = {p:.4f}")

    for arch in ("plain", "halt"):
        v = [r["s"] for r in adm[arch]]
        if v:
            print(f"    {arch:<6} s: {[round(x, 3) for x in v]}  "
                  f"range [{min(v):+.3f}, {max(v):+.3f}]")

    print("\n  Structural check on a fresh probe set (descriptive, the important number):")
    for arch in ("plain", "halt"):
        dn = [r["d_near"] for r in adm[arch]]
        if dn:
            print(f"    {arch:<6} mean distance to nearest reference {np.mean(dn):.3f}   "
                  f"labels {sorted({r['label'] for r in adm[arch]})}")
    print(f"    symbolic reference-to-reference median: {np.median(ref_pairs):.3f}")

    print("\n" + "=" * 70)
    if not gate0:
        verdict = "INCONCLUSIVE (training)"
    elif not gate1:
        verdict = "INCONCLUSIVE (mechanism)"
    elif correct >= total - 1 and p < 0.05:
        verdict = "REPLICATED"
    else:
        verdict = "FAILED"
    print(f"VERDICT: {verdict}")

    # Reported only after the verdict, and explicitly not substituted for it.
    sp = [r["s"] for r in adm["plain"]]
    sh = [r["s"] for r in adm["halt"]]
    if sp and sh:
        gap = min(sh) - max(sp)
        print(f"\n  (post-hoc, not the verdict) E3b's own gap: "
              f"plain max {max(sp):+.3f}, halt min {min(sh):+.3f}, "
              f"{'separable' if gap > 0 else 'overlapping'} by {gap:+.3f}")
    if verdict == "FAILED":
        print("  The lead is dead. Neural side finished; write up the symbolic")
        print("  contribution plus the characterised negative.")


if __name__ == "__main__":
    main()
