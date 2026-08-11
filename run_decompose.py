"""Where does the neural-vs-symbolic distance actually come from?

E3 left one number unexplained: every neural model sits ~0.37 from its nearest symbolic
reference, while two genuinely different symbolic algorithms sit 0.25-0.30 apart. The
label was never supported, but nothing says *which* part of the fingerprint disagrees.

This is a decomposition, not a test. It computes no p-value and makes no claim. It asks:

1. Which interventions carry the neural-vs-symbolic disagreement?
2. Are those the same interventions that carry symbolic *identity* -- the axes that
   separate Bellman-Ford from Dijkstra -- or orthogonal ones? If orthogonal, that is the
   cleanest available statement of why identification fails: the models differ from every
   reference along directions that carry no identity information, and the discriminative
   axes are drowned out.
3. How much comes from the two deletion probes specifically? Constraint 3 says relaxation
   only ever lowers `d`, so edge deletion produces zero response in any real relaxation
   algorithm -- but a network has no such constraint and can respond. That is 14 of 63
   predicate slots on which every neural model must disagree with every reference by
   construction, a constant offset inflating every neural distance and cancelling
   common-mode between neural models.

**Anything that looks like discrimination after dropping probes is a candidate, not a
result.** Dropping probes post-hoc and reporting what appears is the exact selection error
that killed four results here. Such a finding must be pre-registered and replicated on
held-out seeds before it is claimed.

Models and fingerprints are saved so that no future analysis has to retrain.

    python run_decompose.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import PREDICATES, Fingerprint
from fpid.identify import distance, shared_keys
from fpid.interventions import INTERVENTIONS
from fpid.neural import GatedMPNN, neural_ctor
from fpid.train import evaluate, make_dataset, train_model
from run_e1_e2 import deep_probe_graphs, fingerprint_of, true_distance_dataset
from run_e3 import GATE_EPS, HALT_K

DELETIONS = {"delete_tree_edge_near", "delete_tree_edge_far", "delete_nontree_edge"}


def slot_disagreements(a: Fingerprint, b: Fingerprint, stability: float):
    """Jointly-stable keys, split into those that agree and those that do not."""
    keys = shared_keys(a, b, stability)
    bad = [k for k in keys if a.value.get(k) != b.value.get(k)]
    return keys, bad


def drop_interventions(fp: Fingerprint, names: set[str]) -> Fingerprint:
    keep = {k for k in fp.value if k[0] not in names}
    return Fingerprint({k: fp.value[k] for k in keep},
                       {k: fp.stability[k] for k in keep})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--out", default="decompose_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.epochs, args.probe_graphs, args.train_graphs = 1, 60, 3, 8

    rounds, budget = args.nodes, 2 * args.nodes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    rng_d = np.random.default_rng(0)
    train_set = make_dataset(args.train_graphs, args.nodes, rounds, rng_d,
                             density=args.density, min_depth=args.min_depth)
    eval_set = make_dataset(8, args.nodes, rounds, rng_d,
                            density=args.density, min_depth=args.min_depth)
    admit = {k: true_distance_dataset(eval_set, at=k) for k in (rounds, HALT_K)}
    probe_graphs = deep_probe_graphs(args.probe_graphs, args.nodes,
                                     np.random.default_rng(0), args.density,
                                     args.min_depth)

    refs = {n: fingerprint_of(finite_init(c), probe_graphs, budget, args.tol)
            for n, c in ALGORITHMS.items()}

    print(f"device={device}  probe_graphs={args.probe_graphs}  tol={args.tol}\n")

    neural = {}
    for arch, offset, tr_rounds, deploy, fn in (
        ("plain", 700, rounds, None, None),
        ("gated", 800, rounds, None, lambda h, a: GatedMPNN(h, a, eps=GATE_EPS)),
        ("halt", 900, HALT_K, HALT_K, None),
    ):
        for i in range(args.seeds):
            seed = offset + i
            model, _ = train_model(train_set, tr_rounds, seed=seed, hints=False,
                                   epochs=args.epochs, device=device, model_fn=fn)
            mae = evaluate(model, admit[tr_rounds], tr_rounds, device)
            torch.save(model.state_dict(), f"{args.save_dir}/{arch}_{seed}.pt")
            fp = fingerprint_of(neural_ctor(model, device, max_rounds=deploy),
                                probe_graphs, budget, args.tol)
            neural[(arch, seed)] = fp
            print(f"  {arch:<6} seed {seed}: MAE {mae:.3f}  "
                  f"d(bf) {distance(fp, refs['bellman_ford'], args.stability):.3f}")

    with open(f"{args.save_dir}/fingerprints.json", "w") as f:
        json.dump({f"{k[0]}_{k[1]}": {"value": {f"{a}|{b}": v for (a, b), v in
                                                fp.value.items()},
                                      "stability": {f"{a}|{b}": s for (a, b), s in
                                                    fp.stability.items()}}
                   for k, fp in neural.items()}
                  | {f"ref_{n}": {"value": {f"{a}|{b}": v for (a, b), v in fp.value.items()},
                                  "stability": {f"{a}|{b}": s for (a, b), s in
                                                fp.stability.items()}}
                     for n, fp in refs.items()}, f, indent=1)

    st = args.stability
    bf = refs["bellman_ford"]

    # ---- 1. per-intervention contribution, neural vs bellman_ford ----------------
    print("\n" + "=" * 74)
    print("1. WHICH INTERVENTIONS CARRY THE NEURAL-VS-SYMBOLIC DISAGREEMENT?")
    print("   (fraction of that intervention's jointly-stable slots that disagree)\n")

    iv_names = [iv.name for iv in INTERVENTIONS]
    neural_iv = defaultdict(list)
    for fp in neural.values():
        keys, bad = slot_disagreements(fp, bf, st)
        per_iv_tot = defaultdict(int)
        per_iv_bad = defaultdict(int)
        for k in keys:
            per_iv_tot[k[0]] += 1
        for k in bad:
            per_iv_bad[k[0]] += 1
        for iv in iv_names:
            if per_iv_tot[iv]:
                neural_iv[iv].append(per_iv_bad[iv] / per_iv_tot[iv])

    # same decomposition for the symbolic pairs that actually carry identity
    sym_iv = defaultdict(list)
    for a, b in combinations(sorted(refs), 2):
        keys, bad = slot_disagreements(refs[a], refs[b], st)
        tot, bd = defaultdict(int), defaultdict(int)
        for k in keys:
            tot[k[0]] += 1
        for k in bad:
            bd[k[0]] += 1
        for iv in iv_names:
            if tot[iv]:
                sym_iv[iv].append(bd[iv] / tot[iv])

    print(f"    {'intervention':<30}{'neural vs BF':>14}{'symbolic pairs':>16}{'':>6}")
    rows = sorted(iv_names, key=lambda x: -np.mean(neural_iv[x] or [0]))
    for iv in rows:
        n = np.mean(neural_iv[iv]) if neural_iv[iv] else float("nan")
        s = np.mean(sym_iv[iv]) if sym_iv[iv] else float("nan")
        tag = "  <- deletion" if iv in DELETIONS else ""
        print(f"    {iv:<30}{n:>14.3f}{s:>16.3f}{tag}")

    # ---- 2. are the neural axes the identity axes? ------------------------------
    print("\n" + "=" * 74)
    print("2. ARE THOSE THE SAME AXES THAT CARRY SYMBOLIC IDENTITY?\n")
    nv = np.array([np.mean(neural_iv[iv]) if neural_iv[iv] else 0.0 for iv in iv_names])
    sv = np.array([np.mean(sym_iv[iv]) if sym_iv[iv] else 0.0 for iv in iv_names])
    if nv.std() > 0 and sv.std() > 0:
        r = float(np.corrcoef(nv, sv)[0, 1])
        print(f"    correlation across the 9 interventions: r = {r:+.3f}")
        print("    r near 0 or negative => neural models differ from the references along")
        print("    directions that carry no identity information, and the discriminative")
        print("    axes are swamped. r near +1 => they differ along the same axes.")

    # ---- 3. the deletion probes specifically ------------------------------------
    print("\n" + "=" * 74)
    print("3. HOW MUCH IS THE DELETION / MONOTONICITY OFFSET?\n")
    share = []
    for fp in neural.values():
        keys, bad = slot_disagreements(fp, bf, st)
        if bad:
            share.append(sum(1 for k in bad if k[0] in DELETIONS) / len(bad))
    print(f"    deletion probes are {np.mean(share):.1%} of all disagreeing slots "
          f"(they are {len(DELETIONS)}/{len(iv_names)} = "
          f"{len(DELETIONS) / len(iv_names):.1%} of the suite)")

    print("\n    Distances with the deletion probes removed (DIAGNOSTIC ONLY --")
    print("    a discrimination appearing here is a candidate needing pre-registered")
    print("    replication on held-out seeds, not a result):\n")
    refs_nd = {n: drop_interventions(f, DELETIONS) for n, f in refs.items()}
    print(f"    {'model':<14}{'d_near full':>13}{'d_near no-del':>15}"
          f"{'label full':>18}{'label no-del':>18}")
    kept = []
    for (arch, seed), fp in neural.items():
        full = {n: distance(fp, r, st) for n, r in refs.items()}
        nod = {n: distance(drop_interventions(fp, DELETIONS), r, st)
               for n, r in refs_nd.items()}
        lf = min(full, key=lambda k: full[k])
        ln = min(nod, key=lambda k: nod[k])
        kept.append({"arch": arch, "seed": seed, "full": full, "no_del": nod,
                     "label_full": lf, "label_no_del": ln})
        print(f"    {arch + ' ' + str(seed):<14}{full[lf]:>13.3f}{nod[ln]:>15.3f}"
              f"{lf:>18}{ln:>18}")

    sp_full = [distance(refs[a], refs[b], st) for a, b in combinations(sorted(refs), 2)]
    sp_nod = [distance(refs_nd[a], refs_nd[b], st)
              for a, b in combinations(sorted(refs), 2)]
    print(f"\n    symbolic reference-to-reference median: "
          f"{np.median(sp_full):.3f} full, {np.median(sp_nod):.3f} without deletions")

    # ---- 4. which predicates ----------------------------------------------------
    print("\n" + "=" * 74)
    print("4. WHICH PREDICATES DISAGREE?\n")
    pr = defaultdict(list)
    for fp in neural.values():
        keys, bad = slot_disagreements(fp, bf, st)
        tot, bd = defaultdict(int), defaultdict(int)
        for k in keys:
            tot[k[1]] += 1
        for k in bad:
            bd[k[1]] += 1
        for p in PREDICATES:
            if tot[p]:
                pr[p].append(bd[p] / tot[p])
    for p in sorted(PREDICATES, key=lambda x: -np.mean(pr[x] or [0])):
        if pr[p]:
            print(f"    {p:<16}{np.mean(pr[p]):>8.3f}")

    # ---- 5. universally-disagreeing slots ---------------------------------------
    print("\n" + "=" * 74)
    print("5. SLOTS WHERE EVERY NEURAL MODEL DISAGREES WITH EVERY REFERENCE\n")
    print("   These are pure offset: they inflate every neural distance equally and")
    print("   carry no information distinguishing one neural model from another.\n")
    universal = None
    for fp in neural.values():
        for rname, r in refs.items():
            _, bad = slot_disagreements(fp, r, st)
            universal = set(bad) if universal is None else (universal & set(bad))
    for k in sorted(universal or []):
        print(f"    {k[0]:<30} {k[1]}")
    n_slots = len(shared_keys(next(iter(neural.values())), bf, st))
    print(f"\n    {len(universal or [])} such slots out of ~{n_slots} jointly stable "
          f"= {len(universal or []) / max(1, n_slots):.1%} of every neural distance")

    with open(args.out, "w") as f:
        json.dump({"per_intervention_neural": {k: float(np.mean(v))
                                               for k, v in neural_iv.items()},
                   "per_intervention_symbolic": {k: float(np.mean(v))
                                                 for k, v in sym_iv.items()},
                   "per_predicate": {k: float(np.mean(v)) for k, v in pr.items()},
                   "universal_slots": ["|".join(k) for k in sorted(universal or [])],
                   "models": kept}, f, indent=1, default=float)
    print(f"\nsaved: {args.out}, {args.save_dir}/fingerprints.json, "
          f"{args.save_dir}/*.pt")


if __name__ == "__main__":
    main()
