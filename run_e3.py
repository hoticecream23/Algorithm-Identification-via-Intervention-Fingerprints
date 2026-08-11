"""E3: models that genuinely differ in procedure.

The fair test the method never received. B1 failed because shallow- and deep-trained
networks differ in *update-rule generalization*, not in procedure. E1 then failed to
install a procedural difference by supervision at all: a min-aggregating MPNN has no
global readout, cannot learn Dijkstra's argmin over unsettled nodes, and never reaches
admissible accuracy.

So install the difference **by construction**, where no global comparison is needed:

- `plain`  -- MPNN, runs to convergence.            Bellman-Ford-like.
- `gated`  -- freezes a reached node once it stops moving. Dijkstra-like commitment,
              built from a purely local rule.
- `halt`   -- trained and deployed for k rounds, then stops. Truncated-BF-like.

All three are trained hint-free on identical data, so supervision is held constant and
architecture is the only manipulated variable. E2 justifies that choice: hint-free models
reach hint-supervised accuracy and carry the only margins in the project that clear their
own noise floor.

Decision rule fixed in advance in `PREREGISTRATION_E3.md`.

    python run_e3.py
    python run_e3.py --smoke
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import aggregate, probe
from fpid.identify import classify, margin
from fpid.interventions import INTERVENTIONS, firing_round
from fpid.neural import GatedMPNN, MPNN, neural_ctor
from fpid.train import evaluate, make_dataset, pack, train_model
from run_e1_e2 import (
    NOISE_FLOOR,
    deep_probe_graphs,
    drop_predicate,
    family,
    fingerprint_of,
    fisher_exact_2x2,
    true_distance_dataset,
)

# Calibrated on commitment timing and accuracy only, blind to any fingerprint. See
# PREREGISTRATION_E3.md "Calibration of eps". At 1500 epochs this gives MAE 0.708 --
# inside the plain hint-free range, so architecture is not confounded with accuracy --
# with nodes committing gradually across rounds 14-19, i.e. after propagation depth 5-6
# has delivered their values. Larger thresholds (0.05) collapse to committing at round 2
# on garbage, which is a global halt wearing a gate's clothes.
GATE_EPS = 0.002
HALT_K = 8

PREDICTED = {
    "plain": ("relaxation", "bellman_ford"),
    "gated": ("settling", "dijkstra"),
    "halt": ("relaxation", "bf_truncated_k3"),
}


@torch.no_grad()
def freeze_profile(model, dataset, rounds, device):
    """Fraction of nodes committed at each round. 0 everywhere => the gate never fired."""
    if not hasattr(model, "reset"):
        return []
    model.reset()
    adj, w, is_src, target = pack(dataset, device)
    d = target[0].clone()
    out = []
    for _ in range(rounds):
        d = model(d, is_src, w, adj)
        out.append(float(model.frozen.float().mean()))
    return out


def build(arch: str, rounds: int):
    """(model_fn, train_rounds, deploy_max_rounds) for one arm."""
    if arch == "plain":
        return None, rounds, None
    if arch == "gated":
        return (lambda h, a: GatedMPNN(h, a, eps=GATE_EPS)), rounds, None
    if arch == "halt":
        return None, HALT_K, HALT_K
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--eval-graphs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--admit-mae", type=float, default=2.5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--out", default="e3_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.epochs, args.probe_graphs = 2, 60, 3
        args.train_graphs, args.eval_graphs, args.perms = 8, 4, 2000

    rounds = args.rounds if args.rounds else args.nodes
    budget = args.budget if args.budget else 2 * args.nodes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    print(f"device={device}  n={args.nodes}  rounds={rounds}  budget={budget}  "
          f"tol={args.tol}  gate_eps={GATE_EPS}  halt_k={HALT_K}")
    print("pre-registered decision rule: PREREGISTRATION_E3.md\n")

    rng_d = np.random.default_rng(0)
    train_set = make_dataset(args.train_graphs, args.nodes, rounds, rng_d,
                             density=args.density, min_depth=args.min_depth)
    eval_set = make_dataset(args.eval_graphs, args.nodes, rounds, rng_d,
                            density=args.density, min_depth=args.min_depth)
    # One admission set per training horizon: `evaluate` reads row `tr_rounds`, so the
    # halt arm must be graded against the true answer at row k, not at row `rounds`.
    admit_sets = {k: true_distance_dataset(eval_set, at=k) for k in (rounds, HALT_K)}

    probe_graphs = deep_probe_graphs(
        args.probe_graphs, args.nodes, rng, args.density, args.min_depth
    )
    depths = [int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max())
              for g, _ in probe_graphs]
    print(f"probe-graph depths: {depths}  (median {np.median(depths):.1f})")

    refs = {name: fingerprint_of(finite_init(ctor), probe_graphs, budget, args.tol)
            for name, ctor in ALGORITHMS.items()}
    refs_nt = {k: drop_predicate(v, "term_delta") for k, v in refs.items()}

    records = []
    for arch, offset in (("plain", 0), ("gated", 300), ("halt", 400)):
        model_fn, tr_rounds, deploy = build(arch, rounds)
        print(f"\n=== {arch}: train_rounds={tr_rounds} deploy_max_rounds={deploy} ===")
        for i in range(args.seeds):
            seed = offset + i
            model, loss = train_model(
                train_set, tr_rounds, seed=seed, hints=False,
                epochs=args.epochs, device=device, model_fn=model_fn,
            )
            mae = evaluate(model, admit_sets[tr_rounds], tr_rounds, device)
            prof = freeze_profile(model, eval_set, tr_rounds, device)
            admitted = mae < args.admit_mae
            rec = {"arch": arch, "seed": seed, "loss": loss, "mae": mae,
                   "admitted": admitted, "freeze_final": prof[-1] if prof else 0.0,
                   "freeze_profile": prof}
            if admitted:
                ctor = neural_ctor(model, device, max_rounds=deploy)
                fp = fingerprint_of(ctor, probe_graphs, budget, args.tol)
                label, dists = classify(fp, refs, args.stability)
                label_nt, dists_nt = classify(
                    drop_predicate(fp, "term_delta"), refs_nt, args.stability
                )
                fires = [firing_round(ctor(g.copy(), s).run(budget), budget, args.tol)
                         for g, s in probe_graphs]
                rec |= {"label": label, "margin": margin(dists), "family": family(label),
                        "dists": dists, "label_no_term": label_nt,
                        "family_no_term": family(label_nt),
                        "margin_no_term": margin(dists_nt),
                        "fire_rounds": fires}
            records.append(rec)
            tag = (f"-> {rec['label']:<16} margin {rec['margin']:.3f} "
                   f"fire@{np.median(rec['fire_rounds']):.0f}"
                   if admitted else "-> EXCLUDED (MAE)")
            print(f"  seed {seed}: loss {loss:.5f}  MAE {mae:.3f}  "
                  f"frozen {rec['freeze_final']:.2f}  {tag}")

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1, default=float)

    arms = ("plain", "gated", "halt")
    by = {a: [r for r in records if r["arch"] == a] for a in arms}
    adm = {a: [r for r in by[a] if r["admitted"]] for a in arms}

    print("\n\n" + "=" * 70)
    print("GATE 0 -- admission (MAE < %.1f)" % args.admit_mae)
    need = max(1, args.seeds - 1)
    gate0 = True
    for a in arms:
        ok = len(adm[a]) >= need
        gate0 &= ok
        maes = ", ".join(f"{r['mae']:.2f}" for r in by[a])
        print(f"  {a:<6} {len(adm[a])}/{len(by[a])} admitted  "
              f"{'PASS' if ok else 'FAIL'}  (MAE {maes})")

    print("\n" + "=" * 70)
    print("GATE 1 -- mechanism check (is the architectural difference actually active?)")
    fz = [r["freeze_final"] for r in adm["gated"]]
    fz_plain = [r["freeze_final"] for r in adm["plain"]]
    gate_fires = bool(fz) and min(fz) > 0.5
    print(f"  gated  final frozen fraction: "
          f"{'[' + ', '.join(f'{x:.2f}' for x in fz) + ']' if fz else 'n/a'}  "
          f"{'PASS' if gate_fires else 'FAIL'} (need all > 0.50)")
    print(f"  plain  final frozen fraction: {fz_plain or 'n/a (no gate)'}")

    # Per-node commitment must be spread over rounds. If every node freezes on the same
    # round the arm is a global halt wearing a gate's clothes, and it is the `halt` arm
    # in disguise rather than a Dijkstra-like settling model.
    spreads, starts = [], []
    for r in adm["gated"]:
        prof = r["freeze_profile"]
        lo = next((i for i, x in enumerate(prof) if x > 0.25), None)
        hi = next((i for i, x in enumerate(prof) if x > 0.90), None)
        if lo is not None and hi is not None:
            spreads.append(hi - lo)
            starts.append(lo)
    spread_ok = bool(spreads) and min(spreads) >= 3 and min(starts) > 5
    print(f"  gated  commitment 25%->90% spans {spreads} rounds, starts at {starts}  "
          f"{'PASS' if spread_ok else 'FAIL'} (need span >= 3 and start > 5)")
    gate_fires = gate_fires and spread_ok
    halt_fires = [np.median(r["fire_rounds"]) for r in adm["halt"]]
    room = bool(halt_fires) and max(halt_fires) < HALT_K
    print(f"  halt   median firing round: {halt_fires}  vs k={HALT_K}  "
          f"{'PASS' if room else 'FAIL'} (need < k, else no response room)")
    gate1 = gate_fires and room
    print(f"  => {'PASS' if gate1 else 'FAIL -- E3 INCONCLUSIVE (mechanism)'}")

    print("\n" + "=" * 70)
    print("GATE 2 -- discrimination")
    names = sorted(refs)
    print(f"\n  Confusion matrix\n    {'arch':<7} " + " ".join(f"{n[:9]:>10}" for n in names))
    for a in arms:
        row = [sum(1 for r in adm[a] if r["label"] == n) for n in names]
        print(f"    {a:<7} " + " ".join(f"{c:>10}" for c in row))

    pa = sum(1 for r in adm["plain"] if r["family"] == "relaxation")
    pb = len(adm["plain"]) - pa
    ga = sum(1 for r in adm["gated"] if r["family"] == "relaxation")
    gb = len(adm["gated"]) - ga
    p_primary = fisher_exact_2x2(pa, pb, ga, gb)
    print(f"\n  PRIMARY -- gated vs plain, family level")
    print(f"    {'':<7} relaxation  settling")
    print(f"    plain   {pa:>10}  {pb:>8}")
    print(f"    gated   {ga:>10}  {gb:>8}")
    print(f"    matching prediction: {pa + gb}/{pa + pb + ga + gb}   "
          f"Fisher p = {p_primary:.4f}")

    ht = sum(1 for r in adm["halt"] if r["label"] == "bf_truncated_k3")
    pt = sum(1 for r in adm["plain"] if r["label"] == "bf_truncated_k3")
    p_halt = fisher_exact_2x2(ht, len(adm["halt"]) - ht, pt, len(adm["plain"]) - pt)
    print(f"\n  SECONDARY -- halt vs plain, labelled bf_truncated_k3")
    print(f"    halt {ht}/{len(adm['halt'])}   plain {pt}/{len(adm['plain'])}   "
          f"Fisher p = {p_halt:.4f}")

    pool = [r for a in arms for r in adm[a]]
    obs = sum(1 for r in pool if r["label"] == PREDICTED[r["arch"]][1])
    if pool:
        prng = np.random.default_rng(12345)
        arch_col = np.array([r["arch"] for r in pool])
        lab_col = [r["label"] for r in pool]
        hits = 0
        for _ in range(args.perms):
            perm = prng.permutation(arch_col)
            k = sum(1 for a, l in zip(perm, lab_col) if l == PREDICTED[a][1])
            hits += k >= obs
        p_perm = (hits + 1) / (args.perms + 1)
        print(f"\n  SECONDARY -- 3-way exact-label permutation test")
        print(f"    {obs}/{len(pool)} models match their arm's predicted reference   "
              f"p = {p_perm:.4f}  ({args.perms} permutations)")

    print("\n  Margins (unresolved below %.3f):" % NOISE_FLOOR)
    for a in arms:
        for r in adm[a]:
            flag = "  UNRESOLVED" if r["margin"] < NOISE_FLOOR else ""
            print(f"    {a:<6} seed {r['seed']:>3}  {r['label']:<16} "
                  f"margin {r['margin']:.3f}{flag}")

    na = sum(1 for r in adm["plain"] if r["family_no_term"] == "relaxation")
    ng = sum(1 for r in adm["gated"] if r["family_no_term"] == "relaxation")
    p_nt = fisher_exact_2x2(na, len(adm["plain"]) - na, ng, len(adm["gated"]) - ng)
    print(f"\n  Runtime ablation (term_delta dropped), gated vs plain: "
          f"Fisher p = {p_nt:.4f}")

    print("\n" + "=" * 70)
    print("VERDICT (pre-registered ladder)")
    if not gate0:
        verdict = "INCONCLUSIVE (training)"
    elif not gate1:
        verdict = "INCONCLUSIVE (mechanism)"
    else:
        right = pa + gb > (pa + pb + ga + gb) / 2
        resolved = all(
            r["margin"] > NOISE_FLOOR
            for r in adm["plain"] + adm["gated"]
            if (r["arch"] == "plain") == (r["family"] == "relaxation")
        )
        if p_primary < 0.05 and right and resolved:
            verdict = "STRONG"
        elif p_primary < 0.20 and right:
            verdict = "WEAK"
        else:
            verdict = "NULL"
    print(f"  E3: {verdict}")
    if verdict == "STRONG":
        print("  The method works on its actual target: models that genuinely differ in")
        print("  procedure are separated by intervention response alone. B1's negative is")
        print("  then specifically about the proxy chosen, not about the method.")
    elif verdict == "NULL":
        print("  Fingerprints do not separate architectures built to differ in procedure.")
        print("  That is the method's own target, so the neural side is finished.")


if __name__ == "__main__":
    main()
