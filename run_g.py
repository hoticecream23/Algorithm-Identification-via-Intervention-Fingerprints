"""G: do intervention fingerprints discriminate the networks from *each other*?

Every neural experiment in this project measured network -> symbolic reference, and
`THEORY.md` Proposition 5 says that comparison is vacuous: `d_near` 0.37-0.40 against a
reference-to-reference median of 0.303, so the nearest reference is the least-bad of five poor
matches. Nothing in that argument prevents the networks from being separable **from each
other**, and the discrimination claim E3 set out to test never needed the references at all.

`fpid.identify.distance` is fingerprint-to-fingerprint and symmetric, and it has never been
called on two neural fingerprints. B0 measured within-group cross-seed distance (mean 0.042);
no findings document reports a distance between two networks from different arms.

The statistic, the null, the ladder and every control are frozen in `PREREGISTRATION_G.md`,
which was written before this file existed. Nothing here selects.

    python run_g.py --stage 1              # existing checkpoints, CPU, minutes
    python run_g.py --stage 1 --smoke      # plumbing only
    python run_g.py --stage 2              # trains 12 fresh models, needs CUDA
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import Fingerprint
from fpid.identify import distance
from fpid.interventions import firing_round
from run_e1_e2 import NOISE_FLOOR, deep_probe_graphs, fingerprint_of, true_distance_dataset

RULE = "=" * 78
ARMS = ("plain", "gated", "halt")
GATE_EPS = 0.002  # matches run_e3.py / run_decompose.py
HALT_K = 8

# Frozen in PREREGISTRATION_G.md section 4.
PURITY_DISCRIMINATES = 7
PURITY_NULL = 4
ALPHA_LEVEL = 0.05
# Section 6, NC3: pseudo-arm purity at or above this voids the primary.
PSEUDO_VOID = 7


def arm_of(name: str) -> str:
    return name.split("_")[0]


# ---------------------------------------------------------------- exact null


def partitions(items: list, sizes: tuple[int, ...]):
    """Every unordered partition of `items` into blocks of the given sizes, once each.

    The lowest-indexed remaining item is always placed into the next block, which is what
    makes each partition appear exactly once rather than once per block ordering. For
    9 items into three blocks of 3 this yields 9!/(3!^3 3!) = 280 partitions -- small enough
    that the null is enumerated exactly and there is no Monte-Carlo error to report.
    """
    if not sizes:
        yield []
        return
    first, rest = items[0], items[1:]
    for combo in combinations(rest, sizes[0] - 1):
        chosen = set(combo)
        remaining = [x for x in rest if x not in chosen]
        for tail in partitions(remaining, sizes[1:]):
            yield [(first, *combo), *tail]


def group_map(part: list[tuple]) -> dict:
    return {x: i for i, blk in enumerate(part) for x in blk}


def purity(names: list[str], D: np.ndarray, labels: dict[str, str]) -> int:
    """1-NN label purity. Ties are broken *against* the hypothesis (section 4).

    Fingerprint distances are rationals over a small denominator, so exact ties between a
    same-arm and a different-arm neighbour are likely rather than exotic. Scoring a tie as a
    hit would let the statistic collect credit for a measurement that did not resolve the
    comparison, so any tied neighbour from a different arm costs the point.
    """
    hits = 0
    for i, a in enumerate(names):
        others = [j for j in range(len(names)) if j != i]
        best = min(D[i, j] for j in others)
        nearest = [names[j] for j in others if D[i, j] <= best + 1e-12]
        if all(labels[x] == labels[a] for x in nearest):
            hits += 1
    return hits


def gap(names: list[str], D: np.ndarray, labels: dict[str, str]) -> float:
    """Mean between-group distance minus mean within-group distance. Descriptive (S1)."""
    win = [D[i, j] for i, j in combinations(range(len(names)), 2)
           if labels[names[i]] == labels[names[j]]]
    bet = [D[i, j] for i, j in combinations(range(len(names)), 2)
           if labels[names[i]] != labels[names[j]]]
    if not win or not bet:
        return float("nan")
    return float(np.mean(bet) - np.mean(win))


def exact_p(names: list[str], D: np.ndarray, observed: float, stat, sizes: tuple[int, ...]):
    """Exact one-sided p over every relabelling of the items into blocks of `sizes`.

    The null is "the fingerprints are what they are, but the arm labels carry no
    information" -- so the fingerprints and hence `D` are held fixed and only the labelling
    moves. That is the correct null for this question: it asks whether the *grouping* is
    special, not whether the models are.
    """
    total = hits = 0
    for part in partitions(sorted(names), sizes):
        gm = group_map(part)
        total += 1
        if stat(names, D, {x: str(gm[x]) for x in names}) >= observed - 1e-12:
            hits += 1
    return hits / total, total


# ---------------------------------------------------------------- checkpoints


def load_models(save_dir: str, only: set[str] | None = None):
    """(base, model, ctor) per checkpoint. The model itself is needed for the gate check."""
    import torch

    from fpid.neural import MPNN, GatedMPNN, neural_ctor

    out = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*.pt"))):
        base = os.path.basename(path)[:-3]
        if only is not None and base not in only:
            continue
        arch = arm_of(base)
        if arch not in ARMS:
            continue
        model = GatedMPNN(eps=GATE_EPS) if arch == "gated" else MPNN()
        try:
            model.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception as exc:  # noqa: BLE001
            # Loud rather than swallowed, and non-fatal: a stale checkpoint must not take
            # down a run whose other eight objects are fine. It is reported as lost power
            # in the manipulation check rather than silently shrinking the sample.
            print(f"  (skipping {base}: {exc})")
            continue
        model.eval()
        deploy = HALT_K if arch == "halt" else None
        out.append((base, model, neural_ctor(model, "cpu", max_rounds=deploy)))
    return out


def manipulation_check(objs, eval_sets, probe_graphs, rounds, budget, tol, admit_mae):
    """E3's Gate 1, reused unchanged. Runs before any distance is computed.

    E1 read as a null about fingerprints and was a null about MPNN capacity, because the
    independent variable was never checked. This is that check: if the architectural
    differences are not active on these checkpoints, a null here says nothing about
    fingerprints and the primary statistic is not reported at all.
    """
    import torch

    from fpid.train import evaluate, pack

    print(RULE)
    print("MANIPULATION CHECK -- are the architectural differences actually active?")
    print(RULE)

    rows = []
    for base, model, ctor in objs:
        arch = arm_of(base)
        tr_rounds = HALT_K if arch == "halt" else rounds
        mae = evaluate(model, eval_sets[tr_rounds], tr_rounds, "cpu")

        frozen = float("nan")
        if hasattr(model, "reset"):
            with torch.no_grad():
                model.reset()
                adj, w, is_src, target = pack(eval_sets[tr_rounds], "cpu")
                d = target[0].clone()
                for _ in range(tr_rounds):
                    d = model(d, is_src, w, adj)
                frozen = float(model.frozen.float().mean())

        fires = [firing_round(ctor(g.copy(), s).run(budget), budget, tol)
                 for g, s in probe_graphs]
        rows.append({"checkpoint": base, "arm": arch, "mae": mae,
                     "frozen_final": frozen, "median_fire": float(np.median(fires))})
        print(f"  {base:<12} {arch:<6} MAE {mae:6.3f}  frozen "
              f"{'n/a  ' if arch != 'gated' else f'{frozen:5.2f}'}  "
              f"median fire round {np.median(fires):5.1f}")

    mc1 = [r for r in rows if r["mae"] >= admit_mae]
    gated = [r for r in rows if r["arm"] == "gated"]
    halt = [r for r in rows if r["arm"] == "halt"]
    mc2 = bool(gated) and min(r["frozen_final"] for r in gated) > 0.50
    mc3 = bool(halt) and max(r["median_fire"] for r in halt) < HALT_K

    print(f"\n  MC1 admission (MAE < {admit_mae}): "
          f"{len(rows) - len(mc1)}/{len(rows)} admitted"
          + (f"  DROPPED: {[r['checkpoint'] for r in mc1]}" if mc1 else "  PASS"))
    print(f"  MC2 gate fires (gated frozen > 0.50): {'PASS' if mc2 else 'FAIL'}")
    print(f"  MC3 response room (halt fires < k={HALT_K}): {'PASS' if mc3 else 'FAIL'}")
    ok = mc2 and mc3
    if not ok:
        print("\n  => INCONCLUSIVE (mechanism). The primary statistic is NOT reported.")
        print("     A null here would be about the checkpoints, not about fingerprints.")
    else:
        print("\n  => PASS")
    return ok, rows, {r["checkpoint"] for r in mc1}


# ---------------------------------------------------------------- controls


def instrument_health(nodes, density, stability, budget, tol, probe_graphs) -> bool:
    """NC1: the five references must still be separated on this graph draw.

    A positive control on the instrument. If the fingerprint pipeline cannot do the thing it
    is known to do, nothing downstream in this run means anything -- and that failure would
    look exactly like a neural null.
    """
    from fpid.interventions import INTERVENTIONS
    from fpid.separation import minimal_separating_set, unseparated_pairs

    print("\n" + RULE)
    print("NC1 -- instrument health: are the five references still separated?")
    print(RULE)
    refs = {name: fingerprint_of(finite_init(ctor), probe_graphs, budget, tol)
            for name, ctor in ALGORITHMS.items()}
    names = [iv.name for iv in INTERVENTIONS]
    bad = unseparated_pairs(refs, names, stability)
    minimal = minimal_separating_set(refs, names, stability)
    ok = not bad and minimal is not None and len(minimal) <= 3
    print(f"  unseparated pairs: {bad or 'none'}")
    print(f"  minimal separating set: {minimal} (size {len(minimal) if minimal else 'n/a'})")
    print(f"  => {'PASS' if ok else 'FAIL -- the pipeline is broken on this draw'}")
    return ok, refs


# ---------------------------------------------------------------- reporting


def report_primary(label, names, D, labels, sizes, discriminates, null_at, quote_p=True):
    obs = purity(names, D, labels)
    n = len(names)
    p, total = exact_p(names, D, obs, purity, sizes)
    g = gap(names, D, labels)
    pg, _ = exact_p(names, D, g, gap, sizes)

    print(f"\n  {label}")
    print(f"    1-NN arm purity      {obs}/{n}   "
          f"(chance {(sizes[0]-1)/(n-1)*100:.0f}%, i.e. {(sizes[0]-1)*n/(n-1):.2f}/{n})")
    if quote_p:
        print(f"    exact p              {p:.4f}   over {total} partitions "
              f"(floor {1/total:.4f})")
    else:
        print(f"    exact p              {p:.4f}   over {total} partitions -- "
              f"FLOOR IS {1/total:.2f}, NOT QUOTED AS SIGNIFICANCE (section 5)")
    print(f"    S1 between - within  {g:+.4f}   exact p {pg:.4f}   [descriptive]")

    if obs >= discriminates and (p < ALPHA_LEVEL if quote_p else False):
        verdict = "DISCRIMINATES"
    elif obs <= null_at:
        verdict = "NULL"
    else:
        verdict = "INDETERMINATE"
    print(f"    => {verdict}")
    return {"purity": obs, "n": n, "p": p, "partitions": total,
            "gap": g, "gap_p": pg, "verdict": verdict}


def distance_matrix(names, fps, stability, normalise):
    D = np.zeros((len(names), len(names)))
    for i, j in combinations(range(len(names)), 2):
        D[i, j] = D[j, i] = distance(fps[names[i]], fps[names[j]], stability, normalise)
    return D


def show_matrix(names, D):
    print("\n    " + " " * 12 + "".join(f"{x[:10]:>11}" for x in names))
    for i, a in enumerate(names):
        cells = "".join("        -  " if i == j else f"{D[i,j]:11.3f}"
                        for j in range(len(names)))
        print(f"    {a:<12}" + cells)


# ---------------------------------------------------------------- secondaries


def phase_f_embedding(objs, graphs, cal_path, save_dir):
    """S3: 1-NN purity in fitted (alpha, gamma), at two measurement windows.

    A rejected generative model can still be a good coordinate system -- chi2 ~ 1e4 says the
    family does not *contain* these networks, not that the projection is uninformative. But
    `FINDINGS_FAMILY.md` measured `plain_700` at (0.670, 0.283) with a 6-round profile and
    (0.316, 0.864) with an 8-round one, while family members recover identically under the
    same change. So the window is a pre-registered gate, not a footnote: the embedding counts
    as informative only if the arm assignment is the same at both windows.
    """
    from fpid.estimate import ProfileCache, beta_calibration, fit, profiles_of

    print("\n" + RULE)
    print("S3 -- Phase-F (alpha, gamma) embedding, at two measurement windows")
    print(RULE)
    out = {}
    for rounds in (6, 8):
        cal = beta_calibration(graphs[:3], gamma_grid=(0.5, 1.0), cache_path=cal_path)
        cache = ProfileCache(graphs, rounds=rounds)
        pts, names = [], []
        for base, _model, ctor in objs:
            r = fit(profiles_of(ctor, graphs, rounds=rounds), graphs, cal=cal, cache=cache)
            pts.append([r.theta.alpha, r.theta.gamma])
            names.append(base)
            print(f"  [{rounds}r] {base:<12} alpha {r.theta.alpha:6.3f}  "
                  f"gamma {r.theta.gamma:6.3f}  chi2 {r.chi2:10.1f}")
        P = np.asarray(pts)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        labels = {x: arm_of(x) for x in names}
        assign = {x: names[int(np.argmin([D[i, j] if i != j else np.inf
                                          for j in range(len(names))]))]
                  for i, x in enumerate(names)}
        out[rounds] = {
            "params": {n: pts[i] for i, n in enumerate(names)},
            "purity": purity(names, D, labels),
            "nn": {k: arm_of(v) for k, v in assign.items()},
        }
        print(f"  [{rounds}r] 1-NN arm purity {out[rounds]['purity']}/{len(names)}")
    stable = out[6]["nn"] == out[8]["nn"]
    print(f"\n  window stability (identical 1-NN arm assignment at 6r and 8r): "
          f"{'STABLE' if stable else 'UNSTABLE -- the embedding measures the window'}")
    out["stable"] = bool(stable)
    return out


def residual_features(objs, graphs, rounds):
    """S4: gamma_v IQR, gamma_v > 1 rate, negative-entry rate, by arm. Weak by construction."""
    from run_residual_audit import rows_of

    print("\n" + RULE)
    print("S4 -- residual-audit features by arm  [post-hoc read, weak evidence]")
    print(RULE)
    print(f"  {'checkpoint':<12}{'arm':<7}{'g_v IQR':>9}{'g_v>1':>8}{'neg':>8}")
    out = {}
    for base, _model, ctor in objs:
        r = rows_of(ctor, graphs, rounds)
        gv = r["gamma_v"][np.isfinite(r["gamma_v"])]
        n_off = float(np.sum(r["n_off"])) if len(r["n_off"]) else 0.0
        row = {
            "gamma_iqr": float(np.subtract(*np.percentile(gv, [75, 25]))) if gv.size else float("nan"),
            "gamma_gt1": float(np.mean(gv > 1.0)) if gv.size else float("nan"),
            "neg_frac": float(np.sum(r["n_neg"]) / n_off) if n_off > 0 else float("nan"),
        }
        out[base] = row
        print(f"  {base:<12}{arm_of(base):<7}{row['gamma_iqr']:9.3f}"
              f"{row['gamma_gt1']*100:7.1f}%{row['neg_frac']*100:7.1f}%")
    return out


# ---------------------------------------------------------------- stage 2


def train_fresh(args, seeds_per_arm: int):
    """Stage 2: fresh seeds, `run_decompose.py`'s exact training configuration.

    Four per arm rather than three is the whole point of doing this: the plain/gated subset
    then has 8 items and 35 partitions, so an exact p of 0.029 is attainable and section 5's
    ceiling of 0.1 is lifted.
    """
    import torch

    from fpid.neural import GatedMPNN
    from fpid.train import evaluate, make_dataset, train_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rounds = args.nodes
    os.makedirs(args.save_dir, exist_ok=True)

    rng_d = np.random.default_rng(1)  # fresh data draw, not run_decompose's rng(0)
    train_set = make_dataset(args.train_graphs, args.nodes, rounds, rng_d,
                             density=args.density, min_depth=args.min_depth)
    eval_set = make_dataset(8, args.nodes, rounds, rng_d,
                            density=args.density, min_depth=args.min_depth)
    admit = {k: true_distance_dataset(eval_set, at=k) for k in (rounds, HALT_K)}

    print(RULE)
    print(f"STAGE 2 -- training {3 * seeds_per_arm} fresh checkpoints on {device}")
    print(RULE)
    made = []
    for arch, offset, tr_rounds, fn in (
        ("plain", 710, rounds, None),
        ("gated", 810, rounds, lambda h, a: GatedMPNN(h, a, eps=GATE_EPS)),
        ("halt", 910, HALT_K, None),
    ):
        for i in range(seeds_per_arm):
            seed = offset + i
            base = f"{arch}_{seed}"
            path = f"{args.save_dir}/{base}.pt"
            if os.path.exists(path) and not args.retrain:
                print(f"  {base:<12} already present, reusing")
                made.append(base)
                continue
            model, _ = train_model(train_set, tr_rounds, seed=seed, hints=False,
                                   epochs=args.epochs, device=device, model_fn=fn)
            mae = evaluate(model, admit[tr_rounds], tr_rounds, device)
            torch.save(model.state_dict(), path)
            made.append(base)
            print(f"  {base:<12} MAE {mae:6.3f}  saved")
    return made


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--train-graphs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds-per-arm", type=int, default=4)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--admit-mae", type=float, default=2.5)
    ap.add_argument("--graph-seed", type=int, default=None)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--cal-path", default="artifacts/beta_calibration.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-secondary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.probe_graphs, args.epochs, args.seeds_per_arm = 3, 60, 2
        args.train_graphs = 8
    out_path = args.out or f"g_stage{args.stage}_results.json"

    # Stage 2 draws fresh probe graphs as well as fresh seeds. Reusing Stage 1's graph draw
    # would leave the two stages sharing half their randomness, and a "replication" that
    # shares the measurement instrument's own sample is not one.
    graph_seed = args.graph_seed if args.graph_seed is not None else (0 if args.stage == 1 else 21)

    rounds = args.nodes
    budget = 2 * args.nodes
    print(f"stage={args.stage}  n={args.nodes}  probe_graphs={args.probe_graphs}  "
          f"tol={args.tol}  stability={args.stability}  graph_seed={graph_seed}")
    print("pre-registered decision rule: PREREGISTRATION_G.md\n")

    if args.stage == 2:
        wanted = set(train_fresh(args, args.seeds_per_arm))
    else:
        wanted = {f"{a}_{s}" for a, base in (("plain", 700), ("gated", 800), ("halt", 900))
                  for s in range(base, base + 3)}

    probe_graphs = deep_probe_graphs(args.probe_graphs, args.nodes,
                                     np.random.default_rng(graph_seed),
                                     args.density, args.min_depth)
    depths = [int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max())
              for g, _ in probe_graphs]
    print(f"\nprobe-graph depths: {depths}  (median {np.median(depths):.1f})")

    results: dict = {"args": vars(args), "graph_seed": graph_seed, "depths": depths}

    nc1_ok, _refs = instrument_health(args.nodes, args.density, args.stability,
                                      budget, args.tol, probe_graphs)
    results["nc1_instrument_health"] = bool(nc1_ok)
    if not nc1_ok:
        print("\n  Stopping: the fingerprint pipeline is broken on this graph draw.")
        json.dump(results, open(out_path, "w"), indent=1, default=float)
        return

    objs = load_models(args.save_dir, only=wanted)
    if len(objs) < 6:
        print(f"\n  only {len(objs)} checkpoints found under {args.save_dir}; need >= 6")
        return

    from fpid.train import make_dataset

    rng_e = np.random.default_rng(0 if args.stage == 1 else 1)
    eval_set = make_dataset(8, args.nodes, rounds, rng_e,
                            density=args.density, min_depth=args.min_depth)
    eval_sets = {k: true_distance_dataset(eval_set, at=k) for k in (rounds, HALT_K)}

    mech_ok, mech_rows, dropped = manipulation_check(
        objs, eval_sets, probe_graphs, rounds, budget, args.tol, args.admit_mae
    )
    results["manipulation"] = {"rows": mech_rows, "pass": bool(mech_ok),
                              "dropped": sorted(dropped)}
    if not mech_ok:
        results["verdict"] = "INCONCLUSIVE (mechanism)"
        json.dump(results, open(out_path, "w"), indent=1, default=float)
        print(f"\nwrote {out_path}")
        return

    objs = [o for o in objs if o[0] not in dropped]

    print("\n" + RULE)
    print("Fingerprinting checkpoints")
    print(RULE)
    fps: dict[str, Fingerprint] = {}
    for base, _model, ctor in objs:
        fps[base] = fingerprint_of(ctor, probe_graphs, budget, args.tol)
        print(f"  {base}")

    names = sorted(fps)
    labels = {x: arm_of(x) for x in names}
    counts = {a: sum(1 for x in names if labels[x] == a) for a in ARMS}
    sizes = tuple(counts[a] for a in ARMS if counts[a])
    per_arm = sizes[0]
    balanced = len(set(sizes)) == 1

    print("\n" + RULE)
    print("PRIMARY -- 1-NN arm purity in fingerprint space (raw distance, of record)")
    print(RULE)
    D = distance_matrix(names, fps, args.stability, normalise=False)
    show_matrix(names, D)

    within = [D[i, j] for i, j in combinations(range(len(names)), 2)
              if labels[names[i]] == labels[names[j]]]
    print(f"\n  NC2 non-degeneracy: within-arm distances min {min(within):.4f} "
          f"max {max(within):.4f}  (B0 cross-seed mean 0.042, worst 0.094)")
    nc2 = max(within) > 0.0
    print(f"    => {'PASS' if nc2 else 'FAIL -- fingerprints are identical within arm'}")
    results["nc2_nondegenerate"] = bool(nc2)

    if not balanced:
        print(f"\n  Arm sizes {counts} are unbalanced after drops; the exact null is "
              f"enumerated over blocks of {sizes}.")

    results["primary_raw"] = report_primary(
        "three arms, raw distance", names, D, labels, sizes,
        PURITY_DISCRIMINATES if per_arm == 3 else int(np.ceil(0.78 * len(names))),
        PURITY_NULL if per_arm == 3 else int(np.floor(0.45 * len(names))),
    )

    print("\n" + RULE)
    print("NC3 -- pseudo-arm control: regroup by seed index, not architecture")
    print(RULE)
    print("  {plain_700, gated_800, halt_900}, ... -- groups sharing no architectural")
    print("  property. Purity here must sit near chance, or the statistic is not reading")
    print("  architecture and the primary is void.")
    pseudo = {x: str(int(x.split("_")[1]) % 100) for x in names}
    pseudo_purity = purity(names, D, pseudo)
    print(f"\n    pseudo-arm 1-NN purity {pseudo_purity}/{len(names)}")
    void = pseudo_purity >= (PSEUDO_VOID if per_arm == 3 else int(np.ceil(0.78 * len(names))))
    print(f"    => {'VOID -- primary result withdrawn' if void else 'PASS'}")
    results["nc3_pseudo_arm"] = {"purity": pseudo_purity, "void": bool(void)}

    print("\n" + RULE)
    print("CRITICAL SUBSET -- gated vs plain (halt truncates by construction, not by")
    print("learning, so it must not carry the result)")
    print(RULE)
    sub = [x for x in names if labels[x] in ("plain", "gated")]
    if len(sub) >= 4:
        Ds = distance_matrix(sub, fps, args.stability, normalise=False)
        show_matrix(sub, Ds)
        k = sum(1 for x in sub if labels[x] == "plain")
        results["critical_subset"] = report_primary(
            "gated vs plain, raw distance", sub, Ds, labels, (k, len(sub) - k),
            len(sub), int(np.floor(len(sub) / 2)),
            quote_p=(len(sub) >= 8),
        )

    print("\n" + RULE)
    print("S2 -- the same primary under E5's normalised distance  [reported, not of record]")
    print(RULE)
    Dn = distance_matrix(names, fps, args.stability, normalise=True)
    results["s2_normalised"] = report_primary(
        "three arms, normalised distance", names, Dn, labels, sizes,
        PURITY_DISCRIMINATES if per_arm == 3 else int(np.ceil(0.78 * len(names))),
        PURITY_NULL if per_arm == 3 else int(np.floor(0.45 * len(names))),
    )

    if not args.skip_secondary:
        graphs = [g for g, _ in probe_graphs]
        try:
            results["s3_phase_f"] = phase_f_embedding(objs, graphs, args.cal_path,
                                                      args.save_dir)
        except Exception as exc:  # noqa: BLE001
            # Secondaries are descriptive by pre-registration and must never be able to
            # take down the primary, which has already been computed and printed above.
            print(f"  S3 failed: {exc}")
            results["s3_phase_f"] = {"error": str(exc)}
        try:
            results["s4_residual"] = residual_features(objs, graphs, 8)
        except Exception as exc:  # noqa: BLE001
            print(f"  S4 failed: {exc}")
            results["s4_residual"] = {"error": str(exc)}

    print("\n" + RULE)
    print(f"VERDICT -- Stage {args.stage} (pre-registered ladder, PREREGISTRATION_G.md)")
    print(RULE)
    v = results["primary_raw"]["verdict"]
    if void:
        v = "VOID (pseudo-arm control)"
    elif results["s2_normalised"]["verdict"] != results["primary_raw"]["verdict"]:
        print("  Raw and normalised measures disagree -- section 7, S2: that is")
        print("  INDETERMINATE, not 'positive under one measure'.")
        v = "INDETERMINATE"
    results["verdict"] = v
    print(f"  primary (three arms, raw): {results['primary_raw']['verdict']}")
    if "critical_subset" in results:
        print(f"  critical subset (gated vs plain): {results['critical_subset']['verdict']}")
    print(f"  => Stage {args.stage}: {v}")
    if args.stage == 1:
        print("\n  Stage 1 is EXPLORATORY whatever it says. Nine checkpoints are three")
        print("  architectures x three seeds -- three independent design points, not nine,")
        print("  and these fingerprints have already been looked at by the decomposition,")
        print("  Phase F and E5. The verdict of record is Stage 2's.")
        print("      python run_g.py --stage 2")

    print(f"\n  (fingerprint noise floor for reference: {NOISE_FLOOR})")
    json.dump(results, open(out_path, "w"), indent=1, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
