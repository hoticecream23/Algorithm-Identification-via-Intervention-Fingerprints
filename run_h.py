"""H: repair the instrument, in the two places it is known to be broken.

H1 (`ROUTES_TO_POSITIVE.md` §2.2) -- `FINDINGS_DECOMPOSITION.md`'s r = -0.233 is a correlation
over nine points, computed against a symbolic column that measures *mean pairwise slot
disagreement* rather than separating power. It ranks `inject_unreached_node` second-lowest even
though that probe is in the size-2 minimal separating set and is the unique separator of
dijkstra/spfa. That number is quoted in `HANDOFF.md` and `README.md` as the mechanism that
closes the door on probe redesign.

H2 (§2.1, as amended) -- the firing rule does not fire where it claims to. `settle_round`'s
relative threshold is anchored on the largest single-round change a run ever makes, which for a
network is its round-1 sentinel transient; every later move is masked by it. Networks were poked
at ~11% of their active life against ~45% for the references, which pins `poked_settled` to `no`
for every network in the repository.

Statistic, ladder and controls are frozen in `PREREGISTRATION_H.md`. Nothing here selects.

    python run_h.py --part 1        # numpy only, seconds
    python run_h.py --part 2        # existing checkpoints, minutes
    python run_h.py --part all
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.fingerprint import Fingerprint, aggregate, probe
from fpid.identify import classify, distance
from fpid.interventions import INTERVENTIONS, firing_round, settle_round_absolute
from fpid.separation import minimal_separating_set, separation_matrix, unseparated_pairs
from run_decompose import slot_disagreements
from run_e1_e2 import deep_probe_graphs
from run_family_spike import load_checkpoints

RULE = "=" * 78

# Frozen in PREREGISTRATION_H.md.
R_REPRODUCE = -0.233
R_REPRODUCE_TOL = 0.005
ALPHA_LEVEL = 0.05
CLOSED_AT = 7
FIRE_LO, FIRE_HI = 0.35, 0.60
FIRE_NEEDED = 7


# ---------------------------------------------------------------- exact correlation test


def exact_corr_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Pearson r, its exact two-sided permutation p over all n! relabellings, Fisher-z CI.

    n = 9 gives 362,880 permutations, so the null is enumerated rather than sampled and there
    is no Monte-Carlo error to report. The CI is the point of the exercise: a correlation on
    nine points has an interval wide enough to contain conclusions in both directions, and
    quoting the point estimate alone is what made `r = -0.233` look like a mechanism.
    """
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    obs = abs(float(xz @ yz) / n)
    hits = tot = 0
    for p in itertools.permutations(range(n)):
        if abs(float(xz @ yz[list(p)]) / n) >= obs - 1e-12:
            hits += 1
        tot += 1
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1.0 / np.sqrt(max(n - 3, 1))
    ci = (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))
    return r, hits / tot, ci


# ---------------------------------------------------------------- H1


def load_stored_fingerprints(path: str) -> dict[str, Fingerprint]:
    """The exact fingerprints `run_decompose.py` wrote, so H1 changes only one variable."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for name, d in raw.items():
        val = {tuple(k.split("|")): v for k, v in d["value"].items()}
        stab = {tuple(k.split("|")): s for k, s in d["stability"].items()}
        out[name] = Fingerprint(val, stab)
    return out


def part1(args) -> dict:
    print(RULE)
    print("H1 -- is the orthogonality verdict measured against the right quantity?")
    print(RULE)

    fps = load_stored_fingerprints(f"{args.save_dir}/fingerprints.json")
    refs = {k[4:]: v for k, v in fps.items() if k.startswith("ref_")}
    neural = {k: v for k, v in fps.items() if not k.startswith("ref_")}
    bf = refs["bellman_ford"]
    st = args.stability

    # Only the interventions the stored fingerprints actually contain. E6 later added
    # `raise_tree_edge_weight` and `dissociate_keys` to INTERVENTIONS, and they are absent
    # from the decompose run's fingerprints -- including them would enter two all-zero rows
    # into a nine-point correlation and silently change the published quantity. The negative
    # control below is what caught this.
    present = {k[0] for k in bf.value}
    iv_names = [iv.name for iv in INTERVENTIONS if iv.name in present]
    print(f"\n  interventions in the stored fingerprints: {len(iv_names)} of "
          f"{len(INTERVENTIONS)} now defined")

    # ---- the neural column, unchanged from run_decompose.py
    nv_acc = {iv: [] for iv in iv_names}
    for fp in neural.values():
        keys, bad = slot_disagreements(fp, bf, st)
        tot = {iv: 0 for iv in iv_names}
        bd = {iv: 0 for iv in iv_names}
        for k in keys:
            tot[k[0]] = tot.get(k[0], 0) + 1
        for k in bad:
            bd[k[0]] = bd.get(k[0], 0) + 1
        for iv in iv_names:
            if tot[iv]:
                nv_acc[iv].append(bd[iv] / tot[iv])
    nv = np.array([np.mean(nv_acc[iv]) if nv_acc[iv] else 0.0 for iv in iv_names])

    # ---- baseline symbolic column, also unchanged: the negative control
    sr_acc = {iv: [] for iv in iv_names}
    for a, b in itertools.combinations(sorted(refs), 2):
        keys, bad = slot_disagreements(refs[a], refs[b], st)
        tot = {iv: 0 for iv in iv_names}
        bd = {iv: 0 for iv in iv_names}
        for k in keys:
            tot[k[0]] = tot.get(k[0], 0) + 1
        for k in bad:
            bd[k[0]] = bd.get(k[0], 0) + 1
        for iv in iv_names:
            if tot[iv]:
                sr_acc[iv].append(bd[iv] / tot[iv])
    sv_rate = np.array([np.mean(sr_acc[iv]) if sr_acc[iv] else 0.0 for iv in iv_names])

    # ---- the separation-weighted columns
    sep = separation_matrix(refs, iv_names, st)
    sv_pairs = np.array([sum(1 for v in sep.values() if iv in v) for iv in iv_names], float)
    sv_unique = np.array(
        [sum(1 for v in sep.values() if v == [iv]) for iv in iv_names], float
    )
    minimal = minimal_separating_set(refs, iv_names, st)

    print(f"\n  minimal separating set: {minimal}")
    print(f"  reference pairs: {len(sep)}\n")
    print(f"  {'intervention':<30}{'neural':>9}{'sv_rate':>9}{'sv_pairs':>10}{'sv_uniq':>9}")
    order = np.argsort(-nv)
    for i in order:
        star = "  <- minimal set" if minimal and iv_names[i] in minimal else ""
        print(f"  {iv_names[i]:<30}{nv[i]:9.3f}{sv_rate[i]:9.3f}"
              f"{sv_pairs[i]:10.0f}{sv_unique[i]:9.0f}{star}")

    print("\n" + RULE)
    print("NEGATIVE CONTROL -- does sv_rate reproduce the published r = -0.233?")
    r0, p0, ci0 = exact_corr_test(nv, sv_rate)
    ok = abs(r0 - R_REPRODUCE) <= R_REPRODUCE_TOL
    print(f"  r(neural, sv_rate) = {r0:+.3f}   target {R_REPRODUCE:+.3f} "
          f"+/- {R_REPRODUCE_TOL}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  FAIL -- this is not the pipeline that produced the published number.")
        return {"negative_control": False, "r_rate": r0}

    print("\n" + RULE)
    print("RESULT -- the same correlation under three symbolic scores")
    print(RULE)
    out = {"negative_control": True, "interventions": iv_names,
           "nv": nv.tolist(), "sv_rate": sv_rate.tolist(),
           "sv_pairs": sv_pairs.tolist(), "sv_unique": sv_unique.tolist(),
           "minimal_separating_set": minimal}
    for label, col, role in (("sv_rate", sv_rate, "BASELINE (published)"),
                             ("sv_pairs", sv_pairs, "PRIMARY"),
                             ("sv_unique", sv_unique, "SECONDARY")):
        r, p, ci = exact_corr_test(nv, col)
        out[f"r_{label}"] = {"r": r, "p": p, "ci": list(ci)}
        print(f"\n  {role} -- {label}")
        print(f"    r = {r:+.3f}   exact two-sided p = {p:.4f}   "
              f"Fisher-z 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    r, p, _ = exact_corr_test(nv, sv_pairs)
    if p < ALPHA_LEVEL and r < 0:
        verdict = "UPHELD"
    elif p < ALPHA_LEVEL and r > 0:
        verdict = "REVERSED"
    else:
        verdict = "UNSUPPORTED"
    out["verdict"] = verdict

    print("\n" + RULE)
    print(f"H1 VERDICT (pre-registered ladder): {verdict}")
    print(RULE)
    if verdict == "UNSUPPORTED":
        print("  No evidence of any relationship, in either direction, at n = 9.")
        print("  This does NOT show the axes are aligned. It shows that the argument which")
        print("  closed off probe redesign rests on a correlation the data cannot resolve.")
        print("  The question is reopened, not answered.")
    return out


# ---------------------------------------------------------------- H2


def fingerprint_of(ctor, graphs, budget, tol, absolute) -> Fingerprint:
    runs = []
    for g, src in graphs:
        row = {}
        for iv in INTERVENTIONS:
            for p, v in probe(ctor, g, src, iv, budget, tol=tol,
                              absolute_firing=absolute).items():
                row[(iv.name, p)] = v
        runs.append(row)
    return aggregate(runs)


def ref_to_ref_median(refs, stability) -> float:
    names = sorted(refs)
    return float(np.median([distance(refs[a], refs[b], stability)
                            for i, a in enumerate(names) for b in names[i + 1:]]))


def fire_fraction(ctor, graphs, budget, tol, absolute) -> float:
    """Median (firing round / active life). The manipulation check's statistic."""
    out = []
    for g, src in graphs:
        traj = ctor(g.copy(), src).run(budget)
        active = settle_round_absolute(traj, tol)
        if active > 0:
            out.append(firing_round(traj, budget, tol, absolute=absolute) / active)
    return float(np.median(out)) if out else float("nan")


def part2(args) -> dict:
    print("\n" + RULE)
    print("H2 -- does the firing rule fire where it says it does, and does fixing it close")
    print("      Proposition 5's vacuity gap?")
    print(RULE)

    graphs = deep_probe_graphs(args.probe_graphs, args.nodes,
                               np.random.default_rng(0), args.density, args.min_depth)
    budget = 2 * args.nodes
    ckpts = load_checkpoints(args.save_dir)
    ckpts = [(n, c) for n, c in ckpts
             if n.endswith(("700", "701", "702", "800", "801", "802", "900", "901", "902"))]

    print("\n  MANIPULATION CHECK -- fire round as a fraction of active life")
    print(f"  (need {FIRE_NEEDED}/9 networks inside [{FIRE_LO}, {FIRE_HI}] after the repair)\n")
    print(f"    {'executor':<20}{'old':>8}{'repaired':>10}")
    rows = []
    for name, ctor in ALGORITHMS.items():
        old = fire_fraction(finite_init(ctor), graphs, budget, args.tol, False)
        new = fire_fraction(finite_init(ctor), graphs, budget, args.tol, True)
        print(f"    {name:<20}{old:8.2f}{new:10.2f}")
    for name, ctor in ckpts:
        old = fire_fraction(ctor, graphs, budget, args.tol, False)
        new = fire_fraction(ctor, graphs, budget, args.tol, True)
        rows.append({"checkpoint": name, "fire_frac_old": old, "fire_frac_new": new})
        print(f"    {name:<20}{old:8.2f}{new:10.2f}")

    inside = sum(1 for r in rows if FIRE_LO <= r["fire_frac_new"] <= FIRE_HI)
    mc = inside >= FIRE_NEEDED
    print(f"\n    {inside}/{len(rows)} networks inside the window after repair  "
          f"{'PASS' if mc else 'FAIL'}")
    if not mc:
        print("\n  => H2 INCONCLUSIVE (mechanism). No d_near is reported: the repair did not")
        print("     install the change it claims to, so any distance would be a null about")
        print("     the repair rather than about vacuity. This is E1's error, refused.")
        return {"manipulation": {"rows": rows, "pass": False}, "verdict":
                "INCONCLUSIVE (mechanism)"}

    print("\n" + RULE)
    print("NEGATIVE CONTROL -- does the repair disturb the symbolic separation?")
    iv_names = [iv.name for iv in INTERVENTIONS]
    nc = {}
    for absolute in (False, True):
        refs = {n: fingerprint_of(finite_init(c), graphs, budget, args.tol, absolute)
                for n, c in ALGORITHMS.items()}
        bad = unseparated_pairs(refs, iv_names, args.stability)
        minimal = minimal_separating_set(refs, iv_names, args.stability)
        tag = "repaired" if absolute else "original"
        print(f"  [{tag:<9}] unseparated: {bad or 'none'}   minimal: {minimal} "
              f"(size {len(minimal) if minimal else 'n/a'})")
        nc[tag] = {"unseparated": [list(p) for p in bad], "minimal": minimal}
        if absolute:
            refs_new = refs
            ok = not bad and minimal is not None and len(minimal) <= 3
    print(f"  => {'PASS' if ok else 'FAIL -- the repair breaks symbolic separation'}")
    if not ok:
        return {"manipulation": {"rows": rows, "pass": True}, "negative_control": nc,
                "verdict": "INCONCLUSIVE (negative control)"}

    median_new = ref_to_ref_median(refs_new, args.stability)
    print(f"\n  reference-to-reference median, repaired: {median_new:.4f}")

    print("\n" + RULE)
    print("PRIMARY -- d_near vs the reference-to-reference median, under the repair")
    print(RULE)
    recs = []
    for name, ctor in ckpts:
        fp = fingerprint_of(ctor, graphs, budget, args.tol, True)
        label, dists = classify(fp, refs_new, args.stability)
        d_near = min(dists.values())
        closed = d_near < median_new
        recs.append({"checkpoint": name, "label": label, "d_near": d_near,
                     "closed": bool(closed)})
        print(f"  {name:<14} d_near {d_near:.4f}  ({label:<16})  "
              f"{'CLOSED' if closed else 'still-vacuous'}")

    n_closed = sum(1 for r in recs if r["closed"])
    n = len(recs)
    if n_closed >= CLOSED_AT:
        verdict = "CLOSED"
    elif n - n_closed >= CLOSED_AT:
        verdict = "UNCHANGED"
    else:
        verdict = "INDETERMINATE"

    print("\n" + RULE)
    print(f"H2 VERDICT (pre-registered ladder): {verdict}   ({n_closed}/{n} closed)")
    print(RULE)
    if verdict == "CLOSED":
        print("  EXPLORATORY only. Experiment G established these nine are three design")
        print("  points, not nine. The confirmatory rung is the twelve Stage-2 checkpoints")
        print("  at >= 10/12, and it is specified in PREREGISTRATION_H.md.")
    elif verdict == "UNCHANGED":
        print("  The vacuity gap survives a firing rule that demonstrably fires in the right")
        print("  place. Proposition 5 stands, and stands on firmer ground than before.")
    return {"manipulation": {"rows": rows, "pass": True},
            "negative_control": nc, "median_repaired": median_new,
            "records": recs, "verdict": verdict}


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["1", "2", "all"])
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--out", default="h_results.json")
    args = ap.parse_args()

    print("pre-registered decision rule: PREREGISTRATION_H.md\n")
    results = {}
    if args.part in ("1", "all"):
        results["h1"] = part1(args)
    if args.part in ("2", "all"):
        results["h2"] = part2(args)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
