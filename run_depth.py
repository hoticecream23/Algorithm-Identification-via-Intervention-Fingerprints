"""Experiment D -- is Proposition 5's vacuity gap a resolution artefact?

Every neural experiment here drew probe graphs gated on *hop* eccentricity >= 5, which
`FINDINGS_P2_WITNESS.md` showed is not the depth relaxation sees. Measured on the
weighted axis that distribution is depth ~6.0, where `|S_min| = 3` -- one step above the
regime where the instrument provably cannot resolve anything (`<= 4.7`).

Nothing is invalidated by that: weighted depth is provably >= hop depth, so the gate never
admitted a graph that was too shallow. But the headroom is minimal. This asks whether the
vacuity gap survives at weighted depth ~10, where `|S_min| = 2`.

The statistic, both ladder rungs, the repaired firing rule and every threshold are
inherited verbatim from `PREREGISTRATION_H.md`. The probe-graph distribution is the only
independent variable. Read `PREREGISTRATION_DEPTH.md` before interpreting any output.

    python run_depth.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.graphs import weighted_depth
from fpid.identify import classify, shared_keys
from fpid.interventions import INTERVENTIONS
from fpid.separation import minimal_separating_set, unseparated_pairs
from run_e1_e2 import deep_probe_graphs
from run_family_spike import load_checkpoints
from run_h import fingerprint_of, fire_fraction, ref_to_ref_median

RULE = "=" * 78

# --- all inherited from PREREGISTRATION_H.md; nothing below is chosen here ---
FIRE_LO, FIRE_HI = 0.35, 0.60
CLOSED_EXPLORATORY = 7  # of 9
CLOSED_CONFIRMATORY = 10  # of 12
# --- declared in PREREGISTRATION_DEPTH.md ---
DEPTH_FLOOR = 8.0  # G1: mean weighted depth the DEEP arm must reach
SLOT_RATIO_FLOOR = 0.8  # G4: DEEP comparable slots vs CONTROL

ARMS = {
    "CONTROL": {"density": 0.05, "min_depth": 5},
    "DEEP": {"density": 0.02, "min_depth": 8},
}


def cohort_of(base: str) -> str:
    """`plain_700` -> exploratory, `plain_713` -> confirmatory."""
    return "exploratory" if int(base.split("_")[1]) % 100 // 10 == 0 else "confirmatory"


def build_arm(arm: str, n_graphs: int, nodes: int, seed: int):
    cfg = ARMS[arm]
    graphs = deep_probe_graphs(
        n_graphs, nodes, np.random.default_rng(seed), cfg["density"], cfg["min_depth"]
    )
    hop, wdep = [], []
    for g, s in graphs:
        h = g.hop_distances(s)
        fin = h[np.isfinite(h)]
        hop.append(float(fin.max()) if fin.size else 0.0)
        wdep.append(weighted_depth(g, s))
    return graphs, float(np.mean(hop)), float(np.mean(wdep)), min(wdep)


def slot_counts(fp, refs, stability) -> int:
    """Median jointly-stable slot count against the reference library (G4)."""
    return int(np.median([len(shared_keys(fp, r, stability)) for r in refs.values()]))


def run_arm(arm, graphs, ckpts, budget, tol, stability):
    """Everything measured on one arm. Gates are evaluated by the caller."""
    iv_names = [iv.name for iv in INTERVENTIONS]
    refs = {
        n: fingerprint_of(finite_init(c), graphs, budget, tol, True)
        for n, c in ALGORITHMS.items()
    }
    bad = unseparated_pairs(refs, iv_names, stability)
    minimal = minimal_separating_set(refs, iv_names, stability)
    median = ref_to_ref_median(refs, stability)

    recs = []
    for name, ctor in ckpts:
        fp = fingerprint_of(ctor, graphs, budget, tol, True)
        label, dists = classify(fp, refs, stability)
        d_near = min(dists.values())
        recs.append(
            {
                "checkpoint": name,
                "cohort": cohort_of(name),
                "label": label,
                "d_near": float(d_near),
                "closed": bool(d_near < median),
                "slots": slot_counts(fp, refs, stability),
                "fire_frac": fire_fraction(ctor, graphs, budget, tol, True),
            }
        )
    return {
        "arm": arm,
        "unseparated": [list(p) for p in bad],
        "minimal": minimal,
        "median": float(median),
        "records": recs,
    }


def summarise(res, cohort, need):
    rows = [r for r in res["records"] if r["cohort"] == cohort]
    closed = sum(1 for r in rows if r["closed"])
    d_near = float(np.median([r["d_near"] for r in rows]))
    fired = sum(1 for r in rows if FIRE_LO <= r["fire_frac"] <= FIRE_HI)
    slots = int(np.median([r["slots"] for r in rows]))
    return {
        "cohort": cohort,
        "n": len(rows),
        "closed": closed,
        "need": need,
        "median_d_near": d_near,
        "fired_in_window": fired,
        "median_slots": slots,
        "ratio": d_near / res["median"] if res["median"] else float("nan"),
    }


def report(results, n_graphs) -> dict:
    print("\n" + RULE)
    print(f"PASS AT n = {n_graphs} PROBE GRAPHS")
    print(RULE)

    print("\nG1 -- did the depth manipulation land?")
    print(f"  {'arm':<10}{'hop':>8}{'weighted':>11}{'min w':>8}")
    for arm, r in results.items():
        print(f"  {arm:<10}{r['hop']:8.1f}{r['wdepth']:11.1f}{r['min_wdepth']:8.0f}")
    g1 = (
        results["DEEP"]["wdepth"] >= DEPTH_FLOOR
        and results["DEEP"]["wdepth"] > results["CONTROL"]["wdepth"]
    )
    print(f"  => {'PASS' if g1 else 'FAIL'} (need DEEP >= {DEPTH_FLOOR} and > CONTROL)")

    print("\nG3 -- negative control: symbolic separation at each distribution")
    for arm, r in results.items():
        m = r["minimal"]
        print(
            f"  {arm:<10}unseparated: {r['unseparated'] or 'none':<12} "
            f"|S_min| = {len(m) if m else 'n/a'}"
        )
    g3 = all(
        not r["unseparated"] and r["minimal"] is not None and len(r["minimal"]) <= 3
        for r in results.values()
    )
    print(f"  => {'PASS' if g3 else 'FAIL'}")

    print("\nPRIMARY -- d_near vs reference-to-reference median, repaired firing")
    summaries = {}
    for arm, r in results.items():
        print(f"\n  [{arm}]  reference-to-reference median = {r['median']:.4f}")
        print(
            f"    {'cohort':<16}{'n':>4}{'closed':>9}{'need':>7}"
            f"{'med d_near':>13}{'ratio':>9}{'fired':>8}{'slots':>8}"
        )
        for cohort, need in (
            ("exploratory", CLOSED_EXPLORATORY),
            ("confirmatory", CLOSED_CONFIRMATORY),
        ):
            s = summarise(r, cohort, need)
            summaries[(arm, cohort)] = s
            print(
                f"    {cohort:<16}{s['n']:>4}{s['closed']:>9}{s['need']:>7}"
                f"{s['median_d_near']:>13.4f}{s['ratio']:>9.2f}"
                f"{s['fired_in_window']:>8}{s['median_slots']:>8}"
            )

    print("\nG2 -- firing repair still lands at this depth")
    g2 = all(
        s["fired_in_window"] >= s["need"]
        for (arm, _c), s in summaries.items()
        if arm == "DEEP"
    )
    print(f"  => {'PASS' if g2 else 'FAIL'} (DEEP arm, per cohort)")

    print("\nG4 -- comparable-slot guard (the FINDINGS_E5.md artefact)")
    g4 = True
    for cohort in ("exploratory", "confirmatory"):
        c, d = summaries[("CONTROL", cohort)], summaries[("DEEP", cohort)]
        ok = d["median_slots"] >= SLOT_RATIO_FLOOR * c["median_slots"]
        g4 &= ok
        print(
            f"  {cohort:<16}CONTROL {c['median_slots']:>3}  DEEP {d['median_slots']:>3}  "
            f"{'ok' if ok else 'SHRANK -- denominator inflation possible'}"
        )
    print(f"  => {'PASS' if g4 else 'FAIL'}")

    gates = {"G1_depth": g1, "G2_firing": g2, "G3_negative_control": g3, "G4_slots": g4}
    if not all(gates.values()):
        failed = [k for k, v in gates.items() if not v]
        verdict = f"INCONCLUSIVE ({', '.join(failed)})"
    else:
        expl = summaries[("DEEP", "exploratory")]
        conf = summaries[("DEEP", "confirmatory")]
        closed = (
            expl["closed"] >= CLOSED_EXPLORATORY
            and conf["closed"] >= CLOSED_CONFIRMATORY
        )
        verdict = "CLOSED" if closed else "UNCHANGED"

    print("\n" + RULE)
    print(f"VERDICT (n = {n_graphs}, pre-registered ladder): {verdict}")
    print(RULE)
    return {
        "n_graphs": n_graphs,
        "gates": gates,
        "summaries": {f"{a}/{c}": s for (a, c), s in summaries.items()},
        "arms": results,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8, help="primary pass (inherited)")
    ap.add_argument("--secondary-graphs", type=int, default=24, help="declared robustness pass")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--out", default="depth_results.json")
    args = ap.parse_args()
    budget = 2 * args.nodes

    print(RULE)
    print("EXPERIMENT D -- is the vacuity gap a resolution artefact?")
    print("Pre-registered in PREREGISTRATION_DEPTH.md. Ladder: CLOSED / UNCHANGED, no middle.")
    print(RULE)

    ckpts = load_checkpoints(args.save_dir)
    n_e, n_c = (
        sum(1 for n, _ in ckpts if cohort_of(n) == k)
        for k in ("exploratory", "confirmatory")
    )
    print(f"\n  {len(ckpts)} checkpoints: {n_e} exploratory, {n_c} confirmatory")

    passes = []
    for n_graphs in (args.probe_graphs, args.secondary_graphs):
        results = {}
        for arm in ARMS:
            graphs, hop, wdep, min_w = build_arm(arm, n_graphs, args.nodes, args.seed)
            r = run_arm(arm, graphs, ckpts, budget, args.tol, args.stability)
            r.update(hop=hop, wdepth=wdep, min_wdepth=min_w)
            results[arm] = r
        passes.append(report(results, n_graphs))

    primary, secondary = passes
    print("\n" + RULE)
    if primary["verdict"] == secondary["verdict"]:
        final = primary["verdict"]
        print(f"FINAL: {final}   (primary n={primary['n_graphs']} and "
              f"secondary n={secondary['n_graphs']} agree)")
    else:
        final = "INCONCLUSIVE (sample size)"
        print(f"FINAL: {final}")
        print(f"  primary n={primary['n_graphs']}: {primary['verdict']}")
        print(f"  secondary n={secondary['n_graphs']}: {secondary['verdict']}")
        print("  Pre-registered: disagreement between passes is reported as inconclusive,")
        print("  not resolved in either direction.")
    print(RULE)

    with open(args.out, "w") as fh:
        json.dump({"passes": passes, "final": final}, fh, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
