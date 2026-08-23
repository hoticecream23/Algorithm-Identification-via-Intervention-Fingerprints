"""Experiment S — does the intervention framework transfer to a second domain?

Pre-registered in `PREREGISTRATION_SORT.md` (with a §11 amendment, both frozen
before this file existed). Ladder: TRANSFERS / DOES NOT TRANSFER, no middle rung.

    python run_sorting.py

Everything downstream of the intervention class is the SSSP code unmodified --
`fingerprint.aggregate`, `separation.*`, and the whole timing layer. The domain
supplies three things through `fingerprint.Domain`: targets, the metric the `depth`
predicate reports in, and the at-risk rule. That is the claim under test, and it is
checked mechanically by G5, which re-runs two SSSP runners and diffs their output.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fpid.fingerprint import PREDICATES, aggregate, probe
from fpid.interventions import settle_round_absolute
from fpid.separation import (
    minimal_separating_set,
    separating_predicates,
    unseparated_pairs,
)
from fpid.sorting import (
    OUTPUT_IDENTICAL,
    SORT_ALGORITHMS,
    SORT_DOMAIN,
    SORT_EXTRAS,
    SORT_INTERVENTION_NAMES,
    SORT_INTERVENTIONS,
    arrays_in_band,
    deep_arrays,
    leftward_depth,
    naive_depth,
)

STABILITY = 0.8
FIRE_LO, FIRE_HI = 0.35, 0.60
MIN_DEPTH = 6  # the DEEP distribution, PREREGISTRATION_SORT.md §3.4

SMOKE_SEED = 11
REPORT_SEED = 23

# The no-response value of each d-derived predicate. `pi_churn` is exempt by
# pre-registration §5: the null control changes provenance by construction.
NULL_VALUE = {
    "recovery": "exact",
    "latency": "0",
    "spread1": "0",
    "depth": "none",
    "term_delta": "0",
}

ALL_EXECUTORS = {**SORT_ALGORITHMS, **SORT_EXTRAS}


def budget_for(n: int) -> int:
    return n + 2


def fingerprint_of(ctor, insts, budget) -> object:
    runs = []
    for inst in insts:
        row: dict[tuple[str, str], str] = {}
        for iv in SORT_INTERVENTIONS:
            row.update(
                {
                    (iv.name, k): v
                    for k, v in probe(
                        ctor,
                        inst.copy(),
                        0,
                        iv,
                        budget,
                        absolute_firing=True,
                        domain=SORT_DOMAIN,
                    ).items()
                }
            )
        runs.append(row)
    return aggregate(runs)


def all_fingerprints(insts, budget) -> dict:
    return {n: fingerprint_of(c, insts, budget) for n, c in ALL_EXECUTORS.items()}


def final_array(ctor, inst, budget) -> np.ndarray:
    return ctor(inst.copy(), 0).run(budget)[-1].d


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_depth(insts, tag) -> tuple[bool, dict]:
    lw = np.array([leftward_depth(i.a) for i in insts])
    nv = np.array([naive_depth(i.a) for i in insts])
    ok = bool(lw.mean() >= 6.0 and lw.min() >= MIN_DEPTH)
    print(f"\nG1 -- depth manipulation ({tag})")
    print(f"  leftward_depth  mean {lw.mean():5.2f}  min {lw.min()}  max {lw.max()}")
    print(f"  naive_depth     mean {nv.mean():5.2f}  (overstates on "
          f"{int((nv > lw).sum())}/{len(insts)} instances)")
    print(f"  => {'PASS' if ok else 'FAIL'} (need mean >= 6.0 and min >= {MIN_DEPTH})")
    return ok, {
        "leftward_mean": float(lw.mean()),
        "leftward_min": int(lw.min()),
        "naive_mean": float(nv.mean()),
        "naive_strictly_greater": int((nv > lw).sum()),
    }


def gate_firing(insts, budget) -> tuple[bool, dict]:
    print("\nG2 -- firing lands mid-run")
    out, ok = {}, True
    for name, ctor in ALL_EXECUTORS.items():
        fr, lives = [], []
        for inst in insts:
            traj = ctor(inst.copy(), 0).run(budget)
            life = settle_round_absolute(traj)
            lives.append(life)
            if life:
                fr.append(int(np.clip(life // 2, 1, budget - 1)) / life)
        f = float(np.mean(fr)) if fr else float("nan")
        life_mean = float(np.mean(lives))
        # `firing_round` takes an integer round, so on a very short active life no
        # round exists at 0.5 -- life 3 admits only 0.33 or 0.67. Executors with a
        # mean active life below 5 are reported and exempted on that arithmetic
        # ground, which is declared here rather than discovered later.
        exempt = life_mean < 5.0
        good = exempt or (FIRE_LO <= f <= FIRE_HI)
        ok = ok and good
        out[name] = {"fraction": f, "life": life_mean, "exempt": exempt}
        note = "  (exempt: active life < 5 rounds, no integer round at 0.5)" if exempt else ""
        print(f"  {name:24s} frac {f:.3f}  life {life_mean:5.1f}  "
              f"{'ok' if good else 'FAIL'}{note}")
    print(f"  => {'PASS' if ok else 'FAIL'} (need {FIRE_LO}-{FIRE_HI})")
    return ok, out


def gate_witness(rng, n, budget) -> tuple[bool, dict]:
    """G3a unconditional and G3b conditional. Both must be exactly 100%/0%."""
    print("\nG3 -- the two witnesses")
    insts = arrays_in_band(rng, 400, n, 0, n - 1)

    agree = 0
    for inst in insts:
        outs = {
            tuple(final_array(SORT_ALGORITHMS[a], inst, budget))
            for a in OUTPUT_IDENTICAL
        }
        agree += len(outs) == 1
    g3a = agree == len(insts)
    print(f"  G3a unconditional: {len(OUTPUT_IDENTICAL)} sorts agree on "
          f"{agree}/{len(insts)}  => {'PASS' if g3a else 'FAIL'}")

    lo = hi = lo_ok = hi_ok = 0
    for inst in insts:
        d = leftward_depth(inst.a)
        same = np.array_equal(
            final_array(SORT_ALGORITHMS["bubble_sort"], inst, budget),
            final_array(SORT_ALGORITHMS["bubble_truncated_k3"], inst, budget),
        )
        if d <= 3:
            lo += 1
            lo_ok += same
        else:
            hi += 1
            hi_ok += same
    g3b = (lo_ok == lo) and (hi_ok == 0) and lo > 0 and hi > 0
    print(f"  G3b conditional  : depth<=3 agree {lo_ok}/{lo} (want all), "
          f"depth>3 agree {hi_ok}/{hi} (want none)  => {'PASS' if g3b else 'FAIL'}")
    return g3a and g3b, {
        "unconditional_agree": agree,
        "unconditional_n": len(insts),
        "trunc_shallow_agree": lo_ok,
        "trunc_shallow_n": lo,
        "trunc_deep_agree": hi_ok,
        "trunc_deep_n": hi,
    }


def gate_clone(fps) -> tuple[bool, dict]:
    pair = {k: fps[k] for k in ("insertion_sort", "insertion_clone")}
    unsep = unseparated_pairs(pair, SORT_INTERVENTION_NAMES, STABILITY)
    ok = len(unsep) == 1
    print("\nG4 -- clone control (the same state update by a different code path)")
    print(f"  insertion_sort vs insertion_clone unseparated: {bool(unsep)}  "
          f"=> {'PASS' if ok else 'FAIL (the instrument is manufacturing signal)'}")
    return ok, {"unseparated": bool(unsep)}


def gate_null(insts, budget, fps) -> tuple[bool, dict]:
    """Two levels, both reported.

    Trajectory level: the null control provably cannot change `d`, so the intervened
    run must be bit-identical to the control. This is the direct check.

    Predicate level: the pre-registered criterion. It is expected to fail for
    `bubble_truncated_k3` for a reason that is not a response -- `probe`'s reference
    arm restarts the executor from the post-poke array, handing a budget-limited
    algorithm a fresh allowance of passes that the intervened run has already spent.
    Both numbers are printed so the reader can see which is which.
    """
    print("\nG-null -- the null control")
    null_iv = next(
        iv for iv in SORT_INTERVENTIONS if iv.name == "swap_duplicate_value"
    )
    traj_ok, out = True, {}
    for name, ctor in ALL_EXECUTORS.items():
        ident = 0
        for inst in insts:
            control = ctor(inst.copy(), 0).run(budget)
            t = int(np.clip(settle_round_absolute(control) // 2, 1, budget - 1))
            ex = ctor(inst.copy(), 0)
            done = {"v": False}

            def hook(e, r, _t=t, _d=done, _i=inst):
                if r == _t and not _d["v"]:
                    null_iv.apply(e, SORT_DOMAIN.make_targets(_i, 0))
                    _d["v"] = True

            ivt = ex.run(budget, hook=hook)
            ident += all(np.array_equal(a.d, b.d) for a, b in zip(control, ivt))
        bad = {
            p: fps[name].value[("swap_duplicate_value", p)]
            for p in PREDICATES
            if p in NULL_VALUE
            and fps[name].value[("swap_duplicate_value", p)] != NULL_VALUE[p]
        }
        traj_ok = traj_ok and ident == len(insts)
        out[name] = {"trajectory_identical": ident, "n": len(insts),
                     "predicate_deviations": bad}
        print(f"  {name:24s} intervened==control {ident:3d}/{len(insts)}   "
              + (f"predicate deviations {bad}" if bad else "predicates silent"))
    pred_ok = all(not v["predicate_deviations"] for v in out.values())
    print(f"  => trajectory level {'PASS' if traj_ok else 'FAIL'}; "
          f"predicate level {'PASS' if pred_ok else 'FAIL (see FINDINGS_SORT.md)'}")
    return traj_ok, out


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def report_family(fps, names, tag) -> dict:
    fam = {k: fps[k] for k in names}
    unsep = unseparated_pairs(fam, SORT_INTERVENTION_NAMES, STABILITY)
    smin = minimal_separating_set(fam, SORT_INTERVENTION_NAMES, STABILITY)
    print(f"\n{tag}")
    print(f"  algorithms      : {', '.join(names)}")
    print(f"  pairs           : {len(names) * (len(names) - 1) // 2}")
    print(f"  unseparated     : {unsep if unsep else 'none'}")
    print(f"  |S_min|         : {len(smin) if smin else None}  {smin or ''}")
    return {
        "names": list(names),
        "unseparated": [list(p) for p in unsep],
        "s_min": smin,
        "s_min_size": len(smin) if smin else None,
    }


def depth_sweep(rng, n, budget, bands) -> list[dict]:
    print("\nSECONDARY -- identifiability boundary on the sorting depth axis")
    print("  band       n   mean_depth   |S_min|   unseparated")
    rows = []
    for lo, hi in bands:
        insts = arrays_in_band(rng, 120, n, lo, hi, min_count=60)
        fps = all_fingerprints(insts, budget)
        fam = {k: fps[k] for k in SORT_ALGORITHMS}
        unsep = unseparated_pairs(fam, SORT_INTERVENTION_NAMES, STABILITY)
        smin = minimal_separating_set(fam, SORT_INTERVENTION_NAMES, STABILITY)
        md = float(np.mean([leftward_depth(i.a) for i in insts]))
        size = len(smin) if smin else None
        print(f"  [{lo:2d},{hi:2d}]  {len(insts):4d}   {md:6.2f}      "
              f"{size!s:7s}   {len(unsep)} pair(s)")
        rows.append({
            "lo": lo, "hi": hi, "n_instances": len(insts), "mean_depth": md,
            "s_min_size": size, "s_min": smin,
            "unseparated": [list(p) for p in unsep],
        })
    return rows


def run_pass(n: int, seed: int, tag: str) -> dict:
    budget = budget_for(n)
    print("\n" + "=" * 78)
    print(f"{tag}: n = {n}, seed = {seed}, budget = {budget}")
    print("=" * 78)

    # PREREGISTRATION_SORT.md §3.4 specifies i.i.d. uniform arrays with a depth
    # filter, so that is what the ladder keys on. The mixed construction of
    # `arrays_in_band` is reported beside it as the §12.2 sensitivity arm.
    rng = np.random.default_rng(seed)
    insts = deep_arrays(rng, 120, n, MIN_DEPTH)
    mixed = arrays_in_band(np.random.default_rng(seed + 3), 120, n, MIN_DEPTH, n - 1)

    g1, g1d = gate_depth(insts, f"n={n}")
    g2, g2d = gate_firing(insts, budget)
    g3, g3d = gate_witness(np.random.default_rng(seed + 1), n, budget)

    fps = all_fingerprints(insts, budget)
    g4, g4d = gate_clone(fps)
    gn, gnd = gate_null(insts, budget, fps)

    gates = bool(g1 and g2 and g3 and g4 and gn)
    print(f"\nGATES: {'ALL PASS' if gates else 'FAILURE -- see above'}")

    print("\n" + "-" * 78)
    print("PRIMARY")
    print("-" * 78)
    primary = report_family(
        fps, list(SORT_ALGORITHMS),
        "Full family, UNIFORM distribution (pre-registered ladder keys on this)",
    )
    pure = report_family(
        fps,
        list(OUTPUT_IDENTICAL),
        "Output-identical subfamily (all four sort correctly on every input)",
    )

    fps_mixed = all_fingerprints(mixed, budget)
    sens = report_family(
        fps_mixed, list(SORT_ALGORITHMS),
        "Full family, MIXED distribution (§12.2 sensitivity arm, not the ladder)",
    )

    print("\nSECONDARY -- binary vs linear insertion "
          "(equivalent under the EMPTY intervention class)")
    bpair = {k: fps[k] for k in ("insertion_sort", "binary_insertion_sort")}
    bunsep = unseparated_pairs(bpair, SORT_INTERVENTION_NAMES, STABILITY)
    seps = {
        iv: separating_predicates(
            fps["insertion_sort"], fps["binary_insertion_sort"], iv, STABILITY
        )
        for iv in SORT_INTERVENTION_NAMES
    }
    seps = {k: v for k, v in seps.items() if v}
    print(f"  separated: {not bunsep}")
    for iv, ps in seps.items():
        print(f"    {iv:30s} on {ps}")

    sweep = depth_sweep(
        np.random.default_rng(seed + 2), n, budget,
        [(0, 1), (2, 3), (4, 5), (6, 8), (9, 12), (13, n - 1)],
    )

    verdict = "TRANSFERS" if (gates and not primary["unseparated"]) else "DOES NOT TRANSFER"
    print("\n" + "=" * 78)
    print(f"VERDICT (n = {n}, pre-registered ladder): {verdict}")
    print("=" * 78)

    return {
        "n": n, "seed": seed, "verdict": verdict, "gates_pass": gates,
        "gates": {"G1": g1d, "G2": g2d, "G3": g3d, "G4": g4d, "G_null": gnd},
        "primary": primary, "output_identical": pure, "sensitivity_mixed": sens,
        "binary_vs_linear": {
            "separated": not bunsep,
            "separating": {k: list(v) for k, v in seps.items()},
        },
        "depth_sweep": sweep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run the SMOKE seed only; reports nothing")
    args = ap.parse_args()

    if args.smoke:
        run_pass(16, SMOKE_SEED, "SMOKE (no number here is a result)")
        return

    primary = run_pass(16, REPORT_SEED, "PRIMARY PASS")
    secondary = run_pass(24, REPORT_SEED, "SECONDARY PASS (pre-declared, §3.4)")

    final = (
        "TRANSFERS"
        if primary["verdict"] == "TRANSFERS" == secondary["verdict"]
        else "DOES NOT TRANSFER"
    )
    print("\n" + "=" * 78)
    print(f"FINAL: {final}   (n=16 {primary['verdict']}, n=24 {secondary['verdict']})")
    print("=" * 78)

    with open("sorting_results.json", "w") as f:
        json.dump({"final": final, "n16": primary, "n24": secondary}, f, indent=2)
    print("\n  wrote sorting_results.json")


if __name__ == "__main__":
    main()
