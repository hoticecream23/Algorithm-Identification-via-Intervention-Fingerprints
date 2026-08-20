"""Phase F spike: can family parameters be estimated at all?

The neural side of this project died of a diagnosed vacuity: every neural model sits 0.37-0.40
from its nearest symbolic reference while the reference-to-reference median is 0.303, so
nearest-reference classification reports the least-bad of five poor matches (`THEORY.md`
Proposition 5, `FINDINGS_DECOMPOSITION.md`). The repair is to replace classification with
parameter estimation over a family that contains both the algorithms and the networks.

This runner does not do the science. It tests whether the method can work, in the order that
kills it cheapest:

    Gate 0a  embedding      -- does the family reproduce the five references exactly?
    Gate 0b  adequacy       -- does the observable move when each axis moves?
    Gate 1   identifiability-- can distinct parameters be told apart above measurement noise?
    Gate 2   recovery       -- can parameters put in by hand be read back out?
    Gate 2b  replication    -- on a fresh graph stream and a fresh draw, with nothing refitted
    Gate 3   losability     -- does an out-of-family generator fit badly?

Thresholds are frozen in `PREREGISTRATION_FAMILY.md` and were written before Gates 0b-3 ran.
Nothing here selects.

    python run_family_spike.py --gate all
    python run_family_spike.py --gate 0                    # seconds
    python run_family_spike.py --smoke                     # ~1 min, plumbing only
    python run_family_spike.py --reproduce-response-table  # the numbers in response.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import replace
from itertools import combinations

import numpy as np

from fpid.algorithms import ALGORITHMS, UNREACHED, Executor, finite_init
from fpid.estimate import (
    ALPHA_GRID,
    C_GRID,
    GAMMA_GRID,
    K_GRID,
    SELECTION_GRID,
    TAU_GRID,
    ProfileCache,
    beta_calibration,
    c_bucket,
    fit,
    grid_variance,
    profile_vector,
    profiles_of,
    standardise_scale,
    tau_detectable,
)
from fpid.family import FamilyParams, family_ctor, reference_params
from fpid.graphs import INF, Graph, high_diameter_graph, random_connected_graph
from fpid.response import response_profile, response_profile_masked
from run_e1_e2 import deep_probe_graphs

RULE = "=" * 74
REFERENCE = FamilyParams(alpha=1.0, tau=0.0, gamma=1.0, c=0.0, k=None, selection="sync")
AXES = ("alpha", "tau", "gamma", "c", "k", "selection")
GRIDS = {
    "alpha": ALPHA_GRID,
    "tau": TAU_GRID,
    "gamma": GAMMA_GRID,
    "c": C_GRID,
    "k": K_GRID,
    "selection": SELECTION_GRID,
}

# Frozen in PREREGISTRATION_FAMILY.md.
SNR_DISTINGUISHABLE = 3.0
SNR_ADEQUATE = 2.0
CHI2_ADEQUATE = 4.0
CHI2_OUT_OF_FAMILY = 25.0
VE_ADEQUATE = 0.90
HALT_K = 8  # matches run_e3.py's halt arm


# ------------------------------------------------------------------ out-of-family generators


class MeanAgg(Executor):
    """`d_v <- min(d_v, mean over neighbours u of (d_u + w_uv))`. The primary control.

    Out of family for a reason that cannot be absorbed by any parameter: the family's softmin
    is dominated by the smallest message at every finite tau, and as `tau -> inf` it diverges
    to `-inf` rather than approaching the mean. So mean-aggregation is not a limit of the
    family in any direction, while still propagating at the same one-hop-per-round rate as
    Bellman-Ford and putting weight `1/deg(v)` on every neighbour.

    Three earlier candidates were tried and are recorded because the reason they failed is
    itself a result. `max`-aggregation and second-best-aggregation **cannot propagate at all**
    from a single source under a sentinel initialisation: every node's non-minimal message is
    `UNREACHED + w`, which never beats `UNREACHED`, so nothing ever moves and the mask admits
    no row. A global `min(d) + w_bar` rule fails the opposite way -- it converges in one round,
    so no node is ever both reached and still moving. Taking the minimum is not a stylistic
    choice in this task; it is what makes propagation from a single source possible, and any
    negative control has to respect that to be a control rather than a stalled program.
    """

    name = "mean_agg"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source, UNREACHED)
        self.adj = np.isfinite(graph.W)
        self.deg = np.maximum(self.adj.sum(axis=0), 1)

    def _rule(self, d: np.ndarray) -> np.ndarray:
        msg = np.where(self.adj, d[:, None] + self.graph.W, 0.0)
        out = np.minimum(d, msg.sum(axis=0) / self.deg)
        out[self.source] = 0.0
        return out

    def _step(self) -> None:
        self.d = self._rule(self.d)

    def step_fn(self):
        return lambda d: self._rule(d)


class SecondBest(Executor):
    """`d_v <- min(d_v, second-best finite neighbour message)`. The honest hard case.

    Intended as the hard case: same alpha, same gamma, same support size and the same number of
    live rows as a family member, differing only in *which* parent responds.

    **It does not work, and the reason is the finding.** Under a sentinel initialisation the
    second-best message into any frontier node is `UNREACHED + w`, which never beats
    `UNREACHED`, so nothing propagates past the source's immediate neighbourhood and the mask
    admits no row. The degree-1 fallback to the best message does not rescue it. Kept and
    reported rather than deleted: it demonstrates that on this task the *minimum* is what makes
    single-source propagation possible at all, which is why the family is built around a
    softmin rather than a general aggregator.
    """

    name = "second_best"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source, UNREACHED)

    def _rule(self, d: np.ndarray) -> np.ndarray:
        msg = np.sort(d[:, None] + self.graph.W, axis=0)
        best = msg[0]
        second = np.where(np.isfinite(msg[1]), msg[1], best)
        out = np.minimum(d, second)
        out[self.source] = 0.0
        return out

    def _step(self) -> None:
        self.d = self._rule(self.d)

    def step_fn(self):
        return lambda d: self._rule(d)


class MaxAgg(Executor):
    """`d_v <- min(d_v, max_u(d_u + w))`. Relaxes from the worst parent: sanity anchor."""

    name = "max_agg"

    def __init__(self, graph: Graph, source: int):
        super().__init__(graph, source, UNREACHED)

    def _rule(self, d: np.ndarray) -> np.ndarray:
        msg = d[:, None] + self.graph.W
        finite = np.isfinite(msg)
        worst = np.where(finite.any(axis=0), np.where(finite, msg, -INF).max(axis=0), INF)
        out = np.minimum(d, worst)
        out[self.source] = 0.0
        return out

    def _step(self) -> None:
        self.d = self._rule(self.d)

    def step_fn(self):
        return lambda d: self._rule(d)


# `second_best` and `max_agg` are kept and still reported, because a control that provably
# cannot propagate is a finding about the task rather than dead code -- see MeanAgg's docstring.
OUT_OF_FAMILY = {"mean_agg": MeanAgg, "second_best": SecondBest, "max_agg": MaxAgg}
PRIMARY_CONTROL = "mean_agg"


# ------------------------------------------------------------------ design helpers


def lhs(rng: np.random.Generator, n: int, axes=AXES) -> list[FamilyParams]:
    """Latin-hypercube-style design over the parameter grids.

    Every axis is a discrete list of levels, so proper LHS reduces to: build a balanced
    multiset of levels per axis, then permute each axis independently. Marginals are balanced
    by construction and the pairings are random, which is the property the design is for --
    a full outer product is 17,640 points and answers a different question.
    """
    cols = {}
    for a in axes:
        levels = list(GRIDS[a])
        reps = [levels[i % len(levels)] for i in range(n)]
        cols[a] = [reps[i] for i in rng.permutation(n)]
    return [FamilyParams(**{a: cols[a][i] for a in axes}) for i in range(n)]


def within_noise(Z: np.ndarray, sd: np.ndarray, floor: float = 1e-9) -> float:
    """Measurement noise: median split-half profile distance over the graph set.

    Split-half rather than a bootstrap because it measures the thing the fit actually sees --
    the mean profile over a finite graph sample -- and it is deliberately optimistic, which is
    why the distinguishability threshold is 3 rather than 2.

    `floor` is not cosmetic. Some parameter settings are perfectly reproducible across graphs
    -- at `gamma = 0` the update is exactly diagonal, so every instance gives the same profile
    and the split-half distance is 0. Dividing by that reports an SNR of 1e8, which says
    nothing about identifiability and everything about the floor. Callers pass a fraction of
    the reference point's own noise, so every SNR stays bounded and comparable.
    """
    m = Z.shape[0]
    if m < 4:
        return max(float(np.linalg.norm(Z.std(axis=0) / sd)), floor)
    half = m // 2
    ds = []
    for idx in combinations(range(m), half):
        if idx[0] != 0:
            continue  # each complementary split counted once
        rest = [i for i in range(m) if i not in idx]
        a = Z[list(idx)].mean(axis=0)
        b = Z[rest].mean(axis=0)
        ds.append(float(np.linalg.norm((a - b) / sd)))
    return max(float(np.median(ds)) if ds else 0.0, floor)


def noise_floor(z_ref: np.ndarray, sd: np.ndarray, frac: float = 0.05) -> float:
    """A common floor for every SNR denominator, set from the reference point's noise."""
    return frac * within_noise(z_ref, sd)


def snr(za: np.ndarray, zb: np.ndarray, sa: float, sb: float, sd: np.ndarray) -> float:
    d = float(np.linalg.norm((za.mean(axis=0) - zb.mean(axis=0)) / sd))
    denom = float(np.sqrt(max(sa, 1e-12) * max(sb, 1e-12)))
    return d / denom


def make_graphs(n_graphs: int, n_nodes: int, seed: int, density: float, min_depth: int):
    rng = np.random.default_rng(seed)
    return [g for g, _ in deep_probe_graphs(n_graphs, n_nodes, rng, density, min_depth)]


# ------------------------------------------------------------------ gate 0a


def gate_0a(args) -> dict:
    print(RULE)
    print("GATE 0a -- embedding: does the family contain the five references exactly?")
    print(RULE)
    rng = np.random.default_rng(0)
    graphs = [random_connected_graph(args.nodes, rng, density=args.density) for _ in range(10)]
    graphs += [high_diameter_graph(args.nodes, rng) for _ in range(10)]
    budget = args.nodes

    rows, ok = {}, True
    for name in ALGORITHMS:
        ref_ctor = finite_init(ALGORITHMS[name], UNREACHED)
        worst = 0.0
        for g in graphs:
            p = reference_params(name, g.n)
            ref = ref_ctor(g.copy(), 0).run(budget)
            fam = family_ctor(p)(g.copy(), 0).run(budget)
            for a, b in zip(ref, fam):
                worst = max(worst, float(np.max(np.abs(a.d - b.d))))
        rows[name] = worst
        ok &= worst < 1e-9
        print(f"  {name:18s} max|d_family - d_ref| = {worst:.3e}   "
              f"{'exact' if worst < 1e-9 else 'MISMATCH'}")
    print(f"\n  {len(graphs)} graphs x {budget} rounds, both from the UNREACHED sentinel")
    print(f"  => {'PASS' if ok else 'FAIL -- the family does not contain the references'}")
    return {"per_reference_max_abs_error": rows, "pass": bool(ok)}


# ------------------------------------------------------------------ gate 0b


def _axis_points(axis: str) -> list:
    return [v for v in GRIDS[axis] if v != getattr(REFERENCE, axis)]


def gate_0b(args, cache: ProfileCache, sd: np.ndarray) -> dict:
    print("\n" + RULE)
    print("GATE 0b -- observable adequacy: does the profile move when each axis moves?")
    print(RULE)
    print("  The manipulation check, pointed inward. Without it, Gate 1 would report a")
    print("  confound wherever the truth is a dead observable -- E1's error.\n")

    z_ref = cache.get(REFERENCE)
    fl = noise_floor(z_ref, sd)
    s_ref = within_noise(z_ref, sd, fl)
    out, ok = {}, True
    for axis in AXES:
        best, best_at = 0.0, None
        for v in _axis_points(axis):
            p = replace(REFERENCE, **{axis: v})
            z = cache.get(p)
            s = snr(z_ref, z, s_ref, within_noise(z, sd, fl), sd)
            if s > best:
                best, best_at = s, v
        out[axis] = {"max_snr": best, "at": str(best_at)}
        good = best >= SNR_ADEQUATE
        ok &= good
        print(f"  {axis:10s} max marginal SNR {best:8.2f} at {axis}={best_at!s:<6} "
              f"{'ok' if good else 'DEAD'}")
    print(f"\n  => {'PASS' if ok else 'FAIL -- an axis is invisible to this observable'}")
    return {"per_axis": out, "pass": bool(ok)}


# ------------------------------------------------------------------ gate 1


def gate_1(args, cache: ProfileCache, sd: np.ndarray, graphs) -> dict:
    print("\n" + RULE)
    print("GATE 1 -- identifiability: are distinct parameters distinguishable above noise?")
    print(RULE)

    noise: dict[tuple, float] = {}
    fl = noise_floor(cache.get(REFERENCE), sd)

    def zs(p: FamilyParams):
        z = cache.get(p)
        k = p.key()
        if k not in noise:
            noise[k] = within_noise(z, sd, fl)
        return z, noise[k]

    # ---- per-axis level resolution, which the confound matrix cannot separate out
    #
    # The frozen confound metric is "min SNR over pairs >= 2 grid steps apart in the slice".
    # It conflates two different failures: axis i being mistakable for axis j, and axis j
    # simply having levels that are indistinguishable from *each other*. The second dominates
    # here -- with an 8-round profile, k = 8, 12, 24 and None are the same object, because the
    # halt never fires inside the horizon -- so every slice containing k reports 0.00 for a
    # reason that has nothing to do with the axis it is paired with. This pass measures level
    # resolution per axis first, so the confound matrix can be read against it.
    print("\n  Per-axis level resolution: which levels of each axis are distinguishable?")
    print(f"  Indistinguishable classes are levels no pair of which reaches SNR "
          f"{SNR_DISTINGUISHABLE:.0f}.\n")
    resolution: dict[str, dict] = {}
    resolvable: dict[str, list] = {}
    for axis in AXES:
        levels = list(GRIDS[axis])
        M = np.zeros((len(levels), len(levels)))
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                za, sa = zs(replace(REFERENCE, **{axis: levels[i]}))
                zb, sb = zs(replace(REFERENCE, **{axis: levels[j]}))
                M[i, j] = M[j, i] = snr(za, zb, sa, sb, sd)
        # Merge levels transitively while they stay below the threshold.
        classes: list[list] = []
        for i, lv in enumerate(levels):
            for cl in classes:
                if all(M[i, levels.index(x)] < SNR_DISTINGUISHABLE for x in cl):
                    cl.append(lv)
                    break
            else:
                classes.append([lv])
        resolution[axis] = {
            "levels": [str(x) for x in levels],
            "snr": M.tolist(),
            "classes": [[str(x) for x in cl] for cl in classes],
        }
        resolvable[axis] = [cl[0] for cl in classes]
        print(f"    {axis:10s} {len(classes)}/{len(levels)} distinguishable   "
              + "  ".join("{" + ",".join(str(x) for x in cl) + "}" for cl in classes))

    # ---- pairwise slices: the confound matrix
    print("\n  Confound matrix: min SNR over pairs >= 2 grid steps apart within each")
    print(f"  2-D slice through the reference point. Distinguishable at SNR >= "
          f"{SNR_DISTINGUISHABLE:.0f}.\n")
    conf: dict[str, float] = {}
    for a, b in combinations(AXES, 2):
        pts, idx = [], []
        for i, va in enumerate(GRIDS[a]):
            for j, vb in enumerate(GRIDS[b]):
                pts.append(replace(REFERENCE, **{a: va, b: vb}))
                idx.append((i, j))
        worst = float("inf")
        for x in range(len(pts)):
            for y in range(x + 1, len(pts)):
                if abs(idx[x][0] - idx[y][0]) + abs(idx[x][1] - idx[y][1]) < 2:
                    continue
                za, sa = zs(pts[x])
                zb, sb = zs(pts[y])
                worst = min(worst, snr(za, zb, sa, sb, sd))
        conf[f"{a}|{b}"] = worst

    hdr = "  " + " " * 11 + "".join(f"{x[:9]:>10}" for x in AXES)
    print(hdr)
    for a in AXES:
        cells = []
        for b in AXES:
            if a == b:
                cells.append(f"{'-':>10}")
            else:
                key = f"{a}|{b}" if f"{a}|{b}" in conf else f"{b}|{a}"
                cells.append(f"{conf[key]:10.2f}")
        print(f"  {a:>10} " + "".join(cells))
    bad_pairs = {k: v for k, v in conf.items() if v < SNR_DISTINGUISHABLE}
    print(f"\n  axis pairs below SNR {SNR_DISTINGUISHABLE:.0f}: "
          f"{len(bad_pairs)}/{len(conf)}   <- the frozen criterion")
    for k, v in sorted(bad_pairs.items(), key=lambda kv: kv[1]):
        print(f"    {k:24s} min SNR {v:.2f}")

    # Same metric, restricted to levels that are resolvable within their own axis. This is
    # the true confound question, and the gap between the two tables is how much of the
    # frozen criterion's failure is within-axis degeneracy rather than cross-axis confusion.
    print("\n  Restricted confound matrix: only levels resolvable within their own axis.")
    print("  Reported as a decomposition of the frozen criterion above, not a substitute.\n")
    conf_r: dict[str, float] = {}
    for a, b in combinations(AXES, 2):
        la, lb = resolvable[a], resolvable[b]
        pts = [replace(REFERENCE, **{a: va, b: vb}) for va in la for vb in lb]
        worst = float("inf")
        for x in range(len(pts)):
            for y in range(x + 1, len(pts)):
                za, sa = zs(pts[x])
                zb, sb = zs(pts[y])
                worst = min(worst, snr(za, zb, sa, sb, sd))
        conf_r[f"{a}|{b}"] = worst
    print(hdr := "  " + " " * 11 + "".join(f"{x[:9]:>10}" for x in AXES))
    for a in AXES:
        cells = []
        for b in AXES:
            if a == b:
                cells.append(f"{'-':>10}")
            else:
                key = f"{a}|{b}" if f"{a}|{b}" in conf_r else f"{b}|{a}"
                cells.append(f"{conf_r[key]:10.2f}")
        print(f"  {a:>10} " + "".join(cells))
    bad_r = {k: v for k, v in conf_r.items() if v < SNR_DISTINGUISHABLE}
    print(f"\n  restricted pairs below SNR {SNR_DISTINGUISHABLE:.0f}: "
          f"{len(bad_r)}/{len(conf_r)}")
    for k, v in sorted(bad_r.items(), key=lambda kv: kv[1]):
        print(f"    {k:24s} min SNR {v:.2f}")

    # ---- global check on a random design
    lhs_pts = lhs(np.random.default_rng(7), args.lhs)
    pairs, below = 0, []
    for x in range(len(lhs_pts)):
        for y in range(x + 1, len(lhs_pts)):
            if lhs_pts[x].key() == lhs_pts[y].key():
                continue
            za, sa = zs(lhs_pts[x])
            zb, sb = zs(lhs_pts[y])
            s = snr(za, zb, sa, sb, sd)
            pairs += 1
            if s < SNR_DISTINGUISHABLE:
                below.append((lhs_pts[x].key(), lhs_pts[y].key(), s))
    frac_ok = 1.0 - len(below) / max(1, pairs)
    print(f"\n  {args.lhs}-point random design: {pairs} distinct pairs, "
          f"{frac_ok*100:.1f}% at SNR >= {SNR_DISTINGUISHABLE:.0f}")
    for ka, kb, s in sorted(below, key=lambda t: t[2])[:6]:
        print(f"    SNR {s:5.2f}   {ka}  vs  {kb}")
    if len(below) > 6:
        print(f"    ... and {len(below)-6} more")

    # ---- argmin vs fifo, pre-committed prediction
    z_arg, s_arg = zs(replace(REFERENCE, selection="argmin"))
    fifo = replace(REFERENCE, selection="fifo")
    z_fifo = cache.get(fifo)
    s_fifo = within_noise(z_fifo, sd, fl)
    sel_snr = snr(z_arg, z_fifo, s_arg, s_fifo, sd)
    print(f"\n  argmin vs fifo SNR: {sel_snr:.2f}  (pre-committed prediction: < "
          f"{SNR_DISTINGUISHABLE:.0f}, i.e. SPFA and Dijkstra are not separable here)")
    print(f"    prediction {'HELD' if sel_snr < SNR_DISTINGUISHABLE else 'REFUTED (bonus)'}")

    # ---- sentinel invariance
    dev = 0.0
    for p in [REFERENCE, replace(REFERENCE, alpha=0.5), replace(REFERENCE, tau=2.0),
              replace(REFERENCE, gamma=0.5), replace(REFERENCE, selection="argmin")]:
        for U in (UNREACHED, 2 * UNREACHED):
            ctor = family_ctor(p, d_init=U, reached=0.9 * U)
            z = np.stack([
                profile_vector(
                    response_profile_masked(ctor, g, 0, args.rounds, 0.25, 0.05, args.kappa, U),
                    g.n,
                )
                for g in graphs[:3]
            ])
            if U == UNREACHED:
                base = z
            else:
                dev = max(dev, float(np.max(np.abs(z - base))))
    print(f"\n  sentinel invariance, UNREACHED 80 vs 160: max |dz| = {dev:.2e}")
    print("    softmin is translation-equivariant, so only message *gaps* matter")

    # ---- tau detectability
    cal = beta_calibration(
        graphs[:3], gamma_grid=(0.5, 1.0), cache_path=args.cal_path, kappa=args.kappa
    )
    det = tau_detectable(cal)
    print("\n  tau detectability -- smallest grid tau whose off_pr clears the hard-min value")
    print("  (inf = no tau on the grid is detectable at that operating point)\n")
    print("    " + " " * 8 + "".join(f"{f'tau@g={g}':>14}" for g in cal["gamma"]))
    for a in cal["alpha"]:
        cells = "".join(
            f"{det[f'alpha={a},gamma={g}']:>14.2f}" for g in cal["gamma"]
        )
        print(f"    a={a:<6}" + cells)

    passes = (
        len(bad_pairs) == 0
        and frac_ok >= 0.90
        and dev < 1e-6
    )
    print(f"\n  => {'PASS' if passes else 'PARTIAL -- see disqualified axes below'}")
    return {
        "level_resolution": resolution,
        "resolvable_levels": {k: [str(x) for x in v] for k, v in resolvable.items()},
        "confound_matrix": conf,
        "confound_matrix_restricted": conf_r,
        "restricted_pairs_below_threshold": bad_r,
        "axis_pairs_below_threshold": bad_pairs,
        "lhs_pairs": pairs,
        "lhs_frac_distinguishable": frac_ok,
        "lhs_worst": [[list(a), list(b), s] for a, b, s in sorted(below, key=lambda t: t[2])[:20]],
        "argmin_vs_fifo_snr": sel_snr,
        "sentinel_max_dz": dev,
        "tau_detectable": det,
        "pass": bool(passes),
    }


# ------------------------------------------------------------------ gate 2


def k_class(k: int | None, rounds: int) -> str:
    """`k` collapsed to what an `rounds`-round profile can actually see.

    A halt at or beyond the profile horizon never fires inside the window, so k = 8, 12, 24 and
    None give bit-identical profiles at rounds = 8. Freezing a k grid wider than the horizon was
    an error in the pre-registration; see GATE1_DISQUALIFICATIONS.md.
    """
    return str(k) if (k is not None and k < rounds) else "no_halt"


def c_class2(c: float) -> str:
    """The two commitment classes Gate 1 found resolvable, against the three registered."""
    return "zero" if c <= 0.1 else "positive"


def _recovery_row(true_p: FamilyParams, r, det: dict, rounds: int) -> dict:
    a_err = abs(r.theta.alpha - true_p.alpha)
    g_err = abs(r.theta.gamma - true_p.gamma)
    floor = det.get(
        f"alpha={min(ALPHA_GRID, key=lambda v: abs(v-true_p.alpha))},"
        f"gamma={min((0.5,1.0), key=lambda v: abs(v-true_p.gamma))}",
        float("inf"),
    )
    tau_in_range = true_p.tau >= floor and true_p.tau > 0
    return {
        "true": list(true_p.key()),
        "fit": list(r.theta.key()),
        "alpha_err": a_err,
        "gamma_err": g_err,
        "k_exact": r.theta.k == true_p.k,
        "k_class_exact": k_class(r.theta.k, rounds) == k_class(true_p.k, rounds),
        "sel_exact": r.theta.selection == true_p.selection,
        "c_bucket_exact": c_bucket(r.theta.c) == c_bucket(true_p.c),
        "c_class2_exact": c_class2(r.theta.c) == c_class2(true_p.c),
        "tau_floor": floor,
        "tau_in_range": bool(tau_in_range),
        "tau_rel_err": (abs(r.theta.tau - true_p.tau) / true_p.tau) if true_p.tau > 0 else None,
        "tau_refused": "tau_unidentified" in r.flags,
        "gamma_degenerate": "gamma_degenerate" in r.flags,
        "withheld": sorted(r.withheld),
        "alpha_answered": "alpha" not in r.withheld,
        "gamma_answered": "gamma" not in r.withheld,
        "chi2": r.chi2,
        "ve": r.ve,
        "row_sum_deviation": r.row_sum_deviation,
        "flags": r.flags,
    }


def _report_recovery(label: str, rows: list[dict]) -> dict:
    # Frozen scoring: every target counted, including the ones the estimator declined to
    # answer. Reported first, and it is the criterion of record.
    a = np.array([r["alpha_err"] for r in rows])
    g = np.array([r["gamma_err"] for r in rows])
    ok_a = float(np.median(a)) <= 0.05 and float(np.percentile(a, 90)) <= 0.15
    ok_g = float(np.median(g)) <= 0.05 and float(np.percentile(g, 90)) <= 0.15

    # Answered-only scoring, plus coverage. The pre-registration built a withholding mechanism
    # for `gamma_degenerate` and `tau_unidentified` and scored those as refusal rates, but it
    # did not anticipate `no_fittable_rows` -- the case where the mask admits no row at all and
    # the direct estimators have no data. Those rows are refusals too, so they are reported
    # both ways: accuracy on what the method answers, and coverage as the price.
    ans_a = np.array([r["alpha_err"] for r in rows if r["alpha_answered"]])
    ans_g = np.array([r["gamma_err"] for r in rows if r["gamma_answered"]])
    cov_a = len(ans_a) / max(1, len(rows))
    cov_g = len(ans_g) / max(1, len(rows))
    ok_a_ans = ans_a.size > 0 and float(np.median(ans_a)) <= 0.05 and float(
        np.percentile(ans_a, 90)
    ) <= 0.15
    ok_g_ans = ans_g.size > 0 and float(np.median(ans_g)) <= 0.05 and float(
        np.percentile(ans_g, 90)
    ) <= 0.15
    k_rate = float(np.mean([r["k_exact"] for r in rows]))
    kc_rate = float(np.mean([r["k_class_exact"] for r in rows]))
    s_rate = float(np.mean([r["sel_exact"] for r in rows]))
    c_rate = float(np.mean([r["c_bucket_exact"] for r in rows]))
    c2_rate = float(np.mean([r["c_class2_exact"] for r in rows]))

    inr = [r for r in rows if r["tau_in_range"]]
    outr = [r for r in rows if not r["tau_in_range"] and r["true"][1] >= 0]
    tau_err = float(np.median([r["tau_rel_err"] for r in inr])) if inr else float("nan")
    refuse = float(np.mean([r["tau_refused"] for r in outr])) if outr else float("nan")
    deg = [r for r in rows if r["true"][2] <= 0.0]
    deg_rate = float(np.mean([r["gamma_degenerate"] for r in deg])) if deg else float("nan")

    print(f"\n  {label}: {len(rows)} targets")
    print("    -- frozen scoring: all targets, refusals counted as answers (criterion of record)")
    print(f"    alpha       median err {np.median(a):.4f}  p90 {np.percentile(a,90):.4f}"
          f"   {'PASS' if ok_a else 'FAIL'}   (<= 0.05 / 0.15)")
    print(f"    gamma       median err {np.median(g):.4f}  p90 {np.percentile(g,90):.4f}"
          f"   {'PASS' if ok_g else 'FAIL'}   (<= 0.05 / 0.15)")
    print("    -- answered-only, with coverage as the price")
    print(f"    alpha       median err {np.median(ans_a):.4f}  p90 "
          f"{np.percentile(ans_a,90):.4f}   {'PASS' if ok_a_ans else 'FAIL'}"
          f"   coverage {cov_a*100:5.1f}%")
    print(f"    gamma       median err {np.median(ans_g):.4f}  p90 "
          f"{np.percentile(ans_g,90):.4f}   {'PASS' if ok_g_ans else 'FAIL'}"
          f"   coverage {cov_g*100:5.1f}%")
    print(f"    k  frozen   exact {k_rate*100:5.1f}%"
          f"                        {'PASS' if k_rate >= 0.90 else 'FAIL'}   (>= 90%)")
    print(f"    k  amended  class {kc_rate*100:5.1f}%"
          f"                        {'PASS' if kc_rate >= 0.90 else 'FAIL'}   (>= 90%)"
          f"  <- Gate 1 horizon")
    print(f"    selection   exact {s_rate*100:5.1f}%"
          f"                        {'PASS' if s_rate >= 0.95 else 'FAIL'}   (>= 95%)")
    print(f"    c  frozen   3-bucket {c_rate*100:5.1f}%"
          f"                     {'PASS' if c_rate >= 0.80 else 'FAIL'}   (>= 80%)  reported")
    print(f"    c  amended  2-class  {c2_rate*100:5.1f}%"
          f"                     {'PASS' if c2_rate >= 0.80 else 'FAIL'}   (>= 80%)  reported")
    if inr:
        print(f"    tau (n={len(inr)} in detectable range)  median rel err {tau_err:.3f}"
              f"   {'PASS' if tau_err <= 0.40 else 'FAIL'}   (<= 0.40)  reported")
    else:
        print("    tau         no target fell in a detectable range -- reported, not gated")
    if outr:
        print(f"    tau refusal (n={len(outr)} below floor)  flagged {refuse*100:5.1f}%"
              f"       {'PASS' if refuse >= 0.90 else 'FAIL'}   (>= 90%)  reported")
    if deg:
        print(f"    gamma~0 flagged  {deg_rate*100:5.1f}% of {len(deg)}"
              f"              {'PASS' if deg_rate >= 0.90 else 'FAIL'}   (>= 90%)")

    # Can the residual find the failures without being told the answer? If a large chi2 marks
    # the rows the estimator got wrong, the method self-diagnoses -- which is the property the
    # old nearest-reference distance conspicuously lacked.
    chi2 = np.array([r["chi2"] for r in rows])
    print(f"    chi2        median {np.median(chi2):.4f}  p90 {np.percentile(chi2,90):.4f}")
    bad = np.array([r["alpha_err"] > 0.15 or r["gamma_err"] > 0.15 for r in rows])
    caught = float(np.mean(chi2[bad] > CHI2_ADEQUATE)) if bad.any() else float("nan")
    clean = float(np.mean(chi2[~bad] <= CHI2_ADEQUATE)) if (~bad).any() else float("nan")
    print(f"    self-diagnosis: of {int(bad.sum())} targets recovered badly, "
          f"{caught*100:.0f}% have chi2 > {CHI2_ADEQUATE:.0f};")
    print(f"                    of {int((~bad).sum())} recovered well, "
          f"{clean*100:.0f}% have chi2 <= {CHI2_ADEQUATE:.0f}")

    core = ok_a and ok_g and kc_rate >= 0.90 and s_rate >= 0.95
    core_frozen = ok_a and ok_g and k_rate >= 0.90 and s_rate >= 0.95
    print(f"\n    => core, amended k (alpha, gamma, k-class, selection): "
          f"{'PASS' if core else 'FAIL'}")
    print(f"       core, frozen  k                                  : "
          f"{'PASS' if core_frozen else 'FAIL'}")
    if not (ok_a and ok_g):
        print("       alpha or gamma failed: the reframe is dead, per the pre-registration.")
    return {
        "n": len(rows),
        "alpha_median": float(np.median(a)), "alpha_p90": float(np.percentile(a, 90)),
        "gamma_median": float(np.median(g)), "gamma_p90": float(np.percentile(g, 90)),
        "alpha_median_answered": float(np.median(ans_a)) if ans_a.size else float("nan"),
        "alpha_p90_answered": float(np.percentile(ans_a, 90)) if ans_a.size else float("nan"),
        "gamma_median_answered": float(np.median(ans_g)) if ans_g.size else float("nan"),
        "gamma_p90_answered": float(np.percentile(ans_g, 90)) if ans_g.size else float("nan"),
        "alpha_coverage": cov_a, "gamma_coverage": cov_g,
        "alpha_pass_answered": bool(ok_a_ans), "gamma_pass_answered": bool(ok_g_ans),
        "chi2_catches_bad_fits": caught, "chi2_clears_good_fits": clean,
        "k_exact_rate": k_rate, "k_class_rate": kc_rate,
        "selection_exact_rate": s_rate, "c_bucket_rate": c_rate, "c_class2_rate": c2_rate,
        "core_pass_frozen_k": bool(core_frozen),
        "tau_median_rel_err": tau_err, "tau_refusal_rate": refuse,
        "gamma_degenerate_rate": deg_rate,
        "chi2_median": float(np.median(chi2)),
        "core_pass": bool(core),
        "alpha_pass": bool(ok_a), "gamma_pass": bool(ok_g),
        "rows": rows,
    }


def run_recovery(args, graphs, cal, det, grid_var, design_seed: int, label: str) -> dict:
    cache = ProfileCache(graphs, rounds=args.rounds, kappa=args.kappa)
    targets = lhs(np.random.default_rng(design_seed), args.targets)
    rows = []
    for i, p in enumerate(targets):
        profs = profiles_of(family_ctor(p), graphs, rounds=args.rounds, kappa=args.kappa)
        r = fit(profs, graphs, cal=cal, cache=cache, grid_var=grid_var, kappa=args.kappa)
        rows.append(_recovery_row(p, r, det, args.rounds))
        if args.verbose:
            print(f"    [{i+1:>3}/{len(targets)}] true {p.key()}")
            print(f"           fit  {r.theta.key()}  chi2={r.chi2:.4f} {r.flags}")
    return _report_recovery(label, rows)


# ------------------------------------------------------------------ gate 3


def load_checkpoints(save_dir: str):
    """The nine E3 checkpoints. Real out-of-family objects, free -- reported, not gated."""
    import torch

    from fpid.neural import MPNN, GatedMPNN, neural_ctor

    out = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*.pt"))):
        base = os.path.basename(path)[:-3]
        arch = base.split("_")[0]
        model = GatedMPNN() if arch == "gated" else MPNN()
        try:
            model.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad. These fits are descriptive and ungated, so a checkpoint that
            # no longer matches its architecture must not take down the gate ladder that ran
            # before it. Whatever went wrong is printed rather than swallowed.
            print(f"  (skipping {base}: {exc})")
            continue
        model.eval()
        out.append((base, neural_ctor(model, "cpu", HALT_K if arch == "halt" else None)))
    return out


def gate_3(args, graphs, cal, grid_var) -> dict:
    print("\n" + RULE)
    print("GATE 3 -- losability: does an out-of-family generator fit badly?")
    print(RULE)
    print("  Four free parameters can fit anything. If everything fits, this is a tautology.\n")
    cache = ProfileCache(graphs, rounds=args.rounds, kappa=args.kappa)

    res = {}
    holdout = FamilyParams(alpha=0.7, tau=0.0, gamma=0.9, c=0.0, k=None, selection="sync")
    profs = profiles_of(family_ctor(holdout), graphs, rounds=args.rounds, kappa=args.kappa)
    r = fit(profs, graphs, cal=cal, cache=cache, grid_var=grid_var, kappa=args.kappa)
    res["in_family_holdout"] = r.to_dict()
    print(f"  in-family holdout {holdout.key()}")
    print(f"    chi2 {r.chi2:9.4f}   ve {r.ve:7.3f}   rowdev {r.row_sum_deviation:.4f}")

    for name, cls in OUT_OF_FAMILY.items():
        profs = profiles_of(lambda g, s, c=cls: c(g, s), graphs,
                            rounds=args.rounds, kappa=args.kappa)
        r = fit(profs, graphs, cal=cal, cache=cache, grid_var=grid_var, kappa=args.kappa)
        res[name] = r.to_dict()
        gated = " (PRIMARY, must fail)" if name == PRIMARY_CONTROL else " (reported)"
        vac = "no_fittable_rows" in r.flags
        print(f"  {name}{gated}")
        print(f"    chi2 {r.chi2:9.4f}   ve {r.ve:7.3f}   rowdev {r.row_sum_deviation:.4f}"
              f"   -> {'VACUOUS (no fittable rows)' if vac else r.verdict()}")
        if vac:
            print("    A low chi2 here means the generator never moved the state, so an empty")
            print("    profile was matched against an empty profile. Not evidence of fit.")
        worst = sorted(r.per_stat_residual.items(), key=lambda kv: -kv[1])[:3]
        print("    worst statistics: " + ", ".join(f"{k}={v:.1f}" for k, v in worst))

    gm = res[PRIMARY_CONTROL]["chi2"]
    hold = res["in_family_holdout"]["chi2"]
    gm_vacuous = "no_fittable_rows" in res[PRIMARY_CONTROL]["flags"]
    ok = hold <= CHI2_ADEQUATE and gm >= CHI2_OUT_OF_FAMILY and not gm_vacuous
    print(f"\n  => {'PASS' if ok else 'FAIL'}  (holdout chi2 {hold:.2f} <= "
          f"{CHI2_ADEQUATE:.0f} and {PRIMARY_CONTROL} chi2 {gm:.2f} >= "
          f"{CHI2_OUT_OF_FAMILY:.0f}, non-vacuous)")
    if not ok and CHI2_ADEQUATE < gm < CHI2_OUT_OF_FAMILY:
        print(f"     Direction is right -- {PRIMARY_CONTROL} is rejected as family_wrong and")
        print(f"     sits {gm/max(hold,1e-9):.0e}x the holdout residual -- but the frozen")
        print("     5-sigma separation is not met. Resolution, not sign.")

    ckpts = load_checkpoints(args.save_dir)
    if ckpts:
        print(f"\n  Descriptive fits of {len(ckpts)} trained checkpoints -- NOT GATED.")
        print("  Three architectures x three seeds is not a sample, and any lead here needs")
        print("  its own pre-registration and a held-out seed split before it is a result.\n")
        print(f"    {'model':<14}{'alpha':>8}{'gamma':>8}{'tau':>7}{'c':>6}{'k':>6}"
              f"{'sel':>8}{'chi2':>10}{'rowdev':>9}")
        res["checkpoints"] = {}
        for base, ctor in ckpts:
            profs = profiles_of(ctor, graphs, rounds=args.rounds, kappa=args.kappa)
            r = fit(profs, graphs, cal=cal, cache=cache, grid_var=grid_var, kappa=args.kappa)
            res["checkpoints"][base] = r.to_dict()
            t = r.theta
            print(f"    {base:<14}{t.alpha:8.3f}{t.gamma:8.3f}{t.tau:7.2f}{t.c:6.2f}"
                  f"{t.k!s:>6}{t.selection:>8}{r.chi2:10.3f}{r.row_sum_deviation:9.4f}")
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------ response table


def reproduce_response_table(args) -> dict:
    """The numbers quoted in `fpid/response.py`, from a checked-in script for the first time.

    The docstring's claims about `width` and `row_sum` were produced ad hoc and were wrong.
    This is what closes that hole: the table is regenerated rather than asserted.
    """
    print(RULE)
    print("Reference response table (reproduces fpid/response.py's docstring)")
    print(RULE)
    graphs = make_graphs(1, args.nodes, 0, args.density, args.min_depth)
    out = {}
    for name in ALGORITHMS:
        prof = response_profile(
            finite_init(ALGORITHMS[name], UNREACHED), graphs[0], 0, args.rounds
        )
        out[name] = {k: [float(x) for x in v] for k, v in prof.items()}
        print(f"\n  {name}")
        for stat in ("width", "row_sum", "diag_frac"):
            vals = " ".join(f"{v:6.3f}" for v in prof[stat])
            print(f"    {stat:10s} {vals}")
    print("\n  Bellman-Ford reaches width 2 and row_sum 2 from round 3: at a converged node")
    print("  d_v == d_u + w, so a one-sided difference on either argument moves the hard min")
    print("  by the full eps. The 'references are all exactly 1' claim was an artefact.")
    return out


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="all", choices=["0", "1", "2", "3", "all"])
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--lhs", type=int, default=64)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--cal-path", default="artifacts/beta_calibration.json")
    ap.add_argument("--out", default="family_spike_results.json")
    ap.add_argument("--kappa-sweep", action="store_true", help="report kappa sensitivity")
    ap.add_argument("--reproduce-response-table", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.graphs, args.targets, args.lhs, args.rounds = 4, 6, 12, 6

    results: dict = {"args": vars(args)}

    if args.reproduce_response_table:
        results["response_table"] = reproduce_response_table(args)
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "response_reference_table.json"), "w") as f:
            json.dump(results["response_table"], f, indent=1)
        return

    print(f"n={args.nodes}  graphs={args.graphs}  rounds={args.rounds}  "
          f"kappa={args.kappa}  targets={args.targets}")
    print("pre-registered decision rule: PREREGISTRATION_FAMILY.md\n")

    if args.gate in ("0", "all"):
        results["gate_0a"] = gate_0a(args)
        if not results["gate_0a"]["pass"]:
            print("\n  Gate 0a failed. Everything downstream is meaningless; stopping.")
            _save(args, results)
            return

    graphs = make_graphs(args.graphs, args.nodes, 3, args.density, args.min_depth)
    depths = [int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max()) for g in graphs]
    print(f"\n  probe-graph depths: {depths}  (median {np.median(depths):.1f})")

    cache = ProfileCache(graphs, rounds=args.rounds, kappa=args.kappa)
    sd = standardise_scale(cache.get(REFERENCE))

    if args.gate in ("0", "all"):
        results["gate_0b"] = gate_0b(args, cache, sd)

    if args.gate in ("1", "all"):
        results["gate_1"] = gate_1(args, cache, sd, graphs)

    if args.gate in ("2", "3", "all"):
        cal = beta_calibration(
            graphs[:3], gamma_grid=(0.5, 1.0), cache_path=args.cal_path, kappa=args.kappa
        )
        det = tau_detectable(cal)
        gv = grid_variance([cache.get(p) for p in lhs(np.random.default_rng(7), args.lhs)], sd)
        results["grid_variance"] = gv

    if args.gate in ("2", "all"):
        print("\n" + RULE)
        print("GATE 2 -- recovery: are hand-set parameters read back out?")
        print(RULE)
        results["gate_2"] = run_recovery(args, graphs, cal, det, gv, 5, "Gate 2")

        print("\n" + RULE)
        print("GATE 2b -- held-out replication: fresh graphs, fresh draw, nothing refitted")
        print(RULE)
        held = make_graphs(args.graphs, args.nodes, 11, args.density, args.min_depth)
        results["gate_2b"] = run_recovery(args, held, cal, det, gv, 13, "Gate 2b")
        rep = results["gate_2b"]["core_pass"] and results["gate_2"]["core_pass"]
        print(f"\n  => {'REPLICATED' if rep else 'FAILED'}")
        results["gate_2b"]["replicated"] = bool(rep)

        if args.kappa_sweep:
            print("\n  kappa sensitivity (the one hand-picked tolerance)")
            results["kappa_sweep"] = {}
            for kap in (0.5, 1.0, 2.0):
                sub = replace_args(args, kappa=kap)
                r = run_recovery(sub, graphs, cal, det, gv, 5, f"kappa={kap}")
                results["kappa_sweep"][str(kap)] = {
                    k: v for k, v in r.items() if k != "rows"
                }

    if args.gate in ("3", "all"):
        results["gate_3"] = gate_3(args, graphs, cal, gv)

    _verdict(results)
    _save(args, results)


def replace_args(args, **kw):
    import copy

    a = copy.copy(args)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _verdict(results: dict) -> None:
    print("\n" + RULE)
    print("VERDICT (pre-registered ladder, PREREGISTRATION_FAMILY.md)")
    print(RULE)
    g2 = results.get("gate_2")
    if g2 is None:
        print("  Recovery gate not run; no verdict.")
        return
    if not (g2["alpha_pass"] and g2["gamma_pass"]):
        print("  DEAD on the frozen criterion. alpha or gamma failed with every target")
        print("  scored, including the ones the estimator declined to answer.")
        if g2["alpha_pass_answered"] and g2["gamma_pass_answered"]:
            print()
            print("  Scored on answered targets only, both pass:")
            print(f"    alpha p90 {g2['alpha_p90_answered']:.4f} at "
                  f"{g2['alpha_coverage']*100:.0f}% coverage")
            print(f"    gamma p90 {g2['gamma_p90_answered']:.4f} at "
                  f"{g2['gamma_coverage']*100:.0f}% coverage")
            print("  So the failure is coverage, not accuracy: the estimator is exact where")
            print("  the mask admits rows and blind where it does not. That is a real limit")
            print("  and it is not the limit that was registered, so the frozen verdict")
            print("  stands. See FINDINGS_FAMILY.md for what a redesign would have to fix.")
        return
    core = g2["core_pass"] and results.get("gate_2b", {}).get("core_pass", False)
    tau_ok = g2["tau_median_rel_err"] == g2["tau_median_rel_err"] and g2["tau_median_rel_err"] <= 0.40
    c_ok = g2["c_bucket_rate"] >= 0.80
    if core and tau_ok and c_ok:
        print("  FULL. All six axes estimable and replicated. Phase F proceeds as designed.")
    elif core:
        print("  SCOPED. The family is estimable in (alpha, gamma, k, selection) and")
        print("  replicates there. Aggregation temperature and/or standstill commitment are")
        print(f"  not (tau median rel err {g2['tau_median_rel_err']:.3f}, "
              f"c bucket {g2['c_bucket_rate']*100:.0f}%).")
        print("  This is a live pivot, not a death: the estimable axes are the ones that")
        print("  separate a damped discounted relaxation from a hard one.")
    else:
        print("  FAILED TO REPLICATE. Direction may survive; resolution does not. Same")
        print("  failure mode as E3b -- the fifth result in this project to die this way.")


def _save(args, results: dict) -> None:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
