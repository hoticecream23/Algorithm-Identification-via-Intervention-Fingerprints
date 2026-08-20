"""EXPLORATORY: what do the trained networks do that the family cannot express?

Phase F established that the family contains all five reference algorithms exactly and the
trained networks not at all -- six of nine checkpoints at chi2 ~10^4 (`FINDINGS_FAMILY.md`).
That is a clean negative but it does not say *what* is missing. This runner asks that, and it
is the one place in the project where the family's residual is used as a pointer rather than
as a verdict.

**This is exploratory and nothing here is a result.** Its output is a set of candidate
structural violations plus the pre-registration they would need. The project has lost five
findings to best-of-N selection, so every cell of every table is printed and none is chosen;
sign agreement across models is reported instead of a p-value, and nine checkpoints are three
architectures x three seeds, i.e. three independent design points, not nine.

Two stages.

**Stage 1, structural audit.** Four properties the family provably cannot have, tested against
family members as the null:

    H1  |J[v,v]| > 1        self-amplification; family diagonal is in [0,1] always
    H2  J entries < 0       anti-relaxation; lowering a neighbour can never raise you
    H3  off_width > deg(v)  non-local dependence; would mean the observable is wrong
    H4  (alpha_v, gamma_v) heterogeneous across nodes; the family has one scalar pair

**Stage 2, is the heterogeneity a function of anything?** If the per-node discount is
predictable from an observable node property, the missing ingredient has a functional form a
redesigned family could carry. If not, the networks have no compact description in these
coordinates and Phase F is finished for good. `gamma_v = off_row_v / alpha_v` is a ratio, so
the round trend is also checked on the undivided numerator, which is the decisive version.

    python run_residual_audit.py
    python run_residual_audit.py --smoke
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from fpid.algorithms import ALGORITHMS, UNREACHED, finite_init
from fpid.family import FamilyParams, family_ctor
from fpid.graphs import true_sssp
from fpid.neural import MPNN, GatedMPNN, neural_ctor
from fpid.response import jacobian_at, jacobian_pure
from run_e1_e2 import deep_probe_graphs

RULE = "=" * 104
EPS, THETA, KAPPA = 0.25, 0.05, 1.0
REACH = 0.9 * UNREACHED
HALT_K = 8
PREDICTORS = ("deg", "hop", "d_now", "gap", "round")
CHECKPOINTS = (
    "plain_700", "plain_701", "plain_702",
    "gated_800", "gated_801", "gated_802",
    "halt_900", "halt_901", "halt_902",
)


def spearman(x, y) -> float:
    """Hand-rolled rank correlation, matching `run_ood_prediction.py`. The repo avoids scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def sign_agreement(col: np.ndarray) -> int:
    """How many of the models agree on the sign. Reported instead of a p-value.

    A correlation that flips sign between seeds of the same architecture is noise however
    large it is, and this project's history is that effect sizes survive replication while
    directions sometimes do not. Sign agreement makes that visible without pretending the
    nine checkpoints are nine independent draws.
    """
    c = col[np.isfinite(col)]
    return int(max((c > 0).sum(), (c < 0).sum()))


def load_checkpoint(base: str, save_dir: str):
    arch = base.split("_")[0]
    model = GatedMPNN() if arch == "gated" else MPNN()
    model.load_state_dict(torch.load(f"{save_dir}/{base}.pt", map_location="cpu"))
    model.eval()
    return neural_ctor(model, "cpu", HALT_K if arch == "halt" else None)


def rows_of(ctor, graphs, rounds: int) -> dict[str, np.ndarray]:
    """Per-live-row measurements: the unit of analysis for both stages.

    A "live row" is a node that is reached and still moving by more than the probe amplitude,
    the same mask `summarise_masked` uses -- below it the one-sided difference is a secant.
    Everything is kept per row rather than aggregated, because the whole question here is
    whether the rows agree with each other.
    """
    out: dict[str, list[float]] = {k: [] for k in PREDICTORS}
    out.update({k: [] for k in ("alpha_v", "gamma_v", "off_row", "diag", "n_neg", "n_off",
                                "width", "deg_v")})
    for g in graphs:
        deg = np.isfinite(g.W).sum(axis=0)
        hop = g.hop_distances(0)
        dtrue = true_sssp(g, 0)
        for t in range(rounds):
            ex = ctor(g.copy(), 0)
            if t > 0:
                ex.run(t)
            d_now = ex.d.copy()
            if hasattr(ex, "step_fn"):
                step = ex.step_fn()
                d_next = step(d_now)
                J = jacobian_pure(step, d_now, EPS)
            else:
                probe = ctor(g.copy(), 0)
                if t > 0:
                    probe.run(t)
                probe.step()
                d_next = probe.d.copy()
                J = jacobian_at(ctor, g, 0, t, EPS)

            mask = (d_now < REACH) & (np.abs(d_next - d_now) > KAPPA * EPS)
            for v in np.nonzero(mask)[0]:
                dg = float(J[v, v])
                off = J[v].copy()
                off[v] = 0.0
                live = np.abs(off) > THETA
                a = 1.0 - abs(dg)

                out["diag"].append(abs(dg))
                out["off_row"].append(float(np.abs(off).sum()))
                out["width"].append(float(live.sum()))
                out["deg_v"].append(float(deg[v]))
                out["n_neg"].append(float((off[live] < -THETA).sum()))
                out["n_off"].append(float(live.sum()))
                out["alpha_v"].append(a)
                out["gamma_v"].append(float(np.abs(off).sum() / a) if a > 1e-6 else np.nan)
                out["deg"].append(float(deg[v]))
                out["hop"].append(float(hop[v]) if np.isfinite(hop[v]) else np.nan)
                out["d_now"].append(float(d_now[v]))
                out["gap"].append(float(d_now[v] - dtrue[v]))
                out["round"].append(float(t))
    return {k: np.asarray(v, float) for k, v in out.items()}


def stage1(objects, graphs, rounds) -> dict:
    print(RULE)
    print("STAGE 1 -- structural audit. Family members are the null: all four must be absent.")
    print(RULE)
    print(f"  {'object':<22}{'rows':>6}{'|Jvv|>1':>9}{'neg ent':>9}{'neg rows':>10}"
          f"{'w>deg':>7}{'gam>1':>8}{'a_v IQR':>9}{'g_v IQR':>9}")
    res = {}
    for label, ctor in objects:
        r = rows_of(ctor, graphs, rounds)
        n = r["diag"].size
        if n == 0:
            print(f"  {label:<22}{'no live rows':>48}")
            continue
        g_ok = np.isfinite(r["gamma_v"])
        gv = r["gamma_v"][g_ok]
        row = {
            "rows": int(n),
            "diag_gt1_frac": float((r["diag"] > 1.0 + 1e-9).mean()),
            "neg_entry_frac": float(r["n_neg"].sum() / max(1.0, r["n_off"].sum())),
            "neg_row_frac": float((r["n_neg"] > 0).mean()),
            "width_gt_deg_frac": float((r["width"] > r["deg_v"]).mean()),
            "gamma_gt1_frac": float((gv > 1.0).mean()) if gv.size else float("nan"),
            "alpha_iqr": float(np.subtract(*np.percentile(r["alpha_v"], [75, 25]))),
            "gamma_iqr": float(np.subtract(*np.percentile(gv, [75, 25]))) if gv.size else np.nan,
            "diag_max": float(r["diag"].max()),
        }
        res[label] = row
        print(f"  {label:<22}{row['rows']:>6}{row['diag_gt1_frac']*100:8.1f}%"
              f"{row['neg_entry_frac']*100:8.1f}%{row['neg_row_frac']*100:9.1f}%"
              f"{row['width_gt_deg_frac']*100:6.1f}%{row['gamma_gt1_frac']*100:7.1f}%"
              f"{row['alpha_iqr']:9.3f}{row['gamma_iqr']:9.3f}")
    return res


def stage2(graphs, rounds, save_dir) -> dict:
    print()
    print(RULE)
    print("STAGE 2 -- is the heterogeneity a function of anything? All cells shown, none chosen.")
    print(RULE)
    res: dict = {}
    cache = {b: rows_of(load_checkpoint(b, save_dir), graphs, rounds) for b in CHECKPOINTS}

    for target in ("gamma_v", "alpha_v"):
        print(f"\n  Spearman( {target}, predictor )")
        print(f"  {'model':<14}{'n':>6}" + "".join(f"{p:>11}" for p in PREDICTORS))
        tab = []
        for b in CHECKPOINTS:
            r = cache[b]
            ok = np.isfinite(r[target])
            cells = [spearman(r[p][ok & np.isfinite(r[p])], r[target][ok & np.isfinite(r[p])])
                     for p in PREDICTORS]
            tab.append(cells)
            print(f"  {b:<14}{int(ok.sum()):>6}" + "".join(f"{c:>11.3f}" for c in cells))
        A = np.array(tab)
        res[target] = {"models": list(CHECKPOINTS), "predictors": list(PREDICTORS),
                       "spearman": A.tolist()}
        print(f"  {'MEDIAN':<14}{'':>6}" + "".join(f"{v:>11.3f}" for v in np.median(A, axis=0)))
        print(f"  {'SIGN AGREE':<14}{'':>6}"
              + "".join(f"{sign_agreement(c):>9d}/9" for c in A.T))

    # The division check. gamma_v is a ratio, so a trend in it can come from the denominator.
    # off_row involves no division: if it trends with round while alpha_v does not, the trend
    # is in gamma and is real.
    print("\n  Division check -- gamma_v is a ratio, off_row is not")
    print(f"  {'model':<14}{'r(rnd,gamma)':>14}{'r|gamma<=1':>12}{'r(rnd,off_row)':>16}"
          f"{'r(rnd,alpha)':>14}")
    tab = []
    for b in CHECKPOINTS:
        r = cache[b]
        ok = np.isfinite(r["gamma_v"])
        rnd, gv = r["round"][ok], r["gamma_v"][ok]
        keep = gv <= 1.0
        cells = [
            spearman(rnd, gv),
            spearman(rnd[keep], gv[keep]) if keep.sum() > 3 else float("nan"),
            spearman(r["round"], r["off_row"]),
            spearman(r["round"], r["alpha_v"]),
        ]
        tab.append(cells)
        print(f"  {b:<14}{cells[0]:>14.3f}{cells[1]:>12.3f}{cells[2]:>16.3f}{cells[3]:>14.3f}")
    A = np.array(tab)
    res["division_check"] = {
        "columns": ["r_round_gamma", "r_round_gamma_le1", "r_round_off_row", "r_round_alpha"],
        "models": list(CHECKPOINTS), "values": A.tolist(),
        "median": np.median(A, axis=0).tolist(),
        "sign_agreement": [sign_agreement(c) for c in A.T],
    }
    print(f"  {'MEDIAN':<14}{np.median(A[:,0]):>14.3f}{np.median(A[:,1]):>12.3f}"
          f"{np.median(A[:,2]):>16.3f}{np.median(A[:,3]):>14.3f}")
    print(f"  {'SIGN AGREE':<14}{sign_agreement(A[:,0]):>12d}/9{sign_agreement(A[:,1]):>10d}/9"
          f"{sign_agreement(A[:,2]):>14d}/9{sign_agreement(A[:,3]):>12d}/9")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--save-dir", default="artifacts")
    ap.add_argument("--out", default="residual_audit_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.graphs, args.rounds = 3, 5

    rng = np.random.default_rng(3)
    graphs = [g for g, _ in deep_probe_graphs(args.graphs, args.nodes, rng,
                                              args.density, args.min_depth)]
    print(f"n={args.nodes}  graphs={args.graphs}  rounds={args.rounds}")
    print("EXPLORATORY. Nothing here is a result; see FINDINGS_RESIDUAL.md for the")
    print("pre-registration any of it would need.\n")

    objects = [(name, finite_init(ALGORITHMS[name], UNREACHED))
               for name in ("bellman_ford", "dijkstra", "prim")]
    objects += [(f"family a={p.alpha} g={p.gamma}", family_ctor(p))
                for p in (FamilyParams(alpha=0.7, gamma=0.9),
                          FamilyParams(alpha=0.5, tau=2.0, gamma=0.5))]
    objects += [(b, load_checkpoint(b, args.save_dir)) for b in CHECKPOINTS]

    results = {"args": vars(args)}
    results["stage1"] = stage1(objects, graphs, args.rounds)
    results["stage2"] = stage2(graphs, args.rounds, args.save_dir)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
