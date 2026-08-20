"""E5: does sensitivity normalisation close the vacuity gap?

`_delete_nontree_edge`'s own docstring in `fpid/interventions.py` says "every other
response gets normalised against this" -- and nothing in the codebase does. This
implements the missing normalisation as a pairwise mask (a slot is dropped from a
comparison only when it is uninformative for *both* fingerprints -- see
`fpid.identify.informative`) and asks whether it changes Proposition 5's finding that
every neural model sits further from its nearest reference (0.37-0.40) than two
references sit from each other (0.303 median).

Decision rule, gates and outcome ladder are fixed in `PREREGISTRATION_E5E6.md` and were
written before this ran.

    python run_e5.py
    python run_e5.py --smoke
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fpid.algorithms import ALGORITHMS, finite_init
from fpid.identify import distance, classify, margin
from run_e1_e2 import NOISE_FLOOR, deep_probe_graphs, fingerprint_of
from run_family_spike import load_checkpoints
from run_phase_a import build_fingerprints


def ref_to_ref_median(refs: dict, stability: float, normalise: bool) -> float:
    names = sorted(refs)
    vals = [
        distance(refs[a], refs[b], stability, normalise)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return float(np.median(vals))


def null_response_rate(fp, stability: float) -> float:
    """Fraction of stable, non-null-control slots where the fingerprint's response
    differs from its own null-control response -- i.e. how much of the fingerprint is
    "real" signal versus baseline jumpiness, under the pairwise mask's own logic."""
    from fpid.identify import informative

    keys = [k for k in fp.stable_keys(stability) if k[0] != "delete_nontree_edge"]
    if not keys:
        return 0.0
    return sum(1 for k in keys if informative(fp, k)) / len(keys)


def negative_control(args) -> bool:
    print("=" * 70)
    print("NEGATIVE CONTROL -- does normalisation disturb the symbolic separation?")
    print("=" * 70)
    fps = build_fingerprints(25, args.nodes, 0, "high_diameter")
    from fpid.interventions import INTERVENTIONS
    from fpid.separation import minimal_separating_set, unseparated_pairs

    names = [iv.name for iv in INTERVENTIONS]
    for norm in (False, True):
        bad = unseparated_pairs(fps, names, args.stability, normalise=norm)
        minimal = minimal_separating_set(fps, names, args.stability, normalise=norm)
        tag = "normalised" if norm else "raw"
        print(f"  [{tag}] unseparated pairs: {bad or 'none'}")
        print(f"  [{tag}] minimal separating set: {minimal} "
              f"(size {len(minimal) if minimal else 'n/a'})")

    fps_norm = fps
    bad_n = unseparated_pairs(fps_norm, names, args.stability, normalise=True)
    minimal_n = minimal_separating_set(fps_norm, names, args.stability, normalise=True)
    ok = not bad_n and minimal_n is not None and len(minimal_n) <= 3
    print(f"\n  {'PASS' if ok else 'FAIL'} -- unseparated empty and |minimal| <= 3 required")
    if not ok:
        print("  NC FAILED. The normaliser is malformed. Stopping -- this is not a")
        print("  result about sensitivity, it is a bug in the mask.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--probe-graphs", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--min-depth", type=int, default=5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stability", type=float, default=0.8)
    ap.add_argument("--checkpoints", default="artifacts")
    ap.add_argument("--out", default="e5_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.probe_graphs = 3

    if not negative_control(args):
        return

    budget = args.budget if args.budget else 2 * args.nodes
    rng = np.random.default_rng(0)
    probe_graphs = deep_probe_graphs(
        args.probe_graphs, args.nodes, rng, args.density, args.min_depth
    )
    depths = [
        int(g.hop_distances(0)[np.isfinite(g.hop_distances(0))].max())
        for g, _ in probe_graphs
    ]
    print(f"\nprobe-graph depths: {depths} (median {np.median(depths):.1f})")

    print("\n" + "=" * 70)
    print("Building symbolic reference fingerprints")
    refs = {
        name: fingerprint_of(finite_init(ctor), probe_graphs, budget, args.tol)
        for name, ctor in ALGORITHMS.items()
    }
    median_raw = ref_to_ref_median(refs, args.stability, normalise=False)
    median_norm = ref_to_ref_median(refs, args.stability, normalise=True)
    print(f"  reference-to-reference median: raw {median_raw:.4f}  "
          f"normalised {median_norm:.4f}")

    print("\n" + "=" * 70)
    print("PRIMARY -- checkpoint distance to nearest reference, raw vs normalised")
    print("=" * 70)
    checkpoints = load_checkpoints(args.checkpoints)
    if not checkpoints:
        print("  no checkpoints found under", args.checkpoints)
        return

    records = []
    for name, ctor in checkpoints:
        fp = fingerprint_of(ctor, probe_graphs, budget, args.tol)
        label_raw, dists_raw = classify(fp, refs, args.stability, normalise=False)
        label_norm, dists_norm = classify(fp, refs, args.stability, normalise=True)
        d_near_raw = min(dists_raw.values())
        d_near_norm = min(dists_norm.values())
        closed = d_near_norm < median_norm
        rec = {
            "checkpoint": name,
            "label_raw": label_raw,
            "d_near_raw": d_near_raw,
            "margin_raw": margin(dists_raw),
            "label_norm": label_norm,
            "d_near_norm": d_near_norm,
            "margin_norm": margin(dists_norm),
            "closed": closed,
            "null_response_rate": null_response_rate(fp, args.stability),
        }
        records.append(rec)
        print(f"  {name:<14} raw d_near={d_near_raw:.4f} ({label_raw:<16})  "
              f"norm d_near={d_near_norm:.4f} ({label_norm:<16})  "
              f"{'CLOSED' if closed else 'still-vacuous'}")

    print("\n" + "=" * 70)
    print("SECONDARY -- null-control response rate (references, for comparison)")
    for name in sorted(refs):
        rate = null_response_rate(refs[name], args.stability)
        print(f"  {name:<16} {rate:.3f}")

    n_closed = sum(1 for r in records if r["closed"])
    n = len(records)
    print("\n" + "=" * 70)
    print("VERDICT (pre-registered ladder, PREREGISTRATION_E5E6.md)")
    print(f"  {n_closed}/{n} checkpoints CLOSED (normalised d_near < normalised "
          f"reference-to-reference median)")
    if n_closed >= 7:
        verdict = "CLOSED"
        print("  CLOSED. Proposition 5's hypothesis fails under a normalised measure.")
        print("  This is a significant reversal and MUST be replicated on fresh probe")
        print("  graphs before it is written down as anything -- see the ladder note")
        print("  in PREREGISTRATION_E5E6.md.")
    elif n - n_closed >= 7:
        verdict = "UNCHANGED"
        print("  UNCHANGED. The vacuity is real and not a sensitivity artefact.")
        print("  Proposition 5 stands, strengthened.")
    else:
        verdict = "INDETERMINATE"
        print("  INDETERMINATE (neither >=7/9). Reported as such, not as partial")
        print("  success, and not grounds for a follow-up statistic.")

    with open(args.out, "w") as f:
        json.dump(
            {
                "median_raw": median_raw,
                "median_norm": median_norm,
                "records": records,
                "verdict": verdict,
                "noise_floor": NOISE_FLOOR,
            },
            f,
            indent=1,
            default=float,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
