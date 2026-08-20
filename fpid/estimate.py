"""Estimating family parameters from response profiles, with a residual that has a scale.

The old method returned a label and a Hamming distance. `THEORY.md` Proposition 5 shows why
that is vacuous once the object being measured is outside the reference set: the distance has
no scale, so "0.37 from Bellman-Ford" cannot be read as either near or far. This module
returns a parameter vector and a chi-squared, and the chi-squared is in units of the
measurement's own across-instance spread. `chi2 ~ 1` means the family contains the object.
`chi2 > 9` means it does not, and that is now a statement rather than a comparison.

**Two parameters have exact closed forms and are not fitted.** For any update of the form
`d_v <- (1-a) d_v + a * agg(gamma d_u + w)`, on a row that is reached and still moving:

    alpha = 1 - |J[v, v]|            gamma = (off-diagonal row sum) / alpha

Verified to four decimals on 12 hand-set (alpha, tau, gamma) triples. Gridding either of
them would throw away the only clean result available, so `estimate_direct` reads them off
and only `tau` is searched.

**One parameter is mostly unidentifiable, and saying so is the job.** Aggregation temperature
is visible only when `tau` is comparable to the typical best-minus-second-best message gap.
Damping pushes that gap up: measured at gamma=1, the median gap is 3.9 at alpha=1 but 21.8 at
alpha=0.7 and 29.5 at alpha=0.5, because a damped estimate sits far above its fixed point
where the minimum is unambiguous. Measured `off_pr` was flat at 1.0000 for every tau <= 2 at
alpha=0.7. So `off_pr == 1` must be read as "**this update behaves as a hard min at this
operating point**", which is not the same claim as "the aggregation is hard". The estimator
returns `tau = 0` with a `tau_unidentified` flag rather than a point estimate. A correct
refusal is a correct answer; a confident number there would be the same error as reporting a
nearest reference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .family import FamilyParams, family_ctor
from .graphs import Graph
from .response import response_profile_masked

# Profile coordinates, in pack order. `n_fit` and `off_rows` are never nan and carry the
# "no information" signal, which is why nan medians can be imputed to zero without a
# missing-data indicator per statistic.
STATS = (
    "diag_frac",
    "self_med",
    "off_row_med",
    "off_pr",
    "off_rows",
    "off_width",
    "row_sum_med",
    "n_fit",
)
NORMALISED = {"off_rows", "n_fit"}  # divided by n, to keep coordinates order-1

ALPHA_GRID = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
TAU_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
C_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
K_GRID = (3, 5, 8, 12, 24, None)
SELECTION_GRID = ("sync", "argmin")

ROUNDS = 8
EPS = 0.25
THETA = 0.05
KAPPA = 1.0

# off_pr below this is indistinguishable from a hard min at the measured spread. Set from the
# tau sweep in PREREGISTRATION_FAMILY.md: off_pr was 1.0003 at tau=0.5 against 1.0000 at
# tau=0, i.e. inside the noise, and first clearly departed at 1.12.
OFF_PR_FLOOR = 1.02


# Which parameters each flag makes unreportable. A flag that does not withhold anything is a
# note; a flag that does is a refusal, and the difference has to be machine-readable, because
# scoring a refusal as though it were a claim is how a method with honest coverage gets recorded
# as a method with dishonest accuracy.
WITHHOLDS: dict[str, tuple[str, ...]] = {
    "no_fittable_rows": ("alpha", "gamma", "tau"),
    "gamma_degenerate": ("alpha", "tau"),
    "alpha_degenerate": ("gamma",),
    "tau_unidentified": ("tau",),
}


@dataclass
class FitReport:
    theta: FamilyParams
    direct: FamilyParams
    chi2: float
    ve: float
    row_sum_deviation: float
    per_stat_residual: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def withheld(self) -> set[str]:
        """Parameters this fit declines to report.

        `theta` always carries the best-fitting value, because the search still needs a point
        to evaluate and the caller may want it as a diagnostic. But a withheld parameter is not
        an estimate, and anything scoring accuracy must skip it and count it against coverage
        instead. `no_fittable_rows` is the important one: when the mask admits no row, the
        direct estimators have literally no data, and reporting the constructor default (alpha
        = 1) as a confident answer is the single worst thing this module could do.
        """
        out: set[str] = set()
        for f in self.flags:
            out.update(WITHHOLDS.get(f, ()))
        return out

    def verdict(self) -> str:
        """The frozen ladder from PREREGISTRATION_FAMILY.md. No middle rung."""
        return "family_adequate" if (self.chi2 <= 4.0 and self.ve >= 0.90) else "family_wrong"

    def to_dict(self) -> dict:
        return {
            "theta": dict(zip(("alpha", "tau", "gamma", "c", "k", "selection"), self.theta.key())),
            "direct": dict(zip(("alpha", "tau", "gamma", "c", "k", "selection"), self.direct.key())),
            "chi2": self.chi2,
            "ve": self.ve,
            "row_sum_deviation": self.row_sum_deviation,
            "per_stat_residual": self.per_stat_residual,
            "flags": self.flags,
            "withheld": sorted(self.withheld),
            "verdict": self.verdict(),
        }


# ---------------------------------------------------------------- profiles


def profile_one(ctor, graph: Graph, source: int = 0, rounds: int = ROUNDS, **kw) -> dict:
    return response_profile_masked(
        ctor, graph, source, rounds, EPS, THETA, kw.get("kappa", KAPPA)
    )


def profile_vector(prof: dict[str, list[float]], n: int) -> np.ndarray:
    """Pack a profile into a fixed-length vector. `nan` medians impute to 0.0."""
    cols = []
    for name in STATS:
        v = np.asarray(prof[name], dtype=float)
        if name in NORMALISED:
            v = v / n
        cols.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    return np.concatenate(cols)


def profile_set(ctor, graphs: list[Graph], source: int = 0, rounds: int = ROUNDS, **kw):
    """Per-graph profile vectors, stacked. Rows are instances, columns are coordinates."""
    return np.stack(
        [profile_vector(profile_one(ctor, g, source, rounds, **kw), g.n) for g in graphs]
    )


def family_profile_set(params: FamilyParams, graphs: list[Graph], **kw) -> np.ndarray:
    return profile_set(family_ctor(params), graphs, **kw)


class ProfileCache:
    """Memoises family profiles on `(theta, graph-set identity)`.

    The family profile is a pure function of theta and the graph set, so the grid computed for
    Gate 1 is reused by Gate 2's optimiser for free. This is the difference between a two-hour
    spike and a five-minute one.
    """

    def __init__(self, graphs: list[Graph], tag: str = "", **kw):
        self.graphs = graphs
        self.tag = tag
        self.kw = kw
        self._store: dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, params: FamilyParams) -> np.ndarray:
        key = params.key()
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        z = family_profile_set(params, self.graphs, **self.kw)
        self._store[key] = z
        return z


# ---------------------------------------------------------------- scales


def standardise_scale(Z: np.ndarray, floor_frac: float = 0.05) -> np.ndarray:
    """Per-coordinate SD across instances, floored so a constant coordinate cannot blow up.

    A coordinate with zero spread across graphs would give an infinite z-score for any
    deviation, which would let one perfectly-reproducible statistic dominate every distance.
    The floor is a fraction of the median positive SD, so it scales with the measurement
    rather than being an absolute guess.
    """
    sd = Z.std(axis=0, ddof=1) if Z.shape[0] > 1 else np.ones(Z.shape[1])
    pos = sd[sd > 0]
    floor = floor_frac * float(np.median(pos)) if pos.size else 1.0
    return np.maximum(sd, max(floor, 1e-9))


def chi2_of(z_obs: np.ndarray, z_fit: np.ndarray, sd: np.ndarray) -> float:
    return float(np.mean(np.square((z_obs - z_fit) / sd)))


def per_stat_residual(z_obs: np.ndarray, z_fit: np.ndarray, sd: np.ndarray) -> dict[str, float]:
    """chi2 broken out per statistic, so a single bad coordinate is visible not averaged."""
    r = np.square((z_obs - z_fit) / sd)
    per = len(r) // len(STATS)
    return {name: float(np.mean(r[i * per : (i + 1) * per])) for i, name in enumerate(STATS)}


# ---------------------------------------------------------------- direct estimators


def _median_over(prof_list: list[dict], name: str) -> float:
    vals = [v for p in prof_list for v in p[name] if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def estimate_direct(prof_list: list[dict], n: int) -> tuple[FamilyParams, list[str]]:
    """Closed-form per-axis estimates. Each line is independently checkable.

    `prof_list` is one masked profile per graph instance. Everything here is a median over
    instances and rounds, because a selection flip produces a single enormous row response
    that drags any mean by an order of magnitude (`fpid/response.py`).
    """
    flags: list[str] = []

    self_med = _median_over(prof_list, "self_med")
    off_row = _median_over(prof_list, "off_row_med")
    off_pr = _median_over(prof_list, "off_pr")

    if not np.isfinite(self_med):
        return FamilyParams(), ["no_fittable_rows"]

    alpha = float(np.clip(1.0 - self_med, 0.0, 1.0))

    # gamma = off_row / alpha. At gamma -> 0 the update is exactly diagonal, so there is no
    # off-diagonal mass to divide and alpha itself stops being observable too. Report the
    # degeneracy rather than a ratio of two things that are both zero.
    if not np.isfinite(off_row) or off_row <= THETA:
        flags.append("gamma_degenerate")
        gamma = 0.0
    elif alpha < 1e-6:
        flags.append("alpha_degenerate")
        gamma = 1.0
    else:
        gamma = float(np.clip(off_row / alpha, 0.0, 1.0))

    # tau: only meaningful when the off-diagonal participation ratio is clear of 1.
    if not np.isfinite(off_pr) or off_pr < OFF_PR_FLOOR:
        flags.append("tau_unidentified")
        tau = 0.0
    else:
        tau = float("nan")  # filled by the calibration inversion in `fit`

    degenerate = "gamma_degenerate" in flags
    selection, k, c = _estimate_discrete(prof_list, degenerate)
    return FamilyParams(alpha=alpha, tau=tau, gamma=gamma, c=c, k=k, selection=selection), flags


def _estimate_discrete(prof_list: list[dict], degenerate: bool) -> tuple[str, int | None, float]:
    """Seeds for the discrete axes. **These are search seeds, not estimates.**

    Measured behaviour, which is why they are only seeds: at `alpha = 1, tau = 0` a
    synchronous sweep converges within a couple of rounds, so almost no row is still moving
    and `off_rows` peaks as low as one -- indistinguishable from one-sender-per-round. The
    row-count discriminator therefore misreads Bellman-Ford as `argmin`. `fit` scores both
    selections and the whole `c` grid regardless of what this returns, and the reported
    estimate is the fitted one. Seeding is worth doing anyway: it puts the right `k` in the
    candidate set, which is the axis with the widest grid.
    """
    off_rows = np.array([p["off_rows"] for p in prof_list], dtype=float)  # (graphs, rounds)
    diag = np.array([p["diag_frac"] for p in prof_list], dtype=float)

    live = np.median(off_rows, axis=0)
    med_diag = np.median(diag, axis=0)
    peak = float(live.max()) if live.size else 0.0
    selection = "argmin" if peak <= 1.5 else "sync"

    # k: the first round from which nothing responds and the map is the identity, sustained to
    # the end of the profile. A single quiet round mid-run is convergence, not a halt.
    #
    # Suppressed entirely when gamma is degenerate: at gamma = 0 the update is diagonal at
    # every round, so "off-diagonal response stopped" is true from round 1 and this would
    # report a halt at k=1 for an algorithm that never halts. Prim measured exactly that.
    k: int | None = None
    T = live.size
    if not degenerate:
        for t in range(1, T):
            if np.all(live[t:] <= 0.0) and np.all(med_diag[t:] >= 0.999):
                k = t
                break

    # c: commitment shows as diag_frac rising and staying high. This is the weakest reader in
    # the module -- it cannot separate commitment from ordinary convergence, and it was
    # measured to give both false positives and false negatives -- so it is a seed only.
    c = 0.0
    if T >= 4 and k is None:
        rise = float(med_diag[-1] - med_diag[: max(1, T // 3)].mean())
        if rise > 0.30:
            c = 1.0
        elif rise > 0.10:
            c = 0.25
    return selection, k, c


def c_bucket(c: float) -> str:
    if c <= 0.0:
        return "zero"
    return "one" if c >= 1.0 else "interior"


# ---------------------------------------------------------------- tau calibration


def beta_calibration(
    graphs: list[Graph],
    alpha_grid=ALPHA_GRID,
    gamma_grid=(0.5, 1.0),
    tau_grid=TAU_GRID,
    cache_path: str | None = None,
    **kw,
) -> dict:
    """`off_pr = f(tau; alpha, gamma)` on this graph family, for inversion.

    Three-dimensional and not reducible: temperature's observability is conditional on alpha
    and gamma, because those set how far the estimate sits from its fixed point and therefore
    how large the message gaps are. Built with the *same* eps, theta and kappa used on the
    target, since off_pr is a function of the probe amplitude as well as of the update rule.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    table: dict = {"alpha": list(alpha_grid), "gamma": list(gamma_grid), "tau": list(tau_grid),
                   "off_pr": []}
    for a in alpha_grid:
        plane = []
        for g in gamma_grid:
            row = []
            for t in tau_grid:
                prof = [
                    profile_one(family_ctor(FamilyParams(alpha=a, tau=t, gamma=g)), gg, **kw)
                    for gg in graphs
                ]
                row.append(_median_over(prof, "off_pr"))
            plane.append(row)
        table["off_pr"].append(plane)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(table, f, indent=1)
    return table


def invert_off_pr(table: dict, off_pr: float, alpha: float, gamma: float) -> tuple[float, bool]:
    """Read tau back out of off_pr. Returns `(tau, identified)`.

    Monotone interpolation along the nearest (alpha, gamma) plane. `identified` is False when
    the observed off_pr sits below the plane's own floor -- i.e. this operating point could not
    have revealed any temperature, so no value is reported.
    """
    if not np.isfinite(off_pr):
        return 0.0, False
    ai = int(np.argmin(np.abs(np.asarray(table["alpha"]) - alpha)))
    gi = int(np.argmin(np.abs(np.asarray(table["gamma"]) - gamma)))
    taus = np.asarray(table["tau"], dtype=float)
    curve = np.asarray(table["off_pr"][ai][gi], dtype=float)
    ok = np.isfinite(curve)
    if ok.sum() < 2:
        return 0.0, False
    taus, curve = taus[ok], curve[ok]

    # Enforce monotonicity before inverting: off_pr rises with tau, and any local dip is
    # measurement noise rather than a real non-monotonicity.
    curve = np.maximum.accumulate(curve)
    if off_pr <= curve[0] + (OFF_PR_FLOOR - 1.0):
        return 0.0, False
    if off_pr >= curve[-1]:
        return float(taus[-1]), True
    return float(np.interp(off_pr, curve, taus)), True


def tau_detectable(table: dict, snr_ratio: float = OFF_PR_FLOOR - 1.0) -> dict[str, float]:
    """Smallest grid tau whose off_pr clears the hard-min value, per (alpha, gamma).

    Reported, never fitted: this is the honest ceiling on temperature estimation, and it is a
    property of the operating point rather than one global constant. Keyed by the pair because
    minimising over gamma erases the effect -- gamma < 1 compresses the spread between
    competing parents, so softness becomes visible at a lower tau there than at gamma = 1, and
    a per-alpha minimum would report that optimistic corner as if it held everywhere.

    `inf` means no tau on the grid was detectable at that operating point. That is the
    expected entry over most of the box, and it is the result, not a gap in it.
    """
    out: dict[str, float] = {}
    taus = np.asarray(table["tau"], dtype=float)
    for ai, a in enumerate(table["alpha"]):
        for gi, g in enumerate(table["gamma"]):
            curve = np.asarray(table["off_pr"][ai][gi], dtype=float)
            base = curve[0] if np.isfinite(curve[0]) else 1.0
            hit = [
                taus[i]
                for i in range(len(taus))
                if np.isfinite(curve[i]) and curve[i] > base + snr_ratio
            ]
            out[f"alpha={a},gamma={g}"] = float(min(hit)) if hit else float("inf")
    return out


# ---------------------------------------------------------------- fitting


def _golden(f, lo: float, hi: float, iters: int = 20) -> tuple[float, float]:
    """Golden-section minimisation on [lo, hi]. Hand-rolled; the repo avoids scipy."""
    inv = 0.6180339887498949
    b, a = hi, lo
    c, d = b - inv * (b - a), a + inv * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv * (b - a)
            fd = f(d)
    x = c if fc < fd else d
    return x, min(fc, fd)


def fit(
    target_profiles: list[dict],
    graphs: list[Graph],
    cal: dict | None = None,
    cache: ProfileCache | None = None,
    grid_var: float | None = None,
    **kw,
) -> FitReport:
    """Fit family parameters to one target, given its per-instance masked profiles.

    Raw profile dicts rather than packed vectors, deliberately: packing imputes `nan` medians
    to 0.0 so that distances stay well-defined, and a `self_med` of exactly 0.0 is *also* what
    `alpha = 1` produces. The direct estimators must be able to tell "no measurement" from
    "measured zero", so they read the unpacked profiles and only the distance uses the packed
    form.

    The measurement scale comes from the target's own across-instance spread, which is what
    gives chi2 a unit and is the whole repair of the nearest-reference method.

    Sequence: closed forms for alpha and gamma, one pass over discrete candidates, golden
    section on tau for the winner, then a short coordinate polish. alpha and gamma are never
    gridded. Scoring every discrete candidate at every tau would cost ~340 profile evaluations
    per target instead of ~70, for no gain: the discrete axes are read from statistics that
    tau barely moves.
    """
    n = graphs[0].n
    target = np.stack([profile_vector(p, n) for p in target_profiles])
    sd = standardise_scale(target)
    z_obs = target.mean(axis=0)
    cache = cache or ProfileCache(graphs, **kw)

    direct, flags = estimate_direct(target_profiles, n)

    def score(p: FamilyParams) -> float:
        return chi2_of(z_obs, cache.get(p).mean(axis=0), sd)

    # Stage 1: the discrete axes, at the initial tau.
    tau0 = 0.0
    if "tau_unidentified" not in flags and cal is not None:
        tau0, ok = invert_off_pr(
            cal, _median_over(target_profiles, "off_pr"), direct.alpha, direct.gamma
        )
        if not ok:
            flags.append("tau_unidentified")
            tau0 = 0.0

    best = replace(direct, tau=tau0)
    best_s = score(best)
    for sel in SELECTION_GRID:
        for k in _k_candidates(direct.k):
            for c in _c_candidates(direct.c):
                cand = replace(direct, tau=tau0, selection=sel, k=k, c=c)
                s = score(cand)
                if s < best_s:
                    best, best_s = cand, s

    # Stage 2: tau, only if the operating point could reveal it.
    if "tau_unidentified" not in flags:
        lo, hi = (0.0, float(max(TAU_GRID))) if tau0 <= 0 else (
            max(0.0, tau0 - 2.0),
            min(float(max(TAU_GRID)), tau0 + 2.0),
        )
        tstar, s = _golden(lambda t: score(replace(best, tau=t)), lo, hi, 12)
        if s < best_s:
            best, best_s = replace(best, tau=tstar), s

    # Stage 3: coordinate polish. alpha and gamma have closed forms, so the box is tight on
    # purpose -- this absorbs mask-induced bias, it does not search.
    for _ in range(2):
        for name in ("alpha", "gamma"):
            v0 = getattr(best, name)
            lo, hi = max(0.0, v0 - 0.12), min(1.0, v0 + 0.12)
            if hi - lo < 1e-6:
                continue
            # Bind `best` and `name` explicitly. The closure is consumed inside this iteration
            # so late binding would happen to be correct, but both names are reassigned in the
            # loop body below and a reader should not have to verify that to trust the search.
            x, s = _golden(
                lambda v, _b=best, _n=name: score(replace(_b, **{_n: v})), lo, hi, 8
            )
            if s < best_s:
                best, best_s = replace(best, **{name: x}), s

    z_fit = cache.get(best).mean(axis=0)
    ve = float("nan")
    if grid_var is not None and grid_var > 0:
        ve = float(np.clip(1.0 - best_s / grid_var, -10.0, 1.0))

    row_pred = 1.0 - best.alpha * (1.0 - best.gamma)
    row_obs = _median_over(target_profiles, "row_sum_med")
    dev = float(abs(row_obs - row_pred)) if np.isfinite(row_obs) else float("nan")

    return FitReport(
        theta=best,
        direct=direct,
        chi2=best_s,
        ve=ve,
        row_sum_deviation=dev,
        per_stat_residual=per_stat_residual(z_obs, z_fit, sd),
        flags=sorted(set(flags)),
    )


def grid_variance(grid_profiles: list[np.ndarray], sd: np.ndarray) -> float:
    """Mean standardised variance of `z` across a theta grid -- the denominator of `ve`.

    `ve = 1 - chi2 / grid_var` asks "how much of what the family can express does the fit
    account for", so the denominator has to be the family's own dynamic range, measured in the
    same standardised units as the residual. Computed once from Gate 1's grid and passed in;
    without it `ve` is undefined and is reported as `nan` rather than as a number derived from
    the target alone.
    """
    Z = np.stack([z.mean(axis=0) for z in grid_profiles]) / sd
    return float(np.mean(np.var(Z, axis=0)))


def _k_candidates(k: int | None) -> list[int | None]:
    if k is None:
        return [None]
    opts = sorted({k, max(1, k - 1), k + 1} & {x for x in K_GRID if x is not None})
    return [*opts, None] if opts else [None]


def _c_candidates(c: float) -> list[float]:
    """The whole grid, seed first.

    Pruning `c` around the seed was a real search failure, not a saving: the seed reader gives
    false negatives, and a target generated at `c = 1` whose seed read `0` then had `c = 1`
    excluded from the candidate set entirely, so the fit could not recover a value it was
    never allowed to try. Five values is cheap; a systematically unreachable region is not.
    """
    seed = min(C_GRID, key=lambda v: abs(v - c))
    return [seed, *[v for v in C_GRID if v != seed]]


def profiles_of(ctor, graphs: list[Graph], source: int = 0, **kw) -> list[dict]:
    """Per-instance masked profiles for any executor constructor. The input `fit` wants."""
    return [profile_one(ctor, g, source, **kw) for g in graphs]
