# FINDINGS — where the neural-vs-symbolic distance comes from

Run: 2026-08-11, `python run_decompose.py`. Log `decompose_log.txt`, records
`decompose_results.json`, models and fingerprints in `artifacts/`.

> ## ⚠ RETRACTED IN PART — read this before citing anything below
>
> **The headline orthogonality claim in this document is UNSUPPORTED by its own data**
> (Experiment H1, 2026-08-21, `FINDINGS_H.md`). Three independent problems:
>
> - `r = −0.233` is a correlation over **nine points**. Its exact two-sided permutation p over
>   all 362,880 relabellings is **0.5445**, with a 95% Fisher-z interval of **[−0.777, +0.510]**.
>   The data cannot distinguish orthogonality from strong positive alignment.
> - The symbolic column below is a mean pairwise slot-**disagreement rate**, not separating
>   power. Under separating power the correlation is **−0.004**; under unique-separator
>   weighting it is **+0.240**. The result is not robust to the definition of "carries identity".
> - The specific reading that `inject_unreached_node` "barely separates symbolic algorithms" is
>   **false**: it is in the size-2 minimal separating set and is the *unique* separator of
>   dijkstra/spfa.
>
> This document does label itself "no test, no p-value", and that caveat was honest — but the
> conclusion drawn from it ("a complete account of the neural null", "no amount of data fixes an
> orthogonal measurement") is far stronger than the number supports, and it was carried into
> `HANDOFF.md` and `README.md` as the argument that closed off probe redesign. **It cannot bear
> that weight, and `ROUTES_TO_POSITIVE.md` §4 is live again on the merits.**
>
> **The §3 pinned-predicate measurement is also superseded** in its mechanism, though not in its
> observation: `latency` and `poked_settled` really are pinned, but H2 showed the cause is the
> *firing rule* (networks poked at ~11% of their active life, so nothing has settled yet), not
> exact-fixed-point questions. Repaired, the vacuity gap is unchanged.
>
> **What survives:** the vacuity measurement itself — every neural model 0.30–0.40 from its
> nearest reference against a 0.303 reference-to-reference median. That has since survived E5,
> G and H2 and is the load-bearing neural finding. Cite that, not the orthogonality.

Descriptive decomposition, no test, no p-value. 9 hint-free models (3 plain, 3 gated,
3 halt; seeds 700/800/900 series, disjoint from E3's).

**Headline (RETRACTED — see the banner above): the fingerprint's discriminative axes and its
neural-disagreement axes are orthogonal — in fact slightly anti-correlated, r = −0.233.** Where
algorithmic identity actually lives, neural models agree with everything; where neural models
are distinctive, there is no identity to read. That was claimed as a complete account of the
neural null, and as not a resolution problem, a seed problem, or an architecture problem.

**The pre-registered hypothesis that motivated this run was wrong.** Deletion probes were
predicted to dominate the offset (constraint 3: relaxation cannot repair a deletion, but a
network can respond). They are **30.4% of disagreeing slots while being 33.3% of the
suite** — proportionally represented, carrying no excess. The null control
`delete_nontree_edge` contributes least of all nine (0.167). Removing the deletion probes
makes matters *worse*, not better (below).

---

## 1. Which interventions carry the disagreement

Fraction of each intervention's jointly-stable slots that disagree — neural-vs-Bellman-Ford
against the average over symbolic reference pairs.

| Intervention | neural vs BF | symbolic pairs |
|---|---|---|
| `inject_unreached_node` | **0.944** | 0.095 |
| `delete_tree_edge_near` | 0.713 | 0.228 |
| `corrupt_node_down` | 0.648 | 0.197 |
| `insert_improving_edge` | 0.367 | **0.458** |
| `lower_tree_edge_weight` | 0.333 | **0.433** |
| `delete_tree_edge_far` | 0.278 | 0.105 |
| `insert_shortcut_from_source` | 0.278 | **0.406** |
| `corrupt_node_up` | 0.222 | **0.335** |
| `delete_nontree_edge` | 0.167 | 0.000 |

Read the two columns against each other. The four interventions that carry the most
*symbolic* identity — `insert_improving_edge`, `lower_tree_edge_weight`,
`insert_shortcut_from_source`, `corrupt_node_up` — are the four where neural models
disagree **least**. The two where neural models are most distinctive,
`inject_unreached_node` (0.944 vs 0.095) and `corrupt_node_down` (0.648 vs 0.197), are
nearly uninformative between symbolic algorithms.

## 2. The axes are orthogonal

Correlation across the nine interventions between the two columns: **r = −0.233**
[**RETRACTED, H1**: exact permutation p = 0.5445, 95% CI [−0.777, +0.510]; −0.004 under a
separating-power score and +0.240 under unique-separator weighting. See the banner at the top.]

A fingerprint works when the directions along which an unknown differs from the references
are the directions that distinguish references from one another. Here they are unrelated,
and if anything opposed. The measurement is dominated by a large, uninformative signal
while the informative dimensions sit near zero — a signal-to-noise failure, not an
insufficient-data failure. No number of seeds or probe graphs fixes an orthogonal
measurement.

`delete_nontree_edge` is worth pausing on. It is the null control: symbolic algorithms
disagree on it **0.000** of the time, exactly as designed. Neural models disagree 0.167. So
the networks *do* respond where no correct algorithm can — constraint 3's prediction is
confirmed — but the effect is the smallest of all nine, not the dominant one.

## 3. Which predicates disagree — the real offset

| Predicate | disagreement with BF |
|---|---|
| `latency` | **1.000** |
| `poked_settled` | **1.000** |
| `recovery` | 0.335 |
| `spread1` | 0.215 |
| `term_delta` | 0.161 |
| `pi_churn` | 0.115 |
| `depth` | (unstable, dropped) |

Two of seven predicates disagree **always**. And every one of the four slots on which every
neural model disagrees with every one of the five references is a `poked_settled`:

    delete_tree_edge_far           poked_settled
    inject_unreached_node          poked_settled
    insert_improving_edge          poked_settled
    insert_shortcut_from_source    poked_settled

That is 12.5% of every neural distance as pure constant offset, plus `latency` pinned on
top of it — roughly 29% of the fingerprint contributing nothing but a shift.

**This is a measurement artifact of continuous outputs, not a fact about the models.**
`poked_settled` asks whether the poke landed on a node the algorithm had already finished
with, and `latency` asks how many rounds until the intervened run matches the reference
exactly. A symbolic algorithm reaches an exact fixed point and answers both crisply. A
network emits floats that keep drifting, so it answers "not settled" and "never" almost
regardless of which procedure it implements. Both predicates are near-constant on the
neural side and therefore carry no information while consuming two sevenths of the
fingerprint.

## 4. Dropping the deletion probes makes it worse

| Model | `d_near` full | `d_near` no-deletions |
|---|---|---|
| plain 700 / 701 / 702 | 0.438 / 0.424 / 0.424 | 0.476 / 0.464 / 0.476 |
| gated 800 / 801 / 802 | 0.412 / 0.412 / 0.367 | 0.476 / 0.476 / 0.368 |
| halt 900 / 901 / 902 | 0.375 / 0.439 / 0.389 | 0.375 / 0.400 / 0.400 |
| symbolic ref-to-ref median | 0.303 | 0.458 |

Removing them raises neural distances *and* inflates the symbolic reference-to-reference
median from 0.303 to 0.458. The deletion probes were among the few dimensions on which the
references agree with each other, so deleting them removes agreement, not disagreement. The
hypothesis is dead in both directions.

## 5. Incidental, and explicitly not a result

Two of three `halt` models label `bf_truncated_k3` under the full fingerprint (seeds 901,
902), where E3 got 0/6 on seeds 400–405. Pooled that is 2/9. With n=3 and a different seed
block this is most likely noise, and it is recorded here only so that a future
pre-registered test of truncation detection knows the observation exists. It is not
evidence of anything on its own — this project has buried five leads that looked better
than this one.

---

## What this licenses

**The neural null is explained, and the explanation is specific.** E3's p = 1.000 was not
bad luck, insufficient resolution, or the wrong architectures. The instrument measures
neural models along axes that carry no algorithmic identity, and is nearly blind along the
axes that do.

**Two concrete repairs follow, and neither is a new experiment on the same instrument.**

1. **Retire or reformulate `poked_settled` and `latency` for continuous-output models.**
   Both are exact-fixed-point predicates applied to a system that has no exact fixed point.
   A tolerance-based reformulation — "settled" as drift below a threshold sustained over a
   window, "latency" as time to enter and stay within tolerance — would restore two sevenths
   of the fingerprint. This is a measurement fix with a clear invariant behind it, which is
   exactly the kind of thing that must still be pre-registered before being believed.
2. **Measure along axes the networks actually vary on.** `fpid/response.py` estimates the
   Jacobian of one round and finds a structural invariant every symbolic reference satisfies
   exactly and every network violates: a hard `min` update depends on exactly **one** parent
   (median nonzeros per Jacobian row = 1.00 at every support threshold for all five
   references), while a network depends on **2–3** across a 10× threshold range. Soft
   aggregation is a property no reference in the family has, and it is measured without
   reference to any of them.

Both are candidates. Neither is claimed. The next run should be a pre-registered
exploratory/confirmatory split on disjoint seeds, with the decision rule fixed before the
confirmatory set is read — as in `PREREGISTRATION_E1E2.md`.
