# Pre-registration: E5 (sensitivity normalisation) and E6 (Dijkstra/Prim boundary)

**Written:** 2026-08-20, before either experiment ran.
**Status of the project when written:** symbolic complete and positive; neural finished and
negative on all tested approaches (E3, E3b, Phase F); recommended next action is the write-up.

These two experiments exist because `THEORY.md` §7 lists both as open, and because the
write-up should not ship with named holes in its own theory section. Both are cheap
(numpy only, no CUDA, no training), both are re-analyses or small extensions of machinery
that already exists, and **both are therefore exactly the kind of thing that has killed five
results in this project** — a new statistic computed on data that has already been looked at.
Hence a pre-registration rather than a session of poking.

---

## E5. Sensitivity normalisation

### The gap

`fpid/interventions.py:89`, the docstring of `_delete_nontree_edge`, says:

> Null control: remove an edge no shortest path uses. A model whose fingerprint responds
> strongly here is telling us its Lipschitz constant, not its algorithm. **Every other
> response gets normalised against this.**

Nothing in the codebase does this. The probe is measured and folded into the fingerprint
like any other; it is never used as a denominator, a mask, or a control. The only place it
appears outside its own definition is `run_decompose.py:52`, where it is grouped with the
other deletions for an ablation. So the claim in the docstring is unimplemented, and the
concern behind it — a globally sensitive model looking "different" for free — is untested.

### Why normalisation cannot be a ratio

The predicates are categorical by deliberate design (`fpid/fingerprint.py` docstring: a
real-valued metric over response vectors has free parameters and any separation it reports
can be tuned into existence). `recovery ∈ {exact, partial, none}` does not divide.
Normalisation must therefore be a **mask**, not a quotient.

### The rule

For a fingerprint `F` and a slot `k = (i, p)` with `i ≠ delete_nontree_edge`, define

```
informative(F, k)  ⟺  F.value[(i, p)] ≠ F.value[("delete_nontree_edge", p)]
```

A response that matches the executor's own response to a poke which changes nothing is not
a response to the poke.

**The mask is applied pairwise, not per-fingerprint.** Slot `k` is dropped from the
comparison of `A` and `B` iff it is uninformative for *both*:

```
keys_normalised(A, B) = { k ∈ shared_keys(A, B) : informative(A,k) or informative(B,k) }
```

Per-fingerprint masking is wrong and would have been the natural thing to write. Under
`corrupt_node_up`, Bellman-Ford gives `recovery = exact`, which is also what the null control
gives; a per-fingerprint mask drops that slot and destroys the BF/Dijkstra separation that
Phase A's headline rests on. Dijkstra is informative in that slot (`recovery = none`), so the
pairwise rule keeps it. This mirrors the existing pairwise stability rule
(`fpid/identify.py:11-14`), and the fact that the same asymmetry argument applies to both is
the reason to believe the shape is right.

`delete_nontree_edge`'s own slots are excluded from the distance entirely under
normalisation — it is the reference, not a measurement.

### Negative control — must pass, gates everything below

**The symbolic side must be a near no-op.** Real relaxation algorithms do not respond to the
deletion of an edge no shortest path uses, so their null-control responses are the trivial
ones and normalisation should barely bite.

> **NC.** Under normalised distance, on `run_phase_a.py --graphs 25 --nodes 24` at
> stability 0.8: `unseparated_pairs` is still empty, and a minimal separating set still
> exists at size ≤ 3.

If NC fails, the normaliser is malformed and E5 stops. **A failure here is not a result about
sensitivity**; it is a bug in the mask. Record it, fix or abandon, do not report.

### Primary question

Does sensitivity normalisation close the vacuity gap on the nine E3 checkpoints?

This is the question worth asking, because `THEORY.md` Proposition 5 — the finding that ate
the neural side — rests on `d_near ≈ 0.37–0.40` against a reference-to-reference median of
`0.303`. If a substantial part of that distance is the networks responding to everything,
then the vacuity is partly an artefact of an unnormalised measure and Proposition 5's
hypothesis may not actually be satisfied.

Measured on the existing `artifacts/*.pt` checkpoints (`plain_700-702`, `gated_800-802`,
`halt_900-902`). No retraining. Probe graphs at depth ≥ 5, budget and tolerance identical to
E3's so the raw numbers are directly comparable.

**Frozen outcome ladder — two rungs, deliberately no third.** Direction-replicating-without-
resolution is the characteristic signature of a dead lead here (`overshoot`, E3b), so the
ladder leaves it nowhere to land.

| Outcome | Criterion | Reading |
|---|---|---|
| **CLOSED** | normalised `d_near < ` normalised reference-to-reference median for **≥ 7 of 9** checkpoints | Proposition 5's hypothesis fails under a normalised measure. The neural negative is partly an artefact of an unnormalised distance and the neural side reopens. This would be a significant reversal and must then be replicated on fresh probe graphs before it is written down as anything. |
| **UNCHANGED** | `d_near ≥` that median for **≥ 7 of 9** | The vacuity is real and not a sensitivity artefact. Proposition 5 stands, strengthened, and the write-up can quote it having survived the obvious objection. |

Anything else (4–6 of 9) is recorded as **INDETERMINATE** and reported as such — not as a
partial success, and not as grounds for a follow-up statistic.

### Secondary, reported and never gated

- Null-control response rate per checkpoint: the fraction of predicates on which each model
  responds to `delete_nontree_edge` at all. This is the direct measurement of "how jumpy are
  these networks", which has never been quoted, and it is a descriptive number regardless of
  how the primary lands.
- The same rate for all five references. Expected ~0. If it is not ~0, the null control is
  not null and both E5 and every earlier fingerprint that included it need re-examining —
  which would itself be the finding.
- Change in the reference-to-reference median under normalisation. Note that this is the
  denominator of the primary, so it moving is not independently interesting; it is reported
  so the primary's arithmetic is auditable.

### What E5 does *not* claim

It does not revive the OOD claim (B1), which failed for reasons orthogonal to sensitivity.
Even under CLOSED, every neural experiment would need re-running before anything is claimed.

---

## E6. Closing the Dijkstra/Prim boundary

### The gap

Dijkstra and Prim are separated by exactly one intervention, `lower_tree_edge_weight`
(`fpid/interventions.py:120`), whose note reads "Separates Dijkstra from Prim: they key on
`d[u] + w` versus `w`." One probe, one pair. The minimal-separating-set claim — two
interventions suffice at depth ≥ 7 — inherits that fragility: if this single probe is
measuring something incidental, the size-2 result is wrong.

The probe is also indirect. Setting a tree edge's weight to `0.5` moves `w` *and* moves
`d[u] + w`, so both keys change and the two algorithms are separated by the *manner* of their
responses rather than by one responding and the other not.

### The construction

Two new interventions, both targeting the tree edge `(u, v)` into `t.near`, matching the
existing convention.

**`raise_tree_edge_weight`** — `w(u,v) += δ`. Both keys move. This is the control arm.

**`dissociate_keys`** — the actual probe. Simultaneously:

```
w(u, v)   += δ
ex.d[u]    = d_true[u] − δ
```

Dijkstra's key for reaching `v` through `u` is `d[u] + w`, which is
`(d_true[u] − δ) + (w_old + δ) = d_true[u] + w_old` — **exactly invariant**. Prim's key is
`w`, which rose by δ. The two keys are dissociated by construction rather than by
interpretation of a response.

Both edits are computed from the graph and its ground truth only — `d_true[u]`, `w_old`, and
a δ derived below — so every algorithm receives the identical poke and design rule 1
(`fpid/interventions.py:5-8`) holds. Whether a given executor's own `ex.d[u]` happened to
equal `d_true[u]` at firing time is part of its response, not part of the poke.

**Choice of δ.** It must be large enough to clear the measurement tolerance against an
edge-weight scale of 1–10, and small enough that `(u,v)` remains the tree edge into `v` —
otherwise raising `w` changes the ground-truth shortest-path tree and the two arms stop
measuring comparable things. Both bounds are computable from ground truth:

```
δ_max = (best alternative route to v) − d_true[v]
δ     = min(2.0, 0.5 · δ_max)
```

Skip the instance if `δ_max` is not finite and positive, or if `δ < 0.1`. Record the skip
rate; if it exceeds 25% the construction is too fragile for the graph family and E6 is
reported as inapplicable rather than null.

### Mechanism check — must pass before the primary is read

The rule that saved E1 from being written up as the wrong null: verify the manipulation
before reading the measurement.

> **MC.** Dijkstra's response to `dissociate_keys` must be strictly weaker than its response
> to `raise_tree_edge_weight` — measured as the number of predicates on which the intervened
> run differs from the control run — on ≥ 80% of instances. Prim's must be comparable
> between the two arms (within one predicate on ≥ 80%).

If MC fails, the dissociation did not install, and any null is about the construction rather
than about the Dijkstra/Prim boundary. Report as **MANIPULATION FAILED** and stop.

### Primary

> **P.** With the two new interventions in the suite, `separation_matrix[("dijkstra","prim")]`
> contains **≥ 2 distinct interventions** at stability 0.8, on graphs of propagation depth ≥ 5.

Report which predicates carry it, not just the count.

### Regression — mandatory, this is not optional bookkeeping

Adding interventions cannot make a previously-separated pair unseparated, but it *can* change
which minimal set the search returns, and the size-2 claim at depth ≥ 7 is a published number
in `FINDINGS_PHASE_A.md` and `THEORY.md` §7.

- Re-run `run_phase_a.py --graphs 25 --nodes 24`. Report the minimal separating set before
  and after. If its size changes, say so explicitly in the findings and update both documents.
- Re-run `run_identifiability.py --graphs 12 --nodes 24`. The depth boundary (2 at depth ≥ 7,
  3 at depth 5, impossible below ~4) is a headline result and must be re-quoted post-change.

### Exploratory / confirmatory split

Develop the probe — tune δ, confirm MC installs — on **seed 0**, which is the seed every
existing symbolic result uses and therefore the one already looked at. Report P and the
regression on **seed 11**, untouched, in a single pass. Do not tune δ on seed 11.

---

## Execution order

Steps are ordered so that anything which invalidates later work fails first.

### 0. Commit the loose work first

`git status` currently shows the entire Phase F and residual-audit effort **untracked**
against a repo with one commit. This is a real risk and unrelated to E5/E6; do it before
touching any code.

Suggested split — do not squash, the second is exploratory and should be separable:

1. **Phase F**: `fpid/family.py`, `fpid/estimate.py`, `run_family_spike.py`,
   `FINDINGS_FAMILY.md`, `PREREGISTRATION_FAMILY.md`, `GATE1_DISQUALIFICATIONS.md`,
   `family_spike_log.txt`, `family_spike_results.json`, `artifacts/beta_calibration.json`,
   `artifacts/response_reference_table.json`, and the `fpid/response.py` corrections
   (they are Phase F's product — the two retracted `width`/`row_sum` claims).
2. **Residual audit**: `run_residual_audit.py`, `FINDINGS_RESIDUAL.md`,
   `residual_audit_log.txt`, `residual_audit_results.json`.
3. **Docs**: `CLAUDE.md`, `HANDOFF.md`, `README.md`, `fpid/neural.py`.

### 1. E5 — implementation

- `fpid/identify.py`: add `informative(fp, key)` and `keys_normalised(a, b, stability)`; add a
  `normalise: bool = False` parameter threaded through `distance` and `classify`. **Default
  off.** Every existing number in every findings document must remain reproducible by the
  command at the top of that document, so the new path is opt-in.
- `run_e5.py`: the negative control (NC) on symbolic references, then the primary on the nine
  checkpoints, then the secondaries. Reuse the admission / probe-depth / reference-building
  scaffolding from `run_e1_e2.py` as the other runners do. Print every cell — nine checkpoints
  × two distance measures — so nothing can be selected after the fact.
- Emit `e5_log.txt` and `e5_results.json`, matching the existing convention.

### 2. E5 — run and write up

NC first and stop on failure. Then the primary in one pass. `FINDINGS_E5.md`, with the
reproducing command at the top.

### 3. E6 — implementation

- `fpid/interventions.py`: add `_raise_tree_edge_weight` and `_dissociate_keys` plus their
  `Intervention` entries, with notes explaining the key-dissociation argument in the style of
  the existing entries. The δ helper needs `Targets` only.
- `run_e6.py`: MC first, then P, then the regression re-runs. `--seed` defaulting to 11.

### 4. E6 — run and write up

MC first and stop on failure. `FINDINGS_E6.md`. If the minimal separating set changed, update
`FINDINGS_PHASE_A.md`, `FINDINGS_IDENTIFIABILITY.md` and `THEORY.md` §7 in the same pass, and
strike the corresponding entries from `THEORY.md`'s open list.

### 5. Update the cold-start docs

`HANDOFF.md` status table, bottom line, and the "Still open" section; `README.md` findings
table; `CLAUDE.md` quick-start block. Then the write-up proper.

---

## Pre-committed interpretations of failure

Stated now so neither outcome can be reframed later.

- **E5 UNCHANGED** is the expected and useful outcome. It converts "the neural models might
  just be jumpy" from an unanswered objection into a measured and rejected one. It is not a
  disappointment and must not be described as one.
- **E5 CLOSED** would be a reversal of the project's central negative, and the correct
  response is *not* to write it up. It is to replicate it on fresh probe graphs first, and
  then re-run E3, because every neural conclusion downstream of Proposition 5 would be in
  question.
- **E6 primary fails** (still only one separating intervention, MC having passed) means the
  Dijkstra/Prim boundary genuinely rests on a single probe. That is then a stated limitation
  of the symbolic result rather than a hole to be filled, and `THEORY.md` §7 should say so
  explicitly instead of listing it as open work.
- **Neither experiment is permitted to spawn a follow-up statistic on the same data.** If
  something interesting turns up mid-run, it is recorded as exploratory in the findings
  document, as `FINDINGS_RESIDUAL.md` does, and it needs fresh data before it is anything.
