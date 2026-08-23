# Algorithm Identification by Intervention Fingerprints

Can you tell which algorithm a neural reasoner has learned without looking at its weights —
by poking it and watching how it responds?

Delete an edge, corrupt a node's estimate mid-run, inject a value at an unreached node, then
observe how the disturbance propagates. The pattern of responses is an *intervention
fingerprint*, and it can be compared against fingerprints of reference algorithms
(Bellman-Ford, Dijkstra, Prim, SPFA, truncated Bellman-Ford) measured the same way.

New here? Read **[HANDOFF.md](HANDOFF.md)** — it is the cold-start document and carries the
current status, the constraints, and the open threads. **[THEORY.md](THEORY.md)** has the
definitional layer: `I,D`-relative behavioural equivalence, the minimal separating set, and
five propositions with witnesses.

## Status

The symbolic contribution is complete and positive. The neural transfer question was tested
four separate ways — different training targets, hint-free supervision, three architectures
built to differ in procedure, and a held-out replication of the one lead that turned up — and
is negative on all four. A follow-up decomposition proposed a mechanism — that the fingerprint
measures neural models along dimensions carrying no algorithmic identity, and is nearly blind
along the dimensions that do — but Experiment H later found that argument **unsupported by its
own data** (`FINDINGS_H.md`). The negative results stand; that particular explanation of them
does not. See "What does not" below.

The obvious repair was then tried and also failed, which is worth knowing before anyone tries
it again. If a discrete label is the wrong thing to read off a continuous system, replace
classification with **parameter estimation** over a family containing both — damping, softness,
commitment, truncation, selection order. That family does contain all five reference algorithms
exactly, and damping and path-discount are recoverable in closed form to four decimals. It does
**not** contain the trained networks: six of nine checkpoints fit at chi-squared of order 10^4
against an in-family holdout at 0.0000 and a deliberately out-of-family control at 15.0. See
`FINDINGS_FAMILY.md`.

Two remaining objections were closed after that. Was the vacuity gap just an artefact of an
unnormalised distance measure — models that respond to everything looking "different" for
free? No: implementing the sensitivity normalisation the code had always promised but never
performed leaves 7 of 9 checkpoints vacuous under the normalised measure too
(`FINDINGS_E5.md`). Could a probe built to move Dijkstra's and Prim's selection keys
independently finally give that pair a second separating intervention? No — the attempt's own
mechanism check failed: raising an edge weight turns out to be as invisible to relaxation as
deleting one, for the same monotonicity reason, so the boundary still rests on a single probe
(`FINDINGS_E6.md`).

One question survived all of that, because every neural experiment above compares a network to a
*symbolic reference*, and Proposition 5 says that comparison is vacuous. Nothing in that argument
forbids the networks being separable **from each other** — and the discrimination claim never
needed the references at all. Asked directly, with a frozen decision rule and an exact
permutation null over every partition rather than a sampled one: also no. Intervention
fingerprints do not separate networks built to differ in procedure — chance purity on twelve
fresh seeds, and 0/8 on the one pair whose difference is learned rather than imposed by the
deployment harness. Eleven pairs of networks from *different* architectures turn out to have
bit-identical fingerprints (`FINDINGS_G.md`). The neural side is now closed from both directions.

## What holds up

**Algorithm identity is only well-defined relative to an intervention class *and* an input
distribution.** Two of nine interventions suffice to separate all five reference algorithms
on graphs of propagation depth >= 5. Below depth ~4, no subset of the suite separates all
pairs — Dijkstra and SPFA become behaviourally identical, as do Dijkstra and Prim. This
makes the natural research object not "the fingerprint" but *the minimal intervention set
that distinguishes a family of algorithms over a given input distribution*.

**Behavioural identity is not accuracy.** On shallow graphs, truncated Bellman-Ford returns
answers identical to real Bellman-Ford on 11 of 12 instances — no accuracy metric separates
them — and the fingerprint separates them anyway. On deeper graphs, that same truncated
algorithm is correct on 0 of 12. Fingerprints also recover the trained algorithm from real
networks: six MPNNs differing only by seed all identify as Bellman-Ford, and the *worst*
model of the six identifies with the largest margin.

## What does not

**The headline claim is refuted.** Fingerprint similarity does **not** predict OOD
generalization better than validation accuracy when both are measured in-distribution.
Networks differing 5-8x in OOD error are fingerprint-identical on their training
distribution. Four further in-distribution diagnostics also failed to beat validation
accuracy, one of which looked excellent before failing replication on held-out seeds.

The reason is specific and worth knowing: **these networks differ in update-rule
generalization, not in procedure.** They run the same control flow with a rule that degrades
off distribution. Intervention fingerprints probe execution structure, so they are blind to
this. The method should be expected to detect genuinely different procedures — early
halting, premature commitment, depth-limited propagation — and nothing else.

**That expectation was tested directly and also failed.** Three architectures built to
execute differently — plain relaxation, a gated model with Dijkstra-like per-node
commitment, and hard truncation — were trained hint-free on identical data and fingerprinted.
17 of 17 models labelled `bellman_ford`; the mechanism check confirmed the architectural
differences were genuinely active. A follow-up decomposition found the label was never
supported in the first place: every neural model sits 0.30-0.40 from its nearest symbolic
reference, while two *different* symbolic algorithms sit 0.25-0.30 apart on the same measure.
That vacuity result has since survived three independent attacks on the measurement —
sensitivity normalisation (E5), the reference-free reformulation (G), and a firing rule proved
to fire where intended (H2) — and is the load-bearing neural finding.

The decomposition's *other* claim, that the fingerprint's discriminative axes and its
neural-disagreement axes are orthogonal at r = -0.23, was **retracted by H1**: over nine points
that correlation has an exact permutation p of 0.5445 and a 95% interval of [-0.777, +0.510],
and under a separating-power score it is -0.004. It was used to argue that no probe redesign
could help; it cannot bear that weight. See `FINDINGS_E3.md`, `FINDINGS_G.md` and
`FINDINGS_H.md`.

## Layout

`fpid/` is the library; `run_*.py` are the experiments; `FINDINGS_*.md` are the write-ups,
each reproducible from the command at its top.

| Findings | Contents |
|---|---|
| [FINDINGS_PHASE_A.md](FINDINGS_PHASE_A.md) | Symbolic separation gate; minimal separating set |
| [FINDINGS_IDENTIFIABILITY.md](FINDINGS_IDENTIFIABILITY.md) | Identifiability vs graph depth; the truncation result |
| [FINDINGS_NOISE.md](FINDINGS_NOISE.md) | Predicate robustness to decoder noise; differential-measurement property |
| [FINDINGS_SEED_CONTROL.md](FINDINGS_SEED_CONTROL.md) | Fingerprints on real networks; identity decoupled from accuracy |
| [FINDINGS_OOD_NEGATIVE.md](FINDINGS_OOD_NEGATIVE.md) | The negative result on OOD prediction |
| [FINDINGS_SHALLOW_SIGNALS.md](FINDINGS_SHALLOW_SIGNALS.md) | Four alternative in-distribution signals; also null |
| [FINDINGS_E1_E2.md](FINDINGS_E1_E2.md) | Dijkstra-trained discrimination (blocked on MPNN capacity); hint-free regime (positive) |
| [FINDINGS_E3.md](FINDINGS_E3.md) | Architecturally distinct procedures — the fair test, and its null; held-out replication |
| [FINDINGS_DECOMPOSITION.md](FINDINGS_DECOMPOSITION.md) | Proposed mechanism for the neural null: signal and noise as orthogonal axes. **Its orthogonality claim was retracted by `FINDINGS_H.md` — read that first.** The vacuity measurement in the same document stands |
| [FINDINGS_RESIDUAL.md](FINDINGS_RESIDUAL.md) | Exploratory follow-up: what the networks do that the family cannot -- a state-dependent, sometimes amplifying path gain. Nothing claimed; the frozen confirmatory test is stated |
| [FINDINGS_FAMILY.md](FINDINGS_FAMILY.md) | Phase F: estimating parameters instead of classifying. The family contains all five algorithms exactly and the trained networks not at all. Also corrects two wrong claims in `fpid/response.py` |
| [FINDINGS_E5.md](FINDINGS_E5.md) | Sensitivity normalisation: does it close the vacuity gap? No -- 7/9 checkpoints stay vacuous under a normalised measure |
| [FINDINGS_E6.md](FINDINGS_E6.md) | Attempt to close the Dijkstra/Prim boundary with a direct key-dissociation probe; mechanism check failed informatively -- weight raises are monotonically invisible |
| [FINDINGS_G.md](FINDINGS_G.md) | The reference-free question: do fingerprints separate the networks from *each other*? Null at both stages, confirmed on 12 fresh seeds |
| [FINDINGS_H.md](FINDINGS_H.md) | Repairing the instrument: the orthogonality verdict is unsupported (r = -0.004 under separating power), and the firing rule was mis-timing every neural probe -- fixed, gap unchanged |
| [ROUTES_TO_POSITIVE.md](ROUTES_TO_POSITIVE.md) | Ranked menu of routes still open to a positive result, with mechanism, cost and a frozen decision rule where one is possible |

Requires numpy for the symbolic work; PyTorch with CUDA for anything neural.

## A note on the negative results

Several findings in this repository were retracted after failing replication, and the
retractions are recorded alongside the results rather than removed. The recurring cause was
selecting the best of several candidate signals and reporting it before a held-out check —
at n ~ 10 with four candidates, the best one looks strong by chance often enough to be the
default outcome. A mechanistic argument for why a signal *should* work turned out not to
protect against this either.

The methodological rules that came out of it are in
[HANDOFF.md](HANDOFF.md#methodological-rules-learned-the-hard-way), and they are probably
more transferable than the positive results.
