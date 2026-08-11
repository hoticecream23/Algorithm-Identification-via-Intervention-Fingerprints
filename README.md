# Algorithm Identification by Intervention Fingerprints

Can you tell which algorithm a neural reasoner has learned without looking at its weights —
by poking it and watching how it responds?

Delete an edge, corrupt a node's estimate mid-run, inject a value at an unreached node, then
observe how the disturbance propagates. The pattern of responses is an *intervention
fingerprint*, and it can be compared against fingerprints of reference algorithms
(Bellman-Ford, Dijkstra, Prim, SPFA, truncated Bellman-Ford) measured the same way.

 **[THEORY.md](THEORY.md)** has the definitional layer: `I,D`-relative behavioural equivalence, the minimal separating set, and five propositions with witnesses.

## Status

The symbolic contribution is complete and positive. The neural transfer question was tested
four separate ways — different training targets, hint-free supervision, three architectures
built to differ in procedure, and a held-out replication of the one lead that turned up — and
is negative on all four. A follow-up decomposition explains *why*: the fingerprint measures
neural models along dimensions that carry no algorithmic identity, and is nearly blind along
the dimensions that do. That is the current state of the project; see "What does not" below
and `FINDINGS_DECOMPOSITION.md` for the mechanism.

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
differences were genuinely active. A follow-up decomposition found why: every neural model
sits 0.30-0.40 from its nearest symbolic reference, while two *different* symbolic
algorithms sit 0.25-0.30 apart on the same measure — the label was never supported. The
interventions that carry the most algorithmic identity are the ones neural models agree on
most; the ones where neural models are distinctive carry almost no identity information
(correlation across the intervention suite: r = -0.23). See `FINDINGS_E3.md` and
`FINDINGS_DECOMPOSITION.md`.

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
| [FINDINGS_DECOMPOSITION.md](FINDINGS_DECOMPOSITION.md) | Why the neural fingerprint fails: signal and noise are orthogonal |

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
