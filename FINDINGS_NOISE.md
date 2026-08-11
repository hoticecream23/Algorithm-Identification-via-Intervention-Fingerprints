# Noise calibration — can the predicates survive a neural decoder?

Reproduce:

```bash
python run_noise_robustness.py --graphs 10 --nodes 20
python run_noise_robustness.py --graphs 8 --nodes 18 --fixed-tol 1e-6
```

Phase A predicates use exact equality, which is free for a symbolic algorithm and
impossible for a network. This tests the predicate machinery against noisy copies of the
reference algorithms, where ground truth is known — no training required.

## The central design result

**The fingerprint is a differential measurement, so it is invariant to any error that is
common-mode across rollouts.** Every predicate compares the intervened run to the control
run. An error term that lands identically on both cancels exactly, whatever its size.

This is why two of the three noise axes turned out to be nearly unbreakable, and it is a
real property rather than an artifact:

- It is **why the fingerprint can complement validation accuracy**. It measures the
  response operator, not the values, so it is close to orthogonal to accuracy by
  construction. A model with badly-scaled outputs but correct response structure
  fingerprints as correct — and a model with excellent accuracy but the wrong response
  structure does not.
- It is also the **limitation to state honestly**: a fingerprint cannot see value error,
  so it is not a quality metric and must never be reported as one.

Consequence for the noise model: only *input-dependent* error is a real test. A trained
network is deterministic — same input, same output — so its error cancels between rollouts
wherever the computation agrees, and differs only where the intervention actually changed
something. Freshly-resampled per-rollout noise models an ensemble, which is not the object
being fingerprinted.

## Three noise axes

| axis | what it perturbs | common-mode? | verdict |
|---|---|---|---|
| `readout` | observation only, computation exact | yes | cancels |
| `execution` | the state itself, errors compound | yes (shared seed) | cancels |
| `state-dependent` | error is a function of each node's own estimate | **yes, unexpectedly** | cancels; vacuous |
| `input-dependent` | error is a function of the whole state and graph | no | the faithful axis; the only one that degrades |

With tolerance tracking sigma, the first three hold 100% identification to sigma = 2.0
against a weight scale of 1–10. That number is not a robustness result — it is three
different ways of measuring nothing.

**Two successive versions of "the faithful axis" turned out to be vacuous**, which is worth
recording because the failure was not obvious either time:

- `state-dependent` is bounded by sigma while `tol = 4 sigma`, so the tolerance exceeds the
  error by construction.
- Worse, with `--fixed-tol 1e-6` removing that excuse, it produced a mean margin of
  **exactly 0.146 at every sigma, identical to sigma = 0**. Not "robust" — literally no
  effect. Making each node's error a function of that node's own estimate means a node whose
  value the intervention did not change keeps precisely the same error, so it cancels too.

The giveaway both times was a metric that did not move *at all*. A real effect that happens
to be small still jitters the margin; an effect that is identically zero to three decimals
is a bug in the experiment, not a finding about the method.

`input-dependent` keys the perturbation on a checksum of the entire state and graph, so any
intervention shifts the error everywhere — including at nodes whose true value is untouched.
That is how a trained network actually behaves, and it is the only axis that can degrade the
fingerprint.

## The actual robustness result

`--axis input-dependent --fixed-tol 0.05`, against an edge-weight scale of 1–10:

| sigma | accuracy | mean margin |
|---|---|---|
| 0.00 | 100% | 0.146 |
| 0.05 | 100% | 0.132 |
| 0.10 | 100% | 0.129 |
| 0.25 | 100% | 0.084 |
| 0.50 | 100% | 0.085 |
| 1.00 | 100% | 0.084 |
| 2.00 | 100% | 0.077 |

The margin now *moves* — monotonically, roughly halving — which is what a real effect looks
like, in contrast to the three flat sweeps that preceded it. Identification itself survives
input-dependent error up to sigma = 2, i.e. roughly twice the mean edge weight.

Read it as: the fingerprint holds under substantial model error but with steadily falling
confidence, and the **margin, not the accuracy, is the quantity that degrades gracefully**.
That makes margin the thing to report per-model in Phase C — an accuracy of 100% here would
otherwise hide the fact that confidence had halved.

Extrapolating the trend, the method should start misclassifying somewhere past sigma ~ 4
(half the maximum edge weight). Worth confirming rather than assuming, if a Phase C model
turns out to be that noisy.

## The trap: never tie the tolerance to the noise amplitude

Setting `tol = k * sigma` makes any perturbation *bounded* by sigma invisible by
construction, for every sigma. The sweep then produces a flat 100% that looks like a
robustness result and is actually a tautology. This caught three of the four axes here, and
it is easy to miss because the sweep is varying sigma and therefore *looks* like it is
testing something.

The tolerance has to be anchored to an independent scale — the magnitude of real value gaps,
i.e. the edge-weight scale — not to the error it is supposed to absorb. Diagnostic: if a
sweep's margin is constant to three decimals across a 100x range of sigma, the experiment is
measuring nothing.

For Phase C this means calibrating `tol` from the spread of genuine value differences the
model produces, then checking separately that the model's error is below it — two
measurements, not one.

## Mis-calibrated tolerance (`--fixed-tol 1e-6`)

In Phase C the tolerance is an *estimate* of an unknown noise floor, so the mis-calibrated
case is the one that matters:

| axis | accuracy vs sigma |
|---|---|
| readout | collapses to 20–40% at any sigma > 0, flat |
| execution | 80–100% throughout |
| state-dependent | 100% throughout, zero effect (see above) |

The readout collapse is flat in sigma, which is the same signature as bug 4 below: with
`tol` at 1e-6 every round registers a change, the settle round pins to the end of the
budget, and the firing time is wrong regardless of noise scale. **Tolerance calibration is
not a refinement — an under-estimated tolerance destroys identification outright.**

## Bugs found, each of which produced confident-looking wrong numbers

These are worth reading before extending the harness; every one of them survived a run that
looked plausible.

1. **Global stability intersection destroyed the discriminative predicates.** Restricting
   comparison to predicates stable across *every* reference throws away exactly the probe
   that separates one pair — Dijkstra and SPFA lost their only separating predicate,
   collapsed to identical vectors, and were classified by tie-break. Symptom: 80% accuracy
   at sigma = 0, where it must be 100%. Fix: apply stability **pairwise**.

2. **Tolerance mismatch between model and reference.** References were computed at
   `tol=1e-6` and noisy models at `tol=4 sigma`. Tolerance is part of the measurement
   apparatus — it decides which differences count and it shifts the firing round — so the
   two sides are not comparable unless measured identically. Fix: recompute references at
   the model's tolerance.

3. **Size-dependent RNG draws desynchronised the noise streams.** `rng.normal(size=finite.sum())`
   diverges the moment two rollouts differ in how many nodes are reachable, which silently
   converted common-mode noise into independent noise and made every predicate read
   sampling noise. Fix: always draw full width and mask.

4. **A second, tolerance-blind copy of `settle_round`.** `interventions.settle_round` used
   exact equality while `fingerprint.py` had its own tolerance-aware version, and
   `firing_round` called the exact one. Any sigma > 0 then made every round look changed,
   pinning the firing round to `budget // 2` regardless of noise scale. Symptom: accuracy
   collapse that was **flat in sigma** — the giveaway that a threshold was being ignored
   rather than exceeded. Fix: one shared `d_equal`, tolerance threaded through timing.

5. **Absolute thresholds ignore the multiple-comparison problem.** `settle_round` scans one
   comparison per node per round, so noise crosses any per-comparison threshold somewhere in
   nearly every run, dragging the settle round to the end of the budget. Early-terminating
   algorithms are hit hardest — truncated Bellman-Ford really settles at round 3, so one
   spurious late change moved its firing time by an order of magnitude and its fingerprint
   stopped resembling itself. Since truncated-BF is precisely the shortcut model the project
   exists to detect, this is a failure mode aimed at the most important case. Fix: threshold
   carries a `sqrt(2 ln N)` correction, plus a relative term.

   The relative term alone is insufficient: an algorithm whose updates are mostly
   INF -> finite transitions has almost no finite-to-finite deltas, so the relative scale
   collapses to zero and the threshold degenerates back to the absolute one.

## Diagnostic technique worth reusing

When accuracy was collapsing, the decisive move was not more reasoning but dumping *which
predicates differed* between the noisy fingerprint and its own clean reference. That showed
every response suppressed to "no response at all" — which pointed straight at intervention
timing rather than at the predicates, and took one script.

The second useful signal was **flat-in-sigma degradation**. A statistical problem scales
with the noise; a categorical one does not. Flatness meant exact equality was hiding
somewhere.

## For Phase C

- Calibrate `tol` from repeated control rollouts, and measure the symbolic references at
  that same tolerance.
- Expect much of the discriminative power to live in reachability structure (INF vs finite)
  and predecessor pointers, which are categorical and immune to additive value error. CLRS
  decoders emit pointers via argmax, so this should transfer — but verify rather than assume.
- Do not report fingerprint distance as a quality measure. It is deliberately blind to
  value error.
