# OOD prediction — a negative result, and what it bounds

Reproduce:

```bash
python run_ood_prediction.py --seeds 6 --epochs 1500 --probe-graphs 10
```

12 MPNNs, identical architecture and round budget (12), differing only in training
distribution: six on shallow graphs (depth ~2), six on deep graphs (depth ~5-8).
Fingerprinted on shallow graphs only, then evaluated OOD on deep graphs.

## The setup worked; the prediction did not

The premise is cleanly established:

| group | val MAE (shallow) | OOD MAE (deep) |
|---|---|---|
| shallow-trained | 0.815 +/- 0.401 | **6.178** |
| deep-trained | 0.778 +/- 0.273 | **0.719** |

Validation accuracy cannot separate these groups — the distributions overlap heavily —
while OOD error differs by **8.6x**. This is exactly the Model-A/Model-B situation the
project exists to address, reproduced with real networks.

The fingerprint did not resolve it:

| predictor (all measured in-distribution) | Spearman vs OOD MAE |
|---|---|
| validation MAE | +0.364 |
| d(bellman_ford) | +0.406 |
| d(bellman_ford) - d(bf_truncated_k3) | **-0.049** |

All 12 models classify as `bellman_ford`. At n=12, +0.406 versus +0.364 is not a
meaningful difference. **The fingerprint carries no usable OOD signal in this setup.**

## A result I retracted

An earlier run with 6 probe graphs labelled 3/5 shallow-trained models as
`bf_truncated_k3` and 0/5 deep-trained ones — apparently strong support for the
hypothesis. Those labels had margins of 0.006-0.021, i.e. near-exact ties, and raising
the probe count to 10 resolved every one of them to `bellman_ford`.

The margins were flagged as the weak point at the time. They turned out to be the entire
signal. This is the third instance in this project of a result that looked correct and was
measuring nothing, and the pattern is consistent: **when the margin is at the resolution
floor, the label is noise regardless of how sensible the pattern looks.**

## Why it failed — the substantive finding

The neural analogue was mis-specified. Symbolic truncated Bellman-Ford is separable even
at depth 2 because it **halts**: the observable is remaining execution budget, and an
algorithm that has stopped cannot respond to a mid-run intervention at all.

These networks do not halt. Both groups run 12 rounds. The shallow-trained model is not
executing a *different procedure* — it is executing the same procedure with an update rule
that degrades on inputs it never saw.

**The intervention fingerprint probes execution structure, so it is blind to a difference
that lives in the update rule's domain of validity.** Same control flow, same propagation
discipline, same response to every poke — and yet 8.6x worse OOD.

## What this bounds

The claim has to be narrowed. The fingerprint should be expected to predict OOD failure
when failure arises from a *genuinely different procedure* — early halting, premature
commitment of a settled set, depth-limited propagation. It should **not** be expected to
predict failure arising from the same procedure with a poorly-generalising learned update.

That second mode is plausibly the more common one in practice, which is a real problem for
the project as originally framed and needs to be confronted rather than worked around.

Note this does not contradict the symbolic result in `FINDINGS_IDENTIFIABILITY.md`.
Truncated Bellman-Ford really is separable from Bellman-Ford at every depth. The error was
assuming a shallow-trained network is that thing. It is not.

## Diagnostic result: probe distribution was part of it, but the claim still does not survive

The same 10 models fingerprinted on both probe distributions:

| probe set | shallow-trained labels | deep-trained labels | mean margin |
|---|---|---|---|
| shallow (in-distribution) | all `bellman_ford` | all `bellman_ford` | 0.015 |
| deep (OOD distribution) | **3/5 `bf_truncated_k3`** | **0/5** | 0.067 |

On deep probes the asymmetry is clean — no deep-trained model is ever labelled truncated —
and margins are 4.5x larger, above the noise floor that invalidated the earlier result
(0.026-0.146 versus 0.006-0.021). Fisher exact one-sided p ~ 0.083: suggestive at n=10,
not significant.

Continuous predictors remain weak and inconsistent: `d(BF)@shallow` +0.467,
`d(BF)@deep` +0.285, `d(BF)-d(tr)@deep` +0.321, against validation MAE +0.261. No stable
continuous signal.

**Why this does not rescue the project as framed.** Deep probes come from the OOD
distribution. Anyone holding graphs from the target regime can evaluate accuracy on them
directly, which is simpler and strictly more informative. The value proposition was
predicting OOD failure *without* OOD data, and that is exactly the part that failed.

The one genuine residual: fingerprinting consumes **unlabelled** inputs, whereas accuracy
evaluation needs ground truth. For algorithmic reasoning that is worth nothing — ground
truth is free, just run the algorithm. It is a real distinction only in deployment settings
with target-distribution inputs and no labels, which is not this benchmark.

## Standing conclusion

- **Original claim — fingerprint similarity predicts OOD better than validation accuracy,
  measured in-distribution: not supported.** All models look identical in-distribution.
- **Narrowed claim — given unlabelled probes from the target distribution, the fingerprint
  flags the models that will fail: weakly supported** (3/5 versus 0/5, p ~ 0.083, margins
  above the resolution floor).
- The symbolic results in `FINDINGS_IDENTIFIABILITY.md` are unaffected and remain the
  strongest part of the project.

## Appendix: how the diagnostic above was framed in advance

The original null left two explanations undistinguished — either the method cannot see this
class of difference, or the shallow probe distribution could not resolve it. The decision
rule was fixed before running: separation on deep but not shallow probes would mean the
probe distribution was the limit (and would cost the "predict OOD without OOD data"
property); separation on neither would mean the bound is real.

The result was the first branch, which is why the claim is narrowed rather than abandoned —
and why the narrowed version is worth much less. Recording the rule here because it was
stated in advance; three results in this project have been retracted for being selected
after the fact.

## Follow-on

Four in-distribution diagnostics were then tested as alternatives to the fingerprint. All
also failed to beat validation accuracy. See `FINDINGS_SHALLOW_SIGNALS.md`.
