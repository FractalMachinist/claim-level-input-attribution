# Stage 1 scaffold — first results

**Status:** proxy-side plumbing only. No causal (resample) side yet, so nothing
here is validated against a measured effect. One prompt, one question, one seed.
Read these as "the plumbing works and gives a sane-looking signal", not as a
Stage 1 conclusion.

**Setup.** Qwen3-0.6B, **fp32**, A100-80GB (instance 60694029). Prompt = N
random "chaff" tokens + the question "What is the longest bony fish?", thinking
OFF. Proxy = plain gradient of the teacher-forced reply log-prob w.r.t. input
embeddings; per-position score = L2 norm of that gradient. Regions labeled by
character offset (robust to tokenization shifts). Script:
`experiments/stage1_grad_scaffold.py`. Raw JSON: `experiments/runs/`.

## Table

```
   N pos       q_mean   c_mean    q/c  q_share  c_share     logp
   0 before     14.83     0.00    nan    0.574    0.000    -8.46
   5 after      14.82     7.26   2.04    0.449    0.157    -7.67
   5 before     18.71    13.36   1.40    0.444    0.198   -10.77
  15 after      17.19     6.35   2.71    0.381    0.301    -8.73
  15 before     24.72    11.45   2.16    0.385    0.334   -10.64
  30 after      14.99     3.17   4.73    0.369    0.346    -6.84
  30 before     11.28     3.15   3.58    0.351    0.380    -6.37
  60 after      17.55     2.47   7.11    0.315    0.392    -8.17
  60 before     12.43     2.25   5.52    0.310    0.436    -8.08
 120 after      16.39     1.23  13.34    0.299    0.406    -8.17
 120 before     12.42     1.46   8.50    0.269    0.502    -7.97
```
`q_mean`/`c_mean` = mean per-token gradient L2 norm in the question / chaff
region. `q/c` = their ratio. `*_share` = that region's share of the total
gradient norm summed over all input tokens.

## Observations (not yet interpretations backed by causal tests)

1. **Per-token, the question dominates the chaff, and the gap grows with N.**
   q/c rises monotonically 1.4 → 13.3 (before) and 2.0 → 13.3 (after) as N goes
   5 → 120. As chaff is added, each individual chaff token's gradient shrinks
   (c_mean 13.4 → 1.5) while the question's stays roughly flat (q_mean ~11–25).
   This is the direction the README hypothesis predicts.

2. **In aggregate, chaff overtakes the question by sheer count.** chaff c_share
   climbs toward/past 0.5 as N grows; the 8-token question's share falls. By
   N=120 (before), 120 chaff tokens collectively carry more summed gradient
   than the 8 question tokens.

3. **Takeaway for the proxy definition:** sum-of-norms is a poor aggregation —
   it rewards region size. Mean (or a count-normalized measure) tracks the
   epistemic picture far better here. This feeds RESEARCH_PLAN.md's open
   decision on aggregation.

4. **Position is a minor confound at this scale.** before vs after give similar
   q_mean / c_mean and the same q/c trend; no large asymmetry.

## What this is NOT

- No noise floor. No null distribution. No causal yardstick — the whole point
  of Stage 1 is proxy-vs-resample rank agreement, and the resample side is not
  built yet.
- Only L2 norm of the gradient. grad*input, sum-of-abs, IG, LRP: not done.
- Only log-prob objective. logit-difference-vs-foil: not done (needs foils).
- n=1 prompt / 1 seed. No variance estimate.

## Causal side — proxy vs measured effect (the real Stage 1 deliverable)

`experiments/stage1_causal.py`. Intervention = single-token resampling under
teacher forcing (reply R fixed); measured effect per position = mean |ΔlogP| of
R over K=8 replacement draws, and mean per-reply-token KL. Proxy = gradient L2
norm. Agreement = Spearman rank correlation. Consistency check passes: the
gradient-path baseline log-prob exactly matches the teacher-forced baseline.

```
   N pos      rho_all  rho_chaff   rho_KL  q_proxy   q_meas  c_proxy   c_meas
   5 after      0.655      0.700    0.716    14.82    10.21    7.258   0.1854
   5 before     0.499     -0.600    0.487    18.71     9.65   13.357   0.3614
  15 after      0.679      0.407    0.749    17.19    13.72    6.353   0.1499
  15 before     0.383      0.014    0.490    24.72    12.24   11.454   0.2822
  30 after      0.819      0.671    0.891    14.99    11.73    3.171   0.0926
  30 before     0.720      0.454    0.825    11.28    10.58    3.151   0.0868
  60 after      0.796      0.664    0.869    17.55    13.49    2.467   0.0727
  60 before     0.778      0.574    0.881    12.43    10.84    2.252   0.0615
 120 after      0.804      0.706    0.835    16.39    13.71    1.228   0.0335
 120 before     0.726      0.586    0.851    12.42    11.18    1.462   0.0367
```

### Main finding
The gradient proxy predicts the measured causal effect, and the agreement
**improves with N** rather than degrading: rho_all 0.72–0.82 for N≥30, up to
~0.80 at N=120. KL is a better-predicted target than |ΔlogP| (rho_KL 0.83–0.89
for N≥30) — a signal for the "two objectives" decision. Region means track
causally: the question region is both highest-proxy and highest-impact; chaff
proxy is small and chaff measured effect is tiny (→0.03 at N=120). The ranking
the method exists to produce is correct at every scale.

Caveat on magnitude: the proxy *under-states* the question/chaff gap (proxy
~3.6x, measured ~120x at N=30). It gets the order right, not the ratio — which
is what the method needs, but worth remembering before reading proxy magnitudes
as effect sizes.

### The wrinkle (unresolved, flagged honestly)
At small N (5, 15) in the **before** position the correlation collapses:
rho_all 0.38–0.50 and rho_chaff goes to **−0.60 / +0.01** — the proxy ranks
chaff tokens in the wrong order relative to their effect, *there*. The same N in
the **after** position is fine (0.70). So it is a small-N × position
interaction, not random. Candidate causes, not yet separated by the data:
- **Noise floor** — but at N=5 chaff effects are the *largest* in the table
  (c_meas 0.36), so these aren't noise-floor-tiny values; weakens this for N=5.
- **Position / early-token artifact** — chaff-first puts chaff at early
  positions (attention-sink / BOS adjacency); before is systematically worse
  than after at small N.
- **Real proxy weakness at low chaff count**, masked at high N by the easy
  question-vs-chaff separation dominating the correlation.
- **Small-sample Spearman instability** over ~5 chaff points (present, but can't
  explain a *systematic* before-worse-than-after on its own).

What distinguishes them (= the next work): more seeds (variance band),
a resample-to-resample noise floor (calibrates the floor), and the per-token
scatter at n5-before (one outlier token vs broad mis-ranking).

## What this is NOT
- Still **one prompt, one question**. No cross-prompt generalization.
- K=8 draws per position is a thin effect estimate; no variance band yet.
- Only L2 norm of the gradient. grad*input, sum-of-abs, IG, LRP: not done.
- No resample-to-resample noise floor / null distribution yet.

## Next
1. Noise floor: resample-to-resample variance (no-change-but-reseeded) per
   position → is chaff-only rho as good as possible given the floor?
2. Multiple seeds and multiple questions → variance bands, resolve the wrinkle.
3. Add grad*input and the per-swap first-order score; compare proxies.
4. logit-difference-vs-foil objective (needs hand-written foils).
