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

## Next

1. Causal side: resample the chaff tokens, measure per-position reply log-prob
   change + per-position KL under teacher forcing; add a no-change rerun for the
   sampling-noise floor.
2. Rank-correlate proxy vs measured effect (the real Stage 1 deliverable).
3. Add grad*input and the per-swap first-order score.
4. Multiple prompts / seeds for a noise floor and a null on irrelevant tokens.
