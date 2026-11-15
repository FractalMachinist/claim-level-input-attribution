# Stage 2 design notes (raw material, not a committed plan)

Written 2026-11-15 while the Stage 1 SimpleQA run was on the GPU. These are the
things we'll obviously need for Stage 2, so the GPU isn't idle waiting on design.
NOT a full implementation; the specific scaling axes stay open per RESEARCH_PLAN.md.

## What changes from Stage 1

Stage 1 regime: whole prompt is {question + N chaff}; the answer lives in the
model's weights; **chaff should score ~0** (noise floor). The proxy passes if it
keeps chaff near the floor and tracks the causal effect.

Stage 2 regime (RESEARCH_PLAN.md "the obvious failures"): the answer lives **in
the context**. Now there is a region that *should* score high — the gold answer
span — plus distractor context that should score low. The test flips from "does
the proxy correctly ignore irrelevant tokens?" to **"does the proxy correctly
FIND the one span that matters, and does that prediction match the causal
effect of perturbing it?"** If the only way to answer is to read a fact out of
context and the proxy can't detect that span, the idea fails (plan line 39).

## Why SQuAD fits (downloaded: data/squad/dev-v1.1.json, 10,570 Qs)

Each item = (context paragraph, question, gold answer + **character offset**
`answer_start`). The char offset is the critical piece: it locates the exact
token span the proxy *should* light up. So we get a free ground-truth "relevant
region" without hand-labeling — the same reason we used SimpleQA for Stage 1
(recognizable benchmark, no homemade bias to defend).

SQuAD is extractive: the answer is a literal substring of the context. That is
the cleanest possible Stage 2 case — the relevant region is unambiguous.

## Regions for the Stage 2 driver (reuse Stage 1 machinery)

The Stage 1 `build_prompt` labels tokens into regions by char-span overlap. Stage
2 needs the same mechanism with different regions:
- `gold_span`   — the answer tokens inside the context (from answer_start .. +len)
- `context`     — the rest of the paragraph (the distractor region)
- `question`    — the question text
- `template`    — chat scaffolding
(Optionally keep an N-chaff region too, to stack Stage 1's null on top of Stage 2
and watch both at once — a cheap combined test.)

The causal side (`measured_effect`) is UNCHANGED in spirit: single-token resample
under teacher forcing, per position. The new analysis is per-REGION: does
`gold_span` carry the highest measured effect AND the highest proxy, and does the
proxy rank gold_span > context on most prompts?

## Metrics we'll obviously want (per prompt, aggregated across prompts)

- **Hit rate:** is the top-k proxy token inside the gold span? (k=1, and
  span-length-normalized.) This is the headline Stage 2 number.
- **Region contrast:** proxy mean-norm on gold_span vs context; measured effect
  likewise. Does the proxy separate them, and does the causal measure agree?
- **Spearman(proxy, measured)** over context positions — the Stage 1 metric,
  now restricted to the region where it matters.
- **The negative case (plan line 43):** a question the model answers from memory
  with irrelevant context present — proxy should NOT light the context; checked
  against erasing the context. SQuAD + a closed-book SimpleQA item in the same
  context would construct this.

## The confidence/correctness question returns, but differently

Stage 1: no filter (confabulation is fine). Stage 2 is subtler — if the model
can't actually extract the answer, "gold span should score high" has no causal
teeth (perturbing a span the model never read won't move the answer). So Stage 2
likely DOES want to condition on items where the model gets the extraction right
(or at least attends to the span). Open question for the live discussion — do not
assume. Record correctness either way (SQuAD has exact-match / F1 scorers).

## Model size

If 0.6B can't do SQuAD extraction at all (plausible — it's a real comprehension
task, unlike closed-book confabulation), this is exactly the "size is the highly
likely cause of a bad result" case from RESEARCH_PLAN.md: step up (Qwen3-1.7B /
4B / 8B) rather than conclude the proxy failed. The A100-80GB has room for 8B fp32.

## Smallest first run (when the GPU frees up)

1. Filter dev set to short contexts (<~120 tokens) and short answers (1-4 tokens)
   for compact prompts, like the Stage 1 max_q_chars filter.
2. Sanity: does 0.6B even answer SQuAD extractively? Sample ~10 greedy replies.
   If near-zero correct, step the model up before spending on the full run.
3. Then the region-contrast + hit-rate run over ~50 items.
