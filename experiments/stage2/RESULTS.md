# Stage 2 results — extractive attribution on SQuAD (correctness × confidence 2×2)

**Status (2026-11-15):** First Stage 2 result, and the core assumption (row 5) is **supported on both the proxy and the causal side**: the answer tokens causally dominate the reply (perturbing them moves it ~10× more than perturbing the rest of the context, ~18× on items the model gets right), and the gradient proxy predicts that (proxy-vs-causal ρ≈0.42, positive on 96% of items). The proxy orders correctly but understates the magnitude — same as Stage 1. A logged side-observation: attribution concentration covaries with correctness; this is data about the model, *not* a claim that the technique detects wrongness (model correctness is orthogonal to attribution correctness — the "Results, by assumption" section and `PLAN_2x2.md` say why).

**Setup.** Qwen3-0.6B, fp32, A100-80GB. 120 compact SQuAD v1.1 dev items (context ≤400 chars, answer ≤25 chars). Each prompt = `{context}\n\nQuestion: {q}\nAnswer in as few words as possible.`, thinking OFF. Four token regions by character offset: **gold_span** (the answer inside the context, located by SQuAD `answer_start`), **context** (rest of paragraph — the distractor region), **question**, **template** (chat scaffolding). Proxy = plain-gradient L2 norm (same as Stage 1). Causal yardstick = single-token resampling under teacher forcing (same as Stage 1). Procedure: `stage2/run_squad.py`, reusing `lib/`. Raw: `stage2/runs/squad_2x2_n120.json`.

**The 2×2 (Zach's design).** We do *not* filter by correctness or confidence; we run every item and split post-hoc into correct/wrong × confident/unconfident (confidence = teacher-forced log-prob of the model's own reply, median split). The question is whether attribution quality *differs across cells* — which, if it does, is an observation about how attribution relates to correctness, not a wrongness detector (the original framing hoped otherwise; retired, per row 5 point 3 and `PLAN_2x2.md`).

## Results, by assumption

This section has one heading per assumption the STATUS.md table assigns to Stage 2, taken from the table verbatim, whether or not we have tested it. Headings with `Untested` are things Stage 2 is meant to answer and has not — they are here on purpose, so the doc states every question it should answer and an unanswered one is visible rather than silently absent.

### The proxy *finds the relevant span* when the answer lives in the context — not just ignores irrelevant filler. (row 5)
**Supported.** This is the core Stage 2 result, and it is confirmed on *both* sides — the proxy and the causal yardstick — not the proxy alone.

Per-token attribution on the **gold_span** vs the **context** (the distractor region), median ratio across 120 items. The causal column is the measured single-token-resampling effect (same intervention as Stage 1), not the gradient proxy:

| | proxy gold/ctx | **causal gold/ctx** | gold>ctx (proxy / causal) |
|---|---:|---:|---:|
| all items | 1.70 | **10.3** | 96/120 · 99/120 |
| correct (substring) | 1.89 | **18.3** | — |
| wrong (substring) | 1.42 | 1.37 | — |

Three things, in order of importance:

1. **The causal measurement independently says the gold span matters most.** Perturbing the answer tokens moves the reply ~10× more than perturbing the rest of the context (18× on items the model gets right). The proxy says gold>context on 96/120 items; the causal measurement says so on 99/120 — they agree. And **Spearman(proxy, causal effect) over context positions is ρ≈0.42, positive on 96% of items.** So this is not "the gradient score is high on the answer" (a statement about the proxy describing itself); it is "**the answer tokens causally dominate, and the proxy predicts that.**" That is what makes row 5 *supported* rather than merely suggestive.

2. **The proxy gets the ranking right but understates the magnitude** — proxy ratio ~1.7–1.9 where the causal ratio is ~10–18. This is the same behavior seen in Stage 1 (proxy orders correctly, compresses the gap). Read proxy magnitudes as a ranking, not an effect size.

3. **Attribution concentration covaries with correctness** — and the causal side shows this far more sharply than the proxy. Causal gold/ctx is ~18× when the model is right vs ~1.4× when wrong (substring axis; ~22× vs ~3.9× under an F1≥0.5 axis — the wrong-cell magnitude is axis-dependent, the direction is not). This is logged as an **observation about the data**, not as a capability. It is explicitly *not* "the proxy detects when the model is wrong": the measurement is counterfactual influence, not the model's internal sourcing, and a ~1× ratio in the wrong cell is consistent with several internal stories (a consulted-but-vetoed span, redundant parametric+context support, or a correct read that failed downstream). Model correctness is orthogonal to attribution correctness; this technique flags attended-shouldn't / ignored-should, not model wrongness. (This is why the table has no "detects wrongness" row — see `PLAN_2x2.md`.)

*Method note — a metric artifact caught first.* The pre-registered "is the single top proxy token in the gold span?" read ~0% because template/scaffold tokens carry outsized gradient norm (BOS / attention-sink) and dominate any naive top-1 (top token was template on 41/54 items). The fix, used throughout above: compare the gold span only against its real competitor, the context, excluding template. Span labeling uses char **overlap** not containment, so short answers whose boundary falls mid-token are not dropped (0/120 lost here vs 6/60 before).

### The proxy stays predictive as context complexity scales (distractor facts, longer context). (row 6)
**Partial** — the negative case is tested and passes decisively; the distractor-fact and longer-context scaling axes are still untested.

*Negative case (tested, `stage2/run_negative.py`, 80 items).* The direct test of the product claim "flag when the model attended to what it *shouldn't*": give the model a closed-book SimpleQA question it answers from its own weights, with an **irrelevant** context paragraph stapled in front. The proxy should rank that context *below* the question, and the causal measurement should confirm the context barely matters. Two conditions, to see whether topical surface-similarity fools it: **unrelated** context vs **same-topic** context (fluent prose from the question's own broad topic).

| condition | n | proxy ctx/question | causal ctx/question | proxy ranks ctx < question |
|---|---:|---:|---:|---:|
| unrelated | 46 | 0.19 | 0.006 | 46/46 |
| same-topic | 34 | 0.15 | 0.008 | 34/34 |

The proxy ranks the irrelevant context below the question on **80/80 items** (no item where irrelevant context out-scores the question; worst case is a single same-topic *tie*). The causal measurement agrees on 80/80 — perturbing the irrelevant context moves the reply ~0.6–0.8% as much as perturbing the question. And **same-topic context does not fool the proxy** — if anything it scores *lower* than unrelated (0.15 vs 0.19), so surface "looks-relevant" prose is not mistaken for relevance. This is a harder null than Stage 1's random chaff (fluent text, not noise) and the proxy passes it. Note the attribution is not zero on the context (ctx/q ≈ 0.15–0.19, context share of total proxy ≈ 0.26) — fluent prose draws *some* gradient — but it is correctly ranked well below the question, and the causal side confirms that non-zero proxy over-states the context's true ~0 effect (the same proxy-overstates-small-effects caution as Stage 1).

*Still untested:* multiple distractor facts (several candidate spans, ask about one — does the proxy pick the right line?) and longer / multi-paragraph contexts. The test: hold the task fixed, scale context complexity, check the proxy still finds the right span and tracks the causal effect.

### A claim-vs-foil contrast objective (logit-difference) works, not only reply log-probability. (row 7)
**Supported.** `stage1/run_foil_objective.py`, 40 closed-book SimpleQA questions + 30 chaff. For each item we run two attributions on the same inputs: the usual reply-log-probability objective, and the **logit-difference** objective — the gradient of `logit(real answer) − logit(foil)` at the first answer position, where the foil is the first token of a different SimpleQA answer of the *same answer_type* (Person/Place/Date/Number), an automatic same-register alternative (hand-written foils are the eventual refinement; this scales).

| | median |
|---|---:|
| agreement between the two objectives (Spearman on token ranking) | **0.84** |
| log-prob objective: question/chaff mean | 2.16 |
| foil objective: question/chaff mean | 1.60 |
| log-prob objective vs causal effect (Spearman) | 0.59 |
| foil objective vs causal effect (Spearman) | 0.48 |

The two objectives **agree strongly** (ρ=0.84) and both keep chaff at the same share of attribution. The foil objective is slightly weaker on the question/chaff separation (1.60 vs 2.16) and on predicting the causal effect (0.48 vs 0.59) — expected, since a single-token logit contrast is a noisier signal than the whole reply's log-prob. But it behaves correctly: the plan's second objective works and tracks the first. (Consistent with Stage 1's hint that the objective matters — there KL beat |ΔlogP|.) The foil objective is also the one the README's step 3 literally describes (claim vs foil), so having it validated matters for the eventual product, not just for completeness.

### The method survives thinking tokens (gradients through generated reasoning). (row 8)
**Supported (first cut).** `stage2/run_thinking.py`, 30 SimpleQA + 30 chaff, run twice per question — thinking OFF and thinking ON — attributing the **final answer** (tokens after `</think>`) in both. 28/30 produced a real reasoning block (median 416 reasoning tokens before the answer); the answer is cleanly isolated after `</think>` (median 34 tokens).

| | q/chaff mean | question > chaff | chaff share | proxy-vs-causal ρ |
|---|---:|---:|---:|---:|
| thinking OFF | 2.05 | 28/30 | 0.30 | 0.57 |
| thinking ON | 1.60 | 29/30 | 0.24 | 0.65 |

Attribution survives a generated reasoning prefix: the question still dominates the chaff (on 29/30 items, *more* than thinking-off's 28/30), chaff stays low, and the proxy tracks the causal effect at least as well (ρ=0.65 vs 0.57). The question/chaff ratio drops somewhat (1.60 vs 2.05) — expected, because with reasoning present the final answer draws partly on the generated reasoning tokens, diluting the input's share — but the region split does not break.

*Scope of this cut:* we attribute the **final answer only**. The deeper question the plan raises — attributing *through* the generated think tokens (gradients that flow into model-produced reasoning, not input tokens) — is **not** done here. This establishes that a reasoning prefix doesn't break input attribution; it does not yet attribute the reasoning itself.

*(Enabling this run also added batched generation to `lib/` — verified byte-identical to the serial path — which is a speed change only. An earlier thinking run with a 160-token cap was discarded: it truncated every reply before `</think>` closed, so it attributed mid-reasoning tokens, not answers. The cap is now 2048 with per-sequence early-exit.)*

## Incidental
- **0.6B is adequate for Stage 2 extraction** (82/120 substring-correct, 43/120 F1≥0.5) — the model-size lever was not needed.
- Only plain gradients, L2 norm (same proxy-scope choice as Stage 1); single seed; confidence is a median split, not a calibrated threshold.
