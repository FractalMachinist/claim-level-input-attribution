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
**Untested.** This run is the extractive core on compact single-paragraph contexts (≤400 chars). The scaling axes the plan names — multiple distractor facts (team/color/city, ask one), longer contexts — have not been run. The test: hold the task fixed, vary context complexity, and check whether the proxy still picks the right span and still tracks the causal effect. Includes the negative case (question answered from memory with irrelevant context present → the proxy should *not* light the context; plan line 43). This is the natural next Stage 2 run.

### A claim-vs-foil contrast objective (logit-difference) works, not only reply log-probability. (row 7)
**Untested.** Everything in Stage 1 and Stage 2 so far uses the reply log-probability objective. The logit-difference-against-a-foil objective (the other objective the plan commits to at every stage) needs hand-written foils and has not been run here. The test: for each item, write a foil answer, attribute the gold-vs-foil logit difference, and compare to the log-prob attribution. (Stage 1 already hinted the objective matters — KL was better-predicted than |ΔlogP|.)

### The method survives thinking tokens (gradients through generated reasoning). (row 8)
**Untested.** Every run so far is thinking-OFF. The plan puts some Stage 2 runs on a thinking budget and makes thinking the default from Stage 3; the open question (RESEARCH_PLAN.md) is how gradients through generated reasoning tokens are handled. Not attempted here.

## Incidental
- **0.6B is adequate for Stage 2 extraction** (82/120 substring-correct, 43/120 F1≥0.5) — the model-size lever was not needed.
- Only plain gradients, L2 norm (same proxy-scope choice as Stage 1); single seed; confidence is a median split, not a calibrated threshold.
