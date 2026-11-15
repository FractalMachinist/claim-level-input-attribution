# Stage 2 results — extractive attribution on SQuAD (correctness × confidence 2×2)

**Status (2026-11-15):** First Stage 2 result. The gradient proxy **finds the answer span in the context** — when the model answers correctly, the gold span carries ~2× the per-token attribution of the surrounding distractor text. A separate, cooler-than-expected observation: **how concentrated the attribution is on the gold span covaries with whether the model got the answer right.** We are careful below about what that covariation does and does not mean — it is an observation about the data, not a claim that this technique detects model wrongness (it does not; see "Scope").

**Setup.** Qwen3-0.6B, fp32, A100-80GB. 120 compact SQuAD v1.1 dev items (context ≤400 chars, answer ≤25 chars). Each prompt = `{context}\n\nQuestion: {q}\nAnswer in as few words as possible.`, thinking OFF. Four token regions by character offset: **gold_span** (the answer inside the context, located by SQuAD `answer_start`), **context** (rest of paragraph — the distractor region), **question**, **template** (chat scaffolding). Proxy = plain-gradient L2 norm (same as Stage 1). Causal yardstick = single-token resampling under teacher forcing (same as Stage 1). Procedure: `stage2/run_squad.py`, reusing `lib/`. Raw: `stage2/runs/squad_2x2_n120.json`.

**The 2×2 (Zach's design).** We do *not* filter by correctness or confidence; we run every item and split post-hoc into correct/wrong × confident/unconfident (confidence = teacher-forced log-prob of the model's own reply, median split). The question is whether attribution quality *differs across cells* — which, if it does, is an observation about how attribution relates to correctness. (The original framing hoped the wrong cells would show the proxy "detecting" wrongness; that framing was retired — see "Scope.")

## A measurement artifact we caught first (don't skip this)

The pre-registered headline metric was "is the single highest-proxy token inside the gold span?" It came back **~0% across all cells** — which looks like total failure. It is not. The top-1 token is a **template/chat-scaffold token on 41 of 54 items**: those tokens carry outsized gradient norm (a BOS / attention-sink artifact), so the answer span can never win a naive top-1. The fix is to compare the gold span against its *real* competitor — the distractor context — excluding template. All metrics below are template-excluded. (The naive hit rate is retained in the JSON as `naive_top1_hit_rate` for the record.)

A second fix: span labeling now uses token/char **overlap**, not containment, so short answers whose boundary falls mid-token are no longer dropped (6/60 lost in the first run → 0/120 lost here).

## Result

Per-token proxy ratio **gold_span : context** (median across items); `gold>ctx` = fraction of items where the gold span's mean proxy exceeds the context's.

| cell | n | gold/ctx ratio | gold>ctx |
|---|---:|---:|---:|
| correct / confident | 48 | **2.44** | 83% |
| correct / unconfident | 34 | 1.61 | 85% |
| wrong / confident | 12 | **1.05** | 50% |
| wrong / unconfident | 26 | 1.53 | 81% |

(Correctness here = strict substring match of a gold answer in the reply.)

### What is solid
**The proxy finds the answer span.** When the model answers correctly, the gold span carries ~2× the per-token attribution of the surrounding context (2.44 confident, 1.61 unconfident; gold out-scores context on 83–85% of items). This is the core Stage 2 claim — attribution works when the answer lives *in* the context, not just when irrelevant filler should be ignored (Stage 1). It is the in-context analogue of Stage 1's "question beats chaff."

### The interesting observation: attribution concentration covaries with correctness
Across the 2×2, the gold/context ratio is higher when the model answered correctly and lower when it did not — correct cells sit around 2.3–2.4, wrong cells lower. The size of the gap depends on how "wrong" is defined (the strict substring axis, 82/120 correct, and token-F1≥0.5, 43/120 correct, disagree substantially — substring is permissive):

| cell | n (substring) | gold/ctx | n (F1≥0.5) | gold/ctx |
|---|---:|---:|---:|---:|
| correct / confident | 48 | 2.44 | 31 | 2.32 |
| wrong / confident | 12 | 1.05 | 29 | 1.84 |

The substring cut makes the wrong/confident cell look dramatic (1.05); the F1 cut softens it (1.84, still below 2.32). Either way the direction is the same: **more concentrated gold-span attribution goes with being right.** This is a real and genuinely interesting covariation — worth understanding — and nothing more than that is claimed here (small cells: wrong/confident is n=12–29).

### Scope: this is NOT a wrongness detector (why there is no "detects confident failure" claim)
An earlier draft promoted that covariation to an assumption — "the proxy detects when a confidently-wrong model drew from outside the gold span." That was a category error and has been removed. The reasons, which bound what *any* finding in these cells can mean:

- **The proxy measures counterfactual influence, not internal sourcing.** A low gold-span score means "perturbing the answer's location wouldn't move the reply much" — it does **not** establish that the model "drew from elsewhere." Low influence is equally consistent with: the model consulted the span but another computation vetoed it; it answered from parametric memory *and* the context redundantly, so no single location is load-bearing; or it attended correctly and failed downstream.
- **Model correctness is orthogonal to attribution correctness.** The system's job (README) is to flag when the model attended to what it *shouldn't* or ignored what it *should*. A confidently-wrong answer is not in that target set: a model can be wrong for reasons wholly outside our scope (bad recall, a reasoning slip) while attending perfectly well to the gold span. This technique is **not** the right instrument for flagging input→output disconnects like failed recall or misinterpretation, and does not claim to be.
- Consistent with this, the gold span still out-scores context even in the wrong cells (ratio > 1 under F1) — i.e. wrong answers do not stop attending to the answer's location. That is what the ensemble view predicts and what makes "detects wrongness" the wrong claim.

So the covariation is logged as an **observation about the data**, not as a capability of the tool.

### Secondary observations
- **Proxy-vs-causal agreement over context positions** (Spearman) is ρ≈0.39–0.44 across all cells — lower than Stage 1's chaff regime (expected: context tokens have real, varied, correlated effects, a harder ranking than chaff-vs-question), but consistently positive and cell-independent.
- **0.6B is adequate for Stage 2 extraction** (82/120 substring-correct, 43/120 F1≥0.5) — the model-size lever was not needed. The 2×2 cells are populated, if unevenly.

## What Stage 2 has NOT done
- Only plain gradients, L2 norm, log-prob objective (same scope choice as Stage 1).
- No distractor-fact / context-complexity scaling yet (plan's Stage 2 also wants: distractor facts, a negative "answered from memory" case). This run is the extractive core only.
- Single seed; confidence is a median split, not a calibrated threshold.

## Next
1. The negative case: question answered from memory with irrelevant context present — proxy should not light the context (plan line 43). This is squarely in scope (attended-to-what-it-shouldn't) and the natural Stage 2 follow-up.
2. Distractor facts (team/color/city, ask one) — does the proxy pick the right line among several candidate spans?
3. The correctness×attribution covariation is a side-observation worth a cleaner look *if* it becomes relevant, but it is not on the critical path — detecting model wrongness is out of scope (see "Scope").
