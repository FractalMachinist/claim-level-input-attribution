# Stage 2 results — extractive attribution on SQuAD (correctness × confidence 2×2)

**Status (2026-11-15):** First Stage 2 result. The gradient proxy **finds the answer span in the context** — when the model answers correctly, the gold span carries ~2× the per-token attribution of the surrounding distractor text. A sharper claim (that a *confidently wrong* model visibly draws from outside the gold span) appeared in the first cut but **weakened under a stricter correctness definition**; it is a real lead, not a confirmed effect. Details and caveats below, stated straight.

**Setup.** Qwen3-0.6B, fp32, A100-80GB. 120 compact SQuAD v1.1 dev items (context ≤400 chars, answer ≤25 chars). Each prompt = `{context}\n\nQuestion: {q}\nAnswer in as few words as possible.`, thinking OFF. Four token regions by character offset: **gold_span** (the answer inside the context, located by SQuAD `answer_start`), **context** (rest of paragraph — the distractor region), **question**, **template** (chat scaffolding). Proxy = plain-gradient L2 norm (same as Stage 1). Causal yardstick = single-token resampling under teacher forcing (same as Stage 1). Procedure: `stage2/run_squad.py`, reusing `lib/`. Raw: `stage2/runs/squad_2x2_n120.json`.

**The 2×2 (Zach's design).** We do *not* filter by correctness or confidence; we run every item and split post-hoc into correct/wrong × confident/unconfident (confidence = teacher-forced log-prob of the model's own reply, median split). The question is whether attribution quality *differs across cells*. See `stage2/PLAN_2x2.md` for the rationale: "our system works when the model is right" is weak; "the proxy shows a wrong model drew from outside the gold span" is the informative claim.

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

### What looked sharp but isn't (the honest part)
The eye-catching cell is **wrong/confident at 1.05** — the gold span stops being special exactly when the model is confidently wrong. That would be the epistemics payoff: the proxy showing a confidently-wrong model drew from *elsewhere*. Two reasons to not over-claim it:

1. **It is sensitive to how "wrong" is defined.** The strict substring match (82/120 correct) and SQuAD token-F1≥0.5 (43/120 correct) disagree substantially — substring is permissive. Re-cutting the same data with the **F1≥0.5** axis:

   | cell (F1 axis) | n | gold/ctx ratio |
   |---|---:|---:|
   | correct / confident | 31 | 2.32 |
   | wrong / confident | 29 | **1.84** |

   Under F1 the wrong/confident cell is **1.84, not 1.05** — still below correct/confident (2.32), but not the dramatic collapse the substring cut showed. The large gap was partly an artifact of which items the loose axis happened to call "wrong."

2. **Small cells.** wrong/confident is n=12 (substring) / n=29 (F1). A rank-sum of correct/conf vs wrong/conf under the substring axis gives U=430/576 (clear separation *there*), but that is the generous axis.

**Honest summary:** confidently-correct answers reliably concentrate attribution on the gold span (~2.3–2.4×, both axes). Confidently-wrong answers concentrate it *less* (1.05–1.84 depending on axis) — a real directional lead that the proxy *can* see "the answer didn't come from the answer's location," but not a confirmed effect at this sample and not robust to the correctness definition.

### Secondary observations
- **Proxy-vs-causal agreement over context positions** (Spearman) is ρ≈0.39–0.44 across all cells — lower than Stage 1's chaff regime (expected: context tokens have real, varied, correlated effects, a harder ranking than chaff-vs-question), but consistently positive and cell-independent.
- **0.6B is adequate for Stage 2 extraction** (82/120 substring-correct, 43/120 F1≥0.5) — the model-size lever was not needed. The 2×2 cells are populated, if unevenly.

## What Stage 2 has NOT done
- **One correctness axis would be better than two that disagree.** The clean next step is F1-gated cells (or exact-match) at higher N, to firm up or retire the wrong/confident lead. This is the single highest-value Stage 2 follow-up.
- Only plain gradients, L2 norm, log-prob objective (same scope choice as Stage 1).
- No distractor-fact / context-complexity scaling yet (plan's Stage 2 also wants: distractor facts, a negative "answered from memory" case). This run is the extractive core only.
- Single seed; confidence is a median split, not a calibrated threshold.

## Next
1. F1-gated (or exact-match) 2×2 at higher N → confirm or retire the confidently-wrong lead. **Highest value.**
2. The negative case: question answered from memory with irrelevant context present — proxy should not light the context (plan line 43).
3. Distractor facts (team/color/city, ask one) — does the proxy pick the right line?
