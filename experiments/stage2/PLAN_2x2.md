# Stage 2 run plan: the correctness × confidence 2×2 (extractive / SQuAD)

Zach's call (2026-11-15), replacing an earlier pre-gate idea. This is the task spec for the Stage 2 extractive run; if context compacts, resume from here.

## The design: split, don't gate

Do NOT filter SQuAD items by whether the model answers correctly or confidently. Instead **split** every item into a 2×2 and ask whether attribution quality *differs across cells*:

|  | model CORRECT | model WRONG |
|---|---|---|
| **confident** (high reply log-prob) | cell A | cell B |
| **unconfident** | cell C | cell D |

The question the cells answer is **how attribution relates to correctness** — e.g. does the proxy concentrate on the gold span more tightly when the model is right? That is an observation about the data. It is deliberately *not* framed as "the proxy detects when the model is wrong": the proxy measures a token's counterfactual influence on the reply, not the model's internal sourcing, and model correctness is orthogonal to attribution correctness (a model can be wrong while attending to the gold span perfectly well). This technique flags *attended-to-what-it-shouldn't / ignored-what-it-should*, not model wrongness — see `RESULTS.md` → "Scope" for why.

- **Split, not gate.** All items run; we bucket afterward.
- **Record both axes per item regardless** of cell; bucketing is post-hoc, so the cell definitions can be re-cut (e.g. substring vs F1 correctness) without re-running.
- **Rebalance via model size.** If cells are unhelpfully unbalanced (e.g. 0.6B is almost never correct, so A/C are empty), that is the "size is the likely cause" trigger — step up the model (Qwen3-1.7B/4B/8B; A100-80GB fits 8B fp32) until the 2×2 has usable cells.

## Regions (Stage 2 differs from Stage 1)

In Stage 1 the chaff *should* score ~0. In Stage 2 the answer is IN the context, so there is a region that *should* score high:
- `gold_span` — the answer tokens inside the context (from SQuAD `answer_start` .. +len(answer))
- `context` — the rest of the paragraph (distractor region)
- `question` — the question text
- `template` — chat scaffolding

Headline Stage 2 metric: **does the gold span out-score the context** (the per-token proxy ratio gold_span : context), per cell. Compare the answer span only against its real competitor, the distractor context — *exclude template/scaffold tokens*, which carry outsized gradient norm (BOS / attention-sink artifact) and otherwise dominate any naive top-k. Locate the gold span by `answer_start`, and label tokens by char **overlap** (not containment), so a short answer whose boundary falls mid-token is not dropped.

## Correctness & confidence definitions

- **Correct:** model's greedy reply contains the gold answer string (SQuAD-style normalized exact match / substring; SQuAD has multiple acceptable answers — accept any). Record F1 too for borderline.
- **Confident:** teacher-forced log-prob of the model's OWN reply (the `reply_logp` covariate we already record). Bucket by median split within the run, OR a fixed threshold — decide from the distribution, report which.
- Both recorded per item regardless; bucketing is post-hoc.

## Build order (autonomous)

0. **Scale sanity FIRST** (cheap, ~10 items): does the model answer SQuAD extractively at all? Sample greedy replies on short-context items. If near-zero correct at 0.6B, step the model up before the full run — do not spend the full run on an all-wrong-cell result. Follow the failure-risk-early rule: this is the thing most likely to break the plan, so test it first.
1. Driver `stage2/run_squad.py`: reuse `lib/attribution.py` + `lib/causal.py`; new `build_squad_prompt` with the 4 regions; filter to compact items (short context + short answer) for prompt length.
2. Full run over ~50–100 items; per-item record: both axes (correct? by substring AND F1; confident? by reply log-prob), region proxy/measured means, gold:context contrast, spearman(proxy,measured) over context positions.
3. Aggregate PER CELL. Report whether cells differ, as an observation about attribution-vs-correctness.
4. Write `stage2/RESULTS.md`; update `STATUS.md` row 5.

## Reporting cadence
Report to Zach as cells land (he's AFK, long leash). Pull the model-size lever autonomously if the 2×2 is starved; note it in the report. Destroy the box when done or stalled.
