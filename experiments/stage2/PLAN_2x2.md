# Stage 2 run plan: the correctness × confidence 2×2 (extractive / SQuAD)

Zach's call (2026-11-15), replacing an earlier pre-gate idea. This is the live task spec for the autonomous GPU session; if context compacts, resume from here.

## The reframe

Do NOT filter SQuAD items by whether the model answers correctly or confidently. Instead **split** every item into a 2×2 and ask whether attribution quality *differs across cells*:

|  | model CORRECT | model WRONG |
|---|---|---|
| **confident** (high reply logit/log-prob) | cell A | cell B |
| **unconfident** | cell C | cell D |

The weak claim we are avoiding: "our system works as long as the model is right." The strong claim we are chasing: **"when a model gets a SQuAD answer wrong, the proxy shows it drew from outside the `gold_span`"** (cells B/D) — that is an epistemics finding about model behavior, not just a validation of the tool.

- **Split, not gate.** All items run; we bucket afterward.
- **Per-cell analysis may differ** — what's "relevant" isn't the same in a correct cell (gold_span should dominate) vs a wrong cell (where did it draw from instead?).
- **Rebalance via model size.** If cells are unhelpfully unbalanced (e.g. 0.6B is almost never correct, so A/C are empty), that's the "size is the likely cause" trigger — step up the model (Qwen3-1.7B/4B/8B; A100-80GB fits 8B fp32) until the 2×2 has usable cells.

## Regions (Stage 2 differs from Stage 1)

In Stage 1 the chaff *should* score ~0. In Stage 2 the answer is IN the context, so there is a region that *should* score high:
- `gold_span` — the answer tokens inside the context (from SQuAD `answer_start` .. +len(answer))
- `context` — the rest of the paragraph (distractor region)
- `question` — the question text
- `template` — chat scaffolding

Headline Stage 2 metric: **gold-span hit rate** — is the proxy's top-k token inside the gold span? Plus region contrast (proxy & measured effect on gold_span vs context), per cell.

## Correctness & confidence definitions

- **Correct:** model's greedy reply contains the gold answer string (SQuAD-style normalized exact match / substring; SQuAD has multiple acceptable answers — accept any). Record F1 too for borderline.
- **Confident:** teacher-forced log-prob of the model's OWN reply (the `reply_logp` covariate we already record). Bucket by median split within the run, OR a fixed threshold — decide from the distribution, report which.
- Both recorded per item regardless; bucketing is post-hoc.

## Build order (autonomous)

0. **Scale sanity FIRST** (cheap, ~10 items): does the model answer SQuAD extractively at all? Sample greedy replies on short-context items. If near-zero correct at 0.6B, step the model up before the full run — do not spend the full run on an all-wrong-cell result. Follow the failure-risk-early rule: this is the thing most likely to break the plan, so test it first.
1. Driver `stage2/run_squad.py`: reuse `lib/attribution.py` + `lib/causal.py`; new `build_squad_prompt` with the 4 regions; filter to compact items (short context + short answer) for prompt length.
2. Full run over ~50–100 items; per-item record: cell (correct?, confident?), region proxy/measured means, gold-span hit rate, spearman(proxy,measured) over context.
3. Aggregate PER CELL. Report whether cells differ.
4. Write `stage2/RESULTS.md`; update `STATUS.md` row 5 (and 6 if complexity varied).

## Reporting cadence
Report to Zach as cells land (he's AFK, long leash). Pull the model-size lever autonomously if the 2×2 is starved; note it in the report. Destroy the box when done or stalled.
