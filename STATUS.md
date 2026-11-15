# Status

Where the project stands, framed as the **ideal user flow** we're building toward and the **assumptions that flow rests on**. Each assumption names the experiment meant to settle it. If the whole table reads `supported`, the flow is demonstrated end to end — i.e. we're done building. So this table is both a status board and a definition of done.

See `README.md` for the idea and terms (Request / Claims / Dressing / Foil), `RESEARCH_PLAN.md` for the staged experiment sequence, and `experiments/stage1/RESULTS.md` for the Stage 1 findings behind the states below.

---

## Ideal user flow (DRAFT — Zach to revise; synthesized from README)

> This section is the product vision and is the author's to own. The draft below is assembled from the README so the table has something to hang on; replace or correct freely.

A user has relied on an LLM for a consequential answer and wants to know what actually shaped it. They hand the system **one response and its full context** — no hand-built matched pairs, no re-running the model a hundred times themselves.

1. **The system annotates** the context and response into Request, Claims, and Dressing — it finds the load-bearing parts of the answer on its own.
2. **For each claim, it takes foils** — plausible alternative versions of that claim — so "what influenced this" has a concrete contrast to measure against.
3. **It attributes** each claim (vs. its foils) back to the input tokens, using cheap white-box gradient scores rather than hundreds of ablation passes.
4. **It reports** in plain terms — the README's target form is *"The model said X. This depended on A and B. It did not appear to take C into account."* — as text or as annotation around the reply.

The payoff is the Joe/Jane case from the README: two users get materially different answers, and the system shows the email domain drove the difference when it shouldn't have — from a single response each, cheaply enough to run routinely.

---

## Assumptions this flow rests on

States: **supported** (an experiment has demonstrated it) · **partial** (some evidence, not yet conclusive) · **planned** (experiment identified, not run).

| # | Assumption the flow needs | Experiment that tests it | State |
|---|---|---|---|
| 1 | A local gradient score is a usable proxy for a token's causal influence on a claim, when nothing in the prompt *should* matter (irrelevant filler scores near the floor; proxy tracks the measured effect). | `stage1/run_causal.py`, `stage1/run_simpleqa.py` | **supported** (ρ up to 0.79 across 50 SimpleQA prompts; 100% positive at N≥30) |
| 2 | That proxy-vs-causal agreement holds across many prompts, not one cherry-picked example. | `stage1/run_simpleqa.py` (50 SimpleQA Qs, no filter) | **supported** |
| 3 | There is a measurable noise floor / null distribution for attribution on irrelevant tokens, so the report can say a score is *notable*. | `stage1/run_noise_floor.py` | **partial** (floor characterized on single prompt; not yet across the benchmark) |
| 4 | The proxy works regardless of whether the model's answer is correct (confident confabulation included). | `stage1/run_simpleqa.py` (no-filter design) | **supported** |
| 5 | The proxy *finds the relevant span* when the answer lives in the context — not just ignores irrelevant filler. | Stage 2: extractive (SQuAD); `stage2/DESIGN_NOTES.md` | **planned** (data + design staged) |
| 6 | The proxy stays predictive as context complexity scales (distractor facts, longer context). | Stage 2 | **planned** |
| 7 | A claim-vs-foil contrast objective (logit-difference) works, not only reply log-probability. | Stage 1/2: foil objective (needs hand-written foils) | **planned** |
| 8 | The method survives thinking tokens (gradients through generated reasoning). | Stage 2 onward (thinking budget) | **planned** |
| 9 | The method holds under *intentionally* hard-to-reason-about relevance (two people named Mary, scattered formatting). | Stage 3 | **planned** |
| 10 | The annotation step (Request/Claims/Dressing) can be produced, not just assumed. | not yet designed | **planned** |
| 11 | Foils can be generated at useful quality (LLM / user / template). | not yet designed | **planned** |
| 12 | The analysis+report step turns many token attributions into a faithful, readable "depended on A, not C". | Stage 4 analysis step; `RESEARCH_PLAN.md` | **planned** (explicitly unresolved) |
| 13 | It beats or complements the black-box baseline (ContextCite) at matched granularity. | comparison vs ContextCite | **planned** |

### How to read the state column
Rows 1–4 are the Stage 1 core, and they're in. Row 5 (extractive) is the next GPU run and the real test of whether "attribution" means "finds what matters," not just "ignores noise." Rows 7–9 are the plan's deliberately-hard stages. Rows 10–13 are the parts of the full flow we have *not* started — annotation, foil generation, the analysis/report synthesis, and the baseline comparison — and they're where the flow is still a sketch, not a demonstration.
