# Status

Where the project stands, framed as the **ideal user flow** we're building toward and the **assumptions that flow rests on**. Each assumption names the experiment meant to settle it. If the whole table reads `supported`, the flow is demonstrated end to end — i.e. we're done building. So this table is both a status board and a definition of done.

See `README.md` for the idea and terms (Request / Claims / Dressing / Foil), `RESEARCH_PLAN.md` for the staged experiment sequence, and `experiments/stage1/RESULTS.md` for the Stage 1 findings behind the states below.

---

## Ideal user flow

A user has relied on an LLM for a consequential answer and wants to know what actually shaped it. The system gives them that, from the one exchange in full — no hand-built matched pairs, no re-running the model a hundred times.

Actors below are deliberately not all "the system": the README leaves open who annotates, where foils come from, and how the analysis works, and the flow keeps those open rather than assuming the most-automated answer.

1. **Annotate.** The user marks the parts of the answer they find crucial (ideally like highlighting lines in a PDF or HTML doc), splitting out their Request, the LLM's most important Claims, and the persuasive Dressing around those claims. An LLM could do this split for the user, help with it, or add to it — a model may be better than a reader at spotting dressing.
2. **Take foils.** The user and the system can both contribute alternatives — responses *or* dressings — that would have mattered to the user had they been produced. A changed probability, a different recommendation, or a shift in tone can each matter, so we generate foils that are substantially different from the original.
3. **Attribute.** For each claim (against its foils), attribute the model's choice back to the input tokens, using a cheap proxy — currently white-box gradient scores — instead of hundreds of ablation passes.
4. **Analyze.** Turn the proxy scores into claims worth reporting, by a process not yet pinned down — for example a set of causal alterations that strongly support the attributions before they're shown. This step is the project's main open problem.
5. **Report.** Say it in plain terms — the README's example form is *"The model said X. This depended on A and B. It did not appear to take C into account."* — as text or as annotation around the reply.

The payoff is the Joe/Jane case from the README: two users get materially different answers and wouldn't have known; the system shows the email domain drove the difference when it shouldn't have — from a single response each, cheaply enough to run routinely.

---

## Assumptions this flow rests on

States: **supported** (an experiment has demonstrated it) · **partial** (some evidence, not yet conclusive) · **planned** (experiment identified, not run).

| # | Assumption the flow needs | Experiment that tests it | State |
|---|---|---|---|
| 1 | A local gradient score is a usable proxy for a token's causal influence on a claim, when nothing in the prompt *should* matter (irrelevant filler scores near the floor; proxy tracks the measured effect). | `stage1/run_causal.py`, `stage1/run_simpleqa.py` | **supported** (ρ up to 0.79 across 50 SimpleQA prompts; 100% positive at N≥30) |
| 2 | That proxy-vs-causal agreement holds across many prompts, not one cherry-picked example. | `stage1/run_simpleqa.py` (50 SimpleQA Qs, no filter) | **supported** |
| 3 | There is a measurable noise floor / null distribution for attribution on irrelevant tokens, so the report can say a score is *notable*. | `stage1/run_noise_floor.py` | **partial** (floor characterized on single prompt; not yet across the benchmark) |
| 4 | The proxy works regardless of whether the model's answer is correct (confident confabulation included). | `stage1/run_simpleqa.py` (no-filter design) | **supported** |
| 5 | The proxy *finds the relevant span* when the answer lives in the context — not just ignores irrelevant filler. | Stage 2: extractive (SQuAD); `stage2/RESULTS.md` | **supported** (gold span carries ~2× the context's per-token proxy when the model answers correctly, 120 SQuAD items) |
| 5b | The proxy detects when a *confidently wrong* model drew from outside the gold span (the epistemics payoff). | Stage 2: wrong/confident cell | **partial** (directional: gold/ctx 1.05–1.84 vs 2.3 for correct; sensitive to the correctness definition, small cells — see RESULTS) |
| 6 | The proxy stays predictive as context complexity scales (distractor facts, longer context). | Stage 2 | **planned** |
| 7 | A claim-vs-foil contrast objective (logit-difference) works, not only reply log-probability. | Stage 1/2: foil objective (needs hand-written foils) | **planned** |
| 8 | The method survives thinking tokens (gradients through generated reasoning). | Stage 2 onward (thinking budget) | **planned** |
| 9 | The method holds under *intentionally* hard-to-reason-about relevance (two people named Mary, scattered formatting). | Stage 3 | **planned** |
| 10 | The annotation step works — the user can mark Request/Claims/Dressing, and an LLM can assist or stand in well enough to be useful. | not yet designed | **planned** |
| 11 | Foils can be contributed at useful quality, by the user or the system (LLM / templates), for claims *and* dressing. | not yet designed | **planned** |
| 12 | **Analyze:** proxy scores can be turned into attributions worth trusting — e.g. confirmed by causal alterations before they're reported. This is the project's main open problem. | Stage 4 analysis step; `RESEARCH_PLAN.md` | **planned** (explicitly unresolved) |
| 13 | **Report:** the confirmed attributions render as a faithful, readable "depended on A, not C" — text or annotation on the reply. | Stage 4 | **planned** |
| 14 | It beats or complements the black-box baseline (ContextCite) at matched granularity. | comparison vs ContextCite | **planned** |

### How to read the state column
Rows 1–4 are the Stage 1 core, and they're in. Row 5 (extractive) is the next GPU run and the real test of whether "attribution" means "finds what matters," not just "ignores noise." Rows 7–9 are the plan's deliberately-hard stages. Rows 10–14 are the parts of the full flow we have *not* started — annotation, foil generation, the analysis step (row 12, the main open problem), the report, and the baseline comparison — and they're where the flow is still a sketch, not a demonstration.
