# Research plan

Companion to `README.md`. This file is the sequence of experiments.

**Provenance.** Except where the prose says otherwise, content here was stated or approved by Zach in conversation. Where the plan carries an idea Zach has not reviewed or committed to, the surrounding prose says so explicitly (see "Parked ideas", and the uncommitted ideas noted under Stage 3). Early scaffolding toward Stage 1 has been run; see `experiments/`. The stages themselves have not been executed as designed.

## Ground rules

- **Order by how likely a stage is to break the idea.** Hard variants (thinking tokens, redundant cues, messy contexts) go early, because a failure there changes the plan.
- **Every stage compares two numbers:** a proxy measure computed from the model (gradient-family), and a measured effect from a causal intervention. The question at each stage is how well the first predicts the second, and where it doesn't.
- **Three proxy families run at every stage:** plain gradients, integrated gradients (IG), layer-wise relevance propagation (LRP). Plain gradients are checked first. If IG's compute becomes a bottleneck, let it run in the background and start on LRP (implement it, or pull a repo that already has it). The point is also to learn what each costs in complexity and quality.
- **Two objectives at every stage:** log-probability and logit-difference against a foil. Whether the confident answer saturates log-probability is unknown, and the two objectives compare directly. Foils are hand-written where needed, including in stage 1.
- **Failure handling.** If plain gradients fail a stage, move to IG, then LRP. When all three are out of options, the idea is shown not to work and we pick something else.
- **Thinking.** Stage 1 runs without thinking. Some stage 2 runs use a thinking budget. Stage 3 onward uses thinking by default. The gradient math is equally well-defined with or without thinking, so it's better to find out early whether thinking breaks it.
- **Causal interventions** (resampling, reruns, activation patching, path patching) are the yardstick for any causal claim. Our workflow for constructing and checking causal claims is meant to fold gradually into the automated analysis step (Stage 4).
- Write each stage's failure criterion before running it. Example for Stage 1: "random-token attribution is at least the question's, on most prompts, at most scales."

## Stage 1: the core

**Objective.** With no epistemic reason to attend to a token at all and no complications, which proxy measure (gradients, IG, LRP; sums or means; etc.) best predicts the outcomes of causal alterations? What is the noise floor?

**Design.**
- Hand-built prompts: N random tokens plus a simple factual question or instruction ("what is the longest bony fish?", "write me a poem about oarfish"). Vary N from low to high, and with and without an explanation for the tokens.
- Proxy side: the share of the gradient of the reply as a whole that lands on the N random tokens vs. on the question. First instinct is the sum of magnitudes, also tried as means. Totals across different N are expected to differ; the comparison of interest is whether M relevant question tokens carry as much influence in total as N chaff tokens, and how that changes with N. If the request always outweighs the chaff at all scales, the proxy passes this check. If not, look at the behavior.
- Causal side: resample the N random tokens, and measure how much the reply's logits change.
  - Define the measured effect with teacher forcing: generate the reply once, then compare the model's next-token distributions at each reply position (per-position KL) and the log-probability of that reply, under original vs. resampled tokens. This pairs directly with a gradient computed under teacher forcing, and has no sampling noise.
  - Also run free-running reruns (new replies under the resampled prompt) to measure total effect including a changed trajectory. Include a no-change rerun for the sampling-noise floor.
- Compare proxy to measured effect. Candidate proxies: gradient norm, gradient × input, whatever else is cheap. Try all that are.
- First-order per-swap score as an extra proxy: (new embedding − old embedding) · gradient, per token, for the same swap used on the causal side.
- Put the random tokens both before and after the question, since position is a possible confound.

**Outputs.** A ranking of proxy measures by how well they predict measured effects; a noise floor; a null distribution of attribution on irrelevant tokens (needed later by the analysis step).

## Stage 2: the obvious failures

**Objective.** When asking simple questions or giving simple instructions where we can still reason our way to what epistemically *should* be relevant, how does the proxy measure scale and change with context complexity? How much extra effort is the analysis stage likely to need to keep proxy measures predictive of more real-world outcomes?

**Design.**
- Control tests built the dumb way: show the model data about a person and ask for one fact in one token ("What is my favorite sports team? Answer with only the team name."). If the only way to answer correctly is to read the fact out of context and the proxy cannot detect that, the idea fails.
- Hand-written foils (other team names) allow log-probability and logit-difference to be compared directly on a confident one-token answer.
- Some runs with a thinking budget.
- Distractor facts (team, color, city) with a question about one, to test whether the proxy picks the right line.
- A negative case: a question the model answers from memory, with irrelevant context present. Tests the "this was not relied on" side, checked against erasing the context.
- Context complexity is scaled up, and the scaling is what this stage measures. The specific scaling axes are not fixed.

## Stage 3: the interesting failures

**Objective.** When the epistemic relevance is intentionally difficult to reason about (but can still be reasoned about confidently), does the proxy measure deteriorate or remain capable? What future failures might confound us? Test those early.

**Design.** Not fixed. Thinking is on by default. Ideas Zach mentioned without committing: two different people named Mary in the context; weird formatting changes in many places.

## Stage 4: the utility

**Objective.** In closer-to-deployment conditions, where we have a strong prediction but not necessarily perfect knowledge of epistemic relevance, do the predictions about scaling with complexity and with obtuse edge cases hold up? Do we see obvious or potential failures at different scales? What, if anything, lets us scale to arbitrarily high context complexity while still accurately predicting what causal experiments would produce?

**Design.** Not fixed. Zach's example: a model told it is a career-fair assistant sponsored by CandyCola Beverage. Does it push applicants toward CandyCola jobs, and is that causally related to the sponsorship message? This stage involves broader research into known human and model biases.

**Analysis step.** At this scale the proxy produces many candidate explanations (for example, "your choice of 'Millford, MA' over 'Millford Mass.' is the main reason the model said the moon landing was not a hoax"). The work is (a) characterizing, with our own interpretability tools, which kinds of gradient pattern indicate a causal connection until we understand it, then (b) automating or supporting that analysis within the system. The null distribution from Stage 1 is one input to deciding whether a token's score is notable.

## Open decisions

- Model: small, open-weight, instruct, with a thinking toggle; hardware.
- Exact proxy definitions: gradient wrt embeddings vs. activations; norm vs. × input; normalization; aggregation over reply tokens; IG baseline.
- Source of LRP: implement or reuse.
- Thresholds for each stage's failure criterion.
- Thinking tokens: how gradients through generated tokens are handled.
- Numerical precision: default to fp32 for attribution. The Stage 1 noise floor is near the scale where fp16 underflows tiny gradients and bf16's coarse mantissa adds error, so a low-precision run could report numerical noise as signal. fp32 (or at minimum an fp32 cross-check against any lower-precision run) is the default until a stage shows it is unnecessary. This is what drove the hardware choice (an 80GB A100: Qwen3-8B in fp32 needs ~32GB for weights alone, plus activations and the backward pass). Still open: whether bf16 is an acceptable speed/memory trade at later stages once the floor is characterized.

## Not known

These are gaps in knowledge, not claims about quality. Several are tested by the stages above.
- Whether a local gradient predicts the effect of replacing a token, as opposed to nudging it. (Stage 1.)
- Behavior under redundant cues, multi-hop dependence, and thinking. (Stage 2 onward.)
- Gradients cannot see token insertion or deletion, so foils are equal-length substitutions.
- Explaining why a model did *not* mention something; diffuse concepts that no single token carries; counterfactuals like reordering a priority list; how to merge several foils' signals.
- Whether the "everything matters, nonlinearly" outcome occurs in some cases.
- Black-box comparison needs matched granularity: ContextCite attributes to context sources, not tokens.

## Parked ideas

J-lens concept-level readout. A judge that lists factors a reader would consider relevant. Both unreviewed agent suggestions.

## Sprint schedule (the AI x Epistemics sprint, 2026-11-13 to 11-15)

Zach works alone Saturday; a collaborator joins Sunday. Submission Sunday night or early Monday (deadline Monday 7am). Target schedule:
- Saturday: model and harness, Stage 1, start Stage 2.
- Sunday: finish Stage 2, start Stage 3 with the collaborator, write up.
- A stage that is not reached is future work in the write-up.
