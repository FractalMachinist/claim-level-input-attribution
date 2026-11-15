# Claim-level input attribution for LLM epistemics audits

**Status:** Stage 1 of the core hypothesis is **supported** — a plain gradient proxy predicts the causal effect of perturbing input tokens, generalizing across 50 prompts on an external benchmark. See **[STATUS.md](STATUS.md)** for the ideal user flow, the assumptions it rests on, and which are demonstrated vs. planned. Stage 1 findings: [experiments/stage1/RESULTS.md](experiments/stage1/RESULTS.md). The sections below describe the idea as originally framed.

## Problem

A user relies on an LLM for a consequential answer. They want to know whether the model's judgment was shaped by factors that should not matter, or ignored factors that should.

Example: Joe (`@wwf.org`) and Jane (`@army.mil`) ask the same question about a recent executive order. The model's conclusions differ materially. The email domain should not drive that difference.

## Goal

From a single response, without building matched pairs of inputs by hand, produce a readout of this form:

> The model said X. This depended on A and B. It did not appear to take C into account.

Target examples:
- The user stated their priorities and got a vote recommendation that would have been the same whatever they said.
- A recommendation depended heavily on the user's surname or dialect.
- Advice to apologize rested mostly on remembered anxiety about work, and would not have changed if the coworker situation changed.
- The model recommends the name "Dave" for every baby.

## The idea

1. **Annotate.** Divide the full context and response into three regions (see Terms): Request, Claims, Dressing.
2. **Take foils.** For each claim, take one or more foils: alternative versions of the claim. Where foils come from (an LLM, the user, templates) is open.
3. **Attribute.** For each claim and foil, compute which input tokens push the model toward the foil and away from the original.
4. **Analyze and report.** Analyze the attributions and report which inputs the claims depended on, and which relevant inputs they did not. The report could be text or visual annotation around the model's reply. The analysis step is unresolved. One possibility under consideration is automated agentic experimentation and counterfactual testing.

## Hypothesis

Step 3 is done with white-box, gradient-based methods. The hypothesis is that local gradient-based scores are a usable proxy for how much each input token influences a claim. This is untested. If it holds, a small number of backward passes (perhaps one per claim, or one per claim token) highlights the influential tokens among all input tokens, cheaply.

Three candidate ways to compute the proxy:
- **Plain gradients.** The gradient of the claim-vs-foil contrast with respect to the input.
- **Integrated gradients (IG).** The average gradient along a path from a baseline input to the actual input, times (input − baseline). Requires choosing a baseline.
- **Layer-wise relevance propagation (LRP).** Propagates relevance backward through the network using layer-specific rules. Needs no baseline. Transformers need transformer-specific rules.

Black-box ablation-based attribution (ContextCite, below) is the comparison method.

The hypothesis predicts that Request tokens will be highly influential on the claims. If they are not, the hypothesis is wrong or we misunderstand it.

Requires access to model weights.

## Terms

| Term | Definition |
|---|---|
| **Request** | The question or instruction itself, without fluff, explanation, or user data. Changing it obviously changes the response. |
| **Claims** | Response portions material to the request. If altered, the response would be substantially wrong or unhelpful. These are the targets of attribution. |
| **Dressing** | Non-factual response portions that shape the user's takeaway, e.g. whether someone is called a "disgraced" physician or just a physician. Whether a given portion counts as a claim or as dressing depends on the question. Dressing is part of the goal. If something is cut for scope, it will be dressing. |
| **Foil** | An alternative version of a claim, substantially different from the original. May be inaccurate or opposite ("sky is blue" → "sky is red"). The source of a foil is open. A claim may have several. |

## Prior work

Descriptions below were verified by an agent other than the one that drafted this README. The drafting agent did not re-check them.

- Yin & Neubig 2022, *Interpreting Language Models with Contrastive Explanations*. Contrastive gradient attribution, next-token level. arXiv 2202.10419. Code: `github.com/kayoyin/interpret-lm`.
- *Contrastive Attribution in the Wild* (2026). Attributes a wrong-vs-correct logit gap to input tokens and internal states via LRP, with cross-layer attribution graphs for long inputs. arXiv 2604.17761.
- *Rather a Nurse than a Physician — Contrastive Explanations under Investigation*. Uses LRP because naive transformer gradients were reported less faithful. arXiv 2310.11906.
- Kamahi & Yaghoobzadeh, *Counterfactuals As a Means for Evaluating Faithfulness of Attribution Methods in Autoregressive Language Models*. arXiv 2408.11252.
- CELL (IBM), *Contrastive Explanations for Large Language Models*. Minimal prompt edits that change output. arXiv 2406.11785.
- JoPA, *Joint Prompt Attribution* (ACL 2025). Discrete search for prompt-token combinations that explain the whole generation. `aclanthology.org/2025.acl-long.1074`.
- ContextCite (Cohen-Wang et al., NeurIPS 2024). Black-box statement-level attribution to context sources via a linear surrogate fit from extra inference passes. arXiv 2409.00729. Code: `github.com/MadryLab/context-cite`. Units are context sources, goal is grounding.
- *Token-Level Diagnosis of Sycophancy in LLMs with Attribution-Guided Steering*. IG-based token attribution of authority cues. arXiv 2607.28906.
- Turpin et al. 2023, *Language Models Don't Always Say What They Think*. Biasing features shift answers and go unmentioned in chain of thought. arXiv 2305.04388. Source of ready-made planted cues.
- *Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations*. Detects exploitation of protected features using counterfactual prompts. arXiv 2504.05294.

## Elsewhere

Methods, validation, limits, and open decisions belong in `RESEARCH_PLAN.md` (not yet written).
