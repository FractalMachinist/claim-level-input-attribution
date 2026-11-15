#!/usr/bin/env python
"""Reusable attribution machinery -- import this, do not run it.

Shared across stages: model loading, the chaff-vocab sampler, prompt building
with region labels, greedy reply generation, the plain-gradient input
attribution, and the per-region summary. The runnable experiment procedures
live in stageN/run_*.py and import from here.

Precision: fp32 throughout (RESEARCH_PLAN.md: settled, prevents the
low-precision numerical-noise confound at the Stage 1 noise floor).
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"


def load_model(model_name=MODEL, verbose=True):
    """Load tokenizer + fp32 model onto GPU if available. Returns (tok, model)."""
    import time
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    ).eval()
    if torch.cuda.is_available():
        model.cuda()
    if verbose:
        print(f"[load] {model_name} fp32 in {time.time()-t0:.1f}s "
              f"on {model.device}")
    return tok, model


def pick_chaff_vocab(tok, n_candidates=4000, seed=0):
    """Return a list of 'ordinary' token ids to sample chaff from.

    Filters to tokens that decode to short, mostly-alphabetic strings, to avoid
    special tokens and control glyphs that would be their own confound.
    """
    g = torch.Generator().manual_seed(seed)
    vocab_size = len(tok)
    ids = torch.randperm(vocab_size, generator=g)[: n_candidates * 4].tolist()
    keep = []
    special = set(tok.all_special_ids)
    for i in ids:
        if i in special:
            continue
        s = tok.decode([i])
        # ordinary word-ish fragment: letters/space, length 1..12
        if re.fullmatch(r"[ A-Za-z]{1,12}", s):
            keep.append(i)
        if len(keep) >= n_candidates:
            break
    if not keep:
        raise RuntimeError("no chaff candidates survived filter")
    return keep


def build_prompt(tok, question, n_chaff, chaff_vocab, position="before", seed=0,
                 enable_thinking=False):
    """Build input_ids and a region label per position.

    Regions: 'template' (chat scaffolding), 'chaff' (random tokens),
    'question' (the actual question text). We locate the question by encoding it
    alone and finding its contiguous span; everything in the user turn that is
    not question and not chaff is template.
    """
    g = torch.Generator().manual_seed(seed + 12345)
    chaff_ids = [chaff_vocab[int(torch.randint(len(chaff_vocab), (1,), generator=g))]
                 for _ in range(n_chaff)]
    chaff_text = tok.decode(chaff_ids)

    sep = "\n\n"
    if position == "before":
        user_content = (chaff_text + sep + question) if n_chaff else question
    elif position == "after":
        user_content = (question + sep + chaff_text) if n_chaff else question
    else:
        raise ValueError(position)

    msgs = [{"role": "user", "content": user_content}]
    # enable_thinking toggles Qwen3's reasoning format. Default False (empty
    # <think> block) keeps every existing caller's behavior unchanged; Stage 2
    # row 8 flips it True to test attribution with a generated reasoning prefix.
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Character spans of question / chaff *within user_content*, then shifted to
    # their position in the full rendered `text`. The user_content appears
    # verbatim in `text`, so one find() locates it; offsets are exact.
    if position == "before" and n_chaff:
        chaff_lo, chaff_hi = 0, len(chaff_text)
        q_lo = len(chaff_text) + len(sep)
        q_hi = q_lo + len(question)
    elif position == "after" and n_chaff:
        q_lo, q_hi = 0, len(question)
        chaff_lo = len(question) + len(sep)
        chaff_hi = chaff_lo + len(chaff_text)
    else:  # no chaff
        q_lo, q_hi = 0, len(question)
        chaff_lo = chaff_hi = -1

    base = text.find(user_content)
    if base < 0:
        raise RuntimeError("user_content not found verbatim in rendered template")

    def span_in_text(lo, hi):
        return (base + lo, base + hi) if lo >= 0 else (-1, -1)

    q_span = span_in_text(q_lo, q_hi)
    c_span = span_in_text(chaff_lo, chaff_hi) if n_chaff else (-1, -1)

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()  # [(start,end)] per token

    def region_of(a, b):
        # A token [a,b) is in a region if its char span overlaps the region and
        # the token is non-empty (specials map to (0,0) or empty spans).
        if b <= a:
            return "template"
        if q_span[0] >= 0 and a >= q_span[0] and b <= q_span[1]:
            return "question"
        if c_span[0] >= 0 and a >= c_span[0] and b <= c_span[1]:
            return "chaff"
        return "template"

    labels = [region_of(a, b) for (a, b) in offsets]
    return input_ids, labels, user_content


@torch.no_grad()
def generate_reply(model, tok, input_ids, max_new_tokens=16):
    out = model.generate(
        input_ids.unsqueeze(0).to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.eos_token_id,
    )
    reply_ids = out[0, input_ids.shape[0]:].cpu()
    return reply_ids


@torch.no_grad()
def generate_replies_batched(model, tok, input_ids_list, max_new_tokens=16):
    """Greedy-generate replies for many prompts at once. Returns a list of
    reply-id tensors, one per prompt, each stripped of padding and aligned to
    exactly what the per-prompt generate_reply would return.

    Correctness notes for batched causal-LM generation:
    - LEFT-pad: prompts have different lengths; for a decoder-only model the
      generated continuation must start right after each prompt's last real
      token, which only lines up if padding is on the LEFT. Right-padding would
      make the model attend to pad tokens as if they were context -> wrong.
    - An attention_mask marks the real tokens so padding is ignored.
    - Greedy + per-sequence EOS: generate stops a sequence when it emits EOS;
      the batch runs until all sequences stop or the cap is hit. A short reply
      costs little even in a batch with one long reply, because finished
      sequences stop contributing new tokens.
    - We slice each output from its own prompt length (uniform after left-pad)
      and drop trailing pad tokens, so each returned tensor matches the
      single-prompt result.
    """
    device = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    maxlen = max(x.shape[0] for x in input_ids_list)

    batch = torch.full((len(input_ids_list), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(input_ids_list), maxlen), dtype=torch.long)
    for j, ids in enumerate(input_ids_list):
        L = ids.shape[0]
        batch[j, maxlen - L:] = ids       # LEFT-pad
        attn[j, maxlen - L:] = 1
    batch, attn = batch.to(device), attn.to(device)

    out = model.generate(
        batch, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        pad_token_id=pad_id,
    )
    gen = out[:, maxlen:].cpu()  # [B, <=max_new_tokens], continuation only
    replies = []
    for j in range(gen.shape[0]):
        row = gen[j]
        # strip trailing pad tokens (finished sequences are pad-filled after EOS)
        nonpad = (row != pad_id).nonzero().flatten()
        if len(nonpad) == 0:
            replies.append(row[:0])
            continue
        # keep through the last non-pad token (EOS itself is kept if != pad;
        # since pad_id == eos here, a trailing EOS is indistinguishable from pad
        # and is stripped -- matching generate_reply, which also excludes the
        # pad/eos tail for teacher forcing).
        last = int(nonpad[-1].item())
        replies.append(row[: last + 1])
    return replies


def input_grad(model, input_ids, reply_ids):
    """Plain gradient of teacher-forced reply log-prob w.r.t. input embeddings.

    Returns per-input-position gradient L2 norm (len == len(input_ids)) and the
    scalar reply log-prob. Teacher forcing: the reply is fixed; we sum the
    log-probs the model assigns to each reply token given the preceding context.
    """
    device = model.device
    embed = model.get_input_embeddings()

    full = torch.cat([input_ids, reply_ids]).to(device)
    inp_len = input_ids.shape[0]

    inp_emb = embed(full.unsqueeze(0)).detach().clone().requires_grad_(True)
    out = model(inputs_embeds=inp_emb)
    logits = out.logits[0]  # [T, V], fp32

    logp = torch.log_softmax(logits, dim=-1)
    # token at position t is predicted by logits at t-1
    total = 0.0
    for j, tokid in enumerate(reply_ids.tolist()):
        pos = inp_len + j - 1
        total = total + logp[pos, tokid]

    model.zero_grad(set_to_none=True)
    total.backward()

    g = inp_emb.grad[0]  # [T, D]
    per_pos_norm = g[:inp_len].norm(dim=-1)  # [inp_len]
    return per_pos_norm.detach().cpu(), float(total.detach().cpu())


def input_grad_foil(model, input_ids, real_tok, foil_tok):
    """Gradient of the logit-difference (real answer vs foil) w.r.t. inputs.

    The plan's second objective (RESEARCH_PLAN.md: "two objectives at every
    stage: log-probability and logit-difference against a foil"). Instead of the
    reply's log-prob, we attribute the contrast logit(real) - logit(foil) at the
    first answer position -- "which input tokens push the model toward the real
    answer and away from the foil." This is the contrast the README's step 3
    actually calls for (claim vs foil), so it is the more on-target objective.

    Single-token form: real_tok and foil_tok are the first-token ids of the two
    answers. The objective is evaluated at the position that predicts the
    answer's first token, i.e. the last input position (next-token prediction).
    Returns (per-input-position gradient L2 norm, the scalar logit difference).
    """
    device = model.device
    embed = model.get_input_embeddings()
    inp_len = input_ids.shape[0]

    inp_emb = embed(input_ids.unsqueeze(0).to(device)).detach().clone()
    inp_emb.requires_grad_(True)
    out = model(inputs_embeds=inp_emb)
    logits = out.logits[0]  # [T, V] fp32
    # the next token after the prompt is predicted by the last position
    pos = inp_len - 1
    diff = logits[pos, real_tok] - logits[pos, foil_tok]

    model.zero_grad(set_to_none=True)
    diff.backward()

    g = inp_emb.grad[0]  # [T, D]
    per_pos_norm = g[:inp_len].norm(dim=-1)
    return per_pos_norm.detach().cpu(), float(diff.detach().cpu())


def summarize(labels, per_pos_norm):
    import collections
    by = collections.defaultdict(float)
    cnt = collections.Counter()
    for lab, v in zip(labels, per_pos_norm.tolist()):
        by[lab] += v
        cnt[lab] += 1
    total = sum(by.values()) or 1.0
    rows = {}
    for lab in by:
        rows[lab] = {
            "n_tokens": cnt[lab],
            "sum_norm": by[lab],
            "share_of_sum": by[lab] / total,
            "mean_norm": by[lab] / cnt[lab],
        }
    return rows
