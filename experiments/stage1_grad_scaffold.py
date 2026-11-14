#!/usr/bin/env python
"""Stage 1 scaffold: plain gradient of a reply w.r.t. input embeddings.

The smallest test of the core mechanic (README "The idea", step 3). Builds a
prompt of N random "chaff" tokens plus a simple factual question, generates a
short reply with thinking OFF, then computes the plain gradient of the reply's
log-probability (teacher-forced) w.r.t. the input embeddings. Reports how the
gradient magnitude is distributed across prompt regions.

Stage 1 question (RESEARCH_PLAN.md): with no epistemic reason to attend to the
chaff, does the question outweigh the chaff in the proxy, at various N? This
scaffold only establishes the plumbing and prints the region split for one
prompt; the sweep over N, means-vs-sums, position, and the causal (resample)
side come next.

Precision: fp32 throughout (RESEARCH_PLAN.md open-decisions entry). bf16 is
available on this A100 but is deliberately not used here — the Stage 1 noise
floor is where low precision could masquerade as signal.

Design notes / open points (NOT settled — flagged for Zach):
- Objective here is reply log-prob. logit-difference-vs-foil is the other
  objective in the plan; added in the next iteration (needs hand-written foils).
- Aggregation of per-token embedding gradients to a scalar per input position
  uses the L2 norm of the gradient vector at that position. Sum-of-abs and
  grad*input are alternatives the plan wants compared; not done yet.
- "Chaff" tokens are sampled uniformly from a filtered vocab (ascii-ish, no
  specials). Whether that is the right null is itself a Stage 1 question.
"""

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"


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


def build_prompt(tok, question, n_chaff, chaff_vocab, position="before", seed=0):
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
    # enable_thinking=False: Qwen3 non-thinking format (empty <think> block).
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="What is the longest bony fish?")
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--position", choices=["before", "after"], default="before")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print(f"[load] {MODEL} fp32 in {time.time()-t0:.1f}s on {model.device}")

    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)
    input_ids, labels, user_content = build_prompt(
        tok, args.question, args.n_chaff, chaff_vocab, args.position, args.seed
    )
    reply_ids = generate_reply(model, tok, input_ids, args.max_new_tokens)
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)
    print(f"[prompt] {len(input_ids)} tokens; regions: "
          f"{ {l: labels.count(l) for l in set(labels)} }")
    print(f"[reply]  {reply_text!r}")

    per_pos_norm, logp = input_grad(model, input_ids, reply_ids)
    rows = summarize(labels, per_pos_norm)

    print(f"[logp] teacher-forced reply log-prob = {logp:.3f}")
    print(f"{'region':<10}{'n':>5}{'sum_norm':>14}{'share':>9}{'mean_norm':>14}")
    for lab in ("question", "chaff", "template"):
        if lab in rows:
            r = rows[lab]
            print(f"{lab:<10}{r['n_tokens']:>5}{r['sum_norm']:>14.4e}"
                  f"{r['share_of_sum']:>9.3f}{r['mean_norm']:>14.4e}")

    result = {
        "model": MODEL,
        "dtype": "float32",
        "question": args.question,
        "n_chaff": args.n_chaff,
        "position": args.position,
        "seed": args.seed,
        "n_input_tokens": len(input_ids),
        "reply_text": reply_text,
        "reply_logp": logp,
        "region_rows": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
