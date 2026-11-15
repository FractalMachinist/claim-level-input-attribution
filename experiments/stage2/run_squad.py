#!/usr/bin/env python
"""Stage 2 procedure: extractive attribution on SQuAD, split into a 2x2.

Stage 1 showed the proxy keeps irrelevant chaff near the noise floor and tracks
the causal effect. Stage 2 asks the harder question: when the answer lives IN
the context, does the proxy FIND the right span (gold_span) rather than just
ignore filler?

The analysis is a 2x2, NOT a filter (Zach's call 2026-11-15, see PLAN_2x2.md):
  rows = model correct? (greedy reply contains a gold answer)
  cols = model confident? (teacher-forced log-prob of its own reply, split)
Every item runs; we bucket post-hoc and ask whether attribution quality DIFFERS
across cells. The informative claim we chase: "when the model gets SQuAD wrong,
the proxy shows it drew from OUTSIDE the gold_span" (the wrong-answer cells).

Regions: gold_span (answer tokens in context), context (rest of paragraph),
question, template. Headline metric: gold-span hit rate (is the proxy's top
token in the gold span?), plus region contrast, per cell.

Machinery reused unchanged: lib/attribution.py (input_grad, load_model,
generate_reply), lib/causal.py (measured_effect, spearman). Only the prompt
builder is new (4 regions, span located by SQuAD answer_start).

Precision fp32. Run from experiments/:
  python stage2/run_squad.py --squad ../data/squad/dev-v1.1.json \
      --n-items 60 --out stage2/runs/squad_2x2.json
"""

import argparse
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    MODEL, generate_reply, input_grad, load_model, summarize,
)
from lib.causal import measured_effect, spearman  # noqa: E402


def load_squad_items(path, n_items, seed, max_ctx_chars, max_ans_chars):
    """Flatten SQuAD, keep compact items, sample n_items.

    Returns dicts with context, question, the first answer's text+char start
    (for the gold span) and ALL answer texts (for correctness matching).
    """
    d = json.load(open(path))
    items = []
    for art in d["data"]:
        for p in art["paragraphs"]:
            ctx = p["context"]
            if len(ctx) > max_ctx_chars:
                continue
            for qa in p["qas"]:
                answers = qa.get("answers", [])
                if not answers:
                    continue
                a0 = answers[0]
                if len(a0["text"]) > max_ans_chars:
                    continue
                items.append({
                    "question": qa["question"],
                    "context": ctx,
                    "gold_text": a0["text"],
                    "gold_start": a0["answer_start"],
                    "all_answers": list({a["text"] for a in answers}),
                })
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_items]


def build_squad_prompt(tok, item, seed=0):
    """Build input_ids + per-token region labels for a SQuAD item.

    Regions: question / context / gold_span / template. The gold span is located
    inside the context by SQuAD's answer_start char offset. The user turn is
    "{context}\n\nQuestion: {question}\nAnswer in as few words as possible."
    so the question is clearly delimited and the answer is short/extractive.
    """
    ctx = item["context"]
    q = item["question"]
    instr = "\nAnswer in as few words as possible."
    # user_content with context first, then the question.
    ctx_lead = ""
    q_lead = "\n\nQuestion: "
    user_content = ctx_lead + ctx + q_lead + q + instr

    msgs = [{"role": "user", "content": user_content}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    base = text.find(user_content)
    if base < 0:
        raise RuntimeError("user_content not found verbatim in rendered template")

    # char spans within user_content
    ctx_lo = len(ctx_lead)
    ctx_hi = ctx_lo + len(ctx)
    gold_lo = ctx_lo + item["gold_start"]
    gold_hi = gold_lo + len(item["gold_text"])
    q_lo = ctx_hi + len(q_lead)
    q_hi = q_lo + len(q)

    # shift to full text
    def sh(x):
        return base + x
    ctx_span = (sh(ctx_lo), sh(ctx_hi))
    gold_span = (sh(gold_lo), sh(gold_hi))
    q_span = (sh(q_lo), sh(q_hi))

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    def overlaps(a, b, span):
        return a < span[1] and b > span[0]

    def region_of(a, b):
        if b <= a:
            return "template"
        # gold_span takes priority over context (it's inside it). Use OVERLAP,
        # not containment: a token that straddles the span boundary still counts
        # as gold -- otherwise short answers whose first/last char falls
        # mid-token drop the whole item (6/60 lost in the first run).
        if overlaps(a, b, gold_span):
            return "gold_span"
        if a >= q_span[0] and b <= q_span[1]:
            return "question"
        if a >= ctx_span[0] and b <= ctx_span[1]:
            return "context"
        return "template"

    labels = [region_of(a, b) for (a, b) in offsets]
    return input_ids, labels


def normalize(s):
    """SQuAD-style normalization for correctness matching."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def is_correct(reply_text, all_answers):
    """True if any gold answer appears (normalized) in the reply."""
    r = normalize(reply_text)
    for a in all_answers:
        na = normalize(a)
        if na and na in r:
            return True
    return False


def token_f1(reply_text, all_answers):
    """Max SQuAD token-F1 of the reply against any gold answer.

    Recorded alongside the strict substring `correct` flag so the correctness
    axis of the 2x2 can be drawn either way in analysis -- substring match
    mislabels 'time and space' vs gold 'time or space' as wrong; F1 catches it.
    """
    import collections
    pred = normalize(reply_text).split()
    best = 0.0
    for a in all_answers:
        gold = normalize(a).split()
        if not pred or not gold:
            continue
        common = collections.Counter(pred) & collections.Counter(gold)
        same = sum(common.values())
        if same == 0:
            continue
        prec = same / len(pred)
        rec = same / len(gold)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def region_mean(arr, labels, lab):
    idx = [i for i, l in enumerate(labels) if l == lab]
    return float(arr[idx].mean()) if idx else float("nan")


def top_token_region(proxy, labels):
    """Region of the single highest-proxy input token (all regions)."""
    i = int(proxy.argmax())
    return labels[i]


def content_hit(proxy, labels, k=1):
    """Does a gold_span token rank in the top-k AMONG CONTENT tokens?

    Template/chat-scaffold tokens carry outsized gradient norm (BOS /
    attention-sink artifact) and dominate a naive top-1, so the gold span never
    wins against them. The meaningful question is whether the answer span beats
    its actual competitor -- the distractor context. We restrict to content
    regions (gold_span + context + question) and ask whether a gold token is in
    the top-k there. k=1 is the strict headline; we also report the gold span's
    best rank among content tokens.
    """
    content_idx = [i for i, l in enumerate(labels)
                   if l in ("gold_span", "context", "question")]
    if not content_idx:
        return {"content_hit_top1": False, "gold_best_rank_frac": float("nan")}
    # rank content tokens by proxy, descending
    ranked = sorted(content_idx, key=lambda i: float(proxy[i]), reverse=True)
    gold_positions = [rank for rank, i in enumerate(ranked)
                      if labels[i] == "gold_span"]
    hit = any(r < k for r in gold_positions)
    best = (min(gold_positions) / len(ranked)) if gold_positions else float("nan")
    return {"content_hit_top1": hit, "gold_best_rank_frac": best}


def run_one(model, tok, item, max_new_tokens, k_draws, chaff_vocab, seed):
    input_ids, labels = build_squad_prompt(tok, item, seed)
    if "gold_span" not in labels:
        return {"error": "gold_span not tokenized (span straddled tokens)"}
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens)
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)
    if len(reply_ids) == 0:
        return {"error": "empty reply"}

    proxy_norm, grad_logp = input_grad(model, input_ids, reply_ids)
    base_logp, dlogp, dkl = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, k_draws, seed
    )

    correct = is_correct(reply_text, item["all_answers"])
    f1 = token_f1(reply_text, item["all_answers"])
    # Naive top-1 (all regions) -- kept for diagnostics; dominated by template.
    top_region = top_token_region(proxy_norm, labels)
    gold_hit = (top_region == "gold_span")
    # The REAL headline: does the gold span beat its competitor, the context,
    # once the template/sink tokens are excluded?
    ch = content_hit(proxy_norm, labels, k=1)

    rows = summarize(labels, proxy_norm)
    # spearman proxy-vs-measured over CONTEXT positions (incl. gold_span)
    cidx = torch.tensor([i for i, l in enumerate(labels)
                         if l in ("context", "gold_span")])
    rho_ctx = (spearman(proxy_norm[cidx], dlogp[cidx])
               if len(cidx) > 2 else float("nan"))

    return {
        "question": item["question"],
        "gold_text": item["gold_text"],
        "reply_text": reply_text,
        "reply_logp": grad_logp,          # confidence axis
        "correct": correct,               # correctness axis (strict substring)
        "token_f1": f1,                   # softer correctness (SQuAD token-F1)
        "n_input_tokens": len(input_ids),
        "n_reply_tokens": len(reply_ids),
        "gold_span_hit": gold_hit,            # naive top-1 (template-confounded)
        "content_hit_top1": ch["content_hit_top1"],   # gold beats context, top-1
        "gold_best_rank_frac": ch["gold_best_rank_frac"],  # 0=best content token
        "top_token_region": top_region,
        "rho_proxy_dlogp_context": rho_ctx,
        "region_proxy_mean": {
            lab: region_mean(proxy_norm, labels, lab)
            for lab in ("gold_span", "context", "question", "template")
        },
        "region_measured_mean": {
            lab: region_mean(dlogp, labels, lab)
            for lab in ("gold_span", "context", "question", "template")
        },
        "region_share_of_proxy": {
            lab: rows.get(lab, {}).get("share_of_sum", 0.0)
            for lab in ("gold_span", "context", "question", "template")
        },
    }


def summarize_cells(per_item, conf_threshold):
    """Bucket items into the 2x2 and report per-cell attribution quality."""
    cells = {}
    for r in per_item:
        if "error" in r:
            continue
        correct = r["correct"]
        confident = r["reply_logp"] >= conf_threshold
        key = (("correct" if correct else "wrong"),
               ("conf" if confident else "unconf"))
        cells.setdefault(key, []).append(r)

    def agg(rs):
        import statistics
        n = len(rs)

        def med(vals):
            v = [x for x in vals if x == x]  # drop nan
            return statistics.median(v) if v else float("nan")

        # Headline (template-excluded): gold beats context top-1 among content.
        content_hit = sum(1 for r in rs if r["content_hit_top1"]) / n
        # gold span's best rank among content tokens (0=top; lower is better).
        best_rank = med(r["gold_best_rank_frac"] for r in rs)
        # gold vs context mean-norm ratio -- the cleanest "is the answer
        # special" signal, and the one that separated correct from wrong/conf.
        ratio = med(r["region_proxy_mean"]["gold_span"]
                    / r["region_proxy_mean"]["context"]
                    for r in rs if r["region_proxy_mean"]["context"] > 0)
        gold_gt_ctx = sum(
            1 for r in rs
            if r["region_proxy_mean"]["gold_span"]
            > r["region_proxy_mean"]["context"]) / n
        rho_med = med(r["rho_proxy_dlogp_context"] for r in rs)
        # naive (confounded) hit, kept for the record
        naive_hit = sum(1 for r in rs if r["gold_span_hit"]) / n
        return {
            "n": n,
            "content_hit_top1_rate": content_hit,
            "median_gold_best_rank_frac": best_rank,
            "median_gold_to_context_ratio": ratio,
            "frac_gold_mean_gt_context_mean": gold_gt_ctx,
            "median_rho_proxy_dlogp_context": rho_med,
            "naive_top1_hit_rate": naive_hit,
        }

    return {f"{c}/{cf}": agg(rs) for (c, cf), rs in sorted(cells.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", default="../data/squad/dev-v1.1.json")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--max-ctx-chars", type=int, default=400)
    ap.add_argument("--max-ans-chars", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--k-draws", type=int, default=6)
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help="log-prob threshold for 'confident'; default=median split")
    ap.add_argument("--model", default=MODEL,
                    help="override to pull the model-size lever")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", type=int, default=0,
                    help="if >0, just print N greedy replies and exit (scale check)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    tok, model = load_model(args.model)

    items = load_squad_items(args.squad, args.n_items, args.seed,
                             args.max_ctx_chars, args.max_ans_chars)
    print(f"[data] {len(items)} compact SQuAD items")

    # scale sanity: just show greedy replies vs gold, no attribution
    if args.sanity > 0:
        n_ok = 0
        for it in items[:args.sanity]:
            iid, labels = build_squad_prompt(tok, it, args.seed)
            rids = generate_reply(model, tok, iid, args.max_new_tokens)
            rt = tok.decode(rids, skip_special_tokens=True)
            ok = is_correct(rt, it["all_answers"])
            n_ok += ok
            print(f"  [{'OK ' if ok else 'XX '}] gold={it['gold_text']!r:30} "
                  f"reply={rt[:40]!r}")
        print(f"[sanity] {n_ok}/{args.sanity} correct with {args.model}")
        return

    # reuse Stage 1's chaff vocab for the resampling intervention
    from lib.attribution import pick_chaff_vocab
    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)

    per_item = []
    done = 0
    for it in items:
        try:
            r = run_one(model, tok, it, args.max_new_tokens, args.k_draws,
                        chaff_vocab, args.seed)
        except Exception as e:
            r = {"error": repr(e), "question": it["question"]}
        per_item.append(r)
        done += 1
        if done % args.progress_every == 0:
            dt = time.time() - t0
            print(f"[prog] {done}/{len(items)}  {done/dt:.2f}/s")

    ok = [r for r in per_item if "error" not in r]
    n_correct = sum(1 for r in ok if r["correct"])
    print(f"[overall] {len(ok)} scored, {len(per_item)-len(ok)} errored; "
          f"correct: {n_correct}/{len(ok)}")

    # confidence threshold: median split unless given
    if args.conf_threshold is None and ok:
        import statistics
        conf_threshold = statistics.median(r["reply_logp"] for r in ok)
    else:
        conf_threshold = args.conf_threshold or 0.0
    print(f"[confidence] threshold (log-prob) = {conf_threshold:.3f}")

    cells = summarize_cells(per_item, conf_threshold)
    print("\n=== 2x2 cells (correctness x confidence) ===")
    print("  (gold/ctx = median per-token proxy ratio gold_span:context;")
    print("   the key signal is whether this > 1 means the answer is 'special')")
    for key, a in cells.items():
        print(f"[{key:>15}] n={a['n']:>3}  "
              f"gold/ctx={a['median_gold_to_context_ratio']:.2f}  "
              f"gold>ctx on {a['frac_gold_mean_gt_context_mean']:.0%}  "
              f"content_hit={a['content_hit_top1_rate']:.0%}  "
              f"rho(ctx)={a['median_rho_proxy_dlogp_context']:.3f}  "
              f"(naive_hit={a['naive_top1_hit_rate']:.0%})")

    result = {
        "model": args.model, "dtype": "float32", "benchmark": "SQuAD-v1.1-dev",
        "n_items": len(items), "n_scored": len(ok),
        "n_correct": n_correct, "conf_threshold": conf_threshold,
        "max_ctx_chars": args.max_ctx_chars, "max_ans_chars": args.max_ans_chars,
        "k_draws": args.k_draws, "seed": args.seed,
        "elapsed_sec": time.time() - t0,
        "cells": cells,
        "per_item": per_item,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[out] wrote {args.out}")
    else:
        print("[warn] no --out; not saved")


if __name__ == "__main__":
    main()
