#!/usr/bin/env python
"""Row 8: does attribution survive thinking tokens (a generated reasoning prefix)?

Everything so far is thinking-OFF. The plan puts some Stage 2 runs on a thinking
budget and makes thinking the default from Stage 3; the open question
(RESEARCH_PLAN.md) is how gradients through generated reasoning are handled.

This is the FIRST cut: same closed-book SimpleQA + N chaff setup, run twice per
question -- thinking OFF and thinking ON -- and compare the chaff-vs-question
proxy split. The question answered here: *does the presence of a generated
reasoning prefix break the proxy's region split?* We attribute the FINAL ANSWER
only (the tokens after </think>), which isolates that question and is the clean
comparison to thinking-off.

NOT answered here (the deeper follow-up): attributing through the think tokens
themselves -- gradients that flow INTO model-generated reasoning rather than
input tokens. Noted, not attempted.

Reuses lib/ (build_prompt now takes enable_thinking). fp32. Run from experiments/:
  python stage2/run_thinking.py --n-questions 30 --n-chaff 30 \
      --out stage2/runs/thinking.json
"""

import argparse
import ast
import csv
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    MODEL, build_prompt, generate_reply, input_grad, load_model,
    pick_chaff_vocab, summarize,
)
from lib.causal import measured_effect, spearman  # noqa: E402


def load_simpleqa(path, seed, max_q_chars):
    rows = list(csv.DictReader(open(path)))
    out = [{"question": r["problem"], "gold": r["answer"]}
           for r in rows if len(r["problem"]) <= max_q_chars]
    random.Random(seed).shuffle(out)
    return out


def split_think_answer(tok, reply_ids):
    """Return (answer_ids) = tokens after </think>. If no close tag, return the
    whole reply (model never closed the think block -> attribute all of it)."""
    text = tok.decode(reply_ids, skip_special_tokens=False)
    marker = "</think>"
    if marker in text:
        answer_text = text.split(marker, 1)[1]
        ans_ids = tok(answer_text, add_special_tokens=False)["input_ids"]
        # fall back to tail of reply if re-tokenization is empty
        if ans_ids:
            return torch.tensor(ans_ids)
    # no clean close: attribute the last ~min(len,8) tokens as the "answer"
    tail = reply_ids[-8:] if len(reply_ids) > 8 else reply_ids
    return tail


def region_mean(arr, labels, lab):
    idx = [i for i, l in enumerate(labels) if l == lab]
    return float(arr[idx].mean()) if idx else float("nan")


def run_cond(model, tok, question, chaff_vocab, n_chaff, enable_thinking,
             max_new_tokens, k_draws, seed):
    input_ids, labels, _ = build_prompt(
        tok, question, n_chaff, chaff_vocab, "before", seed,
        enable_thinking=enable_thinking,
    )
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens)
    if len(reply_ids) == 0:
        return {"error": "empty reply"}
    full_reply_text = tok.decode(reply_ids, skip_special_tokens=False)

    if enable_thinking:
        answer_ids = split_think_answer(tok, reply_ids)
        had_think = "</think>" in full_reply_text
    else:
        answer_ids = reply_ids
        had_think = False
    if len(answer_ids) == 0:
        return {"error": "empty answer after think split"}

    proxy, logp = input_grad(model, input_ids, answer_ids)
    _, dlogp, _ = measured_effect(
        model, tok, input_ids, answer_ids, chaff_vocab, k_draws, seed
    )
    rows = summarize(labels, proxy)
    q_mean = region_mean(proxy, labels, "question")
    c_mean = region_mean(proxy, labels, "chaff")
    cidx = torch.tensor([i for i, l in enumerate(labels) if l == "chaff"])
    return {
        "thinking": enable_thinking,
        "had_think_block": had_think,
        "n_reply_tokens": len(reply_ids),
        "n_answer_tokens": len(answer_ids),
        "answer_preview": tok.decode(answer_ids, skip_special_tokens=True)[:40],
        "q_mean": q_mean,
        "chaff_mean": c_mean,
        "q_over_chaff": (q_mean / c_mean) if c_mean else float("nan"),
        "chaff_share": rows.get("chaff", {}).get("share_of_sum", 0.0),
        "spearman_proxy_causal": spearman(proxy, dlogp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpleqa", default="../data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--max-q-chars", type=int, default=120)
    ap.add_argument("--max-new-off", type=int, default=12)
    ap.add_argument("--max-new-on", type=int, default=160,
                    help="thinking needs room for the reasoning block + answer")
    ap.add_argument("--k-draws", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    tok, model = load_model(args.model)
    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)
    questions = load_simpleqa(args.simpleqa, args.seed, args.max_q_chars)

    per_item, done = [], 0
    for q in questions:
        if done >= args.n_questions:
            break
        off = run_cond(model, tok, q["question"], chaff_vocab, args.n_chaff,
                       False, args.max_new_off, args.k_draws, args.seed)
        on = run_cond(model, tok, q["question"], chaff_vocab, args.n_chaff,
                      True, args.max_new_on, args.k_draws, args.seed)
        if "error" in off or "error" in on:
            continue
        per_item.append({"question": q["question"], "off": off, "on": on})
        done += 1
        if done % args.progress_every == 0:
            print(f"[prog] {done}/{args.n_questions}  {done/(time.time()-t0):.2f}/s")

    import statistics as st

    def med(xs):
        xs = [x for x in xs if x == x]
        return st.median(xs) if xs else float("nan")

    n_think = sum(r["on"]["had_think_block"] for r in per_item)
    print(f"\n=== thinking OFF vs ON (n={len(per_item)}; "
          f"{n_think} produced a real </think> block) ===")
    for key in ("off", "on"):
        print(f"[{key:>3}] q/chaff mean={med(r[key]['q_over_chaff'] for r in per_item):.2f}  "
              f"chaff share={med(r[key]['chaff_share'] for r in per_item):.3f}  "
              f"proxy-vs-causal rho={med(r[key]['spearman_proxy_causal'] for r in per_item):.3f}")

    result = {
        "model": args.model, "dtype": "float32", "test": "thinking_on_vs_off",
        "n_questions": len(per_item), "n_chaff": args.n_chaff,
        "k_draws": args.k_draws, "seed": args.seed,
        "elapsed_sec": time.time() - t0, "per_item": per_item,
    }
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
