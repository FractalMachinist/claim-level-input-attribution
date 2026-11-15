#!/usr/bin/env python
"""Row 7: the logit-difference-against-a-foil objective (vs reply log-prob).

The plan commits to TWO objectives at every stage: reply log-probability (used
everywhere so far) and the logit-difference of a claim against a foil. The foil
objective is the one the README's step 3 actually describes — "which input
tokens push the model toward the foil and away from the original" — so it is
arguably the more on-target objective, and this checks whether it behaves.

Setup (closed-book, so it is a Stage-1-style regime; lives in stage1/):
SimpleQA question + N chaff. The model's greedy first answer token is the
"real" token; the FOIL token is the first token of a DIFFERENT SimpleQA answer
of the SAME answer_type (Person/Place/Date/Number) — a genuine, same-register
alternative, chosen automatically so the run scales (the plan allows foils from
templates; hand-written foils are the eventual refinement).

Two attributions per prompt, same inputs:
  - log-prob objective  -> lib.input_grad (reply log-prob)
  - foil objective      -> lib.input_grad_foil (logit[real] - logit[foil], at
                           the first answer position)
We compare: do they agree on the region split (question vs chaff), and does the
foil objective keep chaff near the floor? Plus each vs the causal effect.

Reuses lib/ unchanged. fp32. Run from experiments/:
  python stage1/run_foil_objective.py --n-questions 40 --n-chaff 30 \
      --out stage1/runs/foil_objective.json
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
    MODEL, build_prompt, generate_reply, input_grad, input_grad_foil,
    load_model, pick_chaff_vocab, summarize,
)
from lib.causal import measured_effect, spearman  # noqa: E402


def load_simpleqa_typed(path, seed, max_q_chars):
    """SimpleQA rows grouped by answer_type, for same-type foils."""
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if len(r["problem"]) > max_q_chars:
            continue
        meta = ast.literal_eval(r["metadata"])
        out.append({"question": r["problem"], "gold": r["answer"],
                    "answer_type": meta.get("answer_type")})
    random.Random(seed).shuffle(out)
    by_type = {}
    for r in out:
        by_type.setdefault(r["answer_type"], []).append(r)
    return out, by_type


def first_tok(tok, text):
    """First token id of a leading-space-prefixed answer string."""
    ids = tok(" " + text.strip(), add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def region_mean(arr, labels, lab):
    idx = [i for i, l in enumerate(labels) if l == lab]
    return float(arr[idx].mean()) if idx else float("nan")


def run_one(model, tok, q, by_type, chaff_vocab, n_chaff, max_new_tokens,
            k_draws, seed):
    input_ids, labels, _ = build_prompt(
        tok, q["question"], n_chaff, chaff_vocab, "before", seed
    )
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens)
    if len(reply_ids) == 0:
        return {"error": "empty reply"}
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)

    # real first-answer token = the model's actual greedy first reply token
    real_tok = int(reply_ids[0])
    # foil = first token of a different same-type SimpleQA answer
    pool = [r for r in by_type.get(q["answer_type"], []) if r["gold"] != q["gold"]]
    if not pool:
        return {"error": "no same-type foil"}
    foil_ans = random.Random(seed + hash(q["question"]) % 10000).choice(pool)["gold"]
    foil_tok = first_tok(tok, foil_ans)
    if foil_tok is None or foil_tok == real_tok:
        return {"error": "degenerate foil token"}

    # two attributions
    proxy_lp, logp = input_grad(model, input_ids, reply_ids)
    proxy_foil, logit_diff = input_grad_foil(model, input_ids, real_tok, foil_tok)
    # causal yardstick (same as elsewhere)
    _, dlogp, _ = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, k_draws, seed
    )

    def split(proxy):
        rows = summarize(labels, proxy)
        return {
            "q_mean": region_mean(proxy, labels, "question"),
            "chaff_mean": region_mean(proxy, labels, "chaff"),
            "chaff_share": rows.get("chaff", {}).get("share_of_sum", 0.0),
        }

    cidx = torch.tensor([i for i, l in enumerate(labels) if l == "chaff"])
    return {
        "question": q["question"], "gold": q["gold"],
        "answer_type": q["answer_type"], "foil_answer": foil_ans,
        "reply_text": reply_text, "logit_diff": logit_diff,
        "logprob_obj": split(proxy_lp),
        "foil_obj": split(proxy_foil),
        # do the two objectives agree on token ranking?
        "spearman_logprob_vs_foil": spearman(proxy_lp, proxy_foil),
        # each objective vs the causal effect
        "spearman_logprob_vs_causal": spearman(proxy_lp, dlogp),
        "spearman_foil_vs_causal": spearman(proxy_foil, dlogp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpleqa", default="../data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--max-q-chars", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--k-draws", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    tok, model = load_model(args.model)
    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)
    questions, by_type = load_simpleqa_typed(args.simpleqa, args.seed,
                                             args.max_q_chars)

    per_item, done = [], 0
    for q in questions:
        if done >= args.n_questions:
            break
        try:
            r = run_one(model, tok, q, by_type, chaff_vocab, args.n_chaff,
                        args.max_new_tokens, args.k_draws, args.seed)
        except Exception as e:
            r = {"error": repr(e)}
        if "error" not in r:
            per_item.append(r)
            done += 1
        if done and done % args.progress_every == 0:
            print(f"[prog] {done}/{args.n_questions}  "
                  f"{done/(time.time()-t0):.2f}/s")

    import statistics as st

    def med(xs):
        xs = [x for x in xs if x == x]
        return st.median(xs) if xs else float("nan")

    print(f"\n=== foil vs log-prob objective (n={len(per_item)}) ===")
    print(f"agreement between objectives (spearman): "
          f"{med(r['spearman_logprob_vs_foil'] for r in per_item):.3f}")
    print(f"log-prob obj chaff share: "
          f"{med(r['logprob_obj']['chaff_share'] for r in per_item):.3f}  "
          f"q/chaff mean: "
          f"{med(r['logprob_obj']['q_mean']/r['logprob_obj']['chaff_mean'] for r in per_item if r['logprob_obj']['chaff_mean']):.2f}")
    print(f"foil     obj chaff share: "
          f"{med(r['foil_obj']['chaff_share'] for r in per_item):.3f}  "
          f"q/chaff mean: "
          f"{med(r['foil_obj']['q_mean']/r['foil_obj']['chaff_mean'] for r in per_item if r['foil_obj']['chaff_mean']):.2f}")
    print(f"log-prob vs causal: "
          f"{med(r['spearman_logprob_vs_causal'] for r in per_item):.3f}  |  "
          f"foil vs causal: "
          f"{med(r['spearman_foil_vs_causal'] for r in per_item):.3f}")

    result = {
        "model": args.model, "dtype": "float32", "objective_comparison": True,
        "n_questions": len(per_item), "n_chaff": args.n_chaff,
        "k_draws": args.k_draws, "seed": args.seed,
        "elapsed_sec": time.time() - t0, "per_item": per_item,
    }
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
