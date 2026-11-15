#!/usr/bin/env python
"""Stage 1 procedure: plain-gradient region split for one prompt at one N.

The smallest test of the core mechanic (README "The idea", step 3). Builds a
prompt of N random "chaff" tokens plus a simple factual question, generates a
short reply with thinking OFF, computes the plain gradient of the reply's
teacher-forced log-probability w.r.t. the input embeddings, and reports how the
gradient magnitude splits across prompt regions (question / chaff / template).

This establishes the plumbing and the per-prompt region split. The causal
(resample) comparison is run_causal.py; the cross-prompt generalization is
run_simpleqa.py. Machinery lives in lib/attribution.py.

Run from experiments/:  python stage1/run_chaff_sweep.py --n-chaff 30 --out stage1/runs/x.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    MODEL, build_prompt, generate_reply, input_grad, load_model,
    pick_chaff_vocab, summarize,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="What is the longest bony fish?")
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--position", choices=["before", "after"], default="before")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()

    tok, model = load_model()

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
