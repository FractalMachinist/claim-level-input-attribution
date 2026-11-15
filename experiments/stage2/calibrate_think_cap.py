#!/usr/bin/env python
"""Calibrate the thinking-ON generation cap: what fraction of SimpleQA items
fail to close </think> within a given cap? Run before the row-8 thinking run so
the cap is chosen from data, not guessed (the earlier 160 and 512 were guesses).

Records, per question, the reply length and whether the think block closed, at a
high cap. The truncation fraction at a candidate cap is then computable from the
lengths (a reply that closed at length L would also close at any cap >= L).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    build_prompt, generate_reply, load_model, pick_chaff_vocab,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpleqa", default="../data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--max-q-chars", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load_model()
    cv = pick_chaff_vocab(tok, seed=args.seed)
    rows = [r for r in csv.DictReader(open(args.simpleqa))
            if len(r["problem"]) <= args.max_q_chars]
    random.Random(args.seed).shuffle(rows)

    close_marker = "</think>"
    recs, t0 = [], time.time()
    for i, r in enumerate(rows[:args.n]):
        iid, _, _ = build_prompt(tok, r["problem"], args.n_chaff, cv,
                                 "before", args.seed, enable_thinking=True)
        rep = generate_reply(model, tok, iid, args.cap)
        txt = tok.decode(rep, skip_special_tokens=False)
        closed = close_marker in txt
        # length at which </think> closes (chars->tokens approx via retokenize of
        # the pre-close span); simpler: token index of close is what matters, so
        # count tokens up to and including the close by re-decoding incrementally
        close_len = None
        if closed:
            pre = txt.split(close_marker, 1)[0] + close_marker
            close_len = len(tok(pre, add_special_tokens=False)["input_ids"])
        recs.append({"reply_len": len(rep), "closed": closed,
                     "close_len": close_len, "hit_cap": len(rep) >= args.cap})
        if (i + 1) % 10 == 0:
            print(f"[prog] {i+1}/{args.n}  {(i+1)/(time.time()-t0):.2f}/s")

    closed = [r for r in recs if r["closed"]]
    import statistics as st
    print(f"\n=== cap calibration (cap={args.cap}, n={len(recs)}) ===")
    print(f"closed </think>: {len(closed)}/{len(recs)} "
          f"({len(closed)/len(recs):.0%})")
    print(f"hit the cap (truncated): "
          f"{sum(r['hit_cap'] for r in recs)}/{len(recs)}")
    if closed:
        cl = sorted(r["close_len"] for r in closed)
        print(f"close_len: min={cl[0]} median={st.median(cl)} "
              f"p90={cl[int(0.9*len(cl))]} max={cl[-1]}")
        # truncation fraction at several candidate caps
        for c in (256, 512, 1024, 1536, 2048):
            lost = sum(1 for r in recs
                       if (r["close_len"] is None) or (r["close_len"] > c))
            print(f"  cap={c:>4}: {lost}/{len(recs)} would truncate "
                  f"({lost/len(recs):.0%})")
    if args.out:
        json.dump({"cap": args.cap, "n": len(recs), "records": recs},
                  open(args.out, "w"), indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
