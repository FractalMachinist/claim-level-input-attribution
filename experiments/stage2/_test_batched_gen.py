#!/usr/bin/env python
"""Equivalence test: generate_replies_batched must match per-item generate_reply.

Non-negotiable before swapping batched generation into the trusted drivers.
Runs several real SimpleQA prompts (mixed lengths, thinking on and off) through
both paths and reports any mismatch.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    build_prompt, generate_replies_batched, generate_reply, load_model,
    pick_chaff_vocab,
)

tok, model = load_model()
cv = pick_chaff_vocab(tok, seed=0)
rows = [r for r in csv.DictReader(open("../data/simpleqa/simple_qa_test_set.csv"))
        if len(r["problem"]) <= 120]
random.Random(1).shuffle(rows)

for thinking, cap in [(False, 16), (True, 256)]:
    prompts = []
    for r in rows[:6]:
        iid, _, _ = build_prompt(tok, r["problem"], 30, cv, "before", 0,
                                 enable_thinking=thinking)
        prompts.append(iid)
    serial = [generate_reply(model, tok, p, cap) for p in prompts]
    batched = generate_replies_batched(model, tok, prompts, cap)

    print(f"\n=== thinking={thinking} cap={cap} ===")
    n_match = 0
    for i, (s, b) in enumerate(zip(serial, batched)):
        same = (s.shape == b.shape) and bool((s == b).all())
        n_match += same
        if not same:
            print(f"  MISMATCH item {i}: serial_len={len(s)} batched_len={len(b)}")
            print(f"    serial : {tok.decode(s, skip_special_tokens=True)[:50]!r}")
            print(f"    batched: {tok.decode(b, skip_special_tokens=True)[:50]!r}")
    print(f"  {n_match}/{len(serial)} replies identical")
