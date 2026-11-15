#!/usr/bin/env python
"""Stage 2 negative case (row 6, in scope): irrelevant context must NOT light up.

The product claim is "flag when the model attended to what it SHOULDN'T."
The direct test: give the model a closed-book question it answers from its own
weights (SimpleQA -- Stage 1 showed 0.6B confidently answers these), with an
IRRELEVANT context paragraph stapled in front. The proxy should score that
context near the floor, and resampling it should barely move the reply. If the
proxy lights irrelevant-but-real-prose context, that is a real failure -- a
harder null than Stage 1's random chaff, because the distractor is fluent text
with the surface form of relevance.

Two conditions (Zach's call) to see whether topical surface-similarity fools it:
  - UNRELATED: context topic has nothing to do with the question.
  - SAME-TOPIC: context is from the same broad topic (both sports, both science)
    -- surface-similar, still not answer-bearing. The harder distractor.
Compare: does the proxy light SAME-TOPIC context more than UNRELATED?

Context source: SQuAD dev paragraphs (real prose). Question source: SimpleQA
(closed-book). Topic pairing is heuristic (SQuAD article title -> coarse topic).

Regions: irrelevant_context / question / template. NO gold_span -- by
construction nothing in the context should matter. Metric: the context's share
of proxy and its causal effect, vs a floor. Reuses lib/ unchanged.

Precision fp32. Run from experiments/:
  python stage2/run_negative.py --n-items 60 --out stage2/runs/negative.json
"""

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    MODEL, generate_reply, input_grad, load_model, pick_chaff_vocab, summarize,
)
from lib.causal import measured_effect, spearman  # noqa: E402

# Coarse map from SQuAD article title -> a SimpleQA-style topic, for same-topic
# pairing. Titles not listed fall in "Other". Heuristic, documented as such.
SQUAD_TOPIC = {
    # Sports
    "Super_Bowl_50": "Sports",
    # Science and technology
    "Nikola_Tesla": "Science and technology",
    "Computational_complexity_theory": "Science and technology",
    "Steam_engine": "Science and technology", "Prime_number": "Science and technology",
    "Oxygen": "Science and technology", "Packet_switching": "Science and technology",
    "Force": "Science and technology", "Geology": "Science and technology",
    "Immune_system": "Science and technology", "Chloroplast": "Science and technology",
    "Ctenophora": "Science and technology", "Pharmacy": "Science and technology",
    "Apollo_program": "Science and technology", "Construction": "Science and technology",
    "Intergovernmental_Panel_on_Climate_Change": "Science and technology",
    # History
    "Martin_Luther": "History", "Normans": "History", "Huguenot": "History",
    "Genghis_Khan": "History", "Yuan_dynasty": "History", "Black_Death": "History",
    "French_and_Indian_War": "History", "Imperialism": "History",
    "1973_oil_crisis": "History", "Islamism": "History",
    # Geography
    "Warsaw": "Geography", "Victoria_(Australia)": "Geography",
    "Southern_California": "Geography", "Rhine": "Geography", "Kenya": "Geography",
    "Newcastle_upon_Tyne": "Geography", "Jacksonville,_Florida": "Geography",
    "Fresno,_California": "Geography", "Amazon_rainforest": "Geography",
    # Politics
    "Scottish_Parliament": "Politics", "European_Union_law": "Politics",
    "Civil_disobedience": "Politics", "Economic_inequality": "Politics",
    # TV / media
    "Sky_(United_Kingdom)": "TV shows", "Doctor_Who": "TV shows",
    "American_Broadcasting_Company": "TV shows",
    # Other (education / institutions / religion)
    "Teacher": "Other", "Private_school": "Other", "Harvard_University": "Other",
    "University_of_Chicago": "Other", "United_Methodist_Church": "Other",
    "Victoria_and_Albert_Museum": "Art",
}


def load_simpleqa(path, seed, max_q_chars):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if len(r["problem"]) > max_q_chars:
            continue
        topic = ast.literal_eval(r["metadata"]).get("topic")
        out.append({"question": r["problem"], "topic": topic})
    random.Random(seed).shuffle(out)
    return out


def load_squad_contexts(path, max_ctx_chars):
    d = json.load(open(path))
    out = []
    for art in d["data"]:
        topic = SQUAD_TOPIC.get(art["title"], "Other")
        for p in art["paragraphs"]:
            ctx = p["context"]
            if len(ctx) <= max_ctx_chars:
                out.append({"context": ctx, "topic": topic})
    return out


def build_pairs(questions, contexts, n_items, seed):
    """Make UNRELATED and SAME-TOPIC (q, context) pairs."""
    rng = random.Random(seed + 9)
    by_topic = {}
    for c in contexts:
        by_topic.setdefault(c["topic"], []).append(c)
    topics = [t for t, cs in by_topic.items() if cs]

    pairs = []
    for q in questions:
        if len(pairs) >= n_items:
            break
        qt = q["topic"]
        # SAME-TOPIC: a context whose coarse topic matches the question's, if any
        same_pool = by_topic.get(qt, [])
        same = rng.choice(same_pool) if same_pool else None
        # UNRELATED: a context from a different topic
        other_topics = [t for t in topics if t != qt and by_topic[t]]
        unrel = (rng.choice(by_topic[rng.choice(other_topics)])
                 if other_topics else None)
        if same:
            pairs.append({"question": q["question"], "q_topic": qt,
                          "context": same["context"], "cond": "same_topic"})
        if unrel:
            pairs.append({"question": q["question"], "q_topic": qt,
                          "context": unrel["context"], "cond": "unrelated"})
    return pairs[:n_items]


def build_neg_prompt(tok, question, context):
    """{context}\n\nQuestion: {q}\nAnswer...  with regions irrelevant_context /
    question / template. No gold_span: nothing in context should matter."""
    q_lead = "\n\nQuestion: "
    instr = "\nAnswer in as few words as possible."
    user_content = context + q_lead + question + instr
    msgs = [{"role": "user", "content": user_content}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    base = text.find(user_content)
    if base < 0:
        raise RuntimeError("user_content not verbatim in template")
    ctx_span = (base, base + len(context))
    q_lo = base + len(context) + len(q_lead)
    q_span = (q_lo, q_lo + len(question))

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    def region_of(a, b):
        if b <= a:
            return "template"
        if a >= q_span[0] and b <= q_span[1]:
            return "question"
        if a >= ctx_span[0] and b <= ctx_span[1]:
            return "irrelevant_context"
        return "template"

    labels = [region_of(a, b) for (a, b) in offsets]
    return input_ids, labels


def region_mean(arr, labels, lab):
    idx = [i for i, l in enumerate(labels) if l == lab]
    return float(arr[idx].mean()) if idx else float("nan")


def run_one(model, tok, pair, chaff_vocab, max_new_tokens, k_draws, seed):
    input_ids, labels = build_neg_prompt(tok, pair["question"], pair["context"])
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens)
    if len(reply_ids) == 0:
        return {"error": "empty reply", "cond": pair["cond"]}
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)

    proxy, grad_logp = input_grad(model, input_ids, reply_ids)
    base_logp, dlogp, dkl = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, k_draws, seed
    )
    rows = summarize(labels, proxy)

    # The question SHOULD matter, the irrelevant_context should NOT. The key
    # quantities: context's per-token proxy relative to the question's (should
    # be << 1 if the proxy correctly ignores it), and its causal effect.
    q_pm = region_mean(proxy, labels, "question")
    c_pm = region_mean(proxy, labels, "irrelevant_context")
    q_cm = region_mean(dlogp, labels, "question")
    c_cm = region_mean(dlogp, labels, "irrelevant_context")
    return {
        "cond": pair["cond"],
        "q_topic": pair["q_topic"],
        "question": pair["question"],
        "reply_text": reply_text,
        "reply_logp": grad_logp,
        "n_input_tokens": len(input_ids),
        "proxy_question_mean": q_pm,
        "proxy_context_mean": c_pm,
        "proxy_context_to_question": (c_pm / q_pm) if q_pm else float("nan"),
        "causal_question_mean": q_cm,
        "causal_context_mean": c_cm,
        "causal_context_to_question": (c_cm / q_cm) if q_cm else float("nan"),
        "context_share_of_proxy":
            rows.get("irrelevant_context", {}).get("share_of_sum", 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpleqa", default="../data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--squad", default="../data/squad/dev-v1.1.json")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--max-q-chars", type=int, default=120)
    ap.add_argument("--max-ctx-chars", type=int, default=400)
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

    questions = load_simpleqa(args.simpleqa, args.seed, args.max_q_chars)
    contexts = load_squad_contexts(args.squad, args.max_ctx_chars)
    pairs = build_pairs(questions, contexts, args.n_items, args.seed)
    print(f"[data] {len(pairs)} pairs "
          f"({sum(p['cond']=='same_topic' for p in pairs)} same-topic, "
          f"{sum(p['cond']=='unrelated' for p in pairs)} unrelated)")

    per_item, done = [], 0
    for p in pairs:
        try:
            r = run_one(model, tok, p, chaff_vocab, args.max_new_tokens,
                        args.k_draws, args.seed)
        except Exception as e:
            r = {"error": repr(e), "cond": p["cond"]}
        per_item.append(r)
        done += 1
        if done % args.progress_every == 0:
            print(f"[prog] {done}/{len(pairs)}  {done/(time.time()-t0):.2f}/s")

    import statistics as st

    def med(xs):
        xs = [x for x in xs if x == x]
        return st.median(xs) if xs else float("nan")

    print("\n=== negative case: irrelevant context should NOT matter ===")
    print("  (ctx/q < 1 means the proxy/causal correctly ranks the irrelevant")
    print("   context BELOW the question; lower is better)")
    for cond in ("unrelated", "same_topic"):
        rs = [r for r in per_item if r.get("cond") == cond and "error" not in r]
        if not rs:
            continue
        print(f"[{cond:>11}] n={len(rs):>3}  "
              f"proxy ctx/q={med(r['proxy_context_to_question'] for r in rs):.3f}  "
              f"causal ctx/q={med(r['causal_context_to_question'] for r in rs):.3f}  "
              f"ctx share of proxy={med(r['context_share_of_proxy'] for r in rs):.3f}")

    result = {
        "model": args.model, "dtype": "float32",
        "test": "negative_case_irrelevant_context",
        "n_items": len(pairs), "k_draws": args.k_draws, "seed": args.seed,
        "elapsed_sec": time.time() - t0,
        "per_item": per_item,
    }
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
