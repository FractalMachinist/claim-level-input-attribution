#!/usr/bin/env python
"""Stage 1 cross-prompt generalization over SimpleQA.

Stage 1 so far (stage1_causal.py) established, on ONE hand-written question, that
the plain-gradient proxy predicts the measured causal effect of resampling chaff
tokens. This script tests whether that holds ACROSS MANY prompts, using a
recognizable external benchmark instead of hand-written questions, so we don't
have to defend a homemade question set.

Benchmark: SimpleQA (OpenAI; Wei et al. 2024). 4,326 short closed-book factual
questions, answered from the model's own parametric knowledge -- no supporting
context. That is exactly the Stage 1 regime: the whole prompt is
{question tokens + N chaff tokens}, and NOTHING in it should matter except the
question. The answer span lives in the model's weights, not the prompt, which is
what separates this from a Stage 2 extractive (read-it-from-context) task.

NO FILTERING on confidence or correctness (deliberate, see RESEARCH_PLAN.md
discussion): the Stage 1 claim -- "chaff with no epistemic reason to be attended
to scores near the noise floor, and the proxy tracks the causal effect" -- holds
whether or not the model knows the answer. A confidently-confabulated answer is a
valid datapoint: the model produced *some* reply, and we test whether gradient
attribution correctly says it did NOT come from the random chaff. Filtering would
select a friendly subset and invite a cherry-picking critique. We RECORD the
reply's self-confidence (teacher-forced log-prob) as a covariate, but never gate
on it -- so we can later ask "does proxy quality depend on model confidence?"
without having excluded the low-confidence items that would answer it.

Per question we reuse the trusted Stage 1 core unchanged (build_prompt,
generate_reply, input_grad, measured_effect) and record, for each N:
  - region split of the proxy (question / chaff / template),
  - Spearman(proxy, |dLogP|) and Spearman(proxy, KL), over all positions and
    within the chaff region,
  - region means of proxy and measured effect,
  - reply text + self log-prob (the confidence covariate).

The generalization result is the DISTRIBUTION of these across questions: does the
proxy predict the causal effect on most prompts, and does chaff stay near the
floor on most prompts? Aggregates are written alongside the per-question rows.

Precision: fp32 throughout (RESEARCH_PLAN.md: settled, prevents the low-precision
numerical-noise confound at the Stage 1 floor).
"""

import argparse
import ast
import csv
import json
import random
import time

import torch

from stage1_grad_scaffold import (
    MODEL,
    build_prompt,
    generate_reply,
    input_grad,
    pick_chaff_vocab,
    summarize,
)
from stage1_causal import measured_effect, spearman
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_simpleqa(path, n_questions, seed, max_q_chars):
    """Load SimpleQA rows, drop over-long questions, sample n_questions."""
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if len(r["problem"]) <= max_q_chars]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n_questions]:
        meta = ast.literal_eval(r["metadata"])
        out.append({
            "question": r["problem"],
            "gold": r["answer"],
            "topic": meta.get("topic"),
            "answer_type": meta.get("answer_type"),
        })
    return out


def run_one(model, tok, chaff_vocab, question, n_chaff, position,
            max_new_tokens, k_draws, seed):
    """Run the full Stage 1 proxy-vs-causal comparison for one question/N."""
    input_ids, labels, _ = build_prompt(
        tok, question, n_chaff, chaff_vocab, position, seed
    )
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens)
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)
    if len(reply_ids) == 0:
        return None  # empty reply -> nothing to attribute

    proxy_norm, grad_logp = input_grad(model, input_ids, reply_ids)
    base_logp, dlogp, dkl = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, k_draws, seed
    )

    rho_all = spearman(proxy_norm, dlogp)
    rho_kl = spearman(proxy_norm, dkl)
    cidx = torch.tensor([i for i, l in enumerate(labels) if l == "chaff"])
    rho_chaff = (spearman(proxy_norm[cidx], dlogp[cidx])
                 if len(cidx) > 2 else float("nan"))

    rows = summarize(labels, proxy_norm)

    def region_mean(arr, lab):
        idx = [i for i, l in enumerate(labels) if l == lab]
        return float(arr[idx].mean()) if idx else float("nan")

    return {
        "n_chaff": n_chaff,
        "position": position,
        "n_input_tokens": len(input_ids),
        "n_reply_tokens": len(reply_ids),
        "reply_text": reply_text,
        "reply_logp": grad_logp,          # confidence covariate (not a filter)
        "tf_base_logp": base_logp,
        "spearman_proxy_dlogp_all": rho_all,
        "spearman_proxy_dlogp_chaff": rho_chaff,
        "spearman_proxy_kl_all": rho_kl,
        "proxy_region": {
            lab: {"n_tokens": rows[lab]["n_tokens"],
                  "share_of_sum": rows[lab]["share_of_sum"],
                  "mean_norm": rows[lab]["mean_norm"]}
            for lab in rows
        },
        "measured_region_mean": {
            lab: {"dlogp": region_mean(dlogp, lab),
                  "kl": region_mean(dkl, lab)}
            for lab in ("question", "chaff", "template")
        },
    }


def nanmedian(xs):
    import math
    v = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def frac(xs, pred):
    import math
    v = [x for x in xs if x is not None and not math.isnan(x)]
    return (sum(1 for x in v if pred(x)) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--n-chaff", type=int, nargs="+", default=[30])
    ap.add_argument("--positions", nargs="+", default=["before"],
                    choices=["before", "after"])
    ap.add_argument("--max-q-chars", type=int, default=140,
                    help="skip questions longer than this (keep prompts compact)")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--k-draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    if torch.cuda.is_available():
        model.cuda()
    print(f"[load] {MODEL} fp32 in {time.time()-t0:.1f}s on {model.device}")

    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)
    questions = load_simpleqa(args.csv, args.n_questions, args.seed,
                              args.max_q_chars)
    print(f"[data] {len(questions)} questions; N={args.n_chaff}; "
          f"positions={args.positions}; k_draws={args.k_draws}")

    per_q = []
    t_run = time.time()
    total_units = len(questions) * len(args.n_chaff) * len(args.positions)
    done = 0
    for qi, q in enumerate(questions):
        q_rec = {"q_index": qi, "question": q["question"], "gold": q["gold"],
                 "topic": q["topic"], "answer_type": q["answer_type"],
                 "runs": []}
        for n_chaff in args.n_chaff:
            for position in args.positions:
                try:
                    r = run_one(model, tok, chaff_vocab, q["question"],
                                n_chaff, position, args.max_new_tokens,
                                args.k_draws, args.seed)
                except Exception as e:  # keep the sweep alive on a bad prompt
                    r = {"n_chaff": n_chaff, "position": position,
                         "error": repr(e)}
                if r is not None:
                    q_rec["runs"].append(r)
                done += 1
                if done % args.progress_every == 0:
                    dt = time.time() - t_run
                    rate = done / dt
                    eta = (total_units - done) / rate if rate else 0
                    print(f"[prog] {done}/{total_units} units  "
                          f"{rate:.2f}/s  eta {eta/60:.1f}m")
        per_q.append(q_rec)

    # ---- aggregate across prompts, per (N, position) ----
    aggregates = {}
    for n_chaff in args.n_chaff:
        for position in args.positions:
            key = f"N{n_chaff}_{position}"
            runs = [run for q in per_q for run in q["runs"]
                    if run.get("n_chaff") == n_chaff
                    and run.get("position") == position
                    and "error" not in run]
            if not runs:
                continue
            rho_all = [r["spearman_proxy_dlogp_all"] for r in runs]
            rho_kl = [r["spearman_proxy_kl_all"] for r in runs]
            # chaff vs question mean-norm (proxy) across prompts
            chaff_share = [r["proxy_region"].get("chaff", {}).get("share_of_sum")
                           for r in runs]
            q_mean = [r["proxy_region"].get("question", {}).get("mean_norm")
                      for r in runs]
            c_mean = [r["proxy_region"].get("chaff", {}).get("mean_norm")
                      for r in runs]
            # per-prompt: does question out-rank chaff in mean proxy?
            q_gt_c = [(qm is not None and cm is not None and qm > cm)
                      for qm, cm in zip(q_mean, c_mean)]
            aggregates[key] = {
                "n_prompts": len(runs),
                "median_rho_proxy_dlogp": nanmedian(rho_all),
                "frac_rho_dlogp_pos": frac(rho_all, lambda x: x > 0),
                "median_rho_proxy_kl": nanmedian(rho_kl),
                "frac_rho_kl_pos": frac(rho_kl, lambda x: x > 0),
                "median_chaff_share_of_proxy": nanmedian(chaff_share),
                "frac_question_mean_gt_chaff_mean":
                    (sum(q_gt_c) / len(q_gt_c)) if q_gt_c else float("nan"),
            }

    print("\n=== cross-prompt aggregates ===")
    for key, a in aggregates.items():
        print(f"[{key}] prompts={a['n_prompts']}  "
              f"median_rho(proxy,|dLogP|)={a['median_rho_proxy_dlogp']:.3f} "
              f"(>{0}: {a['frac_rho_dlogp_pos']:.0%})  "
              f"median_rho(proxy,KL)={a['median_rho_proxy_kl']:.3f}  "
              f"chaff_share={a['median_chaff_share_of_proxy']:.3f}  "
              f"q>chaff mean on {a['frac_question_mean_gt_chaff_mean']:.0%}")

    result = {
        "model": MODEL, "dtype": "float32", "benchmark": "SimpleQA",
        "n_questions": len(questions), "n_chaff": args.n_chaff,
        "positions": args.positions, "k_draws": args.k_draws,
        "max_q_chars": args.max_q_chars, "max_new_tokens": args.max_new_tokens,
        "seed": args.seed, "no_filter": True,
        "elapsed_sec": time.time() - t0,
        "aggregates": aggregates,
        "per_question": per_q,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[out] wrote {args.out}")
    else:
        print("[warn] no --out; results not saved")


if __name__ == "__main__":
    main()
