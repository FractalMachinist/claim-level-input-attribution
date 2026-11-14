#!/usr/bin/env python
"""Stage 1 causal side: measured per-token effect vs the gradient proxy.

The gradient score from stage1_grad_scaffold.py is only a *proxy*. Stage 1's
real question (RESEARCH_PLAN.md) is whether it predicts the effect of an actual
causal intervention. This script measures that effect directly and reports the
rank correlation between proxy and measured effect.

Intervention = single-token resampling under teacher forcing:
  - Fix the generated reply R (greedy, thinking off), as in the proxy script.
  - Baseline: L0 = teacher-forced log-prob of R under the original prompt.
  - For each input position i, replace token i with K random replacement tokens
    (drawn from the same filtered chaff vocab), recompute the teacher-forced
    log-prob of the SAME R, and record the mean |ΔlogP| over the K draws. Also
    record mean per-reply-token KL between original and perturbed next-token
    distributions at the reply positions.
  - That mean |ΔlogP| (and mean KL) is the measured per-position effect.

Teacher forcing keeps R fixed, so there is no sampling noise: the only variation
is the replacement token draw, which we average over (K draws per position). A
"no-change" control (replace a token with itself) gives ~0 and checks the path.

Proxy vs measured: Spearman rank correlation over all input positions, and
separately within the chaff region (the positions the plan most cares about).
Plain gradient L2 norm is the proxy here; grad*input and others come later.

This is O(T * K) forward passes per prompt (T input tokens, K draws). Small for
Qwen3-0.6B at these lengths; batched K to keep it quick.

Precision: fp32 throughout.
"""

import argparse
import json
import time

import torch

from stage1_grad_scaffold import (
    MODEL,
    build_prompt,
    generate_reply,
    input_grad,
    pick_chaff_vocab,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def teacher_forced_logp_and_dist(model, full_ids, inp_len, reply_ids):
    """Return (logp scalar, reply-position logprob rows [L,V]) for reply R.

    No grad. full_ids is [T] on device; reply occupies the last len(reply_ids).
    """
    with torch.no_grad():
        logits = model(full_ids.unsqueeze(0)).logits[0]  # [T,V] fp32
        logp_all = torch.log_softmax(logits, dim=-1)
        rows = []
        total = 0.0
        for j, tokid in enumerate(reply_ids.tolist()):
            pos = inp_len + j - 1
            total += logp_all[pos, tokid].item()
            rows.append(logp_all[pos])  # [V]
        return total, torch.stack(rows)  # scalar, [L,V]


def measured_effect(model, tok, input_ids, reply_ids, chaff_vocab,
                    k_draws=8, seed=0):
    """Per-position mean |ΔlogP| and mean KL from single-token resampling."""
    device = model.device
    inp_len = input_ids.shape[0]
    full = torch.cat([input_ids, reply_ids]).to(device)

    base_logp, base_rows = teacher_forced_logp_and_dist(
        model, full, inp_len, reply_ids
    )  # base_rows: [L,V] log-probs

    g = torch.Generator().manual_seed(seed + 777)
    n_vocab = len(chaff_vocab)

    dlogp = torch.zeros(inp_len)
    dkl = torch.zeros(inp_len)

    base_p = base_rows.exp()  # [L,V]

    for i in range(inp_len):
        # K perturbed copies of `full`, each differing only at position i.
        batch = full.unsqueeze(0).repeat(k_draws, 1).clone()  # [K,T]
        repl = [chaff_vocab[int(torch.randint(n_vocab, (1,), generator=g))]
                for _ in range(k_draws)]
        batch[:, i] = torch.tensor(repl, device=device)
        with torch.no_grad():
            logits = model(batch).logits  # [K,T,V]
            logp = torch.log_softmax(logits, dim=-1)
        # reply-position rows for each draw
        d_lp = 0.0
        kl = 0.0
        for j, tokid in enumerate(reply_ids.tolist()):
            pos = inp_len + j - 1
            new_lp_tok = logp[:, pos, tokid]            # [K]
            d_lp += (new_lp_tok - base_rows[j, tokid])  # [K]
            # KL(base || new) at this reply position, averaged over draws
            new_row = logp[:, pos, :]                   # [K,V]
            kl_j = (base_p[j].unsqueeze(0) *
                    (base_rows[j].unsqueeze(0) - new_row)).sum(-1)  # [K]
            kl += kl_j
        dlogp[i] = d_lp.abs().mean().item()
        dkl[i] = kl.mean().item() / len(reply_ids)

    return base_logp, dlogp, dkl


def spearman(a, b):
    """Spearman rho without scipy. a,b are 1-D tensors."""
    def rank(x):
        order = x.argsort()
        r = torch.zeros_like(x, dtype=torch.float)
        r[order] = torch.arange(len(x), dtype=torch.float)
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="What is the longest bony fish?")
    ap.add_argument("--n-chaff", type=int, default=30)
    ap.add_argument("--position", choices=["before", "after"], default="before")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--k-draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    if torch.cuda.is_available():
        model.cuda()
    print(f"[load] {MODEL} fp32 in {time.time()-t0:.1f}s on {model.device}")

    chaff_vocab = pick_chaff_vocab(tok, seed=args.seed)
    input_ids, labels, _ = build_prompt(
        tok, args.question, args.n_chaff, chaff_vocab, args.position, args.seed
    )
    reply_ids = generate_reply(model, tok, input_ids, args.max_new_tokens)
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)
    print(f"[prompt] {len(input_ids)} tokens  [reply] {reply_text!r}")

    # proxy
    proxy_norm, grad_logp = input_grad(model, input_ids, reply_ids)
    # measured
    base_logp, dlogp, dkl = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, args.k_draws, args.seed
    )

    import numpy as np
    labs = np.array(labels)
    proxy = proxy_norm
    meas = dlogp

    rho_all = spearman(proxy, meas)
    rho_kl = spearman(proxy, dkl)
    # within chaff region only
    cidx = torch.tensor([i for i, l in enumerate(labels) if l == "chaff"])
    rho_chaff = (spearman(proxy[cidx], meas[cidx]) if len(cidx) > 2
                 else float("nan"))

    print(f"[logp] grad-path={grad_logp:.3f}  tf-base={base_logp:.3f} "
          f"(should match)")
    print(f"[spearman] proxy vs |dLogP|  all={rho_all:.3f}  "
          f"chaff-only={rho_chaff:.3f}")
    print(f"[spearman] proxy vs KL       all={rho_kl:.3f}")

    # region means of the measured effect, to compare with the proxy table
    def region_mean(arr, lab):
        idx = [i for i, l in enumerate(labels) if l == lab]
        return float(arr[idx].mean()) if idx else float("nan")
    print(f"{'region':<10}{'proxy_mean':>12}{'|dLogP|_mean':>14}{'KL_mean':>12}")
    for lab in ("question", "chaff", "template"):
        print(f"{lab:<10}{region_mean(proxy,lab):>12.4f}"
              f"{region_mean(dlogp,lab):>14.4f}{region_mean(dkl,lab):>12.4f}")

    if args.out:
        result = {
            "model": MODEL, "dtype": "float32", "question": args.question,
            "n_chaff": args.n_chaff, "position": args.position,
            "seed": args.seed, "k_draws": args.k_draws,
            "reply_text": reply_text,
            "grad_logp": grad_logp, "tf_base_logp": base_logp,
            "spearman_proxy_dlogp_all": rho_all,
            "spearman_proxy_dlogp_chaff": rho_chaff,
            "spearman_proxy_kl_all": rho_kl,
            "labels": labels,
            "proxy_norm": proxy_norm.tolist(),
            "measured_dlogp": dlogp.tolist(),
            "measured_kl": dkl.tolist(),
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
