#!/usr/bin/env python
"""Stage 1 noise floor: how consistently can the measured effect be measured?

The causal script estimates each position's effect as mean |ΔlogP| over K
replacement draws. That estimate has sampling error (which tokens got drawn).
If a position's effect can't be measured consistently, no proxy can be expected
to rank it correctly -- so the ceiling on proxy-vs-measured rank agreement is
set by the measurement's own self-consistency, not by the proxy.

Method:
  - For each input position, draw TWO independent sets of K replacements and
    compute two independent effect estimates, eA[i] and eB[i].
  - Split-half reliability = Spearman(eA, eB) over positions (and within chaff).
    This is an upper bound on any proxy's achievable rank correlation with the
    "true" effect: a proxy cannot correlate with the true ranking better than
    two independent measurements of it correlate with each other (attenuation).
  - Also report a per-position noise band: std of the per-draw |ΔlogP| across
    the 2K draws, so we can see which positions are simply unmeasurable at K.
  - Corrected proxy agreement: rho(proxy, mean(eA,eB)) / sqrt(reliability) is a
    rough disattenuated estimate of the proxy's true-effect correlation.

Pairs with stage1_causal.py (same intervention, same teacher forcing, fp32).
"""

import argparse
import json
import time

import torch

from stage1_causal import spearman, teacher_forced_logp_and_dist
from stage1_grad_scaffold import (
    MODEL,
    build_prompt,
    generate_reply,
    input_grad,
    pick_chaff_vocab,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def effect_estimate(model, full, inp_len, reply_ids, base_rows,
                    chaff_vocab, k_draws, gen):
    """One |ΔlogP| estimate per position from K draws (independent of others)."""
    device = model.device
    n_vocab = len(chaff_vocab)
    eff = torch.zeros(inp_len)
    for i in range(inp_len):
        batch = full.unsqueeze(0).repeat(k_draws, 1).clone()
        repl = [chaff_vocab[int(torch.randint(n_vocab, (1,), generator=gen))]
                for _ in range(k_draws)]
        batch[:, i] = torch.tensor(repl, device=device)
        with torch.no_grad():
            logp = torch.log_softmax(model(batch).logits, dim=-1)  # [K,T,V]
        d = 0.0
        for j, tokid in enumerate(reply_ids.tolist()):
            pos = inp_len + j - 1
            d = d + (logp[:, pos, tokid] - base_rows[j, tokid])
        eff[i] = d.abs().mean().item()
    return eff


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
    inp_len = input_ids.shape[0]
    full = torch.cat([input_ids, reply_ids]).to(model.device)
    base_logp, base_rows = teacher_forced_logp_and_dist(
        model, full, inp_len, reply_ids
    )

    proxy, _ = input_grad(model, input_ids, reply_ids)

    genA = torch.Generator().manual_seed(args.seed + 101)
    genB = torch.Generator().manual_seed(args.seed + 202)
    eA = effect_estimate(model, full, inp_len, reply_ids, base_rows,
                         chaff_vocab, args.k_draws, genA)
    eB = effect_estimate(model, full, inp_len, reply_ids, base_rows,
                         chaff_vocab, args.k_draws, genB)
    emean = (eA + eB) / 2

    rel_all = spearman(eA, eB)
    rho_proxy = spearman(proxy, emean)
    cidx = [i for i, l in enumerate(labels) if l == "chaff"]
    cidx_t = torch.tensor(cidx)
    rel_chaff = spearman(eA[cidx_t], eB[cidx_t]) if len(cidx) > 2 else float("nan")
    rho_proxy_chaff = (spearman(proxy[cidx_t], emean[cidx_t])
                       if len(cidx) > 2 else float("nan"))

    def disatt(rho, rel):
        return rho / (rel ** 0.5) if rel and rel > 0 else float("nan")

    print(f"[reliability] split-half Spearman(eA,eB)  "
          f"all={rel_all:.3f}  chaff={rel_chaff:.3f}")
    print(f"[proxy vs mean effect]  all={rho_proxy:.3f}  "
          f"chaff={rho_proxy_chaff:.3f}")
    print(f"[disattenuated proxy]   all={disatt(rho_proxy, rel_all):.3f}  "
          f"chaff={disatt(rho_proxy_chaff, rel_chaff):.3f}")
    print("  (disattenuated = proxy agreement corrected for measurement noise;")
    print("   >1 or nan means the measurement floor, not the proxy, is limiting)")

    if args.out:
        json.dump({
            "model": MODEL, "n_chaff": args.n_chaff, "position": args.position,
            "seed": args.seed, "k_draws": args.k_draws,
            "reliability_all": rel_all, "reliability_chaff": rel_chaff,
            "rho_proxy_all": rho_proxy, "rho_proxy_chaff": rho_proxy_chaff,
            "disatt_all": disatt(rho_proxy, rel_all),
            "disatt_chaff": disatt(rho_proxy_chaff, rel_chaff),
            "labels": labels,
            "proxy": proxy.tolist(), "eA": eA.tolist(), "eB": eB.tolist(),
        }, open(args.out, "w"), indent=2)
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
