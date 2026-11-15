#!/usr/bin/env python
"""Stage 1 procedure: measurement noise floor (split-half reliability).

The causal script estimates each position's effect as mean |ΔlogP| over K
replacement draws. That estimate has sampling error. If a position's effect
can't be measured consistently, no proxy can be expected to rank it correctly --
so the ceiling on proxy-vs-measured rank agreement is set by the measurement's
own self-consistency, not by the proxy.

Method: two independent K-draw effect estimates per position (eA, eB).
Split-half reliability = Spearman(eA, eB) is an upper bound on any proxy's
achievable rank correlation with the true effect; proxy agreement divided by
sqrt(reliability) is a rough disattenuated estimate of the true correlation.
Machinery: lib/attribution.py and lib/causal.py.

Run from experiments/:  python stage1/run_noise_floor.py --n-chaff 30 --out stage1/runs/x.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.attribution import (  # noqa: E402
    MODEL, build_prompt, generate_reply, input_grad, load_model, pick_chaff_vocab,
)
from lib.causal import (  # noqa: E402
    effect_estimate, spearman, teacher_forced_logp_and_dist,
)


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

    tok, model = load_model()

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
