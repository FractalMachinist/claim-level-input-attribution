#!/usr/bin/env python
"""Stage 1 procedure: measured causal effect vs the gradient proxy, one prompt.

The gradient score is only a *proxy*. Stage 1's real question is whether it
predicts the effect of an actual causal intervention. This measures that effect
directly (single-token resampling under teacher forcing) and reports the Spearman
rank correlation between proxy and measured effect, overall and within the chaff
region. Machinery: lib/attribution.py (proxy) and lib/causal.py (measurement).

Run from experiments/:  python stage1/run_causal.py --n-chaff 30 --out stage1/runs/x.json
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
from lib.causal import measured_effect, spearman  # noqa: E402


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
    reply_text = tok.decode(reply_ids, skip_special_tokens=True)
    print(f"[prompt] {len(input_ids)} tokens  [reply] {reply_text!r}")

    # proxy
    proxy_norm, grad_logp = input_grad(model, input_ids, reply_ids)
    # measured
    base_logp, dlogp, dkl = measured_effect(
        model, tok, input_ids, reply_ids, chaff_vocab, args.k_draws, args.seed
    )

    rho_all = spearman(proxy_norm, dlogp)
    rho_kl = spearman(proxy_norm, dkl)
    cidx = torch.tensor([i for i, l in enumerate(labels) if l == "chaff"])
    rho_chaff = (spearman(proxy_norm[cidx], dlogp[cidx]) if len(cidx) > 2
                 else float("nan"))

    print(f"[logp] grad-path={grad_logp:.3f}  tf-base={base_logp:.3f} "
          f"(should match)")
    print(f"[spearman] proxy vs |dLogP|  all={rho_all:.3f}  "
          f"chaff-only={rho_chaff:.3f}")
    print(f"[spearman] proxy vs KL       all={rho_kl:.3f}")

    def region_mean(arr, lab):
        idx = [i for i, l in enumerate(labels) if l == lab]
        return float(arr[idx].mean()) if idx else float("nan")
    print(f"{'region':<10}{'proxy_mean':>12}{'|dLogP|_mean':>14}{'KL_mean':>12}")
    for lab in ("question", "chaff", "template"):
        print(f"{lab:<10}{region_mean(proxy_norm,lab):>12.4f}"
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
