#!/usr/bin/env python
"""Reusable causal-measurement machinery -- import this, do not run it.

The causal yardstick for Stage 1 (and the basis for later stages): single-token
resampling under teacher forcing. The gradient proxy (lib/attribution.py) is
only a *proxy*; these functions measure the actual effect of perturbing each
input token, so a procedure can compare the two.

Teacher forcing keeps the generated reply R fixed, so there is no sampling
noise: the only variation is the replacement-token draw, averaged over K draws
per position. Precision: fp32 throughout.
"""

import torch


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


def effect_estimate(model, full, inp_len, reply_ids, base_rows,
                    chaff_vocab, k_draws, gen):
    """One |ΔlogP| estimate per position from K draws (independent of others).

    Used by the noise-floor procedure: two independent calls with different
    generators give a split-half reliability estimate (the ceiling on any
    proxy's achievable rank correlation with the true effect).
    """
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
