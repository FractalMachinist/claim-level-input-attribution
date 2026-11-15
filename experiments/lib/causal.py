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


def _draw_replacements(chaff_vocab, k_draws, gen):
    """K replacement token ids drawn from chaff_vocab with generator `gen`.

    Factored out so the serial loop and the batched impl draw *in the same
    order from the same generator* -- the only way per-position numbers can be
    bit-identical between the two. One position consumes exactly k_draws draws.
    """
    n_vocab = len(chaff_vocab)
    return [chaff_vocab[int(torch.randint(n_vocab, (1,), generator=gen))]
            for _ in range(k_draws)]


def _score_positions(model, full, inp_len, reply_ids, base_rows, base_p,
                     positions, repls, device):
    """Core scorer: given, per position, its K replacement tokens, return
    (dlogp_vals, dkl_vals) lists aligned with `positions`.

    `repls[p]` is the length-k_draws replacement list for position `positions[p]`.
    This is the ONE place the forward pass and the ΔlogP / KL reduction live, so
    the serial and batched impls differ only in how they batch the forwards, not
    in what they compute. The batched impl overrides how the forwards are issued
    by passing a different `_forward` through the module-level hook below.
    """
    reply_tok_list = reply_ids.tolist()
    L = len(reply_tok_list)
    dlogp_vals, dkl_vals = [], []
    for p, i in enumerate(positions):
        k_draws = len(repls[p])
        batch = full.unsqueeze(0).repeat(k_draws, 1).clone()  # [K,T]
        batch[:, i] = torch.tensor(repls[p], device=device)
        with torch.no_grad():
            logp = torch.log_softmax(model(batch).logits, dim=-1)  # [K,T,V]
        d_lp = 0.0
        kl = 0.0
        for j, tokid in enumerate(reply_tok_list):
            pos = inp_len + j - 1
            d_lp += (logp[:, pos, tokid] - base_rows[j, tokid])      # [K]
            new_row = logp[:, pos, :]                                # [K,V]
            kl_j = (base_p[j].unsqueeze(0) *
                    (base_rows[j].unsqueeze(0) - new_row)).sum(-1)   # [K]
            kl += kl_j
        dlogp_vals.append(d_lp.abs().mean().item())
        dkl_vals.append(kl.mean().item() / L)
    return dlogp_vals, dkl_vals


def measured_effect_batched(model, tok, input_ids, reply_ids, chaff_vocab,
                            k_draws=8, seed=0, positions=None, pos_batch=1):
    """Batched-signature twin of measured_effect().

    Returns (base_logp, dlogp, dkl) exactly like measured_effect(), where dlogp
    and dkl are full-length [inp_len] tensors. `positions` (a list of input
    indices) restricts measurement to a subset -- unmeasured positions stay 0.
    `pos_batch` is how many input positions to stack into one model forward.

    STEP 1 (dumb wrapper): pos_batch is ignored and each position's forward is
    issued on its own, reusing the identical core scorer as the serial path --
    so this is a drop-in signature match whose numbers MUST equal the serial
    impl's for the same seed/positions. STEP 2 replaces the forward batching
    with genuine multi-position stacking; the same tests must still pass.
    """
    device = model.device
    inp_len = input_ids.shape[0]
    full = torch.cat([input_ids, reply_ids]).to(device)
    base_logp, base_rows = teacher_forced_logp_and_dist(
        model, full, inp_len, reply_ids
    )
    base_p = base_rows.exp()

    if positions is None:
        positions = list(range(inp_len))

    # Draw replacements for ALL inp_len positions in index order, exactly as the
    # serial loop would, so that drawing for a subset still consumes the RNG in
    # the same positions. Then keep only the requested positions' draws.
    g = torch.Generator().manual_seed(seed + 777)
    all_repls = [_draw_replacements(chaff_vocab, k_draws, g)
                 for _ in range(inp_len)]
    repls = [all_repls[i] for i in positions]

    if pos_batch <= 1 or not _BATCHED_FORWARD["enabled"]:
        dlogp_vals, dkl_vals = _score_positions(
            model, full, inp_len, reply_ids, base_rows, base_p,
            positions, repls, device)
    else:
        dlogp_vals, dkl_vals = _score_positions_stacked(
            model, full, inp_len, reply_ids, base_rows, base_p,
            positions, repls, device, k_draws, pos_batch)

    dlogp = torch.zeros(inp_len)
    dkl = torch.zeros(inp_len)
    for p, i in enumerate(positions):
        dlogp[i] = dlogp_vals[p]
        dkl[i] = dkl_vals[p]
    return base_logp, dlogp, dkl


# Flipped to True in Step 2 to activate genuine multi-position batching.
_BATCHED_FORWARD = {"enabled": False}


def _score_positions_stacked(model, full, inp_len, reply_ids, base_rows, base_p,
                             positions, repls, device, k_draws, pos_batch):
    """Step-2 genuine batching: stack several positions' [K,T] blocks into one
    [P*K, T] forward, then reduce each block exactly as the serial scorer does.

    The reduction math is identical to _score_positions; only the forward is
    shared across positions. Results are per-position ΔlogP/KL aligned with
    `positions`.
    """
    reply_tok_list = reply_ids.tolist()
    L = len(reply_tok_list)
    dlogp_vals = [0.0] * len(positions)
    dkl_vals = [0.0] * len(positions)

    # The reply positions are the contiguous slice [inp_len-1, inp_len-1+L) of
    # the T axis (pos = inp_len + j - 1). We only ever read those rows, so we
    # slice them out of the logits BEFORE the full-vocab log_softmax. That turns
    # the dominant tensor from [P*K, T, V] into [P*K, L, V] -- the [*,T,V]
    # materialization was the memory wall that capped pos_batch and starved the
    # speedup (full-vocab softmax over all T positions is pure wasted bandwidth
    # when only L reply rows matter). Numerically identical: log_softmax is per
    # row over the vocab axis, independent across positions.
    rpos_lo = inp_len - 1
    reply_tok_t = torch.tensor(reply_tok_list, device=device)  # [L]

    for start in range(0, len(positions), pos_batch):
        chunk = list(range(start, min(start + pos_batch, len(positions))))
        P = len(chunk)
        # Build [P*K, T]: block p is full repeated K times, perturbed at its pos.
        block = full.unsqueeze(0).repeat(P * k_draws, 1).clone()  # [P*K, T]
        for bp, p in enumerate(chunk):
            i = positions[p]
            rows = slice(bp * k_draws, (bp + 1) * k_draws)
            block[rows, i] = torch.tensor(repls[p], device=device)
        with torch.no_grad():
            logits = model(block).logits[:, rpos_lo:rpos_lo + L, :]  # [P*K,L,V]
            logp = torch.log_softmax(logits, dim=-1)                 # [P*K,L,V]
        # Vectorized reduction over reply positions j (local index == j after
        # the slice above). base_rows/base_p are [L,V]; broadcast over [*,L,V].
        # ΔlogP of the chosen reply token at each j, summed over j:
        chosen = logp.gather(                       # [P*K, L]
            2, reply_tok_t.view(1, L, 1).expand(logp.shape[0], L, 1)
        ).squeeze(2)
        base_chosen = base_rows.gather(1, reply_tok_t.view(L, 1)).squeeze(1)  # [L]
        d_lp_all = (chosen - base_chosen.unsqueeze(0)).sum(1)   # [P*K]
        # KL(base||new) per j, summed over j:
        kl_all = (base_p.unsqueeze(0) *
                  (base_rows.unsqueeze(0) - logp)).sum(-1).sum(-1)  # [P*K]
        for bp, p in enumerate(chunk):
            sl = slice(bp * k_draws, (bp + 1) * k_draws)
            dlogp_vals[p] = d_lp_all[sl].abs().mean().item()
            dkl_vals[p] = kl_all[sl].mean().item() / L
    return dlogp_vals, dkl_vals


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
