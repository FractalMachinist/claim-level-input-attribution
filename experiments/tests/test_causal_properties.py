"""Correctness alarms for the causal ground-truth measurement.

These test PROPERTIES a correct per-token causal measurement must have -- they
are built to FAIL on a real bug in the serial impl, not merely to confirm that
serial and batched agree. The finite-difference oracle in particular never
reuses the code under test: it recomputes ΔlogP from two independent model
forwards by hand. If that one fails, the whole project's ground truth is wrong.

The same suite runs against measured_effect_batched regardless of whether its
body is the Step-1 dumb wrapper or the Step-2 genuine batching (set via
lib.causal._BATCHED_FORWARD / pos_batch). Nothing here assumes the current
published numbers are correct.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib.causal as causal  # noqa: E402
from lib.causal import (  # noqa: E402
    measured_effect, measured_effect_batched, teacher_forced_logp_and_dist,
)

K = 8
SEED = 0


# --------------------------------------------------------------------------
# Property: the batched-signature twin reproduces the serial impl EXACTLY.
# Alarms if the drop-in interface computes anything different from the serial
# ground truth it stands in for (same seed => bit-identical per position).
# --------------------------------------------------------------------------
def test_batched_matches_serial_full(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    b0, d0, k0 = measured_effect(model, tok, input_ids, reply_ids,
                                 chaff_vocab, k_draws=K, seed=SEED)
    b1, d1, k1 = measured_effect_batched(model, tok, input_ids, reply_ids,
                                         chaff_vocab, k_draws=K, seed=SEED)
    assert abs(b0 - b1) < 1e-6
    assert torch.allclose(d0, d1, atol=1e-6), (d0 - d1).abs().max()
    assert torch.allclose(k0, k1, atol=1e-6), (k0 - k1).abs().max()


# --------------------------------------------------------------------------
# Property: DETERMINISM -- same seed must give the same numbers every call.
# Alarms if any hidden nondeterminism (unseeded RNG, order-dependent state)
# leaks into the measurement.
# --------------------------------------------------------------------------
def test_determinism(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    _, d_a, k_a = measured_effect_batched(model, tok, input_ids, reply_ids,
                                          chaff_vocab, k_draws=K, seed=SEED)
    _, d_b, k_b = measured_effect_batched(model, tok, input_ids, reply_ids,
                                          chaff_vocab, k_draws=K, seed=SEED)
    assert torch.equal(d_a, d_b)
    assert torch.equal(k_a, k_b)


# --------------------------------------------------------------------------
# Property: SEED SENSITIVITY -- different seeds must actually change the draws
# (and thus the numbers). Alarms if the seed is ignored, which would make the
# noise-floor / split-half reliability estimates silently meaningless.
# --------------------------------------------------------------------------
def test_seed_changes_result(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    _, d_a, _ = measured_effect_batched(model, tok, input_ids, reply_ids,
                                        chaff_vocab, k_draws=K, seed=0)
    _, d_b, _ = measured_effect_batched(model, tok, input_ids, reply_ids,
                                        chaff_vocab, k_draws=K, seed=123)
    assert not torch.allclose(d_a, d_b, atol=1e-6), \
        "seed had no effect -- RNG not wired through"


# --------------------------------------------------------------------------
# Property: BATCH-SIZE INVARIANCE -- a position measured alone must give the
# SAME number as when measured together with every other position. Alarms if
# measuring a subset changes a position's value (e.g. RNG consumption that
# depends on how many positions are requested, or cross-position state bleed).
# --------------------------------------------------------------------------
def test_batch_size_invariance(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    inp_len = input_ids.shape[0]
    _, d_all, k_all = measured_effect_batched(
        model, tok, input_ids, reply_ids, chaff_vocab, k_draws=K, seed=SEED)
    for i in (0, inp_len // 2, inp_len - 1):
        _, d_one, k_one = measured_effect_batched(
            model, tok, input_ids, reply_ids, chaff_vocab,
            k_draws=K, seed=SEED, positions=[i])
        assert abs(float(d_one[i]) - float(d_all[i])) < 1e-6, \
            f"pos {i}: alone={float(d_one[i])} in-batch={float(d_all[i])}"
        assert abs(float(k_one[i]) - float(k_all[i])) < 1e-6


# --------------------------------------------------------------------------
# Property: REORDER INVARIANCE -- the value assigned to a position must not
# depend on the ORDER positions are listed in the batch. Alarms if results are
# written back by batch-slot rather than by true position index.
# --------------------------------------------------------------------------
def test_reorder_invariance(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    inp_len = input_ids.shape[0]
    subset = [1, inp_len - 2, 3] if inp_len > 4 else list(range(inp_len))
    _, d_fwd, k_fwd = measured_effect_batched(
        model, tok, input_ids, reply_ids, chaff_vocab,
        k_draws=K, seed=SEED, positions=subset)
    _, d_rev, k_rev = measured_effect_batched(
        model, tok, input_ids, reply_ids, chaff_vocab,
        k_draws=K, seed=SEED, positions=list(reversed(subset)))
    for i in subset:
        assert abs(float(d_fwd[i]) - float(d_rev[i])) < 1e-6, f"pos {i}"
        assert abs(float(k_fwd[i]) - float(k_rev[i])) < 1e-6, f"pos {i}"


# --------------------------------------------------------------------------
# Property (THE ORACLE): finite-difference ground truth.
# For a single position and a single fixed replacement token, resampling must
# change the teacher-forced reply log-prob by EXACTLY what a direct brute-force
# recompute says -- two independent model forwards (original full sequence, and
# full sequence with that one token swapped), reply log-probs summed by hand,
# subtracted. No reuse of the impl under test. This pins correctness, not
# self-consistency, and is the check the causal loop has never had.
#
# We drive measured_effect_batched with k_draws=1 so its single draw is a known
# token, and compare its reported |ΔlogP| against the hand-computed |ΔlogP|.
# --------------------------------------------------------------------------
def _hand_reply_logp(model, full_ids, inp_len, reply_ids):
    """Teacher-forced sum of reply-token log-probs, computed from scratch."""
    with torch.no_grad():
        logits = model(full_ids.unsqueeze(0)).logits[0]
        logp = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for j, tokid in enumerate(reply_ids.tolist()):
        pos = inp_len + j - 1
        total += float(logp[pos, tokid])
    return total


def test_finite_difference_oracle(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    device = model.device
    inp_len = input_ids.shape[0]
    full = torch.cat([input_ids, reply_ids]).to(device)

    # Discover, for each test position, the exact replacement token the impl
    # will draw with k_draws=1 at this seed, by replaying the impl's own RNG.
    # (This is reading the RNG, NOT the measurement; the measured ΔlogP we check
    # against the oracle still comes only from independent forwards.)
    def drawn_token_for(pos):
        g = torch.Generator().manual_seed(SEED + 777)
        tokid = None
        for i in range(inp_len):
            t = chaff_vocab[int(torch.randint(len(chaff_vocab), (1,),
                                              generator=g))]
            if i == pos:
                tokid = t
        return tokid

    base = _hand_reply_logp(model, full, inp_len, reply_ids)

    checked = 0
    for pos in (0, inp_len // 2, inp_len - 1):
        repl = drawn_token_for(pos)
        swapped = full.clone()
        swapped[pos] = repl
        pert = _hand_reply_logp(model, swapped, inp_len, reply_ids)
        oracle_dlogp = abs(pert - base)

        _, d, _ = measured_effect_batched(
            model, tok, input_ids, reply_ids, chaff_vocab,
            k_draws=1, seed=SEED, positions=[pos])
        got = float(d[pos])
        assert abs(got - oracle_dlogp) < 1e-4, (
            f"pos {pos}: impl |dlogP|={got} oracle |dlogP|={oracle_dlogp} "
            f"(replacement token {repl})")
        checked += 1
    assert checked == 3


# --------------------------------------------------------------------------
# Property: KL NON-NEGATIVITY and ZERO-AT-IDENTITY.
# KL(base||new) >= 0 always, and must be ~0 when the "perturbation" leaves the
# sequence unchanged. Alarms if the KL direction/sign is wrong (a KL that can
# go negative, or a nonzero KL for a no-op perturbation, is a broken metric).
# --------------------------------------------------------------------------
def test_kl_nonnegative(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    _, _, dkl = measured_effect_batched(model, tok, input_ids, reply_ids,
                                        chaff_vocab, k_draws=K, seed=SEED)
    assert float(dkl.min()) >= -1e-5, f"negative KL: {float(dkl.min())}"


def test_kl_zero_for_noop(model_tok, chaff_vocab, toy_case):
    """Replace a position with ITS OWN token -> no change -> KL==0, dlogP==0.
    A chaff_vocab of just that token forces the 'perturbation' to be a no-op."""
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    inp_len = input_ids.shape[0]
    pos = inp_len // 2
    own = int(input_ids[pos])
    _, d, kl = measured_effect_batched(
        model, tok, input_ids, reply_ids, [own],
        k_draws=3, seed=SEED, positions=[pos])
    assert abs(float(d[pos])) < 1e-4, f"no-op gave dlogP={float(d[pos])}"
    assert abs(float(kl[pos])) < 1e-4, f"no-op gave KL={float(kl[pos])}"


# --------------------------------------------------------------------------
# Property: base log-prob consistency with the standalone helper. Alarms if the
# returned base_logp drifts from teacher_forced_logp_and_dist (the proxy path
# uses the same quantity; a mismatch would break the proxy-vs-measured compare).
# --------------------------------------------------------------------------
def test_base_logp_matches_helper(model_tok, chaff_vocab, toy_case):
    model, tok = model_tok
    input_ids, reply_ids, _ = toy_case
    device = model.device
    full = torch.cat([input_ids, reply_ids]).to(device)
    ref, _ = teacher_forced_logp_and_dist(
        model, full, input_ids.shape[0], reply_ids)
    base, _, _ = measured_effect_batched(model, tok, input_ids, reply_ids,
                                         chaff_vocab, k_draws=K, seed=SEED)
    assert abs(ref - base) < 1e-6
