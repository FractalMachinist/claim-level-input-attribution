"""Shared fixtures for the causal-measurement correctness suite.

Loads Qwen3-0.6B fp32 once per session (needs a GPU; fp32, small). A tiny toy
prompt built from the project's own build_prompt keeps the finite-difference
oracle cheap while still exercising the real teacher-forcing indexing.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.attribution import (  # noqa: E402
    build_prompt, generate_reply, load_model, pick_chaff_vocab,
)
import lib.causal as causal  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _batching_mode():
    """Run the SAME suite against either impl without touching a test body.

    CAUSAL_BATCHED=1 flips measured_effect_batched to its genuine multi-position
    stacking path (Step 2) by enabling the forward hook and defaulting pos_batch
    to CAUSAL_POS_BATCH (default 4). Unset -> the Step-1 dumb per-position
    wrapper. The suite's assertions are identical either way.
    """
    if os.environ.get("CAUSAL_BATCHED") == "1":
        causal._BATCHED_FORWARD["enabled"] = True
        pb = int(os.environ.get("CAUSAL_POS_BATCH", "4"))
        orig = causal.measured_effect_batched

        def wrapped(*a, **kw):
            kw.setdefault("pos_batch", pb)
            return orig(*a, **kw)

        causal.measured_effect_batched = wrapped
        # Rebind the name the test module imported at import time.
        import tests.test_causal_properties as tmod
        tmod.measured_effect_batched = wrapped
        yield
        causal.measured_effect_batched = orig
        causal._BATCHED_FORWARD["enabled"] = False
    else:
        yield


@pytest.fixture(scope="session")
def model_tok():
    tok, model = load_model(verbose=False)
    return model, tok


@pytest.fixture(scope="session")
def chaff_vocab(model_tok):
    _, tok = model_tok
    return pick_chaff_vocab(tok, seed=0)


@pytest.fixture(scope="session")
def toy_case(model_tok, chaff_vocab):
    """A small real prompt + its greedy reply. Short so the oracle is cheap, but
    built by the real build_prompt so positions/indexing are exactly what the
    experiments use."""
    model, tok = model_tok
    input_ids, labels, _ = build_prompt(
        tok, "What is the capital of France?", n_chaff=6,
        chaff_vocab=chaff_vocab, position="before", seed=0,
    )
    reply_ids = generate_reply(model, tok, input_ids, max_new_tokens=6)
    assert reply_ids.shape[0] >= 2, "need a multi-token reply to exercise KL sum"
    return input_ids, reply_ids, labels
