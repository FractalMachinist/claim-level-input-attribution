# Causal-measurement validation + batching (2026-11-15)

Validation of `lib/causal.py::measured_effect` (the Stage 1/2 causal ground
truth) and a genuine batched twin `measured_effect_batched`.

Hardware: Vast.ai RTX 3060 12GB (sm_86), Qwen/Qwen3-0.6B fp32, torch 2.14.1+cu132.
Test driver: `experiments/tests/test_causal_properties.py` (pytest, 9 properties).

## Step 1 — serial impl correctness: PASS (all 9 properties)

Run against the drop-in `measured_effect_batched` dumb wrapper (per-position
serial scorer; default `pos_batch=1`):

| Property | Result |
|---|---|
| batched-signature twin == serial, bit-for-bit | PASS |
| determinism (same seed → same numbers) | PASS |
| seed sensitivity (different seed → different numbers) | PASS |
| batch-size invariance (position alone == in batch) | PASS |
| reorder invariance (value independent of list order) | PASS |
| **finite-difference oracle** (brute-force ΔlogP, no reuse of impl) | **PASS** (<1e-4) |
| KL non-negativity | PASS |
| KL == 0 and ΔlogP == 0 for a no-op perturbation | PASS |
| base logP matches the standalone teacher-forcing helper | PASS |

The finite-difference oracle is the check the causal loop never had: it runs the
model twice by hand (original full sequence, and with one token swapped),
subtracts teacher-forced reply log-probs, and matches the impl to <1e-4. The
serial causal measurement is correct. **No escalation.**

## Step 2 — genuine batched impl: PASS, same suite unchanged

`measured_effect_batched(..., pos_batch>1)` with `_BATCHED_FORWARD["enabled"]`
stacks P positions' `[K,T]` blocks into one `[P*K,T]` forward, slices the
reply-position rows, and vectorizes the ΔlogP/KL reduction (gather over reply
tokens; broadcast KL). The identical 9-property suite passes with
`CAUSAL_BATCHED=1` (per-position numbers bit-identical to serial).

### Wall-clock, realistic prompt ("What is the longest bony fish?" + N chaff)

| N chaff | T | K | pos_batch | serial | batched | speedup | matches serial |
|---|---|---|---|---|---|---|---|
| 30 | 52 | 8 (published) | 8 | 6.71s | 5.88s | **1.14x** | yes |
| 30 | 52 | 2 | 32 | 2.58s | 1.52s | **1.70x** | yes |
| 60 | 83 | 2 | 32 | 4.88s | 3.37s | **1.45x** | yes |

Speedup is memory-bound, not a bug: the LM-head logits `[P*K, T, V≈152k]` is the
ceiling — at K=8, pos_batch>8 OOMs the 12GB card (a `[128,52,152k]` fp32 tensor
is ~4.9 GiB, produced inside the forward before any slice). The amortizable
overhead (Python loop + kernel launches) is a larger fraction when each
per-position forward is small, so the win grows as K shrinks. On a larger GPU,
higher pos_batch would extend the trend. Correctness is identical throughout.

Reproduce: `cd experiments && pytest tests/test_causal_properties.py` (dumb
wrapper); add `CAUSAL_BATCHED=1` for the genuine batched path.
