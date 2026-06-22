"""
test_pa_small_q_compare.py

Head-to-head GPU latency comparison between the small_q decode kernel
(pa_small_q.cm + pa_small_q_finalization.cm) and the prefill multi_token
kernel (pa_multi_token.cm) on the same q_len > 1 / single-subsequence shape.

Why this exists
---------------
multi_token's WG dispatch is sized for prefill: each WG covers WG_SEQ_LEN =
max_wg_size * q_step (=16*16=256 on Xe2) Q-tokens. When q_len in {2..16} that
WG runs at 1/(WG_SEQ_LEN/q_len) utilisation. small_q dispatches one tile per
TILE_Q q-tokens with DPAS RepeatCount = Q_head_chunk_size * TILE_Q so the
systolic array stays saturated.

Inputs are built once via test_pa_small_q._build_inputs and fed to both
kernels (with multi_token's expected fp16 reshape). Both produce the same
[q_len, heads, head_size] output, so we cross-check correctness too.

Usage:
  RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q_compare.py -vv
  RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q_compare.py \
      -k 'q16 and cmpr0'
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from clops import cl
from test_pa_small_q import (
    DEFAULT_TILE_Q,
    PaSmallQRunner,
    SmallQCase,
    _build_inputs,
    _ceil_div,
)
from test_pa_multiseq import PaMultiTokenRunner


cl.profiling(True)
torch.manual_seed(0)


@dataclass(frozen=True)
class CompareCase:
    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    block_size: int = 256
    past_len: int = 32 * 1024 - 16
    q_len: int = 16
    kv_cache_compression: int = 0
    tile_q: int = DEFAULT_TILE_Q


def _to_small_q_case(c: CompareCase) -> SmallQCase:
    return SmallQCase(
        num_heads=c.num_heads,
        num_kv_heads=c.num_kv_heads,
        head_size=c.head_size,
        block_size=c.block_size,
        past_len=c.past_len,
        q_len=c.q_len,
        kv_cache_compression=c.kv_cache_compression,
        tile_q=c.tile_q,
    )


def _case_id(c: CompareCase) -> str:
    return (
        f"q{c.q_len}_past{c.past_len}_h{c.num_heads}_kv{c.num_kv_heads}"
        f"_hs{c.head_size}_bls{c.block_size}_cmpr{c.kv_cache_compression}_tile{c.tile_q}"
    )


def _run_small_q_perf(
    case: CompareCase, data: dict, *, n_warmup: int = 5, n_iters: int = 50
) -> tuple[torch.Tensor, list[float]]:
    """Time pa_small_q + finalize as one iteration. Returns (output[q,h,hs], iter_ns)."""
    sq = _to_small_q_case(case)
    runner = PaSmallQRunner.create_instance(
        sq.num_heads, sq.num_kv_heads, sq.head_size, sq.block_size,
        sq.sub_block_size, sq.kv_cache_compression, tile_q=sq.tile_q,
    )
    kernels = runner._create_kernels()

    query = data["query"]
    key_cache = data["key_cache"]
    value_cache = data["value_cache"]

    q_len = int(query.shape[0])
    max_context_len = int(data["past_lens"].max().item()) + q_len
    kv_partition_num = _ceil_div(max_context_len, runner.kv_partition_size)
    mapping, tile_count = runner._build_mapping(q_len, runner.tile_q)
    partition_token_rows = tile_count * runner.tile_q

    t_q = cl.tensor(query.detach().numpy())
    t_k = cl.tensor(key_cache.contiguous().detach().numpy())
    t_v = cl.tensor(value_cache.contiguous().detach().numpy())
    t_past_lens = cl.tensor(data["past_lens"].detach().numpy())
    t_block_indices = cl.tensor(data["block_indices"].detach().numpy())
    t_block_indices_begins = cl.tensor(data["block_indices_begins"].detach().numpy())
    t_subsequence_begins = cl.tensor(data["subsequence_begins"].detach().numpy())
    t_mapping = cl.tensor(mapping.detach().numpy())
    t_partition_out = cl.tensor(
        [partition_token_rows, runner.num_heads, kv_partition_num, runner.head_size],
        np.dtype(np.float32),
    )
    t_lse = cl.tensor(
        [partition_token_rows, runner.num_heads, kv_partition_num], np.dtype(np.float32)
    )
    t_out_final = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))

    gws = [tile_count, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [partition_token_rows, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    cl.finish()
    for _ in range(n_warmup):
        kernels.enqueue("cm_pa_small_q", gws, lws,
                        t_q, t_k, t_v, t_past_lens, t_block_indices,
                        t_block_indices_begins, t_subsequence_begins,
                        t_mapping, t_partition_out, t_lse,
                        q_len, tile_count)
        kernels.enqueue("cm_pa_small_q_reduce", gws_2, lws_2,
                        t_partition_out, t_out_final, t_lse,
                        t_subsequence_begins, t_mapping,
                        partition_token_rows, kv_partition_num)
    cl.finish()

    for _ in range(n_iters):
        kernels.enqueue("cm_pa_small_q", gws, lws,
                        t_q, t_k, t_v, t_past_lens, t_block_indices,
                        t_block_indices_begins, t_subsequence_begins,
                        t_mapping, t_partition_out, t_lse,
                        q_len, tile_count)
        kernels.enqueue("cm_pa_small_q_reduce", gws_2, lws_2,
                        t_partition_out, t_out_final, t_lse,
                        t_subsequence_begins, t_mapping,
                        partition_token_rows, kv_partition_num)
    ev_ns = cl.finish()
    if len(ev_ns) != 2 * n_iters:
        raise RuntimeError(f"unexpected event count: {len(ev_ns)} expected {2 * n_iters}")
    iter_ns = [float(ev_ns[2 * i]) + float(ev_ns[2 * i + 1]) for i in range(n_iters)]
    out = torch.from_numpy(t_out_final.numpy())
    return out, iter_ns


def _run_multi_token_perf(
    case: CompareCase, data: dict, *, n_warmup: int = 5, n_iters: int = 50
) -> tuple[torch.Tensor, list[float]]:
    """Time pa_multi_token (single fused kernel) on the same inputs."""
    runner = PaMultiTokenRunner.create_instance(
        case.num_heads, case.num_kv_heads, case.head_size,
        case.block_size, case.kv_cache_compression, is_causal=True,
        sparse_block_size=1, enable_hybrid_dispatch=False,
    )

    query = data["query"]                                # [q_len, heads, hs]
    key_cache = data["key_cache"]
    value_cache = data["value_cache"]
    past_lens = data["past_lens"]
    block_indices = data["block_indices"]
    block_indices_begins = data["block_indices_begins"]
    subsequence_begins = data["subsequence_begins"]

    q_len = int(query.shape[0])
    q_tensor = query.reshape(q_len, runner.num_heads, runner.head_size).contiguous()
    kv_dtype = torch.uint8 if runner.kv_cache_compression != 0 else torch.float16
    kernel_key_cache = runner._format_cache_for_kernel(key_cache.contiguous())
    kernel_value_cache = runner._format_cache_for_kernel(value_cache.contiguous())

    t_q = cl.tensor(q_tensor.to(torch.float16).detach().numpy())
    t_out = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))
    t_key_cache = cl.tensor(kernel_key_cache.to(kv_dtype).detach().numpy())
    t_value_cache = cl.tensor(kernel_value_cache.to(kv_dtype).detach().numpy())
    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

    q_step = runner.q_step
    wg_size = runner.max_wg_size
    wg_seq_len = wg_size * q_step

    selected_sequence_ids = torch.tensor([0], dtype=torch.int32)
    kern_attn_inputs = {
        "query": q_tensor,
        "key_cache": kernel_key_cache,
        "value_cache": kernel_value_cache,
        "past_lens": past_lens,
        "block_indices": block_indices,
        "block_indices_begins": block_indices_begins,
        "subsequence_begins": subsequence_begins,
    }
    blocked_q_starts_and_subseq_mapping, wg_count = runner._build_wg_block_start_and_subseq_mapping(
        kern_attn_inputs, selected_sequence_ids, wg_seq_len,
    )
    if wg_count <= 0:
        raise ValueError("invalid wg mapping")
    t_blocked_q_starts_and_subseq_mapping = cl.tensor(blocked_q_starts_and_subseq_mapping.detach().numpy())

    gws = [1, runner.num_heads, int(wg_count * wg_size)]
    lws = [1, 1, wg_size]
    selected_kernels = runner.kernels_dynamic   # sparse_block_size=1, no optimized variant

    cl.finish()
    for _ in range(n_warmup):
        selected_kernels.enqueue(
            "cm_page_attention", gws, lws,
            t_q, t_key_cache, t_value_cache,
            t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins,
            t_blocked_q_starts_and_subseq_mapping, t_out, q_len,
        )
    cl.finish()

    for _ in range(n_iters):
        selected_kernels.enqueue(
            "cm_page_attention", gws, lws,
            t_q, t_key_cache, t_value_cache,
            t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins,
            t_blocked_q_starts_and_subseq_mapping, t_out, q_len,
        )
    ev_ns = cl.finish()
    if len(ev_ns) != n_iters:
        raise RuntimeError(f"unexpected event count: {len(ev_ns)} expected {n_iters}")
    iter_ns = [float(x) for x in ev_ns]
    out = torch.from_numpy(t_out.numpy()).reshape(q_len, runner.num_heads, runner.head_size)
    return out, iter_ns


def _summarise(iter_ns: list[float], warmup: int = 0) -> dict[str, float]:
    samples = iter_ns[warmup:]
    if not samples:
        raise RuntimeError("no perf samples after warmup")
    avg = sum(samples) / len(samples)
    smin = min(samples)
    smax = max(samples)
    return {"avg_ms": avg * 1e-6, "min_ms": smin * 1e-6, "max_ms": smax * 1e-6}


# ---- Compare matrix ----
# Qwen3-8B shape, past_len=32k - q_len so the total context is exactly 32k. q
# sweeps the spec-decode window {2,4,8,16}; cmpr {fp16, by_token} matches the
# perf cases the small_q kernel already targets. tile_q={1,2} brackets the
# DPAS RepeatCount sweet spot.

_COMPARE_CASES = tuple(
    CompareCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=32 * 1024 - q_len, q_len=q_len,
        kv_cache_compression=cmpr, tile_q=tile_q,
    )
    for q_len in (2, 4, 8, 16)
    for cmpr in (0, 1)
    for tile_q in (1, 2)
)


@pytest.mark.parametrize("case", _COMPARE_CASES, ids=_case_id)
def test_pa_small_q_vs_multi_token(case: CompareCase):
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable cross-kernel perf compare")
    if case.tile_q > case.q_len:
        pytest.skip(f"tile_q={case.tile_q} exceeds q_len={case.q_len}")

    sq_case = _to_small_q_case(case)
    data = _build_inputs(sq_case)

    # Numerical sanity: both kernels should roughly agree (multi_token uses fp16
    # internally; small_q uses fp32 accum + softmax). Tighter tolerance than
    # multi vs SDPA because they share the same quantised KV cache.
    out_sq, iter_sq = _run_small_q_perf(case, data)
    out_mt, iter_mt = _run_multi_token_perf(case, data)

    diff = (out_sq.float() - out_mt.float()).abs().max().item()
    assert diff < 5e-2, (
        f"small_q vs multi_token mismatch: max_abs_diff={diff} (case={_case_id(case)})"
    )

    sq = _summarise(iter_sq)
    mt = _summarise(iter_mt)
    speedup = mt["avg_ms"] / sq["avg_ms"] if sq["avg_ms"] > 0 else float("inf")

    print(
        f"[compare] {_case_id(case)} | "
        f"small_q={sq['avg_ms']:.3f}ms (min={sq['min_ms']:.3f}) | "
        f"multi_token={mt['avg_ms']:.3f}ms (min={mt['min_ms']:.3f}) | "
        f"speedup={speedup:.2f}x | max_abs_diff={diff:.2e}"
    )


# Usage:
#   RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q_compare.py -vv
#   RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q_compare.py -k 'q16 and cmpr0'
