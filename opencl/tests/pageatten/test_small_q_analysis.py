"""
test_small_q_analysis.py

Comprehensive performance analysis for small-q kernel across:
- Different Q_len (3, 4, 8, 16)
- Different TILE_Q (1, 2)
- Different KV cache compression (0=FP16, 1=INT8)
- Cache invalidation vs. cached scenarios

Usage:
  RUN_PA_PERF=1 python -m pytest test_small_q_analysis.py -v -s
  RUN_PA_PERF=1 python -m pytest test_small_q_analysis.py -k "q16 and cmpr1" -v -s
"""

import functools
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from clops import cl
from kv_cache_quant_utils import DEFAULT_SUB_BLOCK_SIZE
from test_pa_small_q import (
    PaSmallQRunner,
    SmallQCase,
    _build_inputs as _build_small_q_inputs,
    _ceil_div,
    get_cm_grf_width,
)


cl.profiling(True)
torch.manual_seed(0)

# Target context length for all tests
PAST_LEN_15K = 15 * 1024


@dataclass(frozen=True)
class SmallQPerfResult:
    """Performance measurement for small_q kernel."""
    q_len: int
    tile_q: int
    cmpr: int

    # Wall-clock timing (includes host overhead)
    wall_avg_ms: float
    wall_min_ms: float
    wall_max_ms: float

    # Pure GPU kernel timing (from profiling events)
    kernel_ms: float = 0.0
    reduce_ms: float = 0.0

    # Performance metrics
    kv_bytes: float = 0.0
    kernel_bw_gbs: float = 0.0

    def __str__(self) -> str:
        return (
            f"Q={self.q_len:2d} TILE={self.tile_q} CMPR={self.cmpr} | "
            f"Wall={self.wall_avg_ms:6.3f}ms "
            f"Kernel={self.kernel_ms:6.3f}ms Reduce={self.reduce_ms:6.3f}ms | "
            f"BW={self.kernel_bw_gbs:6.1f} GB/s"
        )


def _run_bandwidth_measurement_small_q(
    runner: PaSmallQRunner,
    case: SmallQCase,
    loop_cnt: int = 100,
    warmup: int = 8,
) -> Dict[str, float]:
    """Measure small_q kernel bandwidth with profiling events.

    Allocates multiple layers (>8GB total) and rotates through them
    to prevent L2 cache from hiding true DRAM performance.
    """
    kernels = runner._create_kernels()
    data = _build_small_q_inputs(case)

    query = data["query"]
    key_cache = data["key_cache"]
    value_cache = data["value_cache"]
    past_lens = data["past_lens"]
    block_indices = data["block_indices"]
    block_indices_begins = data["block_indices_begins"]
    subsequence_begins = data["subsequence_begins"]

    q_len = int(query.shape[0])
    max_context_len = int(past_lens.max().item()) + q_len
    kv_partition_num = _ceil_div(max_context_len, runner.kv_partition_size)
    mapping, tile_count = runner._build_mapping(q_len, runner.tile_q)
    print("tile_count", tile_count, "kv_partition_num", kv_partition_num)
    partition_token_rows = tile_count * runner.tile_q

    # Allocate multiple layers for cache bypass
    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_po = cl.tensor(
            [partition_token_rows, runner.num_heads, kv_partition_num, runner.head_size],
            np.dtype(np.float32),
        )
        t_out_final = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))
        all_layers.append((t_q, t_k, t_v, t_po, t_out_final))

        mem_size += query.numel() * query.element_size()
        mem_size += key_cache.numel() * key_cache.element_size()
        mem_size += value_cache.numel() * value_cache.element_size()

    if not all_layers:
        raise RuntimeError("Failed to allocate perf input layers")

    # Pre-allocate static tensors
    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
    t_mapping = cl.tensor(mapping.detach().numpy())
    t_lse = cl.tensor(
        [partition_token_rows, runner.num_heads, kv_partition_num],
        np.dtype(np.float32),
    )

    gws = [tile_count, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [partition_token_rows, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    # Clear prior profiling events
    cl.finish()

    # Enqueue kernels with rotating layers
    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_po, t_out_final = all_layers[j]

        kernels.enqueue(
            "cm_pa_small_q",
            gws,
            lws,
            t_q,
            t_k,
            t_v,
            t_past_lens,
            t_block_indices,
            t_block_indices_begins,
            t_subsequence_begins,
            t_mapping,
            t_po,
            t_lse,
            q_len,
            int(tile_count),
        )
        kernels.enqueue(
            "cm_pa_small_q_reduce",
            gws_2,
            lws_2,
            t_po,
            t_out_final,
            t_lse,
            t_subsequence_begins,
            t_mapping,
            partition_token_rows,
            kv_partition_num,
        )

    # Collect profiling events
    latency = cl.finish()
    expected = 2 * loop_cnt
    if len(latency) < expected:
        raise RuntimeError(f"Expected {expected} events, got {len(latency)}")

    # Accumulate timing (skip warmup)
    sq_total = 0.0
    rd_total = 0.0
    runs = 0
    for i in range(loop_cnt):
        sq_ns = float(latency[2 * i])
        rd_ns = float(latency[2 * i + 1])
        if sq_ns <= 0 or rd_ns <= 0 or i < warmup:
            continue
        sq_total += sq_ns
        rd_total += rd_ns
        runs += 1

    if runs <= 0:
        raise RuntimeError("Invalid timing accumulation")

    # Calculate KV bandwidth (padded, like test_pa_decoding)
    # IMPORTANT: All Q tokens share the SAME KV cache, so DON'T multiply by q_len!
    # KV bytes = (past_kv_len) × num_kv_heads × head_size × bytes_per_element × 2 (K+V)
    # This is independent of q_len - different q_len just changes HOW we process KV, not how much we read.
    num_blocks = int(key_cache.shape[0])
    padded_kv_len = num_blocks * case.block_size
    bytes_per_element = 1 if case.kv_cache_compression > 0 else 2

    # KV bytes: only depends on cached KV size, NOT q_len (all Q tokens reuse same KV)
    kv_bytes = padded_kv_len * case.num_kv_heads * case.head_size * bytes_per_element * 2  # 2 for K+V

    # Add metadata for compression (per-token scale/zp, also doesn't scale with q_len)
    if case.kv_cache_compression == 1:  # per-token
        kv_bytes += padded_kv_len * case.num_kv_heads * 2 * 2  # scale + zp per token in KV

    sq_ms = sq_total * 1e-6 / runs
    rd_ms = rd_total * 1e-6 / runs
    bw = kv_bytes / (sq_ms * 1e-3) / 1e9

    return {
        "small_q_ms": sq_ms,
        "small_q_reduce_ms": rd_ms,
        "kv_bytes": kv_bytes,
        "kv_bw_gbs": bw,
        "num_runs": runs,
    }


def _benchmark_small_q(case: SmallQCase) -> SmallQPerfResult:
    """Benchmark small_q kernel with both wall-clock and profiling."""
    runner = PaSmallQRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
        tile_q=case.tile_q,
    )

    data = _build_small_q_inputs(case)
    output = torch.zeros([case.q_len, case.num_heads, case.head_size], dtype=torch.float16)

    # Wall-clock timing
    import time

    def run_once():
        runner(
            data["query"],
            data["key_cache"],
            data["value_cache"],
            data["past_lens"],
            data["block_indices"],
            data["block_indices_begins"],
            data["subsequence_begins"],
            output,
            n_repeats=1,
        )

    # Warmup
    for _ in range(5):
        run_once()
    cl.finish()

    # Measure
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        run_once()
        cl.finish()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000)  # Convert to ms

    wall_avg = sum(samples) / len(samples)
    wall_min = min(samples)
    wall_max = max(samples)

    # Profiling-based kernel timing
    kernel_ms = 0.0
    reduce_ms = 0.0
    bw = 0.0
    kv_bytes = 0.0

    try:
        perf = _run_bandwidth_measurement_small_q(runner, case, loop_cnt=80, warmup=8)
        kernel_ms = perf["small_q_ms"]
        reduce_ms = perf["small_q_reduce_ms"]
        kv_bytes = perf["kv_bytes"]
        bw = perf["kv_bw_gbs"]
    except Exception as e:
        print(f"  [warn] profiling failed: {e}")

    return SmallQPerfResult(
        q_len=case.q_len,
        tile_q=case.tile_q,
        cmpr=case.kv_cache_compression,
        wall_avg_ms=wall_avg,
        wall_min_ms=wall_min,
        wall_max_ms=wall_max,
        kernel_ms=kernel_ms,
        reduce_ms=reduce_ms,
        kv_bytes=kv_bytes,
        kernel_bw_gbs=bw,
    )


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.parametrize("q_len", [3, 4, 8, 16])
@pytest.mark.parametrize("cmpr", [0, 1])
@pytest.mark.parametrize("tile_q", [1, 2])
def test_small_q_perf_sweep(q_len: int, cmpr: int, tile_q: int):
    """Sweep across Q_len, TILE_Q, and compression."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    if tile_q > q_len:
        pytest.skip(f"TILE_Q={tile_q} exceeds Q_len={q_len}")

    case = SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        past_len=PAST_LEN_15K,
        q_len=q_len,
        kv_cache_compression=cmpr,
        tile_q=tile_q,
    )

    result = _benchmark_small_q(case)
    print(f"\n{result}")


def test_small_q_summary_cmpr0():
    """Summary table: FP16 KV cache."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    print("\n" + "=" * 120)
    print("Small-Q Performance Summary (FP16 KV Cache)")
    print("=" * 120)

    results = []
    for q_len in [3, 4, 8, 16]:
        for tile_q in [1, 2]:
            if tile_q > q_len:
                continue
            case = SmallQCase(
                num_heads=32,
                num_kv_heads=8,
                head_size=128,
                block_size=256,
                past_len=PAST_LEN_15K,
                q_len=q_len,
                kv_cache_compression=0,
                tile_q=tile_q,
            )
            result = _benchmark_small_q(case)
            results.append(result)
            print(f"{result}")

    print("=" * 120)


def test_small_q_summary_cmpr1():
    """Summary table: INT8 KV cache."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    print("\n" + "=" * 120)
    print("Small-Q Performance Summary (INT8 Compressed KV Cache)")
    print("=" * 120)

    results = []
    for q_len in [3, 4, 8, 16]:
        for tile_q in [1, 2]:
            if tile_q > q_len:
                continue
            case = SmallQCase(
                num_heads=32,
                num_kv_heads=8,
                head_size=128,
                block_size=256,
                past_len=PAST_LEN_15K,
                q_len=q_len,
                kv_cache_compression=1,
                tile_q=tile_q,
            )
            result = _benchmark_small_q(case)
            results.append(result)
            print(f"{result}")

    print("=" * 120)


def test_small_q_q16_compression_comparison():
    """Direct comparison: Q=16 with/without compression."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    print("\n" + "=" * 120)
    print("Q=16 Compression Analysis (TILE_Q=2)")
    print("=" * 120)

    for cmpr in [0, 1]:
        cmpr_name = "INT8" if cmpr == 1 else "FP16"
        case = SmallQCase(
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=256,
            past_len=PAST_LEN_15K,
            q_len=16,
            kv_cache_compression=cmpr,
            tile_q=2,
        )
        result = _benchmark_small_q(case)
        print(f"\n{cmpr_name:6s}: {result}")

    print("=" * 120)
    print("Key Takeaway: Compare BW with/without INT8 to see if dequant overhead is worth it")
    print("=" * 120)


def test_small_q_perf_bandwidth_q16_default_params():
    """Baseline test matching test_pa_decoding::test_pa_perf_bandwidth_generate_single_subsequence_default_params.

    Single Token (Q=1) parameters:
    - num_heads=32, num_kv_heads=8, head_size=128, block_size=256
    - kv_len=15360 (15K context)
    - kv_cache_compression=1 (INT8)
    - loop_cnt=100, warmup=5

    Small Q Equivalent (Q=16):
    - Same head/block parameters
    - Same past_len=15360
    - Same compression=1 (INT8)
    - TILE_Q=2 (recommended for Arc 140V)
    """
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable bandwidth perf test")

    case = SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        past_len=PAST_LEN_15K,  # 15360 - matches single token test
        q_len=4,               # Speculative window (16 draft tokens)
        kv_cache_compression=1, # INT8 - matches single token test
        tile_q=2,               # Recommended for Arc 140V (64 EU)
    )

    # Create runner
    runner = PaSmallQRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
        tile_q=case.tile_q,
    )

    perf = _run_bandwidth_measurement_small_q(runner, case, loop_cnt=100, warmup=5)

    print("\n" + "=" * 120)
    print("Small-Q Q=16 Baseline Performance (matching test_pa_decoding single_token parameters)")
    print("=" * 120)
    print(f"[perf] "
          f"cm_pa_small_q_bw={perf['kv_bw_gbs']:.3f} GB/s, "
          f"cm_pa_small_q_ms={perf['small_q_ms']:.3f}, "
          f"cm_pa_small_q_reduce_ms={perf['small_q_reduce_ms']:.3f}, "
          f"total_ms={perf['small_q_ms'] + perf['small_q_reduce_ms']:.3f}")
    print("=" * 120)

    # Compare with single token results
    # Note: These are baseline values from test_pa_decoding.py::test_pa_perf_bandwidth_generate_single_subsequence_default_params
    # To update these values, run: RUN_PA_PERF=1 pytest test_pa_decoding.py::test_pa_perf_bandwidth_generate_single_subsequence_default_params -v -s
    single_token_kernel_ms = 0.401
    single_token_bw = 79.6
    single_token_gflops = single_token_bw * 2  # AI=2 for Q=1

    print("\nComparison with Single Token (Q=1) baseline:")
    print(f"  Single Token (Q=1):   kernel={single_token_kernel_ms:.3f}ms, BW={single_token_bw:.1f} GB/s, GFLOPS={single_token_gflops:.1f} (AI=2)")
    print(f"  Small Q (Q=16):       kernel={perf['small_q_ms']:.3f}ms, BW={perf['kv_bw_gbs']:.1f} GB/s, GFLOPS={perf['kv_bw_gbs']*32:.1f} (AI=32)")
    print(f"  Speedup:             {single_token_gflops / (perf['kv_bw_gbs']*32):.2f}x (should approach 1x if both memory-bound)")
    print("=" * 120)


if __name__ == "__main__":
    # Quick test
    case = SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        past_len=PAST_LEN_15K,
        q_len=16,
        kv_cache_compression=0,
        tile_q=2,
    )
    result = _benchmark_small_q(case)
    print(result)
