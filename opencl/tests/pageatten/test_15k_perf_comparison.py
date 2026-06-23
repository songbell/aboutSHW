"""
test_15k_perf_comparison.py

Performance comparison at 15K past length across three scenarios:
1. Single token (q=1): pa_single_token kernel for standard decoding
2. Small q (q=3-16): pa_small_q kernel for speculative decoding
3. Multi token (q=3-16): pa_multi_token kernel (baseline comparison)

Goal: Measure end-to-end latency and bandwidth utilization to understand
      which kernel is optimal for different q_len values at 15K context.

Usage:
  RUN_PA_PERF=1 python -m pytest -s test_15k_perf_comparison.py -v
  RUN_PA_PERF=1 python -m pytest -s test_15k_perf_comparison.py -k "q1" -v
  RUN_PA_PERF=1 python -m pytest -s test_15k_perf_comparison.py -k "cmpr0" -v
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from clops import cl
from test_pa_decoding import PaSingleTokenRunner, DecodingCase
from test_pa_small_q import (
    PaSmallQRunner,
    SmallQCase,
    _build_inputs as _build_small_q_inputs,
)
from test_pa_multiseq import PaMultiTokenRunner

cl.profiling(True)
torch.manual_seed(0)

# Target context length
PAST_LEN_15K = 15 * 1024


@dataclass(frozen=True)
class PerfResult:
    """Performance measurement result."""
    kernel_name: str
    q_len: int
    past_len: int
    kv_cache_compression: int
    tile_q: int  # Only applicable for small_q

    avg_ms: float
    min_ms: float
    max_ms: float

    # Bandwidth calculation
    kv_bytes_read: float  # Bytes read from KV cache
    kv_bandwidth_gbs: float  # GB/s

    def __str__(self) -> str:
        return (
            f"{self.kernel_name:15s} q={self.q_len:2d} past={self.past_len:5d} "
            f"cmpr={self.kv_cache_compression} tile={self.tile_q} | "
            f"avg={self.avg_ms:6.3f}ms min={self.min_ms:6.3f}ms | "
            f"BW={self.kv_bandwidth_gbs:5.1f} GB/s"
        )


def _time_kernel(func, *, n_warmup: int = 5, n_iters: int = 50) -> List[float]:
    """Run kernel with warmup and collect timing samples (in nanoseconds)."""
    for _ in range(n_warmup):
        func()

    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        func()
        cl.finish()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)

    return samples


def _stats_from_samples(samples: List[float]) -> Dict[str, float]:
    """Convert timing samples (ns) to ms statistics."""
    if not samples:
        raise RuntimeError("No samples collected")
    avg = sum(samples) / len(samples)
    return {
        "avg_ms": avg * 1e-6,
        "min_ms": min(samples) * 1e-6,
        "max_ms": max(samples) * 1e-6,
    }


def _calc_kv_bandwidth(
    q_len: int,
    past_len: int,
    num_kv_heads: int,
    head_size: int,
    kv_cache_compression: int,
    time_ms: float,
) -> Tuple[float, float]:
    """
    Calculate KV cache bandwidth.

    Returns:
        (kv_bytes_read, bandwidth_gbs)
    """
    # Each query token reads all past_len K and V vectors
    # K and V are stored separately
    bytes_per_element = 2 if kv_cache_compression == 0 else 1  # fp16 or int8

    # Total bytes read = q_len * past_len * num_kv_heads * head_size * 2 (K+V) * bytes_per_element
    kv_bytes = q_len * past_len * num_kv_heads * head_size * 2 * bytes_per_element

    # Add quantization metadata overhead if compressed
    if kv_cache_compression == 1:  # per-token
        # scale + zp per token
        kv_bytes += q_len * past_len * num_kv_heads * 2 * 2  # 2 float16 per token
    elif kv_cache_compression == 2:  # per-channel
        # scale + zp per sub-block
        # Approximate: past_len tokens / 16 (sub_block_size) * num_kv_heads * 2 * 2
        kv_bytes += (past_len // 16) * num_kv_heads * 2 * 2

    bandwidth_gbs = kv_bytes / (time_ms * 1e-3) / 1e9

    return kv_bytes, bandwidth_gbs


# ============================================================================
# Test Case 1: Single Token (q=1)
# ============================================================================

def test_15k_single_token_q1_cmpr0():
    """Single token decode at 15K context, fp16 KV cache."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    case = DecodingCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        kv_len=PAST_LEN_15K,
        kv_cache_compression=0,
    )

    result = _benchmark_single_token(case)
    print(f"\n{result}")


def test_15k_single_token_q1_cmpr1():
    """Single token decode at 15K context, int8 per-token quantization."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    case = DecodingCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        kv_len=PAST_LEN_15K,
        kv_cache_compression=1,
    )

    result = _benchmark_single_token(case)
    print(f"\n{result}")


def _benchmark_single_token(case: DecodingCase) -> PerfResult:
    """Benchmark pa_single_token kernel."""
    runner = PaSingleTokenRunner(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
    )

    # Build inputs (simplified from test_pa_decoding.py)
    from test_pa_decoding import _build_single_subsequence_inputs
    data = _build_single_subsequence_inputs(case)

    # Create output tensor
    output = torch.zeros_like(data["query"])

    # Timing closure
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
            n_repeats=1,  # Important: pass as keyword argument
        )

    samples = _time_kernel(run_once, n_warmup=5, n_iters=50)
    stats = _stats_from_samples(samples)

    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=1,
        past_len=case.kv_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=stats["avg_ms"],
    )

    return PerfResult(
        kernel_name="single_token",
        q_len=1,
        past_len=case.kv_len,
        kv_cache_compression=case.kv_cache_compression,
        tile_q=1,  # N/A for single token
        avg_ms=stats["avg_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        kv_bytes_read=kv_bytes,
        kv_bandwidth_gbs=bandwidth,
    )


# ============================================================================
# Test Case 2: Small Q (q=3-16) with different TILE_Q
# ============================================================================

@pytest.mark.parametrize("q_len", [3, 4, 8, 16])
@pytest.mark.parametrize("cmpr", [0, 1])
@pytest.mark.parametrize("tile_q", [1, 2])
def test_15k_small_q(q_len: int, cmpr: int, tile_q: int):
    """Small q kernel at 15K context."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    if tile_q > q_len:
        pytest.skip(f"tile_q={tile_q} exceeds q_len={q_len}")

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


def _benchmark_small_q(case: SmallQCase) -> PerfResult:
    """Benchmark pa_small_q + finalization."""
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

    # Create output tensor
    output = torch.zeros([case.q_len, case.num_heads, case.head_size], dtype=torch.float16)

    # Timing closure
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

    samples = _time_kernel(run_once, n_warmup=5, n_iters=50)
    stats = _stats_from_samples(samples)

    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=case.q_len,
        past_len=case.past_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=stats["avg_ms"],
    )

    return PerfResult(
        kernel_name="small_q",
        q_len=case.q_len,
        past_len=case.past_len,
        kv_cache_compression=case.kv_cache_compression,
        tile_q=case.tile_q,
        avg_ms=stats["avg_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        kv_bytes_read=kv_bytes,
        kv_bandwidth_gbs=bandwidth,
    )


# ============================================================================
# Test Case 3: Multi Token (q=3-16)
# ============================================================================

@pytest.mark.parametrize("q_len", [3, 4, 8, 16])
@pytest.mark.parametrize("cmpr", [0, 1])
def test_15k_multi_token(q_len: int, cmpr: int):
    """Multi token kernel at 15K context."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    case = SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        past_len=PAST_LEN_15K,
        q_len=q_len,
        kv_cache_compression=cmpr,
        tile_q=2,  # N/A for multi_token, but needed for data gen
    )

    result = _benchmark_multi_token(case)
    print(f"\n{result}")


def _benchmark_multi_token(case: SmallQCase) -> PerfResult:
    """Benchmark pa_multi_token kernel."""
    runner = PaMultiTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.kv_cache_compression,
        is_causal=True,
        sub_block_size=case.sub_block_size,
    )

    # Use small_q's data builder, then reshape for multi_token
    data = _build_small_q_inputs(case)

    # Multi-token expects flattened shape: [batch_size_in_tokens, num_heads * head_size]
    # data["query"] is [q_len, num_heads, head_size], flatten the last two dims
    query_mt = data["query"].reshape(case.q_len, case.num_heads * case.head_size)
    output_mt = torch.zeros(case.q_len, case.num_heads * case.head_size, dtype=torch.float16)

    # Build kernel inputs dict
    kern_inputs = {
        "query": query_mt,
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "past_lens": data["past_lens"],
        "block_indices": data["block_indices"],
        "block_indices_begins": data["block_indices_begins"],
        "subsequence_begins": data["subsequence_begins"],
    }

    # prefill_seq_indices for the single sequence
    prefill_seq_indices = [0]

    # Timing closure
    def run_once():
        runner(
            kern_inputs,
            output_mt,
            prefill_seq_indices,
            n_repeats=1,
        )

    samples = _time_kernel(run_once, n_warmup=5, n_iters=50)
    stats = _stats_from_samples(samples)

    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=case.q_len,
        past_len=case.past_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=stats["avg_ms"],
    )

    return PerfResult(
        kernel_name="multi_token",
        q_len=case.q_len,
        past_len=case.past_len,
        kv_cache_compression=case.kv_cache_compression,
        tile_q=0,  # N/A
        avg_ms=stats["avg_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        kv_bytes_read=kv_bytes,
        kv_bandwidth_gbs=bandwidth,
    )


# ============================================================================
# Summary Test - Run All and Compare
# ============================================================================

def test_15k_summary_cmpr0():
    """Run all kernels and print comparison table (fp16 KV cache)."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    print("\n" + "=" * 100)
    print("Performance Comparison at 15K Context Length (fp16 KV Cache)")
    print("=" * 100)

    results = []

    # Single token q=1
    print("\n[1/13] Testing single_token q=1...")
    case_st = DecodingCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                           kv_len=PAST_LEN_15K, kv_cache_compression=0)
    results.append(_benchmark_single_token(case_st))

    # Small q with different q_len and tile_q
    for q_len in [3, 4, 8, 16]:
        for tile_q in [1, 2]:
            if tile_q > q_len:
                continue
            idx = len(results) + 1
            print(f"\n[{idx}/13] Testing small_q q={q_len} tile={tile_q}...")
            case_sq = SmallQCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                                past_len=PAST_LEN_15K, q_len=q_len, kv_cache_compression=0, tile_q=tile_q)
            results.append(_benchmark_small_q(case_sq))

    # Multi token
    for q_len in [3, 4, 8, 16]:
        idx = len(results) + 1
        print(f"\n[{idx}/13] Testing multi_token q={q_len}...")
        case_mt = SmallQCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                            past_len=PAST_LEN_15K, q_len=q_len, kv_cache_compression=0, tile_q=2)
        results.append(_benchmark_multi_token(case_mt))

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    for r in results:
        print(r)
    print("=" * 100)

    # Analysis
    print("\nKEY FINDINGS:")

    # Find best kernel for each q_len
    q_lens = sorted(set(r.q_len for r in results))
    for q in q_lens:
        q_results = [r for r in results if r.q_len == q]
        best = min(q_results, key=lambda r: r.avg_ms)
        print(f"\nq={q:2d}: Best kernel = {best.kernel_name:15s} "
              f"(tile_q={best.tile_q}) @ {best.avg_ms:.3f}ms, {best.kv_bandwidth_gbs:.1f} GB/s")

        # Show speedup vs multi_token if not q=1
        if q > 1:
            mt_results = [r for r in q_results if r.kernel_name == "multi_token"]
            if mt_results:
                speedup = mt_results[0].avg_ms / best.avg_ms
                print(f"      Speedup vs multi_token: {speedup:.2f}x")


def test_15k_summary_cmpr1():
    """Run all kernels and print comparison table (int8 quantized KV cache)."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    print("\n" + "=" * 100)
    print("Performance Comparison at 15K Context Length (INT8 Quantized KV Cache)")
    print("=" * 100)

    results = []

    # Single token q=1
    print("\n[1/13] Testing single_token q=1...")
    case_st = DecodingCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                           kv_len=PAST_LEN_15K, kv_cache_compression=1)
    results.append(_benchmark_single_token(case_st))

    # Small q
    for q_len in [3, 4, 8, 16]:
        for tile_q in [1, 2]:
            if tile_q > q_len:
                continue
            idx = len(results) + 1
            print(f"\n[{idx}/13] Testing small_q q={q_len} tile={tile_q}...")
            case_sq = SmallQCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                                past_len=PAST_LEN_15K, q_len=q_len, kv_cache_compression=1, tile_q=tile_q)
            results.append(_benchmark_small_q(case_sq))

    # Multi token
    for q_len in [3, 4, 8, 16]:
        idx = len(results) + 1
        print(f"\n[{idx}/13] Testing multi_token q={q_len}...")
        case_mt = SmallQCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
                            past_len=PAST_LEN_15K, q_len=q_len, kv_cache_compression=1, tile_q=2)
        results.append(_benchmark_multi_token(case_mt))

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (INT8)")
    print("=" * 100)
    for r in results:
        print(r)
    print("=" * 100)


if __name__ == "__main__":
    # Quick manual test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        os.environ["RUN_PA_PERF"] = "1"
        test_15k_summary_cmpr0()
