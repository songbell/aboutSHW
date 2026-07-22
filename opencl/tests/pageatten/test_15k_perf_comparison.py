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

import functools
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
    _ceil_div,
    _run_perf as _small_q_run_perf,
)
from test_pa_multiseq import PaMultiTokenRunner

cl.profiling(True)
torch.manual_seed(0)

# Target context length
PAST_LEN_15K = 15 * 1024


class PaSmallQOnlineRunner(PaSmallQRunner):
    """Runner that uses pa_small_q_ov.cm with configurable partition size."""

    def __init__(self, *args, k_partition_block_num=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_partition_block_num = k_partition_block_num
        self.kv_partition_size = self.block_size * k_partition_block_num

    @classmethod
    def create_instance(cls, *args, k_partition_block_num=1, **kwargs):
        inst = super().create_instance(*args, **kwargs)
        inst.k_partition_block_num = k_partition_block_num
        inst.kv_partition_size = inst.block_size * k_partition_block_num
        return inst

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def _create_kernels_online(
        num_heads, num_kv_heads, head_size, kv_step, block_size, sub_block_size,
        kv_partition_size, reduce_split_step, clean_unused_kvcache,
        kv_cache_compression, xe_arch, q_head_chunks_per_kv_head,
        q_head_chunk_size, tile_q, scale_factor,
    ):
        import os as _os
        src = '\n'.join([
            '#include "pa_small_q_ov.cm"',
            '#include "pa_small_q_finalization.cm"',
        ])
        cwd = _os.path.dirname(_os.path.realpath(__file__))
        return cl.kernels(
            src,
            f'''-cmc -Qxcm_jit_option=""
                        -mCM_printregusage
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DHEADS_NUM={num_heads} -DKV_HEADS_NUM={num_kv_heads} -DHEAD_SIZE={head_size}
                        -DQ_STEP=32 -DKV_STEP={kv_step}
                        -DKV_BLOCK_SIZE={block_size}
                        -DKV_PARTITION_SIZE={kv_partition_size} -DREDUCE_SPLIT_SIZE={reduce_split_step}
                        -DCLEAN_UNUSED_KVCACHE={clean_unused_kvcache}
                        -DKV_CACHE_COMPRESSION={kv_cache_compression}
                        -DSUB_BLOCK_SIZE={sub_block_size}
                        -DXE_ARCH={xe_arch}
                        -DQ_head_chunks_per_kv_head={q_head_chunks_per_kv_head}
                        -DQ_head_chunk_size={q_head_chunk_size}
                        -DTILE_Q={tile_q}
                        -DSCALE_FACTOR={scale_factor}
                        -DKERNEL_NAME=cm_pa_small_q''',
        )

    def _create_kernels(self):
        return self._create_kernels_online(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_step,
            self.block_size,
            self.sub_block_size,
            self.kv_partition_size,
            self.reduce_split_step,
            1,
            self.kv_cache_compression,
            self.xe_arch,
            int(self.q_head_chunks_per_kv_head),
            int(self.q_head_chunk_size),
            self.tile_q,
            self.scale_factor,
        )

    def _enqueue_once(self, kernels, q_tokens, key_cache, value_cache,
                      past_lens, block_indices, block_indices_begins,
                      subsequence_begins, mapping, tile_count, kv_partition_num, out):
        q_len = int(q_tokens.shape[0])
        partitions_per_work_item = 1
        # Recompute partition num with our (possibly larger) partition size
        max_context_len = int(past_lens.max().item()) + q_len
        kv_partition_num = _ceil_div(max_context_len, self.kv_partition_size)

        gws = [int(tile_count),
               self.num_kv_heads * self.q_head_chunks_per_kv_head,
               (kv_partition_num + partitions_per_work_item - 1) // partitions_per_work_item]
        lws = [1, 1, 1]
        gws_2 = [int(tile_count) * self.tile_q, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        t_q = cl.tensor(q_tokens.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_past_lens = cl.tensor(past_lens.detach().numpy())
        t_block_indices = cl.tensor(block_indices.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
        t_mapping = cl.tensor(mapping.detach().numpy())

        partition_token_rows = int(tile_count) * self.tile_q
        padded_partition_num = gws[2] * partitions_per_work_item
        t_partition_out = cl.tensor(
            [partition_token_rows, self.num_heads, padded_partition_num, self.head_size],
            np.dtype(np.float32),
        )
        t_lse = cl.tensor(
            [partition_token_rows, self.num_heads, padded_partition_num], np.dtype(np.float32)
        )
        t_out_final = cl.tensor(out.contiguous().detach().numpy())

        kernels.enqueue("cm_pa_small_q", gws, lws,
                        t_q, t_k, t_v, t_past_lens, t_block_indices,
                        t_block_indices_begins, t_subsequence_begins, t_mapping,
                        t_partition_out, t_lse, int(q_len), int(tile_count))
        kernels.enqueue("cm_pa_small_q_reduce", gws_2, lws_2,
                        t_partition_out, t_out_final, t_lse, t_subsequence_begins,
                        t_mapping, partition_token_rows, padded_partition_num)
        cl.finish()
        out.copy_(torch.from_numpy(t_out_final.numpy()))


@dataclass(frozen=True)
class PerfResult:
    """Performance measurement result."""
    kernel_name: str
    q_len: int
    past_len: int
    kv_cache_compression: int
    tile_q: int  # Only applicable for small_q

    avg_ms: float       # wall-clock per call (INCLUDES host KV re-upload)
    min_ms: float
    max_ms: float

    # Bandwidth: kv_bytes_read / kernel_ms (pure GPU time). BW from wall-clock
    # is meaningless because it's dominated by a fixed host->device numpy
    # re-upload (~3-6ms) that repeats every iteration.
    kv_bytes_read: float
    kv_bandwidth_gbs: float

    # Pure GPU kernel time from profiling events (no host re-upload). This is
    # the real device cost the optimization targets. 0.0 means kernel-only
    # timing was unavailable / failed.
    kernel_ms: float = 0.0
    kernel_bw_gbs: float = 0.0

    def __str__(self) -> str:
        ktxt = (
            f" | kernel={self.kernel_ms:6.3f}ms ({self.kernel_bw_gbs:5.1f} GB/s)"
            if self.kernel_ms > 0 else ""
        )
        return (
            f"{self.kernel_name:15s} q={self.q_len:2d} past={self.past_len:5d} "
            f"cmpr={self.kv_cache_compression} tile={self.tile_q} | "
            f"wall={self.avg_ms:6.3f}ms min={self.min_ms:6.3f}ms"
            f"{ktxt}"
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

    IMPORTANT: `time_ms` MUST be the pure GPU kernel time (from profiling events),
    NOT the wall-clock per-call time. The wall-clock loop re-uploads the whole KV
    cache from numpy every iteration (~3-6ms fixed host cost unrelated to the GPU),
    so dividing kv_bytes by wall-clock time yields a meaningless, deflated BW.
    Pass kernel_ms here.

    Returns:
        (kv_bytes_read, bandwidth_gbs)
    """
    # Logical KV bytes: q_len * (physical KV cache size). This is what the kernel
    # would transfer if every Q token independently pulled the full KV from DRAM.
    # Compared against DRAM peak (~150 GB/s on Arc 140V LPDDR5X):
    #   - well below peak  -> kernel is compute-bound or has slack
    #   - near peak        -> memory-bound, no cross-Q cache reuse
    #   - above peak       -> cache reuse across Q tokens is saving DRAM traffic
    #                         (higher = better reuse). "300 GB/s logical" on a
    #                         150 GB/s DRAM means ~2x reuse.
    # This metric measures kernel efficiency, NOT true DRAM BW. To get true
    # DRAM BW you need hardware counters (VTune / GPA).
    bytes_per_element = 2 if kv_cache_compression == 0 else 1  # fp16 or int8

    kv_bytes = q_len * past_len * num_kv_heads * head_size * 2 * bytes_per_element

    # Add quantization metadata overhead if compressed
    if kv_cache_compression == 1:  # per-token
        kv_bytes += q_len * past_len * num_kv_heads * 2 * 2  # scale + zp (fp16) per token
    elif kv_cache_compression == 2:  # per-channel
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

    # Pure GPU kernel time (no host re-upload), via decoding harness profiling timer.
    # BW is computed from THIS, not from wall-clock time.
    kernel_ms = 0.0
    try:
        from test_pa_decoding import _run_bandwidth_measurement
        perf = _run_bandwidth_measurement(case, loop_cnt=60, warmup=8)
        kernel_ms = perf["cm_sdpa_2nd_ms"] + perf["cm_sdpa_2nd_reduce_ms"]
    except Exception as e:
        print(f"  [warn] single_token kernel-only timing failed: {e}")

    bw_time_ms = kernel_ms if kernel_ms > 0 else stats["avg_ms"]
    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=1,
        past_len=case.kv_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=bw_time_ms,
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
        kernel_ms=kernel_ms,
        kernel_bw_gbs=bandwidth,
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

    # Pure GPU kernel time (no host re-upload): use the small_q perf harness
    # which pre-uploads device tensors ONCE, rotates over many buffers (>L2 so
    # no cache-hit inflation), and times each enqueue via profiling events.
    kernel_ms = 0.0
    try:
        perf = _small_q_run_perf(case, loop_cnt=60, warmup=8)
        kernel_ms = perf["small_q_ms"] + perf["small_q_reduce_ms"]
    except Exception as e:
        print(f"  [warn] small_q kernel-only timing failed for q={case.q_len} tile={case.tile_q}: {e}")

    bw_time_ms = kernel_ms if kernel_ms > 0 else stats["avg_ms"]
    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=case.q_len,
        past_len=case.past_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=bw_time_ms,
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
        kernel_ms=kernel_ms,
        kernel_bw_gbs=bandwidth,
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

    # NOTE: PaMultiTokenRunner.__call__ calls cl.finish() internally every
    # invocation, so we cannot collect kernel-only profiling events across a loop
    # (a second cl.finish() throws std::future_error). Report wall-clock only,
    # leave kernel_ms=0 (BW hidden since wall-clock BW is meaningless — see
    # _calc_kv_bandwidth docstring).
    kv_bytes = (case.q_len * case.past_len * case.num_kv_heads * case.head_size * 2
                * (2 if case.kv_cache_compression == 0 else 1))

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
        kv_bandwidth_gbs=0.0,
        kernel_ms=0.0,
        kernel_bw_gbs=0.0,
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


# ============================================================================
# Test Case 4: Small Q Online Softmax (q=3-16)
# ============================================================================

@pytest.mark.parametrize("q_len", [3, 4, 8, 16])
@pytest.mark.parametrize("cmpr", [0, 1])
@pytest.mark.parametrize("tile_q", [1, 2])
def test_15k_small_q_online(q_len: int, cmpr: int, tile_q: int):
    """Small q online softmax kernel at 15K context."""
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

    result = _benchmark_small_q_online(case)
    print(f"\n{result}")


def _run_perf_with_runner(runner, case: SmallQCase, loop_cnt: int = 60,
                          warmup: int = 8) -> dict[str, float]:
    """Kernel-only timing (via profiling events) for any small_q-style runner.

    Pre-uploads device tensors once; rotates over enough layers (~8 GB total)
    so L2 doesn't hide real memory-bandwidth cost. Times each (small_q +
    reduce) pair via cl.finish() profiling events.
    """
    data = _build_small_q_inputs(case)
    kernels = runner._create_kernels()

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
    partition_token_rows = tile_count * runner.tile_q

    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_partition_out = cl.tensor(
            [partition_token_rows, runner.num_heads, kv_partition_num, runner.head_size],
            np.dtype(np.float32),
        )
        t_out_final = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))
        all_layers.append((t_q, t_k, t_v, t_partition_out, t_out_final))
        mem_size += query.numel() * query.element_size()
        mem_size += key_cache.numel() * key_cache.element_size()
        mem_size += value_cache.numel() * value_cache.element_size()

    if not all_layers:
        raise RuntimeError("Failed to allocate perf input layers")

    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
    t_mapping = cl.tensor(mapping.detach().numpy())
    t_lse = cl.tensor([partition_token_rows, runner.num_heads, kv_partition_num], np.dtype(np.float32))

    cl.finish()

    gws = [tile_count, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [partition_token_rows, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_po, t_out = all_layers[j]
        kernels.enqueue("cm_pa_small_q", gws, lws,
                        t_q, t_k, t_v,
                        t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins,
                        t_mapping, t_po, t_lse,
                        q_len, tile_count)
        kernels.enqueue("cm_pa_small_q_reduce", gws_2, lws_2,
                        t_po, t_out, t_lse, t_subsequence_begins, t_mapping,
                        partition_token_rows, kv_partition_num)

    latency = cl.finish()
    expected = 2 * loop_cnt
    if len(latency) < expected:
        raise RuntimeError(f"Expected at least {expected} profiling events, got {len(latency)}")

    sq_total = 0.0
    rd_total = 0.0
    runs = 0
    for pair_idx in range(loop_cnt):
        kv_ns = float(latency[2 * pair_idx])
        rd_ns = float(latency[2 * pair_idx + 1])
        if kv_ns <= 0 or rd_ns <= 0 or pair_idx < warmup:
            continue
        sq_total += kv_ns
        rd_total += rd_ns
        runs += 1
    if runs <= 0:
        raise RuntimeError("Invalid perf timing accumulation")

    return {
        "small_q_ms": sq_total * 1e-6 / runs,
        "small_q_reduce_ms": rd_total * 1e-6 / runs,
    }


def _benchmark_small_q_online(case: SmallQCase) -> PerfResult:
    """Benchmark pa_small_q_ov + finalization."""
    runner = PaSmallQOnlineRunner(
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

    # Pure GPU kernel time via profiling events (see _run_perf_with_runner).
    kernel_ms = 0.0
    try:
        perf = _run_perf_with_runner(runner, case, loop_cnt=60, warmup=8)
        kernel_ms = perf["small_q_ms"] + perf["small_q_reduce_ms"]
    except Exception as e:
        print(f"  [warn] small_q_online kernel-only timing failed q={case.q_len} tile={case.tile_q}: {e}")

    bw_time_ms = kernel_ms if kernel_ms > 0 else stats["avg_ms"]
    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=case.q_len,
        past_len=case.past_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=bw_time_ms,
    )

    return PerfResult(
        kernel_name="small_q_online",
        q_len=case.q_len,
        past_len=case.past_len,
        kv_cache_compression=case.kv_cache_compression,
        tile_q=case.tile_q,
        avg_ms=stats["avg_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        kv_bytes_read=kv_bytes,
        kv_bandwidth_gbs=bandwidth,
        kernel_ms=kernel_ms,
        kernel_bw_gbs=bandwidth,
    )


# ============================================================================
# Test Case 5: Online Softmax with Large Partition (partition=512,1024,2048)
# ============================================================================

@pytest.mark.parametrize("q_len", [4, 8, 16])
@pytest.mark.parametrize("cmpr", [1])
@pytest.mark.parametrize("k_part_blocks", [1, 2, 4, 8])
def test_15k_online_large_partition(q_len: int, cmpr: int, k_part_blocks: int):
    """Online softmax with larger partition size at 15K context."""
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
        tile_q=1,
    )

    result = _benchmark_online_large_partition(case, k_part_blocks)
    print(f"\n{result}")


def _benchmark_online_large_partition(case: SmallQCase, k_part_blocks: int) -> PerfResult:
    """Benchmark online softmax with larger partition (pre-allocated tensors)."""
    runner = PaSmallQOnlineRunner(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
        tile_q=case.tile_q,
        k_partition_block_num=k_part_blocks,
    )

    data = _build_small_q_inputs(case)
    q_len = case.q_len
    kernels = runner._create_kernels()
    mapping, tile_count = runner._build_mapping(q_len, runner.tile_q)
    max_context_len = int(data["past_lens"].max().item()) + q_len
    kv_partition_num = _ceil_div(max_context_len, runner.kv_partition_size)

    gws = [int(tile_count), runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [int(tile_count) * runner.tile_q, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    t_q = cl.tensor(data["query"].numpy())
    t_k = cl.tensor(data["key_cache"].contiguous().numpy())
    t_v = cl.tensor(data["value_cache"].contiguous().numpy())
    t_past = cl.tensor(data["past_lens"].numpy())
    t_bi = cl.tensor(data["block_indices"].numpy())
    t_bib = cl.tensor(data["block_indices_begins"].numpy())
    t_sb = cl.tensor(data["subsequence_begins"].numpy())
    t_map = cl.tensor(mapping.numpy())
    partition_token_rows = int(tile_count) * runner.tile_q
    t_po = cl.tensor([partition_token_rows, runner.num_heads, kv_partition_num, runner.head_size], np.dtype(np.float32))
    t_lse = cl.tensor([partition_token_rows, runner.num_heads, kv_partition_num], np.dtype(np.float32))
    t_out = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))

    def run_once():
        kernels.enqueue("cm_pa_small_q", gws, lws, t_q, t_k, t_v, t_past, t_bi, t_bib, t_sb, t_map, t_po, t_lse, q_len, int(tile_count))
        kernels.enqueue("cm_pa_small_q_reduce", gws_2, lws_2, t_po, t_out, t_lse, t_sb, t_map, partition_token_rows, kv_partition_num)

    samples = _time_kernel(run_once, n_warmup=5, n_iters=50)
    stats = _stats_from_samples(samples)

    # Pure GPU kernel time via profiling events.
    kernel_ms = 0.0
    try:
        perf = _run_perf_with_runner(runner, case, loop_cnt=60, warmup=8)
        kernel_ms = perf["small_q_ms"] + perf["small_q_reduce_ms"]
    except Exception as e:
        print(f"  [warn] online_large_partition kernel-only timing failed part={256*k_part_blocks}: {e}")

    bw_time_ms = kernel_ms if kernel_ms > 0 else stats["avg_ms"]
    kv_bytes, bandwidth = _calc_kv_bandwidth(
        q_len=case.q_len,
        past_len=case.past_len,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        kv_cache_compression=case.kv_cache_compression,
        time_ms=bw_time_ms,
    )

    part_size = 256 * k_part_blocks
    return PerfResult(
        kernel_name=f"online_p{part_size}",
        q_len=case.q_len,
        past_len=case.past_len,
        kv_cache_compression=case.kv_cache_compression,
        tile_q=case.tile_q,
        avg_ms=stats["avg_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        kv_bytes_read=kv_bytes,
        kv_bandwidth_gbs=bandwidth,
        kernel_ms=kernel_ms,
        kernel_bw_gbs=bandwidth,
    )


if __name__ == "__main__":
    # Quick manual test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        os.environ["RUN_PA_PERF"] = "1"
        test_15k_summary_cmpr0()
