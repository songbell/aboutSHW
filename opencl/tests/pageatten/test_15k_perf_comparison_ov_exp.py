"""15K perf harness for the experimental pa_small_q_ov_exp.cm kernel."""

import os
from itertools import count

import numpy as np
import pytest
import torch

from test_15k_perf_comparison import (
    PAST_LEN_15K,
    PerfResult,
    _build_small_q_inputs,
    _calc_kv_bandwidth,
    _ceil_div,
    _stats_from_samples,
    _time_kernel,
)
from test_ov_exp_kernel_correctness import PaSmallQOvExpRunner
from test_pa_small_q import SmallQCase
from clops import cl


_RESULT_COUNTER = count(1)


def _run_perf_with_runner_exp(runner, case: SmallQCase, loop_cnt: int = 60, warmup: int = 8) -> dict[str, float]:
    """Kernel-only timing for the experimental runner."""
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
    gws, lws, gws_2, lws_2, partition_token_rows, padded_partition_num = runner.dispatch_dims(tile_count, kv_partition_num)

    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_partition_out = cl.tensor(
            [partition_token_rows, runner.num_heads, padded_partition_num, runner.head_size],
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
    t_lse = cl.tensor([partition_token_rows, runner.num_heads, padded_partition_num], np.dtype(np.float32))

    cl.finish()

    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_po, t_out = all_layers[j]
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
            tile_count,
        )
        kernels.enqueue(
            "cm_pa_small_q_reduce",
            gws_2,
            lws_2,
            t_po,
            t_out,
            t_lse,
            t_subsequence_begins,
            t_mapping,
            partition_token_rows,
            padded_partition_num,
        )

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


def _benchmark_small_q_online_exp(case: SmallQCase, kv_partition_size: int | None = None) -> PerfResult:
    """Benchmark the experimental kernel on `case`.

    kv_partition_size pins KV_PARTITION_SIZE directly instead of deriving it as
    block_size * partition_block_num. Both forms reach the same kernel, but at block 256
    the ratio form is the wrong knob: OV picks block and partition from the same
    has_xattention flag (16/128 and 256/256), so a partition_block_num carried over from
    a block-16 sweep silently builds a multi-thousand-token partition.
    """
    old_force = os.environ.get("OV_FORCE_Q_HEAD_CHUNK_SIZE")
    old_shared = os.environ.get("OV_EXP_USE_WG_SHARED_KV")
    os.environ["OV_FORCE_Q_HEAD_CHUNK_SIZE"] = "4"
    os.environ["OV_EXP_USE_WG_SHARED_KV"] = "1"
    try:
        runner = PaSmallQOvExpRunner(
            case.num_heads,
            case.num_kv_heads,
            case.head_size,
            case.block_size,
            case.sub_block_size,
            case.kv_cache_compression,
            tile_q=case.tile_q,
            k_partition_block_num=case.partition_block_num,
            kv_partition_size=kv_partition_size,
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

        kernel_ms = 0.0
        try:
            perf = _run_perf_with_runner_exp(runner, case, loop_cnt=60, warmup=8)
            print(f"  [info] small_q_ov_exp kernel-only timing: {perf['small_q_ms']:.3f}ms + {perf['small_q_reduce_ms']:.3f}ms")
            kernel_ms = perf["small_q_ms"] + perf["small_q_reduce_ms"]
        except Exception as e:
            print(f"  [warn] small_q_ov_exp kernel-only timing failed q={case.q_len} tile={case.tile_q}: {e}")

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
            kernel_name="small_q_ov_exp",
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
    finally:
        if old_force is None:
            os.environ.pop("OV_FORCE_Q_HEAD_CHUNK_SIZE", None)
        else:
            os.environ["OV_FORCE_Q_HEAD_CHUNK_SIZE"] = old_force
        if old_shared is None:
            os.environ.pop("OV_EXP_USE_WG_SHARED_KV", None)
        else:
            os.environ["OV_EXP_USE_WG_SHARED_KV"] = old_shared


@pytest.mark.parametrize("q_len", [6, 16])
@pytest.mark.parametrize("cmpr", [1, 2])
# tile_q=6 is the natural fit for q_len=6 but gives Q_ROW_GROUPS=3, which neither divides
# the 16 KV rows the producer stages nor the 8 marshalling chunks. tile_q=8 wastes 2 of 8
# q-rows and is slower on the current kernel, yet is much faster once the round 2-4 changes
# are enabled. See PROFILING.md "Round 5".
def test_15k_small_q_online_exp_block_size_16(q_len: int, cmpr: int):
    """Profile the experimental pa_small_q_ov_exp kernel at the 6-2-6 point."""
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable perf test")

    case = SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        past_len=PAST_LEN_15K,
        q_len=q_len,
        kv_cache_compression=cmpr,
        tile_q=q_len,
        # KV_PARTITION_SIZE = block_size * partition_block_num, which sets how many fp32
        # partial slices the main kernel writes and the reduce kernel reads back. At 15 k
        # context those partials are the largest single term in the pair's DRAM traffic
        # (16.0 MB of 46.5 MB at pb=16). Paired A/B on *total* time at 192 GRF, 10 rounds:
        #   pb=32   8.1 MB partials, reduce 0.111 ms   baseline
        #   pb=64   4.2 MB partials, reduce 0.040 ms   main +1.7 %, total -4.3 % (10/10 rounds)
        #   pb=128  2.1 MB partials, reduce 0.029 ms   main +5.8 %, total -1.7 %
        # pb=64 is best here, but the trade is shape-dependent -- larger partitions mean
        # fewer workgroups, so a production host should scale this with context length
        # rather than pin it. It also moved with the register file: at 160 GRF pb=64 cost
        # the main kernel 4-7 %, at 192 only 1.7 %.
        # Correctness over pb in {8,16,32,64,128}: prof_results/pb_correctness.py.
        partition_block_num=32,
    )
    result = _benchmark_small_q_online_exp(case)
    result_id = next(_RESULT_COUNTER)
    print(f"\n[Result #{result_id:02d}] {result}")


# KV_PARTITION_SIZE that OV pairs with KV_BLOCK_SIZE=256 (get_partition_size returns
# PA_KV_CACHE_BLOCK_SIZE_XATTN when has_xattention). Mirrors _ov_partition_size in
# test_ov_exp_kernel_correctness.py, which is the file that owns that mapping.
_OV_PARTITION_SIZE_BLK256 = 256


@pytest.mark.parametrize("q_len", [6, 16])
@pytest.mark.parametrize("cmpr", [1, 2])
def test_15k_small_q_online_exp_block_size_256(q_len: int, cmpr: int):
    """Profile the experimental kernel at block 256 -- the xattention cache layout.

    Same 15 k point as the block-16 test above, so the two are directly comparable on
    kernel_ms. The KV traffic is identical; what changes is that a cache block now spans
    STEPS_PER_BLOCK = 16 online tiles instead of exactly one, so consecutive tiles walk a
    within-block token offset rather than re-reading block_indices, and the scale/zp rows
    for a tile sit at group tok_in_blk / SUB_BLOCK_SIZE.

    Correctness for this configuration lives in
    test_ov_exp_kernel_correctness.py::test_ov_exp_correctness_block_size_256.
    """
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
        tile_q=q_len,
        # Unused: kv_partition_size below pins KV_PARTITION_SIZE outright. Left at the
        # dataclass default so it is obvious the ratio form is not in play here.
        partition_block_num=1,
    )
    # 256 is what OV builds. The block-16 test's partition sweep (pb=32..128 at 15 k)
    # found larger partitions win by shrinking the fp32 partials the reduce kernel reads
    # back, and at 15 k a 256-token partition means 60 of them -- so this point is
    # expected to be reduce-heavy, and re-running the sweep here is worthwhile.
    result = _benchmark_small_q_online_exp(case, kv_partition_size=_OV_PARTITION_SIZE_BLK256)
    result_id = next(_RESULT_COUNTER)
    print(f"\n[Result #{result_id:02d}] {result} | block=256 partition={_OV_PARTITION_SIZE_BLK256}")
