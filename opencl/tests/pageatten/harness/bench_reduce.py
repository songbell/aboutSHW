"""Sweep REDUCE_SPLIT_SIZE for the small-q finalization kernel.

head_size / REDUCE_SPLIT_SIZE is the third gws dimension, so this knob sets how many threads
the reduce runs with: q_len * heads * (head_size / split). At the default 64 and q_len=16 that
is 16*32*2 = 1024 threads, each reading kv_partition_num * split floats. Smaller split = more
threads, each doing less; the tradeoff is the per-thread fixed cost (the lse max/sum passes,
which are re-done per split rather than shared).
"""
import os
import sys

sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl")
sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl/tests/pageatten")
os.environ.setdefault("RUN_PA_PERF", "1")

from test_15k_perf_comparison import PAST_LEN_15K
from test_15k_perf_comparison_ov_exp import _run_perf_with_runner_exp
from test_ov_exp_kernel_correctness import PaSmallQOvExpRunner
from test_pa_small_q import SmallQCase

CMPR = int(os.environ.get("BENCH_CMPR", "2"))
BLOCK = int(os.environ.get("BENCH_BLOCK", "256"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "3"))
SPLITS = [int(s) for s in os.environ.get("BENCH_SPLITS", "16,32,64,128").split(",")]
CONFIGS = []
for spec in os.environ.get("BENCH_CONFIGS", "16:512,16:1024").split(","):
    q, part = spec.split(":")
    CONFIGS.append((int(q), int(part)))

print(f"block={BLOCK} cmpr={CMPR}")
print(f"{'q':>3} {'part':>5} {'split':>6} {'threads':>8} {'main':>7} {'reduce':>8} {'total':>7}")
for q_len, part in CONFIGS:
    for split in SPLITS:
        case = SmallQCase(num_heads=32, num_kv_heads=8, head_size=128, block_size=BLOCK,
                          past_len=PAST_LEN_15K, q_len=q_len, kv_cache_compression=CMPR,
                          tile_q=q_len, partition_block_num=1)
        runner = PaSmallQOvExpRunner(case.num_heads, case.num_kv_heads, case.head_size,
                                     case.block_size, case.sub_block_size,
                                     case.kv_cache_compression, tile_q=case.tile_q,
                                     kv_partition_size=part)
        if case.head_size % split != 0:
            continue
        runner.reduce_split_step = split
        try:
            samples = [_run_perf_with_runner_exp(runner, case) for _ in range(ROUNDS)]
        except Exception as e:
            print(f"{q_len:>3} {part:>5} {split:>6} FAILED: {str(e)[:70]}")
            continue
        main = min(s["small_q_ms"] for s in samples)
        red = min(s["small_q_reduce_ms"] for s in samples)
        threads = q_len * case.num_heads * (case.head_size // split)
        print(f"{q_len:>3} {part:>5} {split:>6} {threads:>8} {main:>7.3f} {red:>8.3f} "
              f"{main + red:>7.3f}")
