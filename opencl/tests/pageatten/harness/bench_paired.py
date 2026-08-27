"""Interleaved, min-of-N benchmark for pa_small_q_ov_exp.

The rig throttles hard: the same config measured 0.52 ms early in a session and 1.59 ms
late. Sequential A-then-B therefore charges B for A's heat. This alternates configs round
by round and reports the *minimum* over rounds, which is the least-throttled sample and the
only statistic that is stable here. It also reports the spread so a claim can be checked
against the noise floor before it is believed.
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

HEADS, KV_HEADS, HEAD_SIZE, BLOCK = 32, 8, 128, 256
CMPR = int(os.environ.get("BENCH_CMPR", "2"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))
CONFIGS = []
for spec in os.environ.get("BENCH_CONFIGS", "6:512,16:512,16:1024").split(","):
    q, part = spec.split(":")
    CONFIGS.append((int(q), int(part)))

runners = {}
for q_len, part in CONFIGS:
    case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                      block_size=BLOCK, past_len=PAST_LEN_15K, q_len=q_len,
                      kv_cache_compression=CMPR, tile_q=q_len, partition_block_num=1)
    runners[(q_len, part)] = (case, PaSmallQOvExpRunner(
        case.num_heads, case.num_kv_heads, case.head_size, case.block_size,
        case.sub_block_size, case.kv_cache_compression,
        tile_q=case.tile_q, kv_partition_size=part))

samples = {k: [] for k in runners}
for r in range(ROUNDS):
    for key in CONFIGS:                      # same order each round, but every config
        case, runner = runners[key]          # sees the same thermal history by round r
        perf = _run_perf_with_runner_exp(runner, case)
        samples[key].append((perf["small_q_ms"], perf["small_q_reduce_ms"]))

print(f"\ncmpr={CMPR} block={BLOCK} past={PAST_LEN_15K} rounds={ROUNDS} (min over rounds)")
print(f"{'q':>3} {'part':>5} {'thr':>4} {'main':>7} {'reduce':>7} {'total':>7} "
      f"{'main_spread':>12} {'ms/token':>9}")
base = None
for key in CONFIGS:
    q_len, part = key
    _, runner = runners[key]
    mains = [s[0] for s in samples[key]]
    reds = [s[1] for s in samples[key]]
    main, red = min(mains), min(reds)
    total = main + red
    if base is None:
        base = total
    print(f"{q_len:>3} {part:>5} {runner.wg_threads:>4} {main:>7.3f} {red:>7.3f} {total:>7.3f} "
          f"{min(mains):>5.3f}-{max(mains):<6.3f} {total / q_len:>9.4f}")
