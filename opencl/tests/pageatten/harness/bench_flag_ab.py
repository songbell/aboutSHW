"""A/B any single -D compile flag of pa_small_q_ov_exp across (q_len, partition) points.

Generalised from bench_pipeline_ab.py, which was written for one flag and immediately needed
to be reused for a second.

    BENCH_FLAG=SPLIT_BARRIER BENCH_CONFIGS=16:640,16:384,16:256 \
        python /ceciliapeng/kernel_harness/bench_flag_ab.py

Two things this deliberately does NOT do, both learned the hard way on this kernel:

  * It does not compare against a number from a previous session. The rig measured the same
    config at 0.52 ms cold and 1.59 ms hot, so the arms are alternated inside every round and
    only the minimum is reported.
  * It does not call a delta smaller than the observed spread "no change". That is
    unresolvable, and reporting it as no-change is how a real regression gets waved through.

The flag must reach the build through OV_EXP_EXTRA_FLAGS, which is part of the kernel cache
key -- if it ever stops being one, every row here silently reads +0.0%.
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
FLAG = os.environ.get("BENCH_FLAG", "SPLIT_BARRIER")
ARMS = tuple(int(x) for x in os.environ.get('BENCH_ARMS', '0,1').split(','))
# Applied to BOTH arms. Use it to hold a second knob fixed while A/B-ing the first, e.g. to
# test whether two mechanisms are redundant: if a flag only pays once the prefetch is disabled,
# the prefetch was already hiding the latency it was meant to hide.
COMMON = os.environ.get("BENCH_COMMON_FLAGS", "")

# "q:part" pins past_len at 15k; "q:part:past" sets it explicitly. Use the latter whenever a
# partition is being judged for production: pick_small_q_partition ties the partition to the
# context length, so measuring part=256 at 15k context measures a pair the plugin never
# produces -- 60 partitions and ~480 workgroups instead of the ~32 it would really have.
# Judging a change on such a pair is exactly how the PIPELINE_MARSHAL "3.2 % win" survived in
# the ledger for as long as it did.
CONFIGS = []
for spec in os.environ.get("BENCH_CONFIGS", "16:640,16:384,16:256").split(","):
    parts = spec.split(":")
    q, part = int(parts[0]), int(parts[1])
    CONFIGS.append((q, part, int(parts[2]) if len(parts) > 2 else PAST_LEN_15K))


def flags(arm):
    return f"-D{FLAG}={arm} {COMMON}".strip()


def build(q_len, part, past, arm):
    # The flag must be in the environment both at construction and at first run, since either
    # may be what triggers compilation.
    os.environ["OV_EXP_EXTRA_FLAGS"] = flags(arm)
    case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                      block_size=BLOCK, past_len=past, q_len=q_len,
                      kv_cache_compression=CMPR, tile_q=q_len, partition_block_num=1)
    runner = PaSmallQOvExpRunner(
        case.num_heads, case.num_kv_heads, case.head_size, case.block_size,
        case.sub_block_size, case.kv_cache_compression,
        tile_q=case.tile_q, kv_partition_size=part)
    _run_perf_with_runner_exp(runner, case)          # warm up / force the build
    return case, runner


built = {}
for q_len, part, past in CONFIGS:
    for arm in ARMS:
        built[(q_len, part, past, arm)] = build(q_len, part, past, arm)

samples = {k: [] for k in built}
for _ in range(ROUNDS):
    for q_len, part, past in CONFIGS:
        for arm in ARMS:
            key = (q_len, part, past, arm)
            case, runner = built[key]
            os.environ["OV_EXP_EXTRA_FLAGS"] = flags(arm)
            perf = _run_perf_with_runner_exp(runner, case)
            samples[key].append(perf["small_q_ms"])

print(f"\n{FLAG}: cmpr={CMPR} block={BLOCK} rounds={ROUNDS}"
      f"{' common=' + COMMON if COMMON else ''} (main kernel, min over rounds)")
print(f"{'q':>3} {'past':>6} {'part':>5} {'thr':>4} {('=%d' % ARMS[0]):>8} {('=%d' % ARMS[1]):>8} {'delta':>8} "
      f"{'noise':>7}   verdict")
for q_len, part, past in CONFIGS:
    row = {}
    for arm in ARMS:
        s = samples[(q_len, part, past, arm)]
        row[arm] = (min(s), (max(s) / min(s) - 1.0) * 100.0)
    (m0, s0), (m1, s1) = row[ARMS[0]], row[ARMS[1]]
    delta = (m1 / m0 - 1.0) * 100.0
    noise = max(s0, s1)
    if abs(delta) <= noise:
        verdict = "INCONCLUSIVE (delta < spread)"
    else:
        verdict = "WINS" if delta < 0 else "loses"
    thr = built[(q_len, part, past, ARMS[0])][1].wg_threads
    print(f"{q_len:>3} {past:>6} {part:>5} {thr:>4} {m0:>8.3f} {m1:>8.3f} {delta:>+7.1f}% "
          f"{noise:>6.1f}%   {verdict}")
