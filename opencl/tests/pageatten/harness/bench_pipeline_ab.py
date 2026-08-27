"""A/B PIPELINE_MARSHAL (SLM double-buffering) per (q_len, partition).

PIPELINE_MARSHAL was measured and rejected as a *global* switch: it wins 3.2% at q_len=16 /
part 512 but costs +45% at q_len=6 / part 512. The file records why -- the inflation scales
with chunks-per-thread (MARSHAL_CHUNKS_C / WG_THREADS), which is 1 at q_len=16 (8 threads)
and 3 at q_len=6 (3 threads). Since the rung mechanism now compiles a separate variant per
TILE_Q, the switch does not have to be global.

Two things this has to establish before that is worth doing:
  1. part 640 is what the runtime partition rule actually picks at 15k context, and it was
     never measured -- the recorded data is 512 (-3.2%) and 1024 (+20%), and 640 is between.
  2. q_len=6 must still lose, or the per-rung gate is pointless.

Config 16:512 is included purely as a rig self-check: if it does not reproduce the recorded
-3.2%, the rig disagrees with the ledger and nothing else in this run should be believed.

PIPELINE_MARSHAL is #ifndef-guarded, so this A/Bs through OV_EXP_EXTRA_FLAGS without
touching the kernel source -- both arms are byte-identical apart from that one define.

    BENCH_ROUNDS=5 python /ceciliapeng/kernel_harness/bench_pipeline_ab.py
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
ARMS = (0, 1)

CONFIGS = []
for spec in os.environ.get("BENCH_CONFIGS", "16:640,16:512,6:640").split(","):
    q, part = spec.split(":")
    CONFIGS.append((int(q), int(part)))


def build(q_len, part, pipeline):
    """Build under the arm's flag and immediately exercise it.

    The build is keyed on the option string, so the flag must be in the environment both when
    the runner is constructed and when it first runs -- whichever of the two actually triggers
    compilation. Setting it for both is the only way to be sure the arm measured is the arm
    requested; getting this wrong would silently A/B a kernel against itself.
    """
    os.environ["OV_EXP_EXTRA_FLAGS"] = f"-DPIPELINE_MARSHAL={pipeline}"
    case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                      block_size=BLOCK, past_len=PAST_LEN_15K, q_len=q_len,
                      kv_cache_compression=CMPR, tile_q=q_len, partition_block_num=1)
    runner = PaSmallQOvExpRunner(
        case.num_heads, case.num_kv_heads, case.head_size, case.block_size,
        case.sub_block_size, case.kv_cache_compression,
        tile_q=case.tile_q, kv_partition_size=part)
    _run_perf_with_runner_exp(runner, case)            # warm up / force the build
    return case, runner


built = {}
for q_len, part in CONFIGS:
    for arm in ARMS:
        built[(q_len, part, arm)] = build(q_len, part, arm)

samples = {k: [] for k in built}
for r in range(ROUNDS):
    # Alternate the arms inside each round so both see the same thermal history: this rig
    # measured the same config at 0.52 ms cold and 1.59 ms hot, so A-then-B is meaningless.
    for q_len, part in CONFIGS:
        for arm in ARMS:
            key = (q_len, part, arm)
            case, runner = built[key]
            os.environ["OV_EXP_EXTRA_FLAGS"] = f"-DPIPELINE_MARSHAL={arm}"
            perf = _run_perf_with_runner_exp(runner, case)
            samples[key].append((perf["small_q_ms"], perf["small_q_reduce_ms"]))

print(f"\ncmpr={CMPR} block={BLOCK} past={PAST_LEN_15K} rounds={ROUNDS} (min over rounds)")
print(f"{'q':>3} {'part':>5} {'thr':>4} {'pipe0':>8} {'pipe1':>8} {'delta':>8} "
      f"{'noise':>7}   verdict")
for q_len, part in CONFIGS:
    row = {}
    for arm in ARMS:
        mains = [s[0] for s in samples[(q_len, part, arm)]]
        row[arm] = (min(mains), (max(mains) / min(mains) - 1.0) * 100.0)
    (m0, s0), (m1, s1) = row[0], row[1]
    delta = (m1 / m0 - 1.0) * 100.0
    noise = max(s0, s1)
    # A delta inside the spread is not "no change", it is unresolvable -- calling it the
    # former is how a real regression gets waved through.
    if abs(delta) <= noise:
        verdict = "INCONCLUSIVE (delta < spread)"
    elif delta < 0:
        verdict = "pipeline WINS"
    else:
        verdict = "pipeline loses"
    thr = built[(q_len, part, 0)][1].wg_threads
    print(f"{q_len:>3} {part:>5} {thr:>4} {m0:>8.3f} {m1:>8.3f} {delta:>+7.1f}% "
          f"{noise:>6.1f}%   {verdict}")
