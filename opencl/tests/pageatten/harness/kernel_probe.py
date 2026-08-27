"""Measure what the qq_bias speculative tree mask costs in the *plugin's* pa_small_q.cm.

The sandbox kernel (pa_small_q_ov_exp.cm) has no tree-mask code at all, so the "main < 0.8 ms"
number measured there cannot be compared directly against an in-model measurement of the
plugin, which runs with HAS_QQ_BIAS=1 for the main model. This harness compiles the *plugin*
kernel and A/Bs HAS_QQ_BIAS 0 vs 1 on the same shape, which is the missing number.

qq_bias is filled with 1 (nothing masked), so both builds must produce bit-identical output.
That makes this a pure timing comparison and gives a correctness check for free: if the two
outputs differ, the harness is wrong, not the kernel.
"""
import os
import sys

sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl")
sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl/tests/pageatten")
os.environ.setdefault("RUN_PA_PERF", "1")

import numpy as np

from clops import cl
from test_15k_perf_comparison import PAST_LEN_15K, _build_small_q_inputs, _ceil_div
from test_pa_small_q import SmallQCase

PROBE_DIR = "/ceciliapeng/qq_probe"
Q_LEN = int(os.environ.get("QQ_QLEN", "16"))
PART = int(os.environ.get("QQ_PART", "640"))
CMPR = int(os.environ.get("QQ_CMPR", "2"))
BLOCK = int(os.environ.get("QQ_BLOCK", "256"))
CHUNK = int(os.environ.get("QQ_CHUNK", "4"))
# TILE_Q is compiled in and is >= q_len: OV pads a short spec window up to the nearest
# compiled rung (q_len 6 -> TILE_Q 8), so the padded rows cost a whole extra thread.
TILE_Q = int(os.environ.get("QQ_TILEQ", str(Q_LEN)))
REG_FILE = os.environ.get("QQ_REG_FILE", "192")
PAST = int(os.environ.get("QQ_PAST", str(PAST_LEN_15K)))
# With the runtime-partition kernel the jit constant is only an upper bound and the live
# partition arrives as a scalar, so the two are separate knobs.
PART_MAX = int(os.environ.get("QQ_PART_MAX", str(PART)))
RTPART = bool(os.environ.get("QQ_RTPART"))
LOOPS, WARMUP = 60, 8

HEADS, KV_HEADS, HEAD_SIZE = 32, 8, 128
case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE, block_size=BLOCK,
                  past_len=PAST, q_len=Q_LEN, kv_cache_compression=CMPR,
                  tile_q=Q_LEN, partition_block_num=1)
data = _build_small_q_inputs(case)

kv_step = 16
q_rows = CHUNK * TILE_Q
# RUNTIME_TILE_Q pins ROWS_PER_THREAD at 8, so the thread count is a ceil, not a divide --
# the last thread may own dummy rows.
rows_per_thread = 8
if os.environ.get("QQ_RTQ"):
    wg_threads = -(-q_rows // rows_per_thread)
else:
    wg_threads = q_rows // rows_per_thread if q_rows > 8 else 1
chunks_per_kv = (HEADS // KV_HEADS) // CHUNK
scale = 1.0 / (HEAD_SIZE ** 0.5)

past_lens = data["past_lens"]
max_ctx = int(past_lens.max().item()) + Q_LEN
nparts = _ceil_div(max_ctx, PART)
# One tile per subsequence: TILE_Q == q_len here.
mapping = np.array([0, 0, Q_LEN], dtype=np.int32)
tile_count = 1
rows = tile_count * TILE_Q

# qq_bias: u8 row-major [spec, spec], 1 = allowed. The kernel recovers spec_num as
# isqrt(qq_bias_begins[i+1] - qq_bias_begins[i]).
spec = Q_LEN
qq_bias_np = np.ones(spec * spec, dtype=np.uint8)
qq_begins_np = np.array([0, spec * spec], dtype=np.int32)


def build(has_qq):
    src = f'#include "{PROBE_DIR}/{os.environ.get("QQ_KERNEL","pa_small_q.cm")}"'
    opts = f'''-cmc -Qxcm_jit_option="" -Qxcm_register_file_size={REG_FILE} -I{PROBE_DIR}
        -DHEADS_NUM={HEADS} -DKV_HEADS_NUM={KV_HEADS} -DHEAD_SIZE={HEAD_SIZE}
        -DKV_STEP={kv_step} -DKV_BLOCK_SIZE={BLOCK} -DKV_PARTITION_SIZE={PART_MAX}
        -DKV_CACHE_COMPRESSION={CMPR} -DSUB_BLOCK_SIZE=16 -DXE_ARCH=2
        -DQ_head_chunks_per_kv_head={chunks_per_kv} -DQ_head_chunk_size={CHUNK}
        -DTILE_Q={TILE_Q} -DSCALE_FACTOR={scale} -DHAS_QQ_BIAS={has_qq}
        -DKERNEL_NAME=cm_pa_small_q {os.environ.get("QQ_EXTRA_FLAGS", "")}'''
    return cl.kernels(src, opts)


def run(has_qq, collect_output=False):
    kernels = build(has_qq)
    gws = [tile_count * wg_threads, KV_HEADS * chunks_per_kv, nparts]
    lws = [wg_threads, 1, 1]

    layers, mem = [], 0
    while len(layers) < LOOPS and mem < 8e9:
        layers.append((cl.tensor(data["query"].detach().numpy()),
                       cl.tensor(data["key_cache"].contiguous().detach().numpy()),
                       cl.tensor(data["value_cache"].contiguous().detach().numpy()),
                       cl.tensor(np.zeros([rows, HEADS, nparts, HEAD_SIZE], np.float32))))
        mem += sum(t.numel() * t.element_size()
                   for t in (data["query"], data["key_cache"], data["value_cache"]))

    t_past = cl.tensor(past_lens.detach().numpy())
    t_bi = cl.tensor(data["block_indices"].detach().numpy())
    t_bib = cl.tensor(data["block_indices_begins"].detach().numpy())
    t_sb = cl.tensor(data["subsequence_begins"].detach().numpy())
    t_map = cl.tensor(mapping)
    t_lse = cl.tensor(np.full([rows, HEADS, nparts], -3e38, np.float32))
    t_qq = cl.tensor(qq_bias_np)
    t_qqb = cl.tensor(qq_begins_np)
    cl.finish()

    for i in range(LOOPS):
        q, k, v, po = layers[i % len(layers)]
        args = [q, k, v, t_past, t_bi, t_bib, t_sb]
        if has_qq:
            args += [t_qq, t_qqb]
        args += [t_map, po, t_lse, (PART if RTPART else TILE_Q), tile_count]
        kernels.enqueue("cm_pa_small_q", gws, lws, *args)

    lat = cl.finish()
    tot, n = 0.0, 0
    for i, ns in enumerate(lat[:LOOPS]):
        if i >= WARMUP and float(ns) > 0:
            tot += float(ns); n += 1
    out = layers[(LOOPS - 1) % len(layers)][3].numpy() if collect_output else None
    return tot * 1e-6 / n, out


print(f"plugin pa_small_q.cm  q_len={Q_LEN} part={PART} cmpr={CMPR} block={BLOCK} "
      f"TILE_Q={TILE_Q} chunk={CHUNK} wg_threads={wg_threads} nparts={nparts} reg={REG_FILE}")
res = {}
for has_qq in (0, 1):
    times = [run(has_qq)[0] for _ in range(3)]
    res[has_qq] = min(times)
    print(f"  HAS_QQ_BIAS={has_qq}  main = {min(times):.3f} ms   (runs {['%.3f' % t for t in times]})")

d = res[1] - res[0]
print(f"\nqq_bias tree mask costs {d:.3f} ms = {100.0 * d / res[0]:+.1f}% of the kernel")

# All-ones qq_bias masks nothing, so the two builds must agree exactly.
o0 = run(0, True)[1]
o1 = run(1, True)[1]
same = np.array_equal(o0, o1)
print(f"outputs bit-identical (all-ones qq_bias): {same}"
      f"{'' if same else '   <-- harness bug, timing above is not comparable'}")
