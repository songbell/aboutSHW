"""Equivalence check for the runtime-KV_PARTITION_SIZE kernel.

Compares the stock kernel, compiled with KV_PARTITION_SIZE = P, against the runtime-partition
kernel compiled with an upper bound of MAX and handed P as a scalar. Both must produce
bit-identical partial output and lse for the same inputs.

Sweeps partitions that divide the context evenly and ones that leave a ragged tail, since the
leftover/causal handling is where a runtime partition would diverge first.
"""
import os
import sys

sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl")
sys.path.insert(0, "/ceciliapeng/bell/aboutSHW/opencl/tests/pageatten")
os.environ.setdefault("RUN_PA_PERF", "1")

import numpy as np

from clops import cl
from test_15k_perf_comparison import _build_small_q_inputs, _ceil_div
from test_pa_small_q import SmallQCase

PROBE = "/ceciliapeng/qq_probe"
HEADS, KV_HEADS, HEAD_SIZE = 32, 8, 128
PART_MAX = 640
_DATA = {}


def get_data(q_len, cmpr, block, past):
    key = (q_len, cmpr, block, past)
    if key not in _DATA:
        case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                          block_size=block, past_len=past, q_len=q_len,
                          kv_cache_compression=cmpr, tile_q=q_len, partition_block_num=1)
        _DATA[key] = _build_small_q_inputs(case)
    return _DATA[key]


def run(kernel, q_len, chunk, cmpr, block, past, part, jit_part, rt):
    data = get_data(q_len, cmpr, block, past)
    q_rows = chunk * q_len
    rpt = 8 if q_rows > 8 else q_rows
    wg = q_rows // rpt if q_rows > 8 else 1
    cpk = (HEADS // KV_HEADS) // chunk
    nparts = _ceil_div(past + q_len, part)
    spec = q_len
    kern = cl.kernels(f'#include "{PROBE}/{kernel}"', f'''-cmc -Qxcm_jit_option=""
        -Qxcm_register_file_size=192 -I{PROBE}
        -DHEADS_NUM={HEADS} -DKV_HEADS_NUM={KV_HEADS} -DHEAD_SIZE={HEAD_SIZE}
        -DKV_STEP=16 -DKV_BLOCK_SIZE={block} -DKV_PARTITION_SIZE={jit_part}
        -DKV_CACHE_COMPRESSION={cmpr} -DSUB_BLOCK_SIZE=16 -DXE_ARCH=2
        -DQ_head_chunks_per_kv_head={cpk} -DQ_head_chunk_size={chunk}
        -DTILE_Q={q_len} -DSCALE_FACTOR={1.0 / (HEAD_SIZE ** 0.5)} -DHAS_QQ_BIAS=1
        -DKERNEL_NAME=cm_pa_small_q''')
    po = cl.tensor(np.zeros([q_len, HEADS, nparts, HEAD_SIZE], np.float32))
    lse = cl.tensor(np.full([q_len, HEADS, nparts], -3e38, np.float32))
    kern.enqueue("cm_pa_small_q", [wg, KV_HEADS * cpk, nparts], [wg, 1, 1],
                 cl.tensor(data["query"].detach().numpy()),
                 cl.tensor(data["key_cache"].contiguous().detach().numpy()),
                 cl.tensor(data["value_cache"].contiguous().detach().numpy()),
                 cl.tensor(data["past_lens"].detach().numpy()),
                 cl.tensor(data["block_indices"].detach().numpy()),
                 cl.tensor(data["block_indices_begins"].detach().numpy()),
                 cl.tensor(data["subsequence_begins"].detach().numpy()),
                 cl.tensor(np.ones(spec * spec, np.uint8)),
                 cl.tensor(np.array([0, spec * spec], dtype=np.int32)),
                 cl.tensor(np.array([0, 0, q_len], dtype=np.int32)),
                 po, lse, (part if rt else q_len), 1)
    cl.finish()
    return po.numpy(), lse.numpy()


fails = n = 0
# Ragged tails on purpose: contexts that are not multiples of the partition, and partitions
# that do not divide the cache block size.
for past in (500, 512, 1000, 2048, 4097, 15360):
    for part in (128, 256, 384, 640):
        for q_len, chunk in ((6, 4), (16, 4)):
            for cmpr in (1, 2):
                for block in (16, 256):
                    a = run("pa_small_q.base.cm", q_len, chunk, cmpr, block, past, part, part, False)
                    b = run("pa_small_q.rtpart.cm", q_len, chunk, cmpr, block, past, part, PART_MAX, True)
                    n += 1
                    if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
                        fails += 1
                        print(f"  MISMATCH past={past} part={part} q={q_len} cmpr={cmpr} "
                              f"block={block} maxdiff={np.abs(a[0] - b[0]).max():.3e}")

print(f"\n===== {n - fails}/{n} configurations bit-identical =====")
# Guard against a vacuous pass: different partitions must give different partial buffers.
p1 = run("pa_small_q.rtpart.cm", 6, 4, 2, 256, 2048, 128, PART_MAX, True)
p2 = run("pa_small_q.rtpart.cm", 6, 4, 2, 256, 2048, 640, PART_MAX, True)
print(f"runtime partition actually takes effect (shapes differ): {p1[0].shape != p2[0].shape}")
sys.exit(1 if fails else 0)
