#!/usr/bin/env python3
"""Correctness harness for the experimental pa_small_q_ov_exp.cm kernel."""

import functools
import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'clops'))

from clops import cl
from kv_cache_quant_utils import DEFAULT_SUB_BLOCK_SIZE
from test_pa_small_q import PaSmallQRunner, SmallQCase, _build_inputs as _build_small_q_inputs
from test_pa_multiseq import PaMultiTokenRunner

cl.profiling(True)
torch.manual_seed(0)

# -Qxcm_register_file_size. Valid values are 128 / 160 / 192 / 256; threads resident per
# XVE is 1280 / this, so 192 gives 6 and 160 gives 8.
#
# 192 is not about avoiding spill -- the kernel fits in 150 registers. It is about giving
# IGC enough slack to *batch* the consume phase's SLM reads. At 160 it emits a strict
# LDLDLD... chain, one 512 B SLM read immediately followed by its dependent DPAS, 16 times
# per online tile, so the SLM latency is fully exposed and the workgroup's threads all stall
# on it together behind the barriers. At 192 it schedules LLLLDDDD -- 4-way memory-level
# parallelism. Paired A/B on the main kernel, two batches: -8.4 % and -11.3 %. 256 also
# batches but costs another thread per XVE and measures +2.3 % / -1.0 %.
#
# This flipped sign in round 16: before the K/V prefetch landed, 192 measured -0.4 %.
_DEFAULT_REG_FILE = "192"

MAX_DIFF_TOL = 0.05


@contextmanager
def _temporary_env(set_values: dict[str, str] | None = None, unset_keys: list[str] | None = None):
    """Temporarily set/unset environment variables, then restore."""
    set_values = set_values or {}
    unset_keys = unset_keys or []
    touched_keys = set(set_values.keys()) | set(unset_keys)
    old_values = {k: os.environ.get(k) for k in touched_keys}
    try:
        for key in unset_keys:
            os.environ.pop(key, None)
        for key, value in set_values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class PaSmallQOvExpRunner(PaSmallQRunner):
    """Runner for pa_small_q_ov_exp.cm.

    Reuses the OV-side q-chunk sizing override logic so the experimental kernel
    can be exercised under the same host-side policy as pa_small_q_ov.cm.
    """

    def __init__(self, *args, kv_partition_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # KV_PARTITION_SIZE, decoupled from KV_BLOCK_SIZE. The base runner only knows the
        # ratio form (block_size * k_partition_block_num), which OV does not use: it picks
        # both from the same has_xattention flag, so block 16 pairs with partition 128 and
        # block 256 with partition 256 (see _ov_partition_size). Leaving the ratio in place
        # would silently build a 2048-token partition for block 256.
        if kv_partition_size is not None:
            if int(kv_partition_size) % self.kv_step != 0:
                raise ValueError(
                    f"kv_partition_size ({kv_partition_size}) must be a multiple of "
                    f"kv_step ({self.kv_step})"
                )
            self.kv_partition_size = int(kv_partition_size)
            print(
                f"PaSmallQOvExpRunner: kv_partition_size overridden to "
                f"{self.kv_partition_size} (block_size={self.block_size})"
            )
        # OpenVINO derives Q_head_chunk_size from the model's GQA ratio
        # (get_single_token_q_chunking), so it can be 1, 2, 4 or 8. It feeds
        # Q_ROWS = Q_head_chunk_size * TILE_Q and hence WG_THREADS and the (t, qi)
        # decomposition of a per-thread row index, so it needs coverage -- this was a
        # hardcoded 4 and the env var below was read nowhere, which meant every sweep in
        # this file tested exactly one value of it.
        self.ov_exp_force_q_head_chunk_size = int(
            os.environ.get("OV_FORCE_Q_HEAD_CHUNK_SIZE", "4"))
        self.ov_exp_use_wg_shared_kv = True

        online_tile_steps = 8
        online_tile_size = online_tile_steps * self.kv_step
        max_repeat_count = 8
        reg_file_size = 256
        grf_bytes = 32 if self.xe_arch == 1 else 64
        budget_bytes = reg_file_size * grf_bytes - 1

        bytes_per_q_row = (
            4 * online_tile_size
            + 2 * online_tile_size
            + 4 * self.head_size
            + 2 * self.head_size
        )
        max_q_by_matrix = max(1, budget_bytes // (bytes_per_q_row * self.tile_q))
        self.reduce_split_step = 64
        q_heads_per_kv_head = self.num_heads // self.num_kv_heads
        repeat_count_cap = max(1, max_repeat_count // self.tile_q)
        target_chunk = min(q_heads_per_kv_head, repeat_count_cap, max_q_by_matrix)
        q_head_chunk_size = max(1, target_chunk)
        while q_head_chunk_size > 1 and (q_heads_per_kv_head % q_head_chunk_size) != 0:
            q_head_chunk_size -= 1

        forced_q_head_chunk_size = self.ov_exp_force_q_head_chunk_size
        if forced_q_head_chunk_size > 0:
            if (q_heads_per_kv_head % forced_q_head_chunk_size) != 0:
                raise ValueError(
                    f"OV_FORCE_Q_HEAD_CHUNK_SIZE={forced_q_head_chunk_size} must divide q_heads_per_kv_head={q_heads_per_kv_head}"
                )
            q_head_chunk_size = forced_q_head_chunk_size

        self.q_head_chunk_size = q_head_chunk_size
        self.q_head_chunks_per_kv_head = q_heads_per_kv_head // q_head_chunk_size
        q_rows = self.q_head_chunk_size * self.tile_q
        # Must match ROWS_PER_THREAD / WG_THREADS in pa_small_q_ov_exp.cm. The kernel
        # derives them from the same TILE_Q and Q_head_chunk_size, so nothing is threaded
        # through the build options -- but a mismatch here would be silent, so keep the
        # two definitions side by side.
        self.rows_per_thread = 8 if q_rows > 8 else q_rows

        self.wg_threads = 1
        if (
            self.ov_exp_use_wg_shared_kv
            # compr=1 (K per token) and compr=2 (K per channel) both take the
            # workgroup-shared path; only the K dequantize differs inside the kernel.
            and self.kv_cache_compression in (1, 2)
            and q_rows > 8
        ):
            self.wg_threads = q_rows // self.rows_per_thread
        print(
            "PaSmallQOvExpRunner: "
            f"q_head_chunk_size={self.q_head_chunk_size}, "
            f"q_head_chunks_per_kv_head={self.q_head_chunks_per_kv_head}, "
            f"rows_per_thread={self.rows_per_thread}, "
            f"wg_threads={self.wg_threads}, "
            f"force_q_head_chunk_size={self.ov_exp_force_q_head_chunk_size}, "
            f"wg_shared_kv={int(self.ov_exp_use_wg_shared_kv)}"
            f"tile_q={self.tile_q}, kv_cache_compression={self.kv_cache_compression}"
        )

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _create_kernels_ov_exp(
        num_heads, num_kv_heads, head_size, kv_step, block_size, sub_block_size,
        kv_partition_size, reduce_split_step, clean_unused_kvcache,
        kv_cache_compression, xe_arch, q_head_chunks_per_kv_head,
        q_head_chunk_size, tile_q, scale_factor, source_stamp,
        reg_file_size, extra_flags,
    ):
        # reg_file_size and extra_flags are parameters, not os.environ reads, because the
        # lru_cache key is the argument tuple: anything read from the environment inside this
        # body is invisible to the cache, so two builds differing only by an env var silently
        # return the same kernel. That made an in-process A/B of a -D flag compare a kernel
        # against itself and report a 0.0% delta for a change known to cost 45%.
        src = '\n'.join([
            '#include "pa_small_q_ov_exp.cm"',
            '#include "pa_small_q_finalization.cm"',
        ])
        cwd = os.path.dirname(os.path.realpath(__file__))
        return cl.kernels(
            src,
            f'''-cmc -Qxcm_jit_option=""
                        -mCM_printregusage
                        -Qxcm_register_file_size={reg_file_size} -I{cwd}
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
                        -DOV_EXP_SOURCE_STAMP={source_stamp}
                        -DKERNEL_NAME=cm_pa_small_q {extra_flags}''',
        )

    def _create_kernels(self):
        source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pa_small_q_ov_exp.cm")
        source_stamp = int(os.stat(source_path).st_mtime_ns)
        return self._create_kernels_ov_exp(
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
            source_stamp,
            os.environ.get("OV_EXP_REG_FILE_SIZE", _DEFAULT_REG_FILE),
            os.environ.get("OV_EXP_EXTRA_FLAGS", ""),
        )

    def dispatch_dims(self, tile_count: int, kv_partition_num: int):
        gws = [int(tile_count) * self.wg_threads, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [self.wg_threads, 1, 1]
        #print(f"PaSmallQOvExpRunner.dispatch_dims: gws={gws}, lws={lws}, tile_count={tile_count}, kv_partition_num={kv_partition_num}")
        partition_token_rows = int(tile_count) * self.tile_q
        gws_2 = [partition_token_rows, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]
        return gws, lws, gws_2, lws_2, partition_token_rows, kv_partition_num

    def _enqueue_once(
        self,
        kernels,
        q_tokens: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        past_lens: torch.Tensor,
        block_indices: torch.Tensor,
        block_indices_begins: torch.Tensor,
        subsequence_begins: torch.Tensor,
        mapping: torch.Tensor,
        tile_count: int,
        kv_partition_num: int,
        out: torch.Tensor,
    ):
        q_len = int(q_tokens.shape[0])
        gws, lws, gws_2, lws_2, partition_token_rows, padded_partition_num = self.dispatch_dims(tile_count, kv_partition_num)

        t_q = cl.tensor(q_tokens.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_past_lens = cl.tensor(past_lens.detach().numpy())
        t_block_indices = cl.tensor(block_indices.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
        t_mapping = cl.tensor(mapping.detach().numpy())

        t_partition_out = cl.tensor(
            np.full(
                [partition_token_rows, self.num_heads, padded_partition_num, self.head_size],
                np.nan,
                dtype=np.float32,
            )
        )
        t_lse = cl.tensor(
            np.full(
                [partition_token_rows, self.num_heads, padded_partition_num],
                np.nan,
                dtype=np.float32,
            )
        )
        t_out_final = cl.tensor(out.contiguous().detach().numpy())

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
            t_partition_out,
            t_lse,
            int(q_len),
            int(tile_count),
        )
        kernels.enqueue(
            "cm_pa_small_q_reduce",
            gws_2,
            lws_2,
            t_partition_out,
            t_out_final,
            t_lse,
            t_subsequence_begins,
            t_mapping,
            partition_token_rows,
            padded_partition_num,
        )
        cl.finish()
        out.copy_(torch.from_numpy(t_out_final.numpy()))
        return out, t_partition_out, t_lse


def _ov_partition_size(block_size: int) -> int:
    """KV_PARTITION_SIZE that OV pairs with a given KV_BLOCK_SIZE.

    paged_attention_gen.{hpp,cpp} tie both to has_xattention: KV_BLOCK_SIZE is
    PA_KV_CACHE_BLOCK_SIZE_LEGACY (16) or PA_KV_CACHE_BLOCK_SIZE_XATTN (256), and
    get_partition_size returns 128 for the legacy layout and PA_KV_CACHE_BLOCK_SIZE_XATTN
    (256) for xattention. Those two pairs are the only ones the plugin ever builds.
    """
    return {16: 128, 256: 256}.get(block_size, block_size)


def _run_ov_exp(case: SmallQCase, data: dict, kv_partition_size: int | None = None) -> torch.Tensor:
    runner = PaSmallQOvExpRunner(
        case.num_heads, case.num_kv_heads, case.head_size,
        case.block_size, case.sub_block_size, case.kv_cache_compression,
        tile_q=case.tile_q,
        kv_partition_size=kv_partition_size or _ov_partition_size(case.block_size),
    )
    out = torch.zeros([case.q_len, case.num_heads, case.head_size], dtype=torch.float16)
    runner(
        data["query"], data["key_cache"], data["value_cache"],
        data["past_lens"], data["block_indices"],
        data["block_indices_begins"], data["subsequence_begins"],
        out, n_repeats=1,
    )
    return out


def _run_multi_token_reference(case: SmallQCase, data: dict) -> torch.Tensor:
    """Generate reference output via pa_multi_token kernel on the same inputs."""
    runner = PaMultiTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.kv_cache_compression,
        is_causal=True,
        sub_block_size=case.sub_block_size,
        sparse_block_size=1,
        enable_hybrid_dispatch=False,
    )
    kern_attn_inputs = {
        "query": data["query"],
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "past_lens": data["past_lens"],
        "block_indices": data["block_indices"],
        "block_indices_begins": data["block_indices_begins"],
        "subsequence_begins": data["subsequence_begins"],
    }
    out = torch.zeros([case.q_len, case.num_heads * case.head_size], dtype=torch.float16)
    runner(kern_attn_inputs, out, prefill_seq_indices=[0], n_repeats=1)
    return out.reshape(case.q_len, case.num_heads, case.head_size).contiguous()


def _case_exp(
    q_len: int,
    past_len: int,
    cmpr: int,
    tile_q: int,
    block_size: int = 256,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    kv_partition_size: int | None = None,
) -> tuple[bool, float]:
    case = SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=block_size,
        sub_block_size=sub_block_size,
        past_len=past_len, q_len=q_len, kv_cache_compression=cmpr, tile_q=tile_q,
    )
    data = _build_small_q_inputs(case)

    env_keys = ["OV_EXP_USE_WG_SHARED_KV", "OV_FORCE_Q_HEAD_CHUNK_SIZE"]
    with _temporary_env(unset_keys=env_keys):
        out_base = _run_multi_token_reference(case, data)

    with _temporary_env(
        set_values={
            "OV_EXP_USE_WG_SHARED_KV": "1",
            "OV_FORCE_Q_HEAD_CHUNK_SIZE": "4",
        }
    ):
        out_exp = _run_ov_exp(case, data, kv_partition_size=kv_partition_size)

    diff = (out_exp.float() - out_base.float()).abs()
    max_diff = diff.max().item()
    print(f"  Experimental kernel max_diff={max_diff:.4e}")
    passed = max_diff < MAX_DIFF_TOL and not torch.isnan(out_exp).any().item()
    return passed, max_diff


_PAST_LENS_SWEEP = (64, 128, 144, 192, 1024, 4096, 15360)


@pytest.mark.parametrize(
    "q_len,past_len,cmpr,tile_q,block_size",
    [
        *[(6, past_len, 1, 6, 16) for past_len in _PAST_LENS_SWEEP],
        *[(6, past_len, 2, 6, 16) for past_len in _PAST_LENS_SWEEP],
    ],
)
def test_ov_exp_correctness_vs_multi_token_matrix(
    q_len: int,
    past_len: int,
    cmpr: int,
    tile_q: int,
    block_size: int,
):
    passed, max_diff = _case_exp(
        q_len=q_len,
        past_len=past_len,
        cmpr=cmpr,
        tile_q=tile_q,
        block_size=block_size,
    )
    assert passed, (
        "ov_exp vs pa_multi_token mismatch: "
        f"q_len={q_len} past_len={past_len} cmpr={cmpr} tile_q={tile_q} "
        f"block_size={block_size} max_diff={max_diff:.4e}"
    )


@pytest.mark.parametrize(
    "q_len,past_len,cmpr,tile_q,block_size",
    [
        # The tuned operating point, plus tail tiles. q_len < tile_q leaves some threads
        # owning rows past valid_count, which is the case the causal/partition mask
        # fast-path has to opt out of, so it needs explicit coverage.
        *[(16, past_len, 1, 16, 16) for past_len in _PAST_LENS_SWEEP],
        *[(q_len, past_len, 1, 16, 16)
          for q_len in (13, 10, 5) for past_len in (192, 1024, 15360)],
        # cmpr=2: K quantised per channel, V still per token. Same data layout and same
        # workgroup-shared path; only the K dequantize differs. Partial blocks matter most
        # here -- cmpr=1 neutralises a partial block's tail by zeroing its per-token
        # scale/zp, which a per-channel scale cannot do, so cmpr=2 relies on the
        # causal/partition mask alone. past_lens that are not multiples of the block or
        # partition size cover that.
        *[(16, past_len, 2, 16, 16) for past_len in _PAST_LENS_SWEEP],
        *[(q_len, past_len, 2, 16, 16)
          for q_len in (13, 5) for past_len in (192, 1024, 15360)],
    ],
)
def test_ov_exp_correctness_q16_and_tail_tiles(
    q_len: int,
    past_len: int,
    cmpr: int,
    tile_q: int,
    block_size: int,
):
    """q_len=16 / tile_q=16 is the tuned shape; q_len < tile_q exercises the tail-tile
    path where a workgroup contains threads whose q-rows are all invalid."""
    case = SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=block_size,
        past_len=past_len, q_len=q_len, kv_cache_compression=cmpr, tile_q=tile_q,
    )
    data = _build_small_q_inputs(case)

    env_keys = ["OV_EXP_USE_WG_SHARED_KV", "OV_FORCE_Q_HEAD_CHUNK_SIZE"]
    with _temporary_env(unset_keys=env_keys):
        out_base = _run_multi_token_reference(case, data)
    with _temporary_env(
        set_values={"OV_EXP_USE_WG_SHARED_KV": "1", "OV_FORCE_Q_HEAD_CHUNK_SIZE": "4"}
    ):
        out_exp = _run_ov_exp(case, data)

    max_diff = (out_exp.float() - out_base.float()).abs().max().item()
    print(f"  q_len={q_len} tile_q={tile_q} max_diff={max_diff:.4e}")
    assert max_diff < MAX_DIFF_TOL and not torch.isnan(out_exp).any().item(), (
        "ov_exp mismatch: "
        f"q_len={q_len} past_len={past_len} tile_q={tile_q} max_diff={max_diff:.4e}"
    )


# ---------------------------------------------------------------------------------------
# KV_BLOCK_SIZE = 256, the xattention cache layout.
#
# Everything above runs at block 16, which is degenerate: a cache block is then exactly one
# KV_STEP online tile, so a tile ordinal and a block ordinal are the same number and the
# kernel cannot tell them apart. At block 256 one block spans STEPS_PER_BLOCK = 16 tiles,
# which separates the two: block_indices is indexed by pos / KV_BLOCK_SIZE, and each tile
# additionally carries a within-block token offset (pos % KV_BLOCK_SIZE) that drives the 2D
# descriptors' set_block_y and the address of its slice of the per-token / per-sub-block
# scale and zero-point arrays.
# ---------------------------------------------------------------------------------------

_PAST_LENS_SWEEP_BLK256 = (
    64,      # one partial block, and a partial *first* tile
    200,     # partial tile mid-block (tok_in_blk 208) -- unreachable at block 16, where a
             # partial tile is always the last tile of its block
    250,     # block 0's last tile partial, 10 keys spilling into block 1
    256,     # exact block boundary: block 1 exists but contributes nothing
    272,     # block 1 holds exactly one full tile
    1024,    # 4 whole blocks
    4096,
    15360,
)


@pytest.mark.parametrize(
    "q_len,past_len,cmpr,tile_q",
    [
        *[(6, past_len, cmpr, 6)
          for cmpr in (1, 2) for past_len in _PAST_LENS_SWEEP_BLK256],
        *[(16, past_len, cmpr, 16)
          for cmpr in (1, 2) for past_len in _PAST_LENS_SWEEP_BLK256],
        # q_len < tile_q: threads owning rows past valid_count, on top of the block-256
        # addressing.
        *[(q_len, past_len, cmpr, 16)
          for cmpr in (1, 2) for q_len in (13, 5) for past_len in (200, 250, 1024, 15360)],
    ],
)
def test_ov_exp_correctness_block_size_256(
    q_len: int,
    past_len: int,
    cmpr: int,
    tile_q: int,
):
    """Block 256 / partition 256 -- the pair OV builds when xattention is on."""
    passed, max_diff = _case_exp(
        q_len=q_len,
        past_len=past_len,
        cmpr=cmpr,
        tile_q=tile_q,
        block_size=256,
    )
    assert passed, (
        "ov_exp vs pa_multi_token mismatch at block_size=256: "
        f"q_len={q_len} past_len={past_len} cmpr={cmpr} tile_q={tile_q} "
        f"max_diff={max_diff:.4e}"
    )


@pytest.mark.parametrize(
    "kv_partition_size",
    [
        128,   # partition *smaller* than a block: two partitions share one block, and a
               # partition's first tile starts mid-block
        512,   # partition spanning two blocks: block_indices advances inside one partition
    ],
)
@pytest.mark.parametrize("past_len", [250, 1024, 4096])
@pytest.mark.parametrize("cmpr", [1, 2])
def test_ov_exp_correctness_block_256_partition_not_block(
    cmpr: int,
    past_len: int,
    kv_partition_size: int,
):
    """KV_PARTITION_SIZE != KV_BLOCK_SIZE at block 256.

    OV builds neither pair, but both are what the block ordinal being derived from the key
    position (pos / KV_BLOCK_SIZE) rather than from a blocks-per-partition ratio buys. A
    regression to the ratio form passes the partition == block sweep above and fails here,
    so this is the cheap guard on that.
    """
    passed, max_diff = _case_exp(
        q_len=16,
        past_len=past_len,
        cmpr=cmpr,
        tile_q=16,
        block_size=256,
        kv_partition_size=kv_partition_size,
    )
    assert passed, (
        "ov_exp vs pa_multi_token mismatch at block_size=256: "
        f"past_len={past_len} cmpr={cmpr} kv_partition_size={kv_partition_size} "
        f"max_diff={max_diff:.4e}"
    )


@pytest.mark.parametrize(
    "sub_block_size",
    [
        16,    # GROUPS_PER_BLOCK 16 -- a tile's group index is its position in the block
        64,    # GROUPS_PER_BLOCK 4 -- four tiles per group
        256,   # GROUPS_PER_BLOCK 1 -- one scale/zp row for the whole block
    ],
)
@pytest.mark.parametrize("past_len", [200, 250, 1024, 15360])
def test_ov_exp_correctness_block_256_per_channel_sub_blocks(
    past_len: int,
    sub_block_size: int,
):
    """cmpr=2 (K per channel) at block 256, over the sub-block granularities.

    SUB_BLOCK_SIZE decides how many scale/zp groups a block carries, and at block 256 a
    tile sits at group tok_in_blk / SUB_BLOCK_SIZE rather than always at group 0. Reading
    the wrong group dequantises K with another sub-block's scale, which the per-token
    cmpr=1 path cannot expose.
    """
    passed, max_diff = _case_exp(
        q_len=16,
        past_len=past_len,
        cmpr=2,
        tile_q=16,
        block_size=256,
        sub_block_size=sub_block_size,
    )
    assert passed, (
        "ov_exp vs pa_multi_token mismatch at block_size=256 cmpr=2: "
        f"past_len={past_len} sub_block_size={sub_block_size} max_diff={max_diff:.4e}"
    )
