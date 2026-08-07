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
from test_pa_small_q import PaSmallQRunner, SmallQCase, _build_inputs as _build_small_q_inputs
from test_pa_multiseq import PaMultiTokenRunner

cl.profiling(True)
torch.manual_seed(0)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ov_exp_force_q_head_chunk_size = 4
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
        self.workgroup_q_groups = 1
        if (
            self.ov_exp_use_wg_shared_kv
            and self.kv_cache_compression == 1
            and (self.q_head_chunk_size * self.tile_q) > 8
        ):
            self.workgroup_q_groups = (self.q_head_chunk_size * self.tile_q) // 8
        print(
            "PaSmallQOvExpRunner: "
            f"q_head_chunk_size={self.q_head_chunk_size}, "
            f"q_head_chunks_per_kv_head={self.q_head_chunks_per_kv_head}, "
            f"workgroup_q_groups={self.workgroup_q_groups}, "
            f"force_q_head_chunk_size={self.ov_exp_force_q_head_chunk_size}, "
            f"wg_shared_kv={int(self.ov_exp_use_wg_shared_kv)}"
            f"tile_q={self.tile_q}, kv_cache_compression={self.kv_cache_compression}"
        )

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _create_kernels_ov_exp(
        num_heads, num_kv_heads, head_size, kv_step, block_size, sub_block_size,
        kv_partition_size, reduce_split_step, clean_unused_kvcache,
        kv_cache_compression, xe_arch, q_head_chunks_per_kv_head,
        q_head_chunk_size, tile_q, scale_factor, source_stamp,
    ):
        src = '\n'.join([
            '#include "pa_small_q_ov_exp.cm"',
            '#include "pa_small_q_finalization.cm"',
        ])
        cwd = os.path.dirname(os.path.realpath(__file__))
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
                        -DOV_EXP_SOURCE_STAMP={source_stamp}
                        -DKERNEL_NAME=cm_pa_small_q''',
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
        )

    def dispatch_dims(self, tile_count: int, kv_partition_num: int):
        gws = [int(tile_count) * self.workgroup_q_groups, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [self.workgroup_q_groups, 1, 1]
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


def _run_ov_exp(case: SmallQCase, data: dict) -> torch.Tensor:
    runner = PaSmallQOvExpRunner(
        case.num_heads, case.num_kv_heads, case.head_size,
        case.block_size, case.sub_block_size, case.kv_cache_compression,
        tile_q=case.tile_q,
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
) -> tuple[bool, float]:
    case = SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=block_size,
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
        out_exp = _run_ov_exp(case, data)

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
        *[(16, past_len, 1, 16, 16) for past_len in _PAST_LENS_SWEEP],
        *[(16, past_len, 1, 8, 16) for past_len in _PAST_LENS_SWEEP],
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


if __name__ == "__main__":
    for q_len, tile_q in [(6, 6)]:
        for past_len in _PAST_LENS_SWEEP:
            passed, max_diff = _case_exp(
                q_len=q_len,
                past_len=past_len,
                cmpr=1,
                tile_q=tile_q,
                block_size=16,
            )
            print({
                "q_len": q_len,
                "past_len": past_len,
                "cmpr": 1,
                "tile_q": tile_q,
                "block_size": 16,
                "passed": passed,
                "max_diff": max_diff,
            })