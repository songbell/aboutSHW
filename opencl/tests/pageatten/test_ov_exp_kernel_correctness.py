#!/usr/bin/env python3
"""Correctness harness for the experimental pa_small_q_ov_exp.cm kernel."""

import functools
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'clops'))

from clops import cl
from test_ov_kernel_correctness import _run_baseline
from test_pa_small_q import PaSmallQRunner, SmallQCase, _build_inputs as _build_small_q_inputs

cl.profiling(True)
torch.manual_seed(0)


class PaSmallQOvExpRunner(PaSmallQRunner):
    """Runner for pa_small_q_ov_exp.cm.

    Reuses the OV-side q-chunk sizing override logic so the experimental kernel
    can be exercised under the same host-side policy as pa_small_q_ov.cm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        q_heads_per_kv_head = self.num_heads // self.num_kv_heads
        repeat_count_cap = max(1, max_repeat_count // self.tile_q)
        target_chunk = min(q_heads_per_kv_head, repeat_count_cap, max_q_by_matrix)
        q_head_chunk_size = max(1, target_chunk)
        while q_head_chunk_size > 1 and (q_heads_per_kv_head % q_head_chunk_size) != 0:
            q_head_chunk_size -= 1

        forced_q_head_chunk_size = int(os.environ.get("OV_FORCE_Q_HEAD_CHUNK_SIZE", "0"))
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
            os.environ.get("OV_EXP_USE_WG_SHARED_KV", "0") == "1"
            and self.kv_cache_compression == 1
            and (self.q_head_chunk_size * self.tile_q) > 8
        ):
            self.workgroup_q_groups = (self.q_head_chunk_size * self.tile_q) // 8

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _create_kernels_ov_exp(
        num_heads, num_kv_heads, head_size, kv_step, block_size, sub_block_size,
        kv_partition_size, reduce_split_step, clean_unused_kvcache,
        kv_cache_compression, xe_arch, q_head_chunks_per_kv_head,
        q_head_chunk_size, tile_q, scale_factor,
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
                        -DKERNEL_NAME=cm_pa_small_q''',
        )

    def _create_kernels(self):
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
        )

    def dispatch_dims(self, tile_count: int, kv_partition_num: int):
        gws = [int(tile_count) * self.workgroup_q_groups, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [self.workgroup_q_groups, 1, 1]
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
            [partition_token_rows, self.num_heads, padded_partition_num, self.head_size],
            np.dtype(np.float32),
        )
        t_lse = cl.tensor(
            [partition_token_rows, self.num_heads, padded_partition_num], np.dtype(np.float32)
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


def _case_exp(q_len: int, past_len: int, cmpr: int, tile_q: int) -> tuple[bool, float]:
    case = SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=past_len, q_len=q_len, kv_cache_compression=cmpr, tile_q=tile_q,
    )
    data = _build_small_q_inputs(case)
    out_base = _run_baseline(case, data)
    out_exp = _run_ov_exp(case, data)

    diff = (out_exp.float() - out_base.float()).abs()
    max_diff = diff.max().item()
    passed = max_diff < 0.05 and not torch.isnan(out_exp).any().item()
    return passed, max_diff


@pytest.mark.parametrize("cmpr", [0, 1])
@pytest.mark.parametrize("tile_q", [1, 2])
def test_ov_exp_smoke(cmpr: int, tile_q: int):
    passed, max_diff = _case_exp(q_len=16, past_len=15360, cmpr=cmpr, tile_q=tile_q)
    assert passed, f"experimental kernel mismatch: max_diff={max_diff:.4e}"


if __name__ == "__main__":
    for cmpr in [0, 1]:
        for tile_q in [1, 2, 4]:
            if tile_q > 16:
                continue
            passed, max_diff = _case_exp(q_len=16, past_len=15360, cmpr=cmpr, tile_q=tile_q)
            print({"cmpr": cmpr, "tile_q": tile_q, "passed": passed, "max_diff": max_diff})