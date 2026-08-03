"""
test_pa_small_q.py

Functional + perf harness for the small-q decode kernel (pa_small_q.cm /
pa_small_q_finalization.cm). The kernel handles q_len > 1 inside one
subsequence — the spec-decoding / draft-validation path where one sequence
carries `spec_num` query tokens (e.g. q=2..16) against a long past KV.
multi_token's WG utilisation collapses to 1/q_len in this shape.

The TILE_Q optimisation (see kernel preamble) packs TILE_Q q-tokens per SG so
DPAS RepeatCount = q_head_chunk_size * TILE_Q stays saturated and K/V loads
amortise across the spec window. The host-packed mapping is i32 triples
(orig_seq_idx, q_start_in_subseq, valid_count) of length 3 * tile_count.

Layout invariants this test asserts:
  output  : f32 [tile_count * TILE_Q, head, partition, head_size]
  lse     : f32 [tile_count * TILE_Q, head, partition]
  Causal  : row t in a tile sees past + q_start + t + 1 keys.

Usage:
  python -m py_compile test_pa_small_q.py
  timeout 180s python -m pytest -s -q test_pa_small_q.py -vv
  RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q.py \
      -k 'perf and q16 and bls256 and cmpr0 and tile2'
"""

import functools
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from kv_cache_quant_utils import (
    DEFAULT_SUB_BLOCK_SIZE,
    dequant_per_channel as _dequant_per_channel,
    dequant_per_token as _dequant_per_token,
    quant_per_channel as _quant_per_channel,
    quant_per_token as _quant_per_token,
)


cl.profiling(True)
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)


def get_cm_grf_width() -> int:
    cm_kernels = cl.kernels(
        r'''
        extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
            info[0] = CM_GRF_WIDTH;
        }''',
        "-cmc",
    )
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return int(t_info.numpy()[0])


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-3) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    bad = torch.where(~close_mask)
    raise AssertionError(
        "Tensor mismatch\n"
        f"indices={bad}\n"
        f"actual[bad][:8]={actual[bad][:8]}\n"
        f"expected[bad][:8]={expected[bad][:8]}\n"
        f"max_abs_diff={(actual - expected).abs().max().item()}"
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Mirrors PagedAttentionGeneratorSmallQ::TILE_Q in the OV plugin: tied to the
# DPAS RepeatCount budget (Q_head_chunk_size * TILE_Q ≤ 8). TILE_Q=2 with
# q_chunk_size=4 hits the systolic-array sweet spot for Qwen3-8B-shape (32:8).
DEFAULT_TILE_Q = 2


@dataclass(frozen=True)
class SmallQCase:
    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    block_size: int = 256
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE
    past_len: int = 4096        # past KV length (number of cached keys before this round)
    q_len: int = 16             # number of new query tokens (spec window)
    kv_cache_compression: int = 0
    tile_q: int = DEFAULT_TILE_Q
    partition_block_num: int = 8   # number of blocks per partition (for KV partitioning)


class PaSmallQRunner:
    """Host harness for pa_small_q. Mirrors PaSingleTokenRunner shape but extends
    the q dimension via TILE_Q and a packed (orig_seq, q_start, valid_count)
    mapping, matching the OV-side PagedAttentionGeneratorSmallQ contract."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        sub_block_size: int,
        kv_cache_compression: int,
        tile_q: int = DEFAULT_TILE_Q,
        k_partition_block_num: int = 8,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.sub_block_size = sub_block_size
        self.kv_cache_compression = kv_cache_compression
        self.tile_q = int(tile_q)
        if self.tile_q < 1:
            raise ValueError(f"tile_q must be >= 1, got {self.tile_q}")
        if self.block_size % self.sub_block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be divisible by sub_block_size ({self.sub_block_size})"
            )

        self.cm_grf_width = get_cm_grf_width()
        self.xe_arch = 1 if self.cm_grf_width == 256 else 2
        self.kv_step = 8 if self.xe_arch == 1 else 16

        self.k_partition_block_num = k_partition_block_num
        self.kv_partition_size = int(self.block_size * self.k_partition_block_num)
        print(f"PaSmallQRunner: kv_partition_size={self.kv_partition_size} (block_size={self.block_size} * k_partition_block_num={self.k_partition_block_num})")
        self.reduce_split_step = 64

        # Mirror OV's get_single_token_q_chunking with tile_q_factor = TILE_Q so
        # the chunking solver shrinks q_head_chunk_size when the rS / Pmat /
        # Omat tiles, scaled by TILE_Q, would exceed the GRF budget.
        max_repeat_count = 8
        reg_n = 8 if self.xe_arch == 1 else 16
        reg_file_size = 256
        grf_bytes = 32 if self.xe_arch == 1 else 64

        kv_partition_step_num = self.kv_partition_size // self.kv_step
        rs_cols = kv_partition_step_num * reg_n
        budget_bytes = reg_file_size * grf_bytes - 1

        if self.tile_q == 1:
            bytes_per_q_row = 4 * rs_cols
        else:
            # rS:f32 + Pmat:f16 over partition cols; Omat:f32 + Qmat:f16 over head_size cols.
            bytes_per_q_row = (
                4 * self.kv_partition_size
                + 2 * self.kv_partition_size
                + 4 * self.head_size
                + 2 * self.head_size
            )
        max_q_by_matrix = max(1, budget_bytes // (bytes_per_q_row * self.tile_q))

        q_heads_per_kv_head = self.num_heads // self.num_kv_heads
        repeat_count_cap = max(1, max_repeat_count // self.tile_q)
        target_chunk = min(repeat_count_cap, max_q_by_matrix)
        q_head_chunk_size = max(1, min(q_heads_per_kv_head, target_chunk))
        while q_head_chunk_size > 1 and (q_heads_per_kv_head % q_head_chunk_size) != 0:
            q_head_chunk_size -= 1

        self.q_head_chunks_per_kv_head = q_heads_per_kv_head // q_head_chunk_size
        self.q_head_chunk_size = q_head_chunk_size
        self.scale_factor = 1.0 / (self.head_size ** 0.5)

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        sub_block_size: int,
        kv_cache_compression: int,
        tile_q: int = DEFAULT_TILE_Q,
        k_partition_block_num: int = 8,
    ):
        return PaSmallQRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            sub_block_size,
            kv_cache_compression,
            tile_q=tile_q,
            k_partition_block_num=k_partition_block_num,
        )

    @staticmethod
    @functools.cache
    def _create_kernels_cached(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        kv_step: int,
        block_size: int,
        sub_block_size: int,
        kv_partition_size: int,
        reduce_split_step: int,
        clean_unused_kvcache: int,
        kv_cache_compression: int,
        xe_arch: int,
        q_head_chunks_per_kv_head: int,
        q_head_chunk_size: int,
        tile_q: int,
        scale_factor: float,
    ):
        src = "\n".join(
            [
                '#include "pa_small_q.cm"',
                '#include "pa_small_q_finalization.cm"',
            ]
        )
        cwd = os.path.dirname(os.path.realpath(__file__))
        return cl.kernels(
            src,
            f'''-cmc -Qxcm_jit_option=""
                        -mCM_printregusage -mdump_asm -g2
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
                        -DSCALE_FACTOR={scale_factor}''',
        )

    def _create_kernels(self):
        return self._create_kernels_cached(
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
            int(self.tile_q),
            self.scale_factor,
        )

    @staticmethod
    def _build_mapping(q_len: int, tile_q: int) -> tuple[torch.Tensor, int]:
        """Pack (orig_seq=0, q_start, valid_count) triples for one subsequence."""
        triples: list[int] = []
        for q_start in range(0, q_len, tile_q):
            valid = min(tile_q, q_len - q_start)
            triples.extend([0, q_start, valid])
        tile_count = len(triples) // 3
        if tile_count == 0:
            raise ValueError("q_len must be > 0")
        return torch.tensor(triples, dtype=torch.int32), tile_count

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
        """One full enqueue (small_q + finalize). `out` is read back into shape
        [q_len, num_heads, head_size]."""
        q_len = int(q_tokens.shape[0])
        gws = [int(tile_count), self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [1, 1, 1]
        # Finalize fans (tile_count * TILE_Q, head, head_size_split). Lanes whose
        # t_in_tile >= valid_count early-return.
        gws_2 = [int(tile_count) * self.tile_q, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        t_q = cl.tensor(q_tokens.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_past_lens = cl.tensor(past_lens.detach().numpy())
        t_block_indices = cl.tensor(block_indices.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
        t_mapping = cl.tensor(mapping.detach().numpy())

        partition_token_rows = int(tile_count) * self.tile_q
        t_partition_out = cl.tensor(
            [partition_token_rows, self.num_heads, kv_partition_num, self.head_size],
            np.dtype(np.float32),
        )
        t_lse = cl.tensor(
            [partition_token_rows, self.num_heads, kv_partition_num], np.dtype(np.float32)
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
            int(q_len),                     # q_len_unused
            int(tile_count),                # selected_token_count == tile count
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
            partition_token_rows,           # selected_token_count for finalize == tile_count * TILE_Q
            kv_partition_num,
        )
        cl.finish()
        out.copy_(torch.from_numpy(t_out_final.numpy()))
        return out, t_partition_out, t_lse

    def __call__(
        self,
        query: torch.Tensor,                 # [q_len, num_heads, head_size]
        key_cache: torch.Tensor,             # block-paged k cache (already quantised if applicable)
        value_cache: torch.Tensor,
        past_lens: torch.Tensor,             # i32 [1]
        block_indices: torch.Tensor,         # i32 [block_count]
        block_indices_begins: torch.Tensor,  # i32 [2]
        subsequence_begins: torch.Tensor,    # i32 [2] = [0, q_len]
        out: torch.Tensor,                   # [q_len, num_heads, head_size] f16
        n_repeats: int = 1,
    ) -> torch.Tensor:
        if out.dtype != torch.float16:
            raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")

        kernels = self._create_kernels()

        q_len = int(query.shape[0])
        if q_len < 1:
            raise ValueError("q_len must be >= 1")
        # Partition count is sized against tile-wide kv_len_max = past + q_len (causal cap of the last row).
        max_context_len = int(past_lens.max().item()) + q_len
        kv_partition_num = _ceil_div(max_context_len, self.kv_partition_size)

        mapping, tile_count = self._build_mapping(q_len, self.tile_q)

        for _ in range(n_repeats):
            self._enqueue_once(
                kernels,
                query,
                key_cache,
                value_cache,
                past_lens,
                block_indices,
                block_indices_begins,
                subsequence_begins,
                mapping,
                tile_count,
                kv_partition_num,
                out,
            )
        return out


# ---------- input builders ----------


def _build_inputs(case: SmallQCase):
    """Build a single-subsequence small-q test fixture. Returns kernel-shaped
    tensors plus the SDPA reference output for q_len rows."""
    batch = 1
    low, high = -127, 128

    full_kv_len = case.past_len + case.q_len
    aligned_kv_len = _ceil_div(full_kv_len, case.block_size) * case.block_size
    total_blk_num = aligned_kv_len // case.block_size

    # BLHS canonical
    q_blhs = torch.randint(low, high, [batch, case.q_len, case.num_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    k_blhs = torch.randint(low, high, [batch, aligned_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    v_blhs = torch.randint(low, high, [batch, aligned_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high

    # BHLS for SDPA reference and physical block layout.
    q_bhls = q_blhs.transpose(1, 2).contiguous()
    k_bhls = k_blhs.transpose(1, 2).contiguous()
    v_bhls = v_blhs.transpose(1, 2).contiguous()

    # Wipe trailing tail keys/values past the real kv_len.
    k_bhls.view(torch.uint16)[:, :, full_kv_len:, :] = 0
    v_bhls.view(torch.uint16)[:, :, full_kv_len:, :] = 0xFE00

    # Build [block_num, num_kv_heads, block_size, head_size] then quantise if needed.
    k_blocks = (
        k_bhls[0].transpose(0, 1)
        .reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size)
        .transpose(1, 2).contiguous()
    )
    v_blocks = (
        v_bhls[0].transpose(0, 1)
        .reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size)
        .transpose(1, 2).contiguous()
    )

    if case.kv_cache_compression == 1:
        k_cache = _quant_per_token(k_blocks)
        k_ref_blocks = _dequant_per_token(k_cache, case.head_size, case.block_size)
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    elif case.kv_cache_compression == 2:
        k_cache = _quant_per_channel(k_blocks, case.sub_block_size)
        k_ref_blocks = _dequant_per_channel(k_cache, case.head_size, case.block_size, case.sub_block_size)
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    else:
        k_cache = k_blocks
        v_cache = v_blocks
        k_ref_blocks = k_blocks
        v_ref_blocks = v_blocks

    # Permute physical block placement to exercise the block_indices indirection.
    block_indices = torch.randperm(total_blk_num, dtype=torch.int32)
    key_cache = torch.empty_like(k_cache)
    value_cache = torch.empty_like(v_cache)
    key_cache[block_indices.to(dtype=torch.long)] = k_cache
    value_cache[block_indices.to(dtype=torch.long)] = v_cache

    # SDPA reference uses logical-order K/V (post-dequant), causal over real kv_len.
    k_ref_bhls = (
        k_ref_blocks.transpose(1, 2)
        .reshape(batch, aligned_kv_len, case.num_kv_heads, case.head_size)
        .transpose(1, 2).contiguous()
    )
    v_ref_bhls = (
        v_ref_blocks.transpose(1, 2)
        .reshape(batch, aligned_kv_len, case.num_kv_heads, case.head_size)
        .transpose(1, 2).contiguous()
    )

    # Causal mask: each new q row sees past + q_in_subseq + 1 keys.
    attn_mask = torch.full([batch, 1, case.q_len, full_kv_len], float("-inf"), dtype=torch.float16)
    for q_in_subseq in range(case.q_len):
        causal_kv_len = case.past_len + q_in_subseq + 1
        attn_mask[:, :, q_in_subseq, :causal_kv_len] = 0.0

    expected = F.scaled_dot_product_attention(
        q_bhls,
        k_ref_bhls[:, :, :full_kv_len, :],
        v_ref_bhls[:, :, :full_kv_len, :],
        attn_mask,
        dropout_p=0.0,
        enable_gqa=(case.num_heads > case.num_kv_heads),
    ).transpose(1, 2).contiguous()  # [batch, q_len, num_heads, head_size]

    past_lens = torch.tensor([case.past_len], dtype=torch.int32)
    block_indices_begins = torch.tensor([0, total_blk_num], dtype=torch.int32)
    subsequence_begins = torch.tensor([0, case.q_len], dtype=torch.int32)

    # Kernel expects q in shape [q_len, num_heads, head_size].
    query_tokens = q_bhls.transpose(1, 2).reshape(case.q_len, case.num_heads, case.head_size).contiguous()

    return {
        "query": query_tokens,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "past_lens": past_lens,
        "block_indices": block_indices,
        "block_indices_begins": block_indices_begins,
        "subsequence_begins": subsequence_begins,
        "expected": expected[0],  # [q_len, num_heads, head_size]
    }


# ---------- functional smoke tests ----------


_COMPRESSION_NAMES = {0: "fp16", 1: "by_token", 2: "by_channel"}


def _case_id(case: SmallQCase) -> str:
    return (
        f"q{case.q_len}"
        f"_past{case.past_len}"
        f"_h{case.num_heads}"
        f"_kv{case.num_kv_heads}"
        f"_hs{case.head_size}"
        f"_bls{case.block_size}"
        f"_sbls{case.sub_block_size}"
        f"_cmpr{_COMPRESSION_NAMES.get(case.kv_cache_compression, case.kv_cache_compression)}"
        f"_tile{case.tile_q}"
    )


# Smoke matrix: small q ∈ {2,4,8,16} × {fp16, by_token, by_channel} on Qwen3-shaped
# attention; one extra small case with head_size=64 to exercise non-128 head sizes.
SMALL_Q_SMOKE_CASES = tuple(
    SmallQCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        past_len=2048,
        q_len=q_len,
        kv_cache_compression=cmpr,
        tile_q=DEFAULT_TILE_Q,
    )
    for q_len in (2, 4, 8, 16)
    for cmpr in (0, 1, 2)
) + tuple(
    SmallQCase(
        num_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=256,
        past_len=513,
        q_len=q_len,
        kv_cache_compression=cmpr,
        tile_q=DEFAULT_TILE_Q,
    )
    for q_len in (2, 4)
    for cmpr in (0, 1, 2)
) + tuple(
    # Tail-tile coverage: q_len % TILE_Q != 0 forces a final tile with valid_count<TILE_Q.
    # Dummy rows must reach finalize as lse=-inf so the reduction ignores them.
    SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=2048, q_len=q_len, kv_cache_compression=cmpr, tile_q=DEFAULT_TILE_Q,
    )
    for q_len in (3, 5, 7, 15)
    for cmpr in (0, 1)
) + tuple(
    # Partition-boundary coverage: past_len chosen so KV crosses kv_partition boundaries
    # and the last partition is short. Exercises leftover_size / per-row mask straddling.
    SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=past_len, q_len=q_len, kv_cache_compression=cmpr, tile_q=DEFAULT_TILE_Q,
    )
    for past_len in (255, 257, 511, 1023)
    for q_len in (2, 4)
    for cmpr in (0, 1)
)


@pytest.mark.parametrize("case", SMALL_Q_SMOKE_CASES, ids=_case_id)
def test_pa_small_q_smoke_matches_sdpa(case: SmallQCase):
    """Functional: small_q output must match torch SDPA over the same masked KV."""
    data = _build_inputs(case)

    runner = PaSmallQRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
        tile_q=case.tile_q,
    )

    output = torch.zeros([case.q_len, case.num_heads, case.head_size], dtype=torch.float16)
    output = runner(
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

    assert torch.isfinite(output).all().item()
    tol = (2e-2, 2e-2) if case.kv_cache_compression > 0 else (1e-2, 1e-3)
    _check_close(output, data["expected"], atol=tol[0], rtol=tol[1])


# Tile sweep — same shape, vary TILE_Q. Only TILE_Q values that divide DPAS budget
# (q_chunk_size * TILE_Q ≤ 8) and stay within q_len make sense.
TILE_Q_SWEEP_CASES = tuple(
    SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=2048, q_len=16, kv_cache_compression=0, tile_q=tile_q,
    )
    for tile_q in (1, 2, 4)
)


@pytest.mark.parametrize("case", TILE_Q_SWEEP_CASES, ids=_case_id)
def test_pa_small_q_tile_q_sweep(case: SmallQCase):
    data = _build_inputs(case)
    runner = PaSmallQRunner.create_instance(
        case.num_heads, case.num_kv_heads, case.head_size, case.block_size,
        case.sub_block_size, case.kv_cache_compression, tile_q=case.tile_q,
    )
    output = torch.zeros([case.q_len, case.num_heads, case.head_size], dtype=torch.float16)
    output = runner(
        data["query"], data["key_cache"], data["value_cache"],
        data["past_lens"], data["block_indices"], data["block_indices_begins"],
        data["subsequence_begins"], output, n_repeats=1,
    )
    assert torch.isfinite(output).all().item()
    _check_close(output, data["expected"], atol=1e-2, rtol=1e-3)


# ---------- perf benchmark ----------


def _run_perf(case: SmallQCase, loop_cnt: int = 100, warmup: int = 5) -> dict[str, float]:
    """Time the (small_q + finalize) pair the same way pa_2nd_token.py does for the
    single-token path. KV cache size and intermediate buffer size are reported as
    proxy for memory bandwidth."""
    data = _build_inputs(case)
    runner = PaSmallQRunner.create_instance(
        case.num_heads, case.num_kv_heads, case.head_size, case.block_size,
        case.sub_block_size, case.kv_cache_compression, tile_q=case.tile_q,
    )
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
    partition_token_rows = tile_count * runner.tile_q

    # Allocate enough rotating layers to cover ~8 GB so cache effects don't dominate.
    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_partition_out = cl.tensor(
            [partition_token_rows, runner.num_heads, kv_partition_num, runner.head_size],
            np.dtype(np.float32),
        )
        t_out_final = cl.tensor([q_len, runner.num_heads, runner.head_size], np.dtype(np.float16))
        all_layers.append((t_q, t_k, t_v, t_partition_out, t_out_final))

        mem_size += query.numel() * query.element_size()
        mem_size += key_cache.numel() * key_cache.element_size()
        mem_size += value_cache.numel() * value_cache.element_size()

    if len(all_layers) == 0:
        raise RuntimeError("Failed to allocate perf input layers")

    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
    t_mapping = cl.tensor(mapping.detach().numpy())
    t_lse = cl.tensor([partition_token_rows, runner.num_heads, kv_partition_num], np.dtype(np.float32))

    cl.finish()  # discard any prior profiling events.

    gws = [tile_count, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [partition_token_rows, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_partition_out, t_out_final = all_layers[j]
        kernels.enqueue(
            "cm_pa_small_q", gws, lws,
            t_q, t_k, t_v,
            t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins,
            t_mapping, t_partition_out, t_lse,
            q_len, tile_count,
        )
        kernels.enqueue(
            "cm_pa_small_q_reduce", gws_2, lws_2,
            t_partition_out, t_out_final, t_lse,
            t_subsequence_begins, t_mapping,
            partition_token_rows, kv_partition_num,
        )

    latency = cl.finish()

    # KV cache bytes touched once per iter; intermediate partition buffer bytes the same.
    new_kv_len = int(key_cache.shape[0] * case.block_size)
    if case.kv_cache_compression > 0:
        # u8 + per-token/channel scale+zp ≈ head_size + 4 bytes per (block, head).
        kvcache_size = new_kv_len * case.num_kv_heads * (case.head_size + 4) * 2
    else:
        kvcache_size = new_kv_len * case.num_kv_heads * case.head_size * 2 * 2
    intermedia_size = partition_token_rows * case.num_heads * kv_partition_num * (case.head_size + 1) * 4

    expected_event_count = 2 * loop_cnt
    if len(latency) < expected_event_count:
        raise RuntimeError(f"Expected at least {expected_event_count} events, got {len(latency)}")

    small_q_total = 0.0
    reduce_total = 0.0
    runs = 0
    for pair_idx in range(loop_cnt):
        kv_ns = float(latency[2 * pair_idx])
        red_ns = float(latency[2 * pair_idx + 1])
        if kv_ns <= 0 or red_ns <= 0:
            continue
        if pair_idx < warmup:
            continue
        small_q_total += kv_ns
        reduce_total += red_ns
        runs += 1

    if runs <= 0 or small_q_total <= 0 or reduce_total <= 0:
        raise RuntimeError("Invalid perf timing accumulation")

    return {
        "num_runs": float(runs),
        "tile_count": float(tile_count),
        "small_q_bw_gbs": float(kvcache_size * runs / small_q_total),
        "small_q_reduce_bw_gbs": float(intermedia_size * runs / reduce_total),
        "small_q_ms": float(small_q_total * 1e-6 / runs),
        "small_q_reduce_ms": float(reduce_total * 1e-6 / runs),
    }


# Perf matrix: q ∈ {2,4,8,16} × {fp16, u8 by_token} × {tile_q 1, 2}, all on
# Qwen3-8B-shape attention with past_len=32k for a realistic spec-decode shape.
SMALL_Q_PERF_CASES = tuple(
    SmallQCase(
        num_heads=32, num_kv_heads=8, head_size=128, block_size=256,
        past_len=32 * 1024 - q_len, q_len=q_len,
        kv_cache_compression=cmpr, tile_q=tile_q,
    )
    for q_len in (2, 4, 8, 16)
    for cmpr in (0, 1)
    for tile_q in (1, 2)
)


@pytest.mark.parametrize("case", SMALL_Q_PERF_CASES, ids=lambda c: "perf_" + _case_id(c))
def test_pa_small_q_perf_bandwidth(case: SmallQCase):
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable bandwidth perf test")
    if case.tile_q > case.q_len:
        pytest.skip(f"tile_q={case.tile_q} exceeds q_len={case.q_len}")

    perf = _run_perf(case, loop_cnt=50, warmup=5)
    print(
        f"[perf] {_case_id(case)} | "
        f"tiles={int(perf['tile_count'])} | "
        f"small_q_ms={perf['small_q_ms']:.3f} reduce_ms={perf['small_q_reduce_ms']:.3f} | "
        f"kv_bw={perf['small_q_bw_gbs']:.1f} GB/s reduce_bw={perf['small_q_reduce_bw_gbs']:.1f} GB/s"
    )
    assert perf["small_q_ms"] > 0.0
    assert perf["small_q_reduce_ms"] > 0.0


# Usage:
#   python -m py_compile test_pa_small_q.py
#   python -m pytest --collect-only -q test_pa_small_q.py | head
#   timeout 180s python -m pytest -s -q test_pa_small_q.py -vv -k 'smoke and q4 and cmpr0'
#   timeout 600s python -m pytest -s -q test_pa_small_q.py -vv -k 'tile_q_sweep'
#   RUN_PA_PERF=1 timeout 600s python -m pytest -s -q test_pa_small_q.py \
#       -k 'perf and q16 and bls256 and cmpr0 and tile2'
