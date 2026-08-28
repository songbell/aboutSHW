"""
test_pa_small_q_multiseq.py

Multi-sequence correctness coverage for the new small-q kernel (pa_small_q_ov_exp.cm).

Every existing small-q test (test_pa_small_q.py, test_ov_exp_kernel_correctness.py,
test_15k_perf_comparison_ov_exp.py, the full-matrix sweep) uses
subsequence_begins=[0, q_len] -- exactly one sequence. Recorded as an open item in the
ledger ("multi-sequence coverage for the small-q path"): production batches multiple
sequences into one dispatch, and the mechanism *looked* right by inspection but nothing
ever exercised it.

How production batches multiple sequences (pa_small_q_ov_exp.cm):
  - `mapping` carries one (orig_seq_idx, q_start_in_subseq, valid_count) triple per TILE,
    across ALL sequences in the batch -- not one triple per sequence.
  - `subsequence_begins`, `past_lens`, `block_indices_begins` each carry one entry per
    sequence (plus subsequence_begins' trailing end marker).
  - The GWS partition dimension (kv_partition_num) is sized by the LONGEST sequence in the
    batch. Every workgroup computes its OWN `seq_kv_partition_num` from
    `past_lens[orig_seq_idx]` and early-outs (writing zero output + LSE=-inf, so the reduce
    kernel's uniform loop still sees a well-formed empty partition) once its partition index
    exceeds what that particular sequence needs.
  - The finalize kernel resolves each tile's true output row via
    `subsequence_begins[orig_seq_idx] + q_in_subseq`, not via the padded per-tile row index
    used for the intermediate partition buffer -- so it is already sequence-aware by
    construction, not something layered on afterward.

What each test below stresses:
  - test_multiseq_uniform_past_len: N sequences, same past_len, different q_len. Exercises
    orig_seq_idx / q_start bookkeeping across sequence boundaries with a single shared
    partition (no early-out involved yet).
  - test_multiseq_heterogeneous_past_len: the actual previously-untested mechanism. Sequences
    with very different past_len share one dispatch, forcing very different
    seq_kv_partition_num per sequence against one batch-wide kv_partition_num.
  - test_multiseq_multi_tile_sequence: a sequence whose q_len needs more than one TILE_Q
    tile, interleaved with single-tile sequences -- multiple mapping triples for one
    orig_seq_idx, non-contiguous with other sequences' triples.
  - test_multiseq_compression_and_block_size: repeats the heterogeneous-past shape across
    both compression modes and both block sizes this kernel supports.
  - test_multiseq_order_independence: the same set of sequences dispatched in two different
    orig_seq_idx orderings must produce the same per-sequence outputs. Cheap and catches
    indexing bugs (off-by-one on orig_seq_idx, wrong block_indices_begins accumulation) that
    a single fixed ordering could hide.

All of these call PaSmallQOvExpRunner._enqueue_once directly, bypassing __call__ (which is
single-sequence only via _build_mapping(q_len, tile_q)'s implicit orig_seq=0). Nothing about
the runner or kernel needed to change for this: _enqueue_once already takes mapping /
subsequence_begins / past_lens / block_indices_begins as plain data, and the finalize kernel
already routes through subsequence_begins[orig_seq_idx] -- only the *input builder* used by
every other test file was single-sequence-only.
"""
import os

import torch
import torch.nn.functional as F
import pytest

from kv_cache_quant_utils import (
    dequant_per_channel as _dequant_per_channel,
    dequant_per_token as _dequant_per_token,
    quant_per_channel as _quant_per_channel,
    quant_per_token as _quant_per_token,
)
from test_pa_small_q import _ceil_div, _check_close
from test_ov_exp_kernel_correctness import PaSmallQOvExpRunner

os.environ.setdefault("RUN_PA_CORRECTNESS", "1")

HEADS, KV_HEADS, HEAD_SIZE = 32, 8, 128


def _build_multiseq_inputs(seqs, num_heads, num_kv_heads, head_size, block_size,
                            sub_block_size, kv_cache_compression, seed=0):
    """seqs: list of (past_len, q_len) pairs, one per sequence.

    Mirrors test_pa_small_q._build_inputs (single sequence), generalised to N independent
    sequences concatenated into one dispatch: each sequence gets its own random K/V, its own
    causal SDPA reference, and its own slice of a batch-wide block table. block_indices is
    permuted GLOBALLY across every sequence's blocks together, so the indirection is exercised
    across sequence boundaries too, not just within one sequence's own block range.

    Each sequence draws from its own torch.Generator seeded by (seed, past_len, q_len) rather
    than a single global RNG consumed in list order. A global RNG would make sequence i's data
    depend on every OTHER sequence's shape ahead of it in the list (each torch.randint call
    consumes a different amount of RNG stream depending on shape) -- so the exact same
    sequence, at a different position in the batch, would silently get different random K/V/Q.
    Per-sequence generators make a sequence's data depend only on its own shape, which is what
    test_multiseq_order_independence relies on to compare the same sequence across two
    different batch orderings.
    """
    low, high = -127, 128
    all_k_cache, all_v_cache, all_query, all_expected = [], [], [], []
    block_indices_begins = [0]
    subsequence_begins = [0]
    past_lens = []
    blk_offset = 0

    for past_len, q_len in seqs:
        gen = torch.Generator().manual_seed(hash((seed, past_len, q_len)) % (2**31))
        full_kv_len = past_len + q_len
        aligned_kv_len = _ceil_div(full_kv_len, block_size) * block_size
        total_blk_num = aligned_kv_len // block_size

        q_blhs = torch.randint(low, high, [1, q_len, num_heads, head_size], dtype=torch.int32, generator=gen).to(torch.float16) / high
        k_blhs = torch.randint(low, high, [1, aligned_kv_len, num_kv_heads, head_size], dtype=torch.int32, generator=gen).to(torch.float16) / high
        v_blhs = torch.randint(low, high, [1, aligned_kv_len, num_kv_heads, head_size], dtype=torch.int32, generator=gen).to(torch.float16) / high

        q_bhls = q_blhs.transpose(1, 2).contiguous()
        k_bhls = k_blhs.transpose(1, 2).contiguous()
        v_bhls = v_blhs.transpose(1, 2).contiguous()
        # Wipe trailing tail keys/values past the real kv_len for this sequence.
        k_bhls.view(torch.uint16)[:, :, full_kv_len:, :] = 0
        v_bhls.view(torch.uint16)[:, :, full_kv_len:, :] = 0xFE00

        k_blocks = (k_bhls[0].transpose(0, 1)
                    .reshape(total_blk_num, block_size, num_kv_heads, head_size)
                    .transpose(1, 2).contiguous())
        v_blocks = (v_bhls[0].transpose(0, 1)
                    .reshape(total_blk_num, block_size, num_kv_heads, head_size)
                    .transpose(1, 2).contiguous())

        if kv_cache_compression == 1:
            k_cache = _quant_per_token(k_blocks)
            k_ref_blocks = _dequant_per_token(k_cache, head_size, block_size)
            v_cache = _quant_per_token(v_blocks)
            v_ref_blocks = _dequant_per_token(v_cache, head_size, block_size)
        elif kv_cache_compression == 2:
            k_cache = _quant_per_channel(k_blocks, sub_block_size)
            k_ref_blocks = _dequant_per_channel(k_cache, head_size, block_size, sub_block_size)
            v_cache = _quant_per_token(v_blocks)
            v_ref_blocks = _dequant_per_token(v_cache, head_size, block_size)
        else:
            k_cache = k_blocks
            v_cache = v_blocks
            k_ref_blocks = k_blocks
            v_ref_blocks = v_blocks

        k_ref_bhls = (k_ref_blocks.transpose(1, 2)
                      .reshape(1, aligned_kv_len, num_kv_heads, head_size)
                      .transpose(1, 2).contiguous())
        v_ref_bhls = (v_ref_blocks.transpose(1, 2)
                      .reshape(1, aligned_kv_len, num_kv_heads, head_size)
                      .transpose(1, 2).contiguous())

        attn_mask = torch.full([1, 1, q_len, full_kv_len], float("-inf"), dtype=torch.float16)
        for q_in_subseq in range(q_len):
            causal_kv_len = past_len + q_in_subseq + 1
            attn_mask[:, :, q_in_subseq, :causal_kv_len] = 0.0
        expected = F.scaled_dot_product_attention(
            q_bhls,
            k_ref_bhls[:, :, :full_kv_len, :],
            v_ref_bhls[:, :, :full_kv_len, :],
            attn_mask,
            dropout_p=0.0,
            enable_gqa=(num_heads > num_kv_heads),
        ).transpose(1, 2).contiguous()

        all_k_cache.append(k_cache)
        all_v_cache.append(v_cache)
        all_query.append(q_bhls.transpose(1, 2).reshape(q_len, num_heads, head_size).contiguous())
        all_expected.append(expected[0])
        past_lens.append(past_len)
        blk_offset += total_blk_num
        block_indices_begins.append(blk_offset)
        subsequence_begins.append(subsequence_begins[-1] + q_len)

    key_cache_all = torch.cat(all_k_cache, dim=0)
    value_cache_all = torch.cat(all_v_cache, dim=0)
    total_blk_num_all = key_cache_all.shape[0]
    block_indices = torch.randperm(total_blk_num_all, dtype=torch.int32)
    key_cache = torch.empty_like(key_cache_all)
    value_cache = torch.empty_like(value_cache_all)
    key_cache[block_indices.to(dtype=torch.long)] = key_cache_all
    value_cache[block_indices.to(dtype=torch.long)] = value_cache_all

    return {
        "query": torch.cat(all_query, dim=0),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "past_lens": torch.tensor(past_lens, dtype=torch.int32),
        "block_indices": block_indices,
        "block_indices_begins": torch.tensor(block_indices_begins, dtype=torch.int32),
        "subsequence_begins": torch.tensor(subsequence_begins, dtype=torch.int32),
        "expected": torch.cat(all_expected, dim=0),
    }


def _build_multiseq_mapping(seqs, tile_q):
    """One (orig_seq_idx, q_start, valid_count) triple per tile, across all sequences --
    generalises test_pa_small_q.PaSmallQRunner._build_mapping's single-sequence form."""
    triples = []
    for orig_seq_idx, (_, q_len) in enumerate(seqs):
        for q_start in range(0, q_len, tile_q):
            valid = min(tile_q, q_len - q_start)
            triples.extend([orig_seq_idx, q_start, valid])
    tile_count = len(triples) // 3
    if tile_count == 0:
        raise ValueError("every sequence must have q_len > 0")
    return torch.tensor(triples, dtype=torch.int32), tile_count


def _run_multiseq(seqs, kv_cache_compression, tile_q, kv_partition_size, block_size=256,
                   sub_block_size=16, seed=0):
    data = _build_multiseq_inputs(seqs, HEADS, KV_HEADS, HEAD_SIZE, block_size,
                                   sub_block_size, kv_cache_compression, seed=seed)
    mapping, tile_count = _build_multiseq_mapping(seqs, tile_q)
    max_context_len = max(past + q for past, q in seqs)
    kv_partition_num = _ceil_div(max_context_len, kv_partition_size)

    runner = PaSmallQOvExpRunner(HEADS, KV_HEADS, HEAD_SIZE, block_size, sub_block_size,
                                  kv_cache_compression, tile_q=tile_q,
                                  kv_partition_size=kv_partition_size)
    kernels = runner._create_kernels()
    total_q = sum(q for _, q in seqs)
    out = torch.zeros([total_q, HEADS, HEAD_SIZE], dtype=torch.float16)
    out, _, _ = runner._enqueue_once(
        kernels, data["query"], data["key_cache"], data["value_cache"], data["past_lens"],
        data["block_indices"], data["block_indices_begins"], data["subsequence_begins"],
        mapping, tile_count, kv_partition_num, out)
    return out, data["expected"]


def _tol(kv_cache_compression):
    return (2e-2, 2e-2) if kv_cache_compression > 0 else (1e-2, 1e-3)


def test_multiseq_uniform_past_len():
    """3 sequences, identical past_len, different q_len, one tile each (q_len <= tile_q),
    one shared partition. Isolates orig_seq_idx/q_start bookkeeping from the partition
    early-out mechanism (covered separately below)."""
    seqs = [(2048, 3), (2048, 5), (2048, 7)]
    tile_q, part = 8, 4096
    out, expected = _run_multiseq(seqs, kv_cache_compression=2, tile_q=tile_q,
                                   kv_partition_size=part, seed=0)
    assert torch.isfinite(out.float()).all()
    _check_close(out.float(), expected.float(), *_tol(2))


def test_multiseq_heterogeneous_past_len():
    """4 sequences with very different past_len sharing one dispatch. seq_kv_partition_num
    ranges from 1 (shortest) to 24 (longest) against one batch kv_partition_num=24 -- every
    workgroup outside a sequence's own range must early-out to zero/-inf rather than
    contribute garbage to that sequence's reduce."""
    seqs = [(128, 6), (2000, 6), (5000, 6), (15000, 6)]
    tile_q, part = 6, 640
    out, expected = _run_multiseq(seqs, kv_cache_compression=2, tile_q=tile_q,
                                   kv_partition_size=part, seed=1)
    assert torch.isfinite(out.float()).all()
    _check_close(out.float(), expected.float(), *_tol(2))

    # Per-sequence breakdown, so a failure points at which sequence's early-out broke rather
    # than just "the batch is wrong somewhere".
    q_offsets = [0]
    for _, q in seqs:
        q_offsets.append(q_offsets[-1] + q)
    for i, (past, q) in enumerate(seqs):
        lo, hi = q_offsets[i], q_offsets[i + 1]
        _check_close(out[lo:hi].float(), expected[lo:hi].float(), *_tol(2))


def test_multiseq_multi_tile_sequence():
    """A sequence needing 3 tiles (q_len=10, tile_q=4 -> valid_counts 4,4,2), interleaved
    with single-tile sequences before and after it in the mapping."""
    seqs = [(1024, 3), (1024, 10), (1024, 2)]
    tile_q, part = 4, 4096
    mapping, tile_count = _build_multiseq_mapping(seqs, tile_q)
    assert tile_count == 1 + 3 + 1  # seq0: 1 tile, seq1: 3 tiles, seq2: 1 tile
    # orig_seq_idx column of the middle sequence's three triples must all be 1, in order.
    triples = mapping.view(-1, 3).tolist()
    mid_triples = [t for t in triples if t[0] == 1]
    assert [t[1] for t in mid_triples] == [0, 4, 8]      # q_start
    assert [t[2] for t in mid_triples] == [4, 4, 2]       # valid_count

    out, expected = _run_multiseq(seqs, kv_cache_compression=2, tile_q=tile_q,
                                   kv_partition_size=part, seed=2)
    assert torch.isfinite(out.float()).all()
    _check_close(out.float(), expected.float(), *_tol(2))


@pytest.mark.parametrize("kv_cache_compression,block_size,sub_block_size", [
    (1, 16, 16),
    (2, 16, 16),
    (1, 256, 16),
    (2, 256, 16),
])
def test_multiseq_compression_and_block_size(kv_cache_compression, block_size, sub_block_size):
    """The heterogeneous-past shape again, across both compression modes and both block
    sizes the kernel supports -- production runs one of these four combinations depending on
    has_xattention, and multi-sequence batching should not be specific to any one of them."""
    seqs = [(100, 6), (900, 6), (3000, 6)]
    tile_q = 6
    part = 128 if block_size == 16 else 384
    out, expected = _run_multiseq(seqs, kv_cache_compression=kv_cache_compression,
                                   tile_q=tile_q, kv_partition_size=part,
                                   block_size=block_size, sub_block_size=sub_block_size,
                                   seed=3)
    assert torch.isfinite(out.float()).all()
    _check_close(out.float(), expected.float(), *_tol(kv_cache_compression))


def test_multiseq_order_independence():
    """The same three sequences, dispatched in two different orig_seq_idx orderings, must
    give each sequence the same answer regardless of its position in the batch. Cheap and
    catches indexing bugs (off-by-one on orig_seq_idx, wrong block_indices_begins
    accumulation) that a single fixed ordering could hide."""
    a, b, c = (300, 6), (4000, 6), (12000, 6)
    tile_q, part = 6, 640

    out1, exp1 = _run_multiseq([a, b, c], kv_cache_compression=2, tile_q=tile_q,
                                kv_partition_size=part, seed=7)
    out2, exp2 = _run_multiseq([c, a, b], kv_cache_compression=2, tile_q=tile_q,
                                kv_partition_size=part, seed=7)

    # Each sequence's random K/V/Q depends only on (seed, past_len, q_len), not on its
    # position in the seqs list (see _build_multiseq_inputs), so out1's [a,b,c] slices and
    # out2's [c,a,b] slices cover the exact same underlying data once re-sliced back to
    # (a, b, c) -- a real indexing bug would show up as a mismatch here, not as "different
    # input data" noise.
    q = 6
    out1_by_seq = {"a": out1[0:q], "b": out1[q:2 * q], "c": out1[2 * q:3 * q]}
    out2_by_seq = {"c": out2[0:q], "a": out2[q:2 * q], "b": out2[2 * q:3 * q]}
    for name in ("a", "b", "c"):
        _check_close(out1_by_seq[name].float(), out2_by_seq[name].float(), atol=0, rtol=0)
