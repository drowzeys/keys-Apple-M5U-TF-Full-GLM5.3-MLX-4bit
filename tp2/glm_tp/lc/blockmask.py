# The Metal source below is adapted from oMLX 0.8.0rc1, omlx/patches/glm_moe_dsa/sparse_mla.py (topk_indices_to_block_masks),
# Copyright (c) 2026 Apple Inc., Apache License 2.0; changed to read the row and key counts from an input.
"""Block masks for oMLX's exact block-token attention, one compiled kernel for
every chunk and context length.

oMLX's `topk_indices_to_block_masks` (omlx/patches/glm_moe_dsa/sparse_mla.py)
turns the indexer's top-k rows into block and token masks for prefill chunks of
more than 8 rows below 11264 keys. Its template carries the chunk length, the
key count and the block counts, so a chunk at a new context length compiles a
kernel (36-85 ms measured): every follow-up turn of a conversation between 2k
and 11k tokens. Same arithmetic here, with those numbers read from an input.
Masks are integers: the reference implementation is compared bit for bit in the development tests.
"""

import sys

import mlx.core as mx

from . import config as C

_KERNEL = None
_ORIG = None
STATS = {"calls": 0, "passed on": 0}

_SOURCE = r"""
    const uint L = dims[0];
    const uint K = dims[1];
    const uint TOPK_ROWS = dims[2];
    const uint Q_BLOCKS = dims[3];
    const uint K_BLOCKS = dims[4];
    const uint PREFIX_ROWS = dims[5];
    const uint elem = thread_position_in_grid.x;
    const uint total = B * L * TOPK;
    if (elem >= total) {
      return;
    }

    const uint j = elem % TOPK;
    const uint q_pos = (elem / TOPK) % L;
    const uint b = elem / (TOPK * L);

    const uint q_abs = (K - L) + q_pos;
    if (CAUSAL_PREFIX_INDICES && q_pos < PREFIX_ROWS) {
      const uint valid_length = min(uint(K), q_abs + 1);
      const uint prefix_blocks =
          (valid_length + uint(K_BLOCK) - 1) / uint(K_BLOCK);
      if (j >= prefix_blocks) {
        return;
      }

      const uint q_block = q_pos / Q_BLOCK;
      const uint k_block = j;
      if (q_block >= Q_BLOCKS || k_block >= K_BLOCKS) {
        return;
      }

      const uint block_idx = (b * Q_BLOCKS + q_block) * K_BLOCKS + k_block;
      block_mask[block_idx] = true;

      const uint block_start = k_block * uint(K_BLOCK);
      const uint tokens_in_block =
          min(uint(K_BLOCK), valid_length - block_start);
      const uint full_bits =
          K_BLOCK == 32 ? 0xffffffffu : ((1u << K_BLOCK) - 1u);
      const uint bits = tokens_in_block == K_BLOCK
          ? full_bits
          : ((1u << tokens_in_block) - 1u);
      const uint token_idx = (b * L + q_pos) * K_BLOCKS + k_block;
      block_token_mask[token_idx] = bits;
      return;
    }

    const uint topk_row = COMPACT_PREFIX_TOPK ? q_pos - PREFIX_ROWS : q_pos;
    const uint topk_elem = (b * TOPK_ROWS + topk_row) * TOPK + j;
    const uint k_pos = topk_indices[topk_elem];
    if (k_pos >= K) {
      return;
    }

    if (CAUSAL && k_pos > q_abs) {
      return;
    }

    const uint q_block = q_pos / Q_BLOCK;
    const uint k_block = k_pos / K_BLOCK;
    if (q_block >= Q_BLOCKS || k_block >= K_BLOCKS) {
      return;
    }

    const uint block_idx = (b * Q_BLOCKS + q_block) * K_BLOCKS + k_block;
    block_mask[block_idx] = true;

    const uint bit = 1u << (k_pos - k_block * K_BLOCK);
    const uint token_idx = (b * L + q_pos) * K_BLOCKS + k_block;
    device atomic_uint* atomic_token_mask =
        reinterpret_cast<device atomic_uint*>(block_token_mask);
    atomic_fetch_or_explicit(
        &atomic_token_mask[token_idx],
        bit,
        memory_order_relaxed);
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="g53_topk_indices_to_block_masks_rt",
            input_names=["topk_indices", "dims"],
            output_names=["block_mask", "block_token_mask"],
            source=_SOURCE,
        )
    return _KERNEL


def block_masks(topk_indices, *, L=None, K, q_block_size=32, k_block_size=16, causal=True,
                causal_prefix_indices=False, causal_prefix_rows=0, stream=None):
    """Same contract as oMLX's topk_indices_to_block_masks."""
    if (not C.LC_REDUCE or not mx.metal.is_available() or topk_indices.ndim != 4 or topk_indices.shape[1] != 1
            or k_block_size <= 0 or k_block_size > 32 or q_block_size <= 0):
        STATS["passed on"] += 1
        return _ORIG(topk_indices, L=L, K=K, q_block_size=q_block_size, k_block_size=k_block_size, causal=causal,
                     causal_prefix_indices=causal_prefix_indices, causal_prefix_rows=causal_prefix_rows, stream=stream)
    B, _, L_in, topk = topk_indices.shape
    L = L_in if L is None else L
    causal_prefix_rows = max(0, min(causal_prefix_rows, L))
    compact = causal_prefix_indices and causal_prefix_rows > 0 and L_in == L - causal_prefix_rows
    if (L != L_in and not compact) or K <= 0 or topk <= 0:
        return None
    q_blocks = (L + q_block_size - 1) // q_block_size
    k_blocks = (K + k_block_size - 1) // k_block_size
    STATS["calls"] += 1
    dims = mx.array([L, K, L_in, q_blocks, k_blocks, causal_prefix_rows], dtype=mx.uint32)
    block_mask, block_token_mask = _kernel()(
        inputs=[topk_indices.astype(mx.uint32), dims],
        template=[("B", B), ("TOPK", topk), ("Q_BLOCK", q_block_size), ("K_BLOCK", k_block_size),
                  ("CAUSAL", bool(causal)), ("CAUSAL_PREFIX_INDICES", bool(causal_prefix_indices)),
                  ("COMPACT_PREFIX_TOPK", bool(compact))],
        grid=(B * L * topk, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 1, q_blocks, k_blocks), (B, 1, L, k_blocks)],
        output_dtypes=[mx.bool_, mx.uint32],
        init_value=0,
        stream=stream or mx.gpu,
    )
    return block_mask, block_token_mask


def install():
    global _ORIG
    mod = sys.modules.get("omlx.patches.glm_moe_dsa.sparse_mla")
    if mod is None or not hasattr(mod, "topk_indices_to_block_masks"):
        return False
    if getattr(mod.topk_indices_to_block_masks, "_g53", False):
        return True
    _ORIG = mod.topk_indices_to_block_masks
    block_masks._g53 = True
    mod.topk_indices_to_block_masks = block_masks
    return True


def uninstall():
    global _ORIG
    mod = sys.modules.get("omlx.patches.glm_moe_dsa.sparse_mla")
    if mod is not None and _ORIG is not None:
        mod.topk_indices_to_block_masks = _ORIG
    _ORIG = None
