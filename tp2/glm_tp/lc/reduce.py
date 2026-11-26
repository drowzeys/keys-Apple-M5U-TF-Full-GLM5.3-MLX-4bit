# The Metal source below is adapted from oMLX 0.8.0rc1, omlx/patches/glm_moe_dsa/sparse_mla.py (fused_index_score_reduce),
# Copyright (c) 2026 Apple Inc., Apache License 2.0; changed to read the row and key counts from an input.
"""Head-summed indexer scores for a verify pass, one compiled kernel for every
context length.

oMLX's `fused_index_score_reduce` (omlx/patches/glm_moe_dsa/sparse_mla.py)
is a Metal kernel whose template carries the number of rows L and of keys K.
MLX compiles a kernel per template, so in decoding, where K grows with every
pass, each verify pass with more than one row compiles a new one: 38-41 ms
measured on the M5 Ultra, more than half a pass, and the reason a two-row pass
cost 100 ms beyond 2k tokens of context where it should cost 65.

The kernel below does the same arithmetic in the same order (per key: fp32
sum over the heads, in head order, of relu(score) * weight; future keys -inf
when causal) and reads L and K from an input instead. the development tests
compares it bit for bit with oMLX's kernel.

oMLX routes passes of fewer than 16 rows to this reduction, and prefill chunks
while the context is under 4096 keys.
"""

import sys

import mlx.core as mx

from . import config as C

MAX_ROWS = 1 << 20      # every row count (oMLX uses this reduction below 16 rows, and for chunks under 4096 keys)
_KERNEL = None
_ORIG = None
STATS = {"calls": 0, "passed on": 0}

_SOURCE = r"""
    const uint L = dims[0];
    const uint K = dims[1];
    const uint elem = thread_position_in_grid.x;
    const uint total = B * L * K;
    if (elem >= total) {
      return;
    }

    const uint k_pos = elem % K;
    const uint q_pos = (elem / K) % L;
    const uint b = elem / (L * K);

    if (CAUSAL && k_pos > K - L + q_pos) {
      out[elem] = static_cast<T>(-INFINITY);
      return;
    }

    float acc = 0.0f;
    #pragma clang loop unroll(full)
    for (uint h = 0; h < H; ++h) {
      const uint score_idx = ((b * H + h) * L + q_pos) * K + k_pos;
      const uint weight_idx = (b * H + h) * L + q_pos;
      const float s = static_cast<float>(head_scores[score_idx]);
      const float w = static_cast<float>(weights[weight_idx]);
      acc += metal::max(s, 0.0f) * w;
    }
    out[elem] = static_cast<T>(acc);
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="g53_index_score_reduce_rt",
            input_names=["head_scores", "weights", "dims"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def reduce_rows(head_scores, weights, *, causal=False, stream=None):
    """Same contract as oMLX's fused_index_score_reduce."""
    if not C.LC_REDUCE and _ORIG is not None:          # switched off through the control file
        STATS["passed on"] += 1
        return _ORIG(head_scores, weights, causal=causal, stream=stream)
    if (not mx.metal.is_available() or head_scores.ndim != 4 or weights.ndim != 4
            or weights.shape[-1] != 1 or head_scores.shape[:3] != weights.shape[:3]
            or head_scores.shape[2] > MAX_ROWS):
        STATS["passed on"] += 1
        return _ORIG(head_scores, weights, causal=causal, stream=stream) if _ORIG is not None else None
    B, H, L, K = head_scores.shape
    STATS["calls"] += 1
    tg = 256 if K > 32768 else 512
    return _kernel()(
        inputs=[head_scores, weights, mx.array([L, K], dtype=mx.uint32)],
        template=[("T", head_scores.dtype), ("B", B), ("H", H), ("CAUSAL", bool(causal))],
        grid=(B * L * K, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, 1, L, K)],
        output_dtypes=[head_scores.dtype],
        stream=stream or mx.gpu,
    )[0]


def install():
    """Point oMLX's indexer at reduce_rows. Returns False when its module is not there."""
    global _ORIG
    mod = sys.modules.get("omlx.patches.glm_moe_dsa.deepseek_v32")
    if mod is None or not hasattr(mod, "fused_index_score_reduce"):
        return False
    if getattr(mod.fused_index_score_reduce, "_g53", False):
        return True
    _ORIG = mod.fused_index_score_reduce
    reduce_rows._g53 = True
    mod.fused_index_score_reduce = reduce_rows
    return True


def uninstall():
    global _ORIG
    mod = sys.modules.get("omlx.patches.glm_moe_dsa.deepseek_v32")
    if mod is not None and _ORIG is not None:
        mod.fused_index_score_reduce = _ORIG
    _ORIG = None
