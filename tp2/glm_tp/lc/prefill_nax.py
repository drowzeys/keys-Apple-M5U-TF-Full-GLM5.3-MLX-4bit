"""Prefill chunks on the M5 tensor units: sparse attention with the rope term, and
the indexer's scores.

oMLX 0.8.0rc1 ships two tensor-unit ("NAX") Metal kernels that its GLM model
code does not call:

* `omlx/patches/glm_moe_dsa/sparse_mla_nax.py`: sparse MLA attention over each
  row's selected keys, for latents without a rope part. GLM-5.3 as packed here
  has a 64-wide rope part, so the kernel cannot be used as it is.
* `omlx/patches/glm_moe_dsa/indexer_nax.py`: the indexer's head-summed scores.

Measured on this Mac at the shapes of a 2048-row chunk (development benchmark,
measured on an M5 Ultra): the native sparse attention kernel 34-41 ms per layer
(15-16 TFLOPS) against 8-10 ms (53-68 TFLOPS) for the tensor-unit one; index
scores 28 TFLOPS against 81-92.

Here: (1) the attention kernel is built at run time from oMLX's source with a
second, 64-wide product added to the scores (the rope part of query and key,
read from their own arrays; values stay the 512-wide latent), nothing else
changed; (2) both are put in front of oMLX's calls for chunks of
G53_PF_MIN_ROWS rows or more.

NOT bit-identical to the paths they replace. Both keep the arithmetic of the
native kernels (fp32 accumulation, the same formulas) but sum each dot product
in another order, so results agree to the rounding of the 16-bit outputs; with
the index scores, keys at the margin of the 2048 selected can differ (as they
already do between two runs of the native path). Decode and verify passes do
not come here.
"""

import sys

import mlx.core as mx

from . import config as C

STATS = {"attention": 0, "attention passed on": 0, "scores": 0, "scores passed on": 0}
_KERNEL = None
_ORIG = {}
_OK = None


def available():
    global _OK
    if _OK is None:
        try:
            from omlx.custom_kernels.nax import is_nax_available

            _OK = bool(is_nax_available())
        except Exception:
            _OK = False
    return _OK


def _source():
    """oMLX's kernel source with the rope product added. Raises when the source is not the one this was written for."""
    from omlx.patches.glm_moe_dsa import sparse_mla_nax as N

    s = N._SOURCE
    edits = [
        ("    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;\n",
         "    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;\n"
         "    const device T* qp0 = qpe + (ulong(head0 + fm) * L + qi) * 64 + fn;\n"
         "    const device T* qp1 = qpe + (ulong(head0 + fm + 8) * L + qi) * 64 + fn;\n"),
        ("            const device T* kr[2][2];\n",
         "            const device T* kr[2][2];\n            const device T* pr[2][2];\n"),
        ("                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;\n",
         "                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;\n"
         "                    pr[tn][i] = kpe + ulong(max(kp, 0)) * 64 + fn;\n"),
        ("                qk_op.run(qa, kb, sc);\n            }\n        }\n",
         "                qk_op.run(qa, kb, sc);\n            }\n"
         "            for (short kk = 0; kk < 64; kk += 16) {\n"
         "                for (short j = 0; j < 4; ++j) {\n"
         "                    qa[j] = qp0[kk + j];\n"
         "                    qa[4 + j] = qp1[kk + j];\n"
         "                }\n"
         "                for (short tn = 0; tn < 2; ++tn) {\n"
         "                    for (short i = 0; i < 2; ++i) {\n"
         "                        for (short j = 0; j < 4; ++j) {\n"
         "                            kb[tn * 8 + i * 4 + j] = pr[tn][i][kk + j];\n"
         "                        }\n"
         "                    }\n"
         "                }\n"
         "                qk_op.run(qa, kb, sc);\n"
         "            }\n        }\n"),
    ]
    for a, b in edits:
        if s.count(a) != 1:
            raise RuntimeError("oMLX's sparse_mla_nax source is not the expected one")
        s = s.replace(a, b)
    return N._HEADER, s


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        header, source = _source()
        _KERNEL = mx.fast.metal_kernel(
            name="g53_sparse_mla_nax_rope",
            input_names=["q", "kv", "idx", "params", "scale", "qpe", "kpe"],
            output_names=["out"],
            header=header,
            source=source,
        )
    return _KERNEL


def sparse_attention(q_latent, q_pe, kv_latent, k_pe, topk_indices, scale):
    """[1, H, L, 512] attention output over each row's selected keys, scores = scale * (latent + rope products).
    q_latent [1, H, L, 512], q_pe [1, H, L, 64], kv_latent [1, 1, K, 512], k_pe [1, 1, K, 64],
    topk_indices [1, 1, L, TOPK] (a slot past the row's own position, or >= K, is unused)."""
    _, H, L, _ = q_latent.shape
    K = kv_latent.shape[2]
    idx = topk_indices[0, 0]
    if idx.dtype != mx.int32:
        idx = idx.astype(mx.int32)
    params = mx.array([L, K, idx.shape[-1], K - L], dtype=mx.int32)
    out = _kernel()(
        inputs=[q_latent[0], kv_latent[0, 0], idx, params, mx.array([scale], mx.float32), q_pe[0], k_pe[0, 0]],
        template=[("T", q_latent.dtype)],
        grid=(L * 256, H // 32, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(H, L, 512)],
        output_dtypes=[q_latent.dtype],
    )[0]
    return out[None]


def _attention_hook(q_latent, q_pe, kv_latent, k_pe, topk_indices, scale, *, topk_valid_prefix=False,
                    causal_prefix_indices=False, topk_length=None, causal_prefix_rows=0, stream=None):
    """In front of oMLX's sparse_mla_attention (same contract)."""
    K = kv_latent.shape[2] if kv_latent.ndim == 4 else 0
    ok = (C.PF_ATTN and available() and q_latent.ndim == 4 and q_latent.shape[0] == 1
          and q_latent.shape[2] >= C.PF_MIN_ROWS and q_latent.shape[1] % 32 == 0 and q_latent.shape[-1] == 512
          and q_pe.shape[-1] == 64 and k_pe.shape[-1] == 64 and kv_latent.shape[-1] == 512
          and topk_indices.ndim == 4 and topk_indices.shape[2] == q_latent.shape[2] and causal_prefix_rows == 0
          and topk_length is None and q_latent.dtype in (mx.bfloat16, mx.float16)
          and kv_latent.dtype == q_latent.dtype and q_pe.dtype == q_latent.dtype and k_pe.dtype == q_latent.dtype)
    if ok:
        STATS["attention"] += 1
        return sparse_attention(q_latent, q_pe, kv_latent, k_pe, topk_indices, scale)
    STATS["attention passed on"] += 1
    if K < _ORIG["min_k"]:
        return None            # below oMLX's own threshold: its block-token path, as before
    return _ORIG["attention"](q_latent, q_pe, kv_latent, k_pe, topk_indices, scale,
                              topk_valid_prefix=topk_valid_prefix, causal_prefix_indices=causal_prefix_indices,
                              topk_length=topk_length, causal_prefix_rows=causal_prefix_rows, stream=stream)


def _scores_hook(queries, keys, weights, *, causal=False, unused_causal_prefix_topk=0,
                 skip_causal_future_store=False, causal_q_offset=-1, stream=None):
    """In front of oMLX's fused_indexer_scores (same contract; masked scores are -1e30 instead of unwritten or -inf)."""
    ok = (C.PF_INDEX and available() and queries.ndim == 4 and keys.ndim == 4 and weights.ndim == 3
          and queries.shape[0] == 1 and keys.shape[:2] == (1, 1) and queries.shape[3] == 128 and keys.shape[3] == 128
          and queries.shape[2] >= C.PF_MIN_ROWS and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16
          and weights.dtype == mx.bfloat16 and keys.shape[2] >= 4096)
    if not ok:
        STATS["scores passed on"] += 1
        return _ORIG["scores"](queries, keys, weights, causal=causal,
                               unused_causal_prefix_topk=unused_causal_prefix_topk,
                               skip_causal_future_store=skip_causal_future_store,
                               causal_q_offset=causal_q_offset, stream=stream)
    from omlx.patches.glm_moe_dsa import indexer_nax as N

    _, H, L, _ = queries.shape
    K = keys.shape[2]
    q = queries[0].transpose(1, 0, 2)                      # [L, H, 128]
    k = keys[0, 0]
    w = weights[0]
    first = (causal_q_offset if causal_q_offset >= 0 else K - L) if causal else K       # position of query row 0
    step = N.max_rows_per_call(K)
    parts = []
    for r in range(0, L, step):
        e = min(L, r + step)
        out = N.indexer_scores_nax(q[r:e], k, w[r:e], first + r, K, 1)
        if out is None:
            STATS["scores passed on"] += 1
            return _ORIG["scores"](queries, keys, weights, causal=causal,
                                   unused_causal_prefix_topk=unused_causal_prefix_topk,
                                   skip_causal_future_store=skip_causal_future_store,
                                   causal_q_offset=causal_q_offset, stream=stream)
        parts.append(out)
    STATS["scores"] += 1
    s = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    return s[None, None]


def install():
    """Returns the list of what was put in place."""
    on = []
    if not available():
        return on
    model = sys.modules.get("omlx.patches.glm_moe_dsa.glm_moe_dsa_model")
    base = sys.modules.get("omlx.patches.glm_moe_dsa.deepseek_v32")
    if C.PF_ATTN and model is not None and "attention" not in _ORIG:
        try:
            # Compile now (3-4 s on first use otherwise, inside the first long prompt).
            z = mx.zeros((1, 64, 16, 512), mx.bfloat16)
            mx.eval(sparse_attention(z, z[..., :64], mx.zeros((1, 1, 64, 512), mx.bfloat16),
                                     mx.zeros((1, 1, 64, 64), mx.bfloat16),
                                     mx.zeros((1, 1, 16, 2048), mx.uint32), 0.05))
        except Exception as e:
            print(f"[tp2] tensor-unit prefill attention not installed: {e!r}", flush=True)
        else:
            _ORIG["attention"] = model.sparse_mla_attention
            _ORIG["min_k_fn"] = model._native_sparse_mla_default_min_k
            _ORIG["min_k"] = int(model._native_sparse_mla_default_min_k())
            model.sparse_mla_attention = _attention_hook
            model._native_sparse_mla_default_min_k = lambda: "0"      # every selecting chunk asks the hook first
            on.append("attention")
    if C.PF_INDEX and base is not None and "scores" not in _ORIG:
        _ORIG["scores"] = base.fused_indexer_scores
        base.fused_indexer_scores = _scores_hook
        on.append("index scores")
    return on


def uninstall():
    model = sys.modules.get("omlx.patches.glm_moe_dsa.glm_moe_dsa_model")
    base = sys.modules.get("omlx.patches.glm_moe_dsa.deepseek_v32")
    if "attention" in _ORIG and model is not None:
        model.sparse_mla_attention = _ORIG.pop("attention")
        model._native_sparse_mla_default_min_k = _ORIG.pop("min_k_fn")
        _ORIG.pop("min_k", None)
    if "scores" in _ORIG and base is not None:
        base.fused_indexer_scores = _ORIG.pop("scores")
