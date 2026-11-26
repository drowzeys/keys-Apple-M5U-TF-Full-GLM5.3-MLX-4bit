"""8-bit latent KV cache for oMLX's GLM-5.3 attention (MLX, works with and without TP).

The MLA latent (512 per token per layer) is stored with mx.quantize (affine, 8 bits, group 64:
512 B + 32 B of scales/biases instead of 1024 B); the rope key (64) and the indexer keys stay
bf16. Per token: 57.7 KB instead of 95.2 KB of KV (78 layers; indexer keys on 21 of them).

Where the latent is read:
- decode (L == 1) and short verify passes (MTP): the top-k rows are gathered from the 8-bit
  cache first and only those rows are dequantized (2048 x 512 per layer);
- prefill chunks: the whole latent is dequantized for that layer and chunk (transient), then
  oMLX's sparse attention (or the tensor-unit hook) runs unchanged on bf16.
The attention forward is oMLX's own source with those three reads swapped, recompiled at install
(the install refuses if the source no longer matches, so an oMLX update cannot change it silently).
The cache grows by doubling, so a long prefill copies O(n) bytes, not O(n^2 / step).

    import kv8; kv8.install()          # after the model is built; then model.make_cache() gives 8-bit latents
"""
import inspect
import os
import sys
import textwrap

import mlx.core as mx

GROUP, BITS = 64, 8


class Q8Latent:
    """A read-only view of the first `n` tokens of the quantized latent: (w, s, b) + logical shape."""

    def __init__(self, w, s, b, d):
        self.w, self.s, self.b = w, s, b
        self.shape = (w.shape[0], w.shape[1], w.shape[2], d)

    def full(self):
        return mx.dequantize(self.w, self.s, self.b, group_size=GROUP, bits=BITS)

    def gather_axis2(self, idx):
        """idx [B, 1, K] (int) -> dequantized rows [B, 1, K, D]."""
        def g(a):
            i = idx[..., None]
            return mx.take_along_axis(a, mx.broadcast_to(i, i.shape[:-1] + (a.shape[-1],)), axis=2)
        return mx.dequantize(g(self.w), g(self.s), g(self.b), group_size=GROUP, bits=BITS)

    def rows(self, idx):
        """idx [L, K] -> [L, K, D] (batch 1)."""
        return mx.dequantize(self.w[0, 0][idx], self.s[0, 0][idx], self.b[0, 0][idx], group_size=GROUP, bits=BITS)


def _full(x):
    return x.full() if isinstance(x, Q8Latent) else x


def _gather(x, idx):
    if isinstance(x, Q8Latent):
        return x.gather_axis2(idx[..., 0])
    return mx.take_along_axis(x, mx.broadcast_to(idx, idx.shape[:-1] + (x.shape[-1],)), axis=2)


def _rows(x, idx):
    return x.rows(idx) if isinstance(x, Q8Latent) else x[0, 0][idx]


class Q8LatentCache:
    """Latent cache: keys = 8-bit latent (w, s, b), values = bf16 rope key. Trimmable (MTP)."""

    def __init__(self):
        self.w = self.s = self.b = self.pe = None
        self.offset = 0
        self.d = None

    # oMLX ties the indexer cache to the latent cache with mx.depends(cache[0].keys, ...)
    @property
    def keys(self):
        return self.w

    @keys.setter
    def keys(self, v):
        self.w = v

    @property
    def values(self):
        return self.pe

    def _grow(self, B, n_new, dtype_s, pe_dim):
        need = self.offset + n_new
        cap = 0 if self.w is None else self.w.shape[2]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap, 1024)
        new_cap = (new_cap + 255) // 256 * 256

        def grow(a, last, dtype):
            z = mx.zeros((B, 1, new_cap, last), dtype)
            if a is not None and self.offset:
                z[:, :, : self.offset, :] = a[:, :, : self.offset, :]
            return z

        self.w = grow(self.w, self.d // (32 // BITS), mx.uint32)
        self.s = grow(self.s, self.d // GROUP, dtype_s)
        self.b = grow(self.b, self.d // GROUP, dtype_s)
        self.pe = grow(self.pe, pe_dim, dtype_s)

    def update_and_fetch(self, kv_latent, k_pe):
        B, _, n, d = kv_latent.shape
        self.d = d
        w, s, b = mx.quantize(kv_latent, group_size=GROUP, bits=BITS)
        self._grow(B, n, s.dtype, k_pe.shape[-1])
        p, q = self.offset, self.offset + n
        self.w[:, :, p:q, :] = w
        self.s[:, :, p:q, :] = s
        self.b[:, :, p:q, :] = b
        self.pe[:, :, p:q, :] = k_pe
        self.offset = q
        return Q8Latent(self.w[:, :, :q], self.s[:, :, :q], self.b[:, :, :q], d), self.pe[:, :, :q]

    @property
    def state(self):
        return [a for a in (self.w, self.s, self.b, self.pe) if a is not None]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self):
        return self.w is None

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state)

    def make_mask(self, *args, **kwargs):  # KVCache's, which reads only self.offset
        from mlx_lm.models.cache import KVCache

        return KVCache.make_mask(self, *args, **kwargs)


_SWAPS = [
    # L == 1: gather the top-k rows, then dequantize only them
    ("""                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )""", """                kv_latent = _g53_gather(kv_latent, idx)"""),
    # L == 1 without a selection (context <= index_topk): the whole (short) latent
    ("""            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            q_nope = self.embed_q(q_nope)""", """            kv_latent = _g53_full(kv_latent)
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            q_nope = self.embed_q(q_nope)"""),
    # short verify passes: per-row gathered rows
    ("""            kv_rows = kv_latent[0, 0][idx]  # (L, topk, latent)""",
     """            kv_rows = _g53_rows(kv_latent, idx)  # (L, topk, latent)"""),
    # prefill and everything after: bf16 latent for this layer and chunk
    ("""        direct_sparse_mla_min_k = int(""", """        kv_latent = _g53_full(kv_latent)
        direct_sparse_mla_min_k = int("""),
]


def install():
    """Recompile oMLX's GlmMoeDsaAttention.__call__ with the 8-bit reads and make Model.make_cache
    use Q8LatentCache for the latent. Idempotent."""
    mod = sys.modules.get("omlx.patches.glm_moe_dsa.glm_moe_dsa_model")
    if mod is None:
        raise RuntimeError("oMLX's glm_moe_dsa patch is not applied")
    cls = mod.GlmMoeDsaAttention
    if getattr(cls.__call__, "_g53_kv8", False):
        return True
    src = textwrap.dedent(inspect.getsource(cls.__call__))
    for old, new in _SWAPS:  # the swaps are written at file indentation; the source is dedented by 4
        cnt = src.count(_dedent4(old))
        if cnt != 1:
            raise RuntimeError(f"kv8: oMLX attention source changed (swap found {cnt} times): {old.strip()[:60]}")
        src = src.replace(_dedent4(old), _dedent4(new))
    # compile in the module's own namespace (not a copy): later hooks on module globals such as
    # sparse_mla_attention (the tensor-unit prefill) must stay visible to the recompiled forward
    mod.__dict__.update(_g53_full=_full, _g53_gather=_gather, _g53_rows=_rows)
    ns = {}
    exec(compile(src, "<kv8 GlmMoeDsaAttention.__call__>", "exec"), mod.__dict__, ns)
    fn = ns["__call__"]
    fn._g53_kv8 = True
    fn._g53_stock = cls.__call__
    cls.__call__ = fn

    from mlx_lm.models.cache import CacheList, KVCache

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(Q8LatentCache()))
            else:
                caches.append(CacheList(Q8LatentCache(), KVCache()))
        return caches

    global _STOCK_MAKE, _Q8_MAKE, _MOD
    _STOCK_MAKE, _Q8_MAKE, _MOD = mod.Model.make_cache, make_cache, mod
    mod.Model.make_cache = make_cache
    return True


_STOCK_MAKE = _Q8_MAKE = _MOD = None


def enable(on=True):
    """Switch Model.make_cache between 8-bit and bf16 latents (the attention handles both)."""
    _MOD.Model.make_cache = _Q8_MAKE if on else _STOCK_MAKE


def _dedent4(s):
    return "\n".join(line[4:] if line.startswith("    ") else line for line in s.split("\n"))

