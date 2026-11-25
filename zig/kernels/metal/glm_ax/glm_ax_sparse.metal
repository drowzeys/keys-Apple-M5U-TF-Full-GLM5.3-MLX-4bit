// Full GLM-5.3's sparse MLA for prompt chunks on the tensor units: glm_sparse_nax.metal (GLM-5.3-Flash's, 512-wide
// keys) with the 64-wide rope part added. Keys are the cache's 576-wide rows (512 latent, then 64 roped dims) and the
// queries [rows, 64 heads, 576] the same; a simdgroup takes 128 latent dims and 16 of the rope's, values the latent.
// Exact bf16 products, fp32 sums, probabilities as 3 bf16 parts (exact).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

#ifndef GLM_HEADS // TP2 builds one Mac's 32 heads
#define GLM_HEADS 64
#endif
constant constexpr int HEADS = GLM_HEADS, RANK = 512, KS = 576, GROUP = 16, BLOCK = 32, SLICE = RANK / 4;
constant constexpr float NO_SCORE = -3.4028234663852886e38f;

// A transposed right operand's element e: the row it comes from (here a key) and its column (a latent dim).
#ifndef TF_SIMD_FRAGS
inline short t_key(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short t_dim(short e, short2 home) { return home.x + (e & 3); }
#else
inline short t_key(short e, short2 home) { return home.x + TF_COL(e); }
inline short t_dim(short e, short2 home) { return home.y + (e >> 2) * 8; }
#endif
// A plain operand's or an accumulator's element e: its row (a head, or a key for the values) and column.
inline short p_row(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short p_col(short e, short2 home) { return home.x + TF_COL(e); }

// The max (or sum) of a head's values across the four lanes that hold its other keys.
inline float row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}
inline float row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// This simdgroup's 128 latent dims of 16 heads' queries as eight 16-dim fragments, and its 16 rope dims as a ninth.
inline void sparse_queries(thread frag<bfloat> (&q)[9], const device bfloat* qrow, int d0, uint s, short2 home) {
  TF_UNROLL
  for (short t = 0; t < 8; t++) frag_get(q[t], qrow + d0, KS, 0, 16 * t, home);
  frag_get(q[8], qrow + RANK + 16 * int(s), KS, 0, 0, home);
}

// One row's entries [lo, hi) (an entry outside [0, key_length) is no key): scores, online softmax, values into `acc`.
inline void sparse_pass(thread frag<float> (&acc)[4][2], thread float (&m)[2], thread float (&l)[2], thread const frag<bfloat> (&q)[9],
                        const device bfloat* keys, const device int32_t* list, int lo, int hi, int key_length, float scale,
                        threadgroup float (&partial)[4][32][16], threadgroup int (&keyrow)[BLOCK], uint s, uint lane, short2 home, int d0) {
  for (int j0 = lo; j0 < hi; j0 += BLOCK) {
    if (s == 0) {
      const int j = j0 + int(lane);
      const int k = j < hi ? list[j] : -1;
      keyrow[lane] = k >= 0 && k < key_length ? k : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: this simdgroup's dims
    frag<float> lo_s = frag<float>(0), hi_s = frag<float>(0);
#ifndef TF_SIMD_FRAGS
    // a lane's four dims of each of its four keys, 8 bytes a load
    const device bfloat* kr[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) kr[i] = keys + long(max(keyrow[8 * i + home.y], 0)) * KS + d0 + home.x;
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      const frag<bfloat> b0 = frag<bfloat>(*(const device bfloat4*)(kr[0] + 16 * t), *(const device bfloat4*)(kr[1] + 16 * t));
      const frag<bfloat> b1 = frag<bfloat>(*(const device bfloat4*)(kr[2] + 16 * t), *(const device bfloat4*)(kr[3] + 16 * t));
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
    {
      // the rope part: this simdgroup's 16 of the 64 roped dims
      const int r = RANK + 16 * int(s) - d0;
      const frag<bfloat> b0 = frag<bfloat>(*(const device bfloat4*)(kr[0] + r), *(const device bfloat4*)(kr[1] + r));
      const frag<bfloat> b1 = frag<bfloat>(*(const device bfloat4*)(kr[2] + r), *(const device bfloat4*)(kr[3] + r));
      mma_16x32<false, true>(lo_s, hi_s, q[8], b0, b1);
    }
#else
    TF_UNROLL
    for (short t = 0; t < 8; t++) {
      frag<bfloat> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int k0 = keyrow[t_key(e, home)], k1 = keyrow[16 + t_key(e, home)];
        const int d = d0 + 16 * t + t_dim(e, home);
        b0[e] = keys[long(max(k0, 0)) * KS + d];
        b1[e] = keys[long(max(k1, 0)) * KS + d];
      }
      mma_16x32<false, true>(lo_s, hi_s, q[t], b0, b1);
    }
    {
      frag<bfloat> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int k0 = keyrow[t_key(e, home)], k1 = keyrow[16 + t_key(e, home)];
        const int d = RANK + 16 * int(s) + t_dim(e, home);
        b0[e] = keys[long(max(k0, 0)) * KS + d];
        b1[e] = keys[long(max(k1, 0)) * KS + d];
      }
      mma_16x32<false, true>(lo_s, hi_s, q[8], b0, b1);
    }
#endif
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      partial[s][lane][e] = lo_s[e];
      partial[s][lane][8 + e] = hi_s[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[16];
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const float v = ((partial[0][lane][i] + partial[1][lane][i]) + partial[2][lane][i]) + partial[3][lane][i];
      const short e = i & 7;
      const int key = (i < 8 ? 0 : 16) + p_col(e, home);
      sc[i] = keyrow[key] >= 0 ? v * scale : NO_SCORE;
    }
    // the online softmax, a head at a time (elements 0-3 and 8-11 are this lane's first head, 4-7 and 12-15 its second)
    float factor[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      float bmax = NO_SCORE;
      TF_UNROLL
      for (short i = 0; i < 4; i++) bmax = max(bmax, max(sc[4 * h + i], sc[8 + 4 * h + i]));
      const float new_max = max(m[h], row_max(bmax));
      factor[h] = fast::exp(m[h] - new_max);
      m[h] = new_max;
    }
    frag<bfloat> p1[2], p2[2], p3[2]; // keys 0-15 and 16-31: the probabilities' three bf16 parts
    float bsum[2] = {0.0f, 0.0f};
    TF_UNROLL
    for (short i = 0; i < 16; i++) {
      const short e = i & 7, h = e >> 2, kb = i < 8 ? 0 : 1;
      const float p = (sc[i] > NO_SCORE || isnan(sc[i])) ? fast::exp(sc[i] - m[h]) : 0.0f;
      bsum[h] += p;
      const bfloat a = bfloat(p);
      const float r1 = p - float(a);
      const bfloat b = bfloat(r1);
      p1[kb][e] = a;
      p2[kb][e] = b;
      p3[kb][e] = bfloat(r1 - float(b));
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) l[h] = l[h] * factor[h] + row_sum(bsum[h]);
    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) // a factor of 1 leaves every value as it is
    TF_UNROLL
    for (short c = 0; c < 4; c++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[c][0][e] *= factor[e >> 2];
        acc[c][1][e] *= factor[e >> 2];
      }
    }
    // values: keys 0-15 then 16-31, each part in turn, into this simdgroup's 128 dims
    TF_UNROLL
    for (short kb = 0; kb < 2; kb++) {
#ifndef TF_SIMD_FRAGS
      const device bfloat* va = keys + long(max(keyrow[16 * kb + home.y], 0)) * KS + d0 + home.x;
      const device bfloat* vb = keys + long(max(keyrow[16 * kb + home.y + 8], 0)) * KS + d0 + home.x;
#endif
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        frag<bfloat> v0, v1;
#ifndef TF_SIMD_FRAGS
        v0 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c), *(const device bfloat4*)(vb + 32 * c));
        v1 = frag<bfloat>(*(const device bfloat4*)(va + 32 * c + 16), *(const device bfloat4*)(vb + 32 * c + 16));
#else
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const int k = max(keyrow[16 * kb + p_row(e, home)], 0);
          const int d = d0 + 32 * c + p_col(e, home);
          v0[e] = keys[long(k) * KS + d];
          v1[e] = keys[long(k) * KS + d + 16];
        }
#endif
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          if (keyrow[16 * kb + p_row(e, home)] < 0) v0[e] = v1[e] = bfloat(0);
        }
        mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);
        mma_16x32<false, false>(acc[c][0], acc[c][1], p2[kb], v0, v1);
#ifndef GX_P2
        mma_16x32<false, false>(acc[c][0], acc[c][1], p3[kb], v0, v1);
#endif
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void sparse_clear(thread frag<float> (&acc)[4][2], thread float (&m)[2], thread float (&l)[2]) {
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    acc[c][0] = frag<float>(0);
    acc[c][1] = frag<float>(0);
  }
  m[0] = m[1] = NO_SCORE;
  l[0] = l[1] = 0.0f;
}

// out [rows, 64, 512] over rows' `width` keys (an entry outside [0, key_length) is no key); grid (4 head groups, rows).
// meta: (width, key_length, position of row 0, -); ql [rows, 64, 576].
[[kernel]] void gx_sparse_nax(const device bfloat* ql [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                               const device int32_t* indices [[buffer(2)]], constant float& scale [[buffer(3)]],
                               constant int4& meta [[buffer(4)]], device bfloat* out [[buffer(5)]],
                               uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float partial[4][32][16]; // each simdgroup's scores, by lane (every simdgroup's lanes hold the same cells)
  threadgroup int keyrow[BLOCK];
  const int width = meta.x, key_length = meta.y, pos = meta.z;
  const int row = int(tg.y), h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> q[9];
  sparse_queries(q, ql + (long(row) * HEADS + h0) * KS, d0, s, home);
  frag<float> acc[4][2];
  float m[2], l[2]; // the running max and sum of this lane's two heads
  sparse_clear(acc, m, l);
  // a row at position p lists its keys within its first p + 1 entries
  sparse_pass(acc, m, l, q, keys, indices + long(row) * width, 0, min(width, pos + row + 1), key_length, scale, partial, keyrow, s, lane, home, d0);
  device bfloat* obase = out + (long(row) * HEADS + h0) * RANK + d0;
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> o = acc[c][j];
      TF_UNROLL
      for (short e = 0; e < 8; e++) o[e] = l[e >> 2] == 0.0f ? 0.0f : o[e] / l[e >> 2];
      frag_put(o, obase, RANK, 0, 32 * c + 16 * j, home);
    }
  }
}



// ---------------------------------------------------------------------------------------------------------------
// The DSA indexer's scores for a prompt block on the tensor units: glm_index_nax.metal's M5 kernel (GLM-5.3-Flash's,
// over pooled blocks of 4) with each token's own index key and the causal bound a row each.
// score[r][k] = sum_h (w_h * wscale) relu(q_h . key_k), heads in order, for k <= p0 + r; q roped (gx_index_qrope).
constant constexpr int IX_HEADS = 32, IX_DIM = 128, IX_PAD = IX_DIM + 8;

struct gx_score_args {
  uint p0, q_stride, w_stride, s_stride, rows;
  float wscale;
  uint pad0, pad1;
};

// A fragment of q at column c from a lane's two row starts (8-byte loads on M5, 4-byte before).
inline frag<bfloat> gx_ix_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  return f;
#endif
}

// Grid (64-key tiles, 16-row tiles of the block / simdgroups): a threadgroup of sgs simdgroups, 16 rows each.
[[kernel]] void gx_index_nax(const device bfloat* iq [[buffer(0)]], const device bfloat* iw [[buffer(1)]],
                             const device bfloat* keys [[buffer(2)]], device float* scores [[buffer(3)]],
                             constant gx_score_args& a [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]],
                             uint sgs [[simdgroups_per_threadgroup]]) {
  threadgroup bfloat tile[64 * IX_PAD];
  const int tile_rows = 16 * int(sgs);
  const int b0 = int(tg.x) * 64, r0 = int(tg.y) * tile_rows;
  const int rows = min(tile_rows, int(a.rows) - r0);
  if (rows <= 0) return;
  const int last = int(a.p0) + r0 + rows - 1; // the tile's last row sees keys through its own position
  if (b0 > last) return;
  const uint t = sg * 32 + lane;
  for (uint i = t; i < 64 * (IX_DIM / 8); i += 32 * sgs) { // the 64 keys, 16 bytes a load (past the last: the last)
    const uint rr = i / (IX_DIM / 8), cc = (i % (IX_DIM / 8)) * 8;
    const int b = min(b0 + int(rr), last);
    *(threadgroup uint4*)(tile + rr * IX_PAD + cc) = *(const device uint4*)(keys + size_t(b) * IX_DIM + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int tm = 16 * int(sg);
  const device bfloat* qr[2];
  const device bfloat* wr[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = tm + home.y + 8 * h;
    const long at = long(r0 + (r < rows ? r : 0)); // a dead row reads a live one; its scores are never stored
    qr[h] = iq + at * a.q_stride;
    wr[h] = iw + at * a.w_stride;
  }
  frag<float> total[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) total[j] = frag<float>(0);
  for (int hd = 0; hd < IX_HEADS; hd++) {
    frag<float> s[4];
    TF_UNROLL
    for (short j = 0; j < 4; j++) s[j] = frag<float>(0);
    TF_UNROLL
    for (short q = 0; q < IX_DIM / 16; q++) {
      const frag<bfloat> qa = gx_ix_frag(qr[0] + hd * IX_DIM, qr[1] + hd * IX_DIM, 16 * q, home);
      TF_UNROLL
      for (short n = 0; n < 2; n++) {
        frag<bfloat> k0, k1;
        frag_get_t(k0, (const threadgroup bfloat*)tile, IX_PAD, 32 * n, 16 * q, home);
        frag_get_t(k1, (const threadgroup bfloat*)tile, IX_PAD, 32 * n + 16, 16 * q, home);
        mma_16x32<false, true>(s[2 * n], s[2 * n + 1], qa, k0, k1);
      }
    }
    const float w0 = float(wr[0][hd]) * a.wscale, w1 = float(wr[1][hd]) * a.wscale;
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) total[j][e] = fma(e < 4 ? w0 : w1, metal::max(s[j][e], 0.0f), total[j][e]);
    }
  }
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = tm + home.y + (e >> 2) * 8;
      const int b = b0 + 16 * j + home.x + TF_COL(e);
      if (r < rows && b <= int(a.p0) + r0 + r) scores[size_t(r0 + r) * a.s_stride + b] = total[j][e];
    }
  }
}

// Decode rows on the tensor units (glm_sparse_nax.metal's split and combine, 576-wide keys and queries): block b's
// unnormalized `span` list entries: po [rows,blocks,64,512], max pm, sum pl [rows,blocks,64]; grid (4,blocks,rows).
// meta: (width, key_length, span, position of row 0); a row lists min(width, its position + 1) keys.
[[kernel]] void gx_sparse_split(const device bfloat* ql [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                 const device int32_t* indices [[buffer(2)]], constant float& scale [[buffer(3)]],
                                 constant int4& meta [[buffer(4)]], device float* po [[buffer(5)]], device float* pm [[buffer(6)]],
                                 device float* pl [[buffer(7)]], uint s [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 grid [[threadgroups_per_grid]]) {
  threadgroup float partial[4][32][16];
  threadgroup int keyrow[BLOCK];
  const int key_length = meta.y, span = meta.z;
  const int row = int(tg.z), b = int(tg.y), blocks = int(grid.y), h0 = int(tg.x) * GROUP, d0 = int(s) * SLICE;
  const int width = min(meta.x, meta.w + row + 1);
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> q[9];
  sparse_queries(q, ql + (long(row) * HEADS + h0) * KS, d0, s, home);
  frag<float> acc[4][2];
  float m[2], l[2];
  sparse_clear(acc, m, l);
  const int lo = b * span;
  sparse_pass(acc, m, l, q, keys, indices + long(row) * meta.x, lo, min(width, lo + span), key_length, scale, partial, keyrow, s, lane, home, d0);
  const long at = (long(row) * blocks + b) * HEADS + h0; // this block's first head
  TF_UNROLL
  for (short c = 0; c < 4; c++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) frag_put(acc[c][j], po + at * RANK + d0, RANK, 0, 32 * c + 16 * j, home);
  }
  if (s == 0 && home.x == 0) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      pm[at + home.y + 8 * h] = m[h];
      pl[at + home.y + 8 * h] = l[h];
    }
  }
}

// Decode rows: heads' blocks merged in order, out [rows,64,512] bf16; grid (64 heads, rows), 128 threads x 4 dims.
[[kernel]] void gx_sparse_combine(const device float* po [[buffer(0)]], const device float* pm [[buffer(1)]],
                                   const device float* pl [[buffer(2)]], constant int& blocks [[buffer(3)]],
                                   device bfloat* out [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
                                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int head = int(tg.x), row = int(tg.y);
  const long first = long(row) * blocks * HEADS + head; // block 0's (row, head) entry; block b's is HEADS * b further
  float most = NO_SCORE;
  for (int b = 0; b < blocks; b++) most = max(most, pm[first + long(b) * HEADS]);
  float total = 0.0f;
  float4 o = float4(0.0f);
  for (int b = 0; b < blocks; b++) {
    const long at = first + long(b) * HEADS;
    const float f = fast::exp(pm[at] - most);
    total += pl[at] * f;
    o += *(const device float4*)(po + at * RANK + 4 * t) * f;
  }
  const float4 v = total == 0.0f ? float4(0.0f) : o / total;
  *(device bfloat4*)(out + (long(row) * HEADS + head) * RANK + 4 * t) = bfloat4(v);
}

// gx_sparse_combine with the head's unembed (glm_ax.metal's gx_heads_gemv, 4-bit groups of 64) fused: the merged
// 512 latent dims, rounded to bf16 as the unfused path stores them, stay in threadgroup memory; out [rows, 64, N]
// bf16. Grid (64 heads, rows), 256 threads; W/S/B a [64 * N, 512] affine matrix; meta (blocks, N, -, -).
[[kernel]] void gx_combine_unembed(const device float* po [[buffer(0)]], const device float* pm [[buffer(1)]],
                                   const device float* pl [[buffer(2)]], constant int4& meta [[buffer(3)]],
                                   const device uint* W [[buffer(4)]], const device bfloat* S [[buffer(5)]],
                                   const device bfloat* B [[buffer(6)]], device bfloat* out [[buffer(7)]],
                                   uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float ol[RANK];
  const int head = int(tg.x), row = int(tg.y), blocks = meta.x, N = meta.y;
  if (t < 128) {
    const long first = long(row) * blocks * HEADS + head;
    float most = NO_SCORE;
    for (int b = 0; b < blocks; b++) most = max(most, pm[first + long(b) * HEADS]);
    float total = 0.0f;
    float4 o = float4(0.0f);
    for (int b = 0; b < blocks; b++) {
      const long at = first + long(b) * HEADS;
      const float f = fast::exp(pm[at] - most);
      total += pl[at] * f;
      o += *(const device float4*)(po + at * RANK + 4 * t) * f;
    }
    const float4 v = float4(bfloat4(total == 0.0f ? float4(0.0f) : o / total));
    ol[4 * t] = v.x;
    ol[4 * t + 1] = v.y;
    ol[4 * t + 2] = v.z;
    ol[4 * t + 3] = v.w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // gx_heads_gemv's sums, an output a simdgroup at a time (K = 512: 64 words, two a lane)
  for (int n = int(sg); n < N; n += 8) {
    const ulong r = ulong(head) * N + n;
    const device uint* wr = W + r * (RANK / 8);
    const device bfloat* sr = S + r * (RANK / 64);
    const device bfloat* br = B + r * (RANK / 64);
    float acc = 0.0f;
    for (uint i = 0; i < 2; ++i) {
      const uint wi = lane + 32 * i;
      const uint wv = wr[wi];
      const float sc = float(sr[wi / 8]), bi = float(br[wi / 8]);
      float dot = 0.0f, xs = 0.0f;
      for (uint j = 0; j < 8; ++j) {
        const float xv = ol[wi * 8 + j];
        dot = fma(float((wv >> (4 * j)) & 15u), xv, dot);
        xs += xv;
      }
      acc = fma(sc, dot, fma(bi, xs, acc));
    }
    acc = simd_sum(acc);
    if (lane == 0) out[(long(row) * HEADS + head) * N + n] = bfloat(acc);
  }
}
