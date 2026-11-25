// Prompt chunks' sparse MLA on the tensor units, by MetalPerformancePrimitives' cooperative tensors. The method is
// oMLX 0.8.0rc1's tensor-unit sparse MLA attention (omlx/patches/glm_moe_dsa/sparse_mla_nax.py, Apache-2.0) as
// Splash ports it with GLM-5.3's rotary term (runtime/metal/kernels/shared/listed_attention.metal,
// attention_bf16_listed_kv1_g64, Apache-2.0): a threadgroup a (row, half of the 64 heads), its eight simdgroups two
// groups of 16 heads by four quarters of a 128-key tile; each simdgroup scores its 16 heads against its quarter's 32
// keys over all 576 values (no reduction across simdgroups), the quarters exchange their rows' maxima, the
// probabilities go to threadgroup memory as two fp16 pieces, and each quarter multiplies them by its quarter of the
// 512 latent values. Here keys are glm_ax's cache rows (512 latent, then 64 roped) read in place through the row's
// list, queries [rows, 64, 576] the same order, out [rows, 64, 512] bf16.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int GM_D = 576, GM_RANK = 512, GM_HEADS = 64, GM_BK = 128;

// The caches' rows (GLM_AX_K8=1 builds these with GX_K8): the latent cache's are bf16 [576] (512 latent, 64 roped),
// or 8-bit codes [576] then nine half scales (the eight 64-wide latent groups, the roped 64), 608 bytes; the index
// cache's are bf16 [128], or 8-bit codes [128] then two half scales (its two 64-wide halves), 136 bytes. Codes are
// round(v / scale), scale = the group's largest |v| / 127 (as a half).
#ifndef GX_KACCESS
#define GX_KACCESS
// GX_K8: the latent cache 8-bit; GX_K8I: the index cache 8-bit (each on its own, GLM_AX_K8=lat / idx / 1 for both)
#ifdef GX_K8
typedef uchar gx_kt;
constant constexpr long GX_KROW = 608;
inline float gx_ks(const device gx_kt* row, int d) { return float(((const device half*)(row + 576))[d >> 6]); }
inline bfloat gx_kget(const device gx_kt* keys, long r, int d) {
  const device gx_kt* row = keys + r * GX_KROW;
  return bfloat(float(as_type<char>(row[d])) * gx_ks(row, d));
}
inline bfloat4 gx_kget4(const device gx_kt* keys, long r, int d) {
  const device gx_kt* row = keys + r * GX_KROW;
  return bfloat4(float4(*(const device char4*)(row + d)) * gx_ks(row, d));
}
#else
typedef bfloat gx_kt;
constant constexpr long GX_KROW = 576;
inline bfloat gx_kget(const device gx_kt* keys, long r, int d) { return keys[r * 576 + d]; }
inline bfloat4 gx_kget4(const device gx_kt* keys, long r, int d) { return *(const device bfloat4*)(keys + r * 576 + d); }
#endif
#ifdef GX_K8I
typedef uchar gx_it;
constant constexpr long GX_IROW = 136;
inline float gx_iget(const device gx_it* row, uint i) { return float(as_type<char>(row[i])) * float(((const device half*)(row + 128))[i >> 6]); }
#else
typedef bfloat gx_it;
constant constexpr long GX_IROW = 128;
inline float gx_iget(const device gx_it* row, uint i) { return float(row[i]); }
#endif
// A group's 8-bit scale from its largest |v| (a half), and v's code under it.
inline half gx_qscale(float amax) { return half(amax / 127.0f); }
inline char gx_qcode(float v, half s) { return float(s) > 0.0f ? char(clamp(rint(v / float(s)), -127.0f, 127.0f)) : char(0); }
#endif

// meta: (list width, key length, position of row 0, -): row r lists min(width, its position + 1) keys at
// indices + r * width, entries outside [0, key length) no key. Grid (2 halves, rows), 256 threads.
[[kernel]] void gx_sparse_mpp(device const bfloat *queries [[buffer(0)]], device const gx_kt *keys [[buffer(1)]],
                              device const int *indices [[buffer(2)]], constant float &scale [[buffer(3)]],
                              constant int4 &meta [[buffer(4)]], device bfloat *out [[buffer(5)]],
                              uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int key_at[2][GM_BK];
  threadgroup float red_max[2][4][16];
  threadgroup float red_sum[2][4][16];
  threadgroup half p_hi[2][GM_BK / 16][32 * 8];
  threadgroup half p_lo[2][GM_BK / 16][32 * 8];

  const int row = int(group.y);
  const uint half_heads = group.x;
  const int pos = meta.z + row;
  const int count = min(meta.x, pos + 1);
  const int key_length = meta.y;
  device const int *list = indices + long(row) * meta.x;
  const int tiles = (count + GM_BK - 1) / GM_BK;

  const float scale_log2 = scale * 1.44269504088896341f;
  const uint tid = sg * 32 + lane;
  const int hg = int(sg) / 4;
  const int j4 = int(sg) % 4;
  const int head0 = int(half_heads) * 32 + hg * 16;
  const short qid = short(lane >> 2);
  const short fm = short((qid & 4) | ((lane >> 1) & 3));
  const short fn = short(((qid & 2) | (lane & 1)) * 4);
  const bool row_writer = (lane & 9u) == 0u;

  constexpr auto qk_desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroup> qk_op;
  constexpr auto pv_desc = matmul2d_descriptor(16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroup> pv_op;

  auto pa = pv_op.get_left_input_cooperative_tensor<half, bfloat, float>();
  auto pm = pv_op.get_left_input_cooperative_tensor<half, bfloat, float>();
  auto pb = pv_op.get_right_input_cooperative_tensor<half, bfloat, float>();
  auto o0 = pv_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
  auto o1 = pv_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
  auto o2 = pv_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
  auto o3 = pv_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
  for (short e = 0; e < 16; ++e) {
    o0[e] = 0.0f;
    o1[e] = 0.0f;
    o2[e] = 0.0f;
    o3[e] = 0.0f;
  }
  float m_run[2] = {-FLT_MAX, -FLT_MAX};
  float l_run[2] = {0.0f, 0.0f};

  device const bfloat *query = queries + long(row) * GM_HEADS * GM_D;
  device const bfloat *qr0 = query + long(head0 + fm) * GM_D + fn;
  device const bfloat *qr1 = query + long(head0 + fm + 8) * GM_D + fn;

  // tile t's keys, a thread a key: its cache row, or -1 where the list holds none
  const auto stage = [&](int t, int b) {
    if (tid < uint(GM_BK)) {
      const int at = t * GM_BK + int(tid);
      const int k = at < count ? list[at] : -1;
      key_at[b][tid] = (k >= 0 && k < key_length) ? k : -1;
    }
  };

  if (tiles > 0) stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = 0; t < tiles; ++t) {
    const int buf = t & 1;
    const int tile_keys = min(GM_BK, count - t * GM_BK);

    auto qa = qk_op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb = qk_op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto sc = qk_op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
    for (short e = 0; e < 16; ++e) sc[e] = 0.0f;
    if (j4 * 32 < tile_keys) {
      long ks[2][2];
      for (short tn = 0; tn < 2; ++tn)
        for (short i = 0; i < 2; ++i) {
          const int k = key_at[buf][j4 * 32 + tn * 16 + fm + i * 8];
          ks[tn][i] = long(max(k, 0));
        }
      _Pragma("clang loop unroll_count(4)")
      for (short kk = 0; kk < GM_D; kk += 16) {
        for (short j = 0; j < 4; ++j) {
          qa[j] = qr0[kk + j];
          qa[4 + j] = qr1[kk + j];
        }
        for (short tn = 0; tn < 2; ++tn)
          for (short i = 0; i < 2; ++i)
          {
            const bfloat4 kv4 = gx_kget4(keys, ks[tn][i], fn + kk);
            for (short j = 0; j < 4; ++j) kb[tn * 8 + i * 4 + j] = kv4[j];
          }
        qk_op.run(qa, kb, sc);
      }
    }
    bool ok[2][4];
    for (short tn = 0; tn < 2; ++tn)
      for (short j = 0; j < 4; ++j) ok[tn][j] = key_at[buf][j4 * 32 + tn * 16 + fn + j] >= 0 && j4 * 32 + tn * 16 + fn + j < tile_keys;
    float pmx[2];
    for (short i = 0; i < 2; ++i) {
      float m = -FLT_MAX;
      for (short tn = 0; tn < 2; ++tn)
        for (short j = 0; j < 4; ++j) m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
      m = max(m, simd_shuffle_xor(m, ushort(1)));
      m = max(m, simd_shuffle_xor(m, ushort(8)));
      pmx[i] = m;
    }
    if (row_writer) {
      red_max[hg][j4][fm] = pmx[0];
      red_max[hg][j4][fm + 8] = pmx[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t + 1 < tiles) stage(t + 1, buf ^ 1);

    float factor[2];
    float mnew[2];
    for (short i = 0; i < 2; ++i) {
      const int r = fm + i * 8;
      const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]), max(red_max[hg][2][r], red_max[hg][3][r]));
      const float candidate = m == -FLT_MAX ? -FLT_MAX : m * scale_log2;
      mnew[i] = max(m_run[i], candidate);
      factor[i] = fast::exp2(m_run[i] - mnew[i]);
      m_run[i] = mnew[i];
    }
    float rs[2] = {0.0f, 0.0f};
    for (short tn = 0; tn < 2; ++tn) {
      vec<half, 8> h, lo;
      for (short i = 0; i < 2; ++i)
        for (short j = 0; j < 4; ++j) {
          const float e = ok[tn][j] ? fast::exp2(float(sc[tn * 8 + i * 4 + j]) * scale_log2 - mnew[i]) : 0.0f;
          const half hi = half(e);
          h[i * 4 + j] = hi;
          lo[i * 4 + j] = half(e - float(hi));
          rs[i] += e;
        }
      const int step = j4 * 2 + tn;
      *reinterpret_cast<threadgroup vec<half, 8> *>(&p_hi[hg][step][lane * 8]) = h;
      *reinterpret_cast<threadgroup vec<half, 8> *>(&p_lo[hg][step][lane * 8]) = lo;
    }
    for (short i = 0; i < 2; ++i) {
      rs[i] += simd_shuffle_xor(rs[i], ushort(1));
      rs[i] += simd_shuffle_xor(rs[i], ushort(8));
    }
    if (row_writer) {
      red_sum[hg][j4][fm] = rs[0];
      red_sum[hg][j4][fm + 8] = rs[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
      for (short e = 0; e < 16; ++e) {
        const float f = factor[(e >> 2) & 1];
        o0[e] *= f;
        o1[e] *= f;
        o2[e] *= f;
        o3[e] *= f;
      }
    }
    for (short i = 0; i < 2; ++i) {
      const int r = fm + i * 8;
      const float sum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
      l_run[i] = l_run[i] * factor[i] + sum;
    }
    const int steps = (tile_keys + 15) / 16;
    for (short step = 0; step < steps; ++step) {
      const vec<half, 8> h = *reinterpret_cast<const threadgroup vec<half, 8> *>(&p_hi[hg][step][lane * 8]);
      const vec<half, 8> lo = *reinterpret_cast<const threadgroup vec<half, 8> *>(&p_lo[hg][step][lane * 8]);
      for (short e = 0; e < 8; ++e) {
        pa[e] = h[e];
        pm[e] = lo[e];
      }
      const int k0 = max(key_at[buf][step * 16 + fm], 0), k1 = max(key_at[buf][step * 16 + fm + 8], 0);

      for (short np = 0; np < 4; ++np) {
        for (short tn = 0; tn < 2; ++tn) {
          const int at = j4 * 128 + np * 32 + tn * 16 + fn;
          const bfloat4 a4 = gx_kget4(keys, long(k0), at), b4 = gx_kget4(keys, long(k1), at);
          for (short j = 0; j < 4; ++j) {
            pb[tn * 8 + j] = a4[j];
            pb[tn * 8 + 4 + j] = b4[j];
          }
        }
        if (np == 0) {
          pv_op.run(pa, pb, o0);
          pv_op.run(pm, pb, o0);
        } else if (np == 1) {
          pv_op.run(pa, pb, o1);
          pv_op.run(pm, pb, o1);
        } else if (np == 2) {
          pv_op.run(pa, pb, o2);
          pv_op.run(pm, pb, o2);
        } else {
          pv_op.run(pa, pb, o3);
          pv_op.run(pm, pb, o3);
        }
      }
    }
  }

  for (short i = 0; i < 2; ++i) {
    device bfloat *o = out + (long(row) * GM_HEADS + head0 + fm + i * 8) * GM_RANK + j4 * 128 + fn;
    const float denominator = l_run[i] > 0.0f ? l_run[i] : 1.0f;
    for (short tn = 0; tn < 2; ++tn)
      for (short j = 0; j < 4; ++j) {
        const short e = tn * 8 + i * 4 + j;
        o[tn * 16 + j] = bfloat(o0[e] / denominator);
        o[32 + tn * 16 + j] = bfloat(o1[e] / denominator);
        o[64 + tn * 16 + j] = bfloat(o2[e] / denominator);
        o[96 + tn * 16 + j] = bfloat(o3[e] / denominator);
      }
  }
}
