// GLM-5.3 (glm_moe_dsa) on MLX affine weights: norms, MLA in latent space over each row's key list, the DSA indexer.
// Rows are bf16 as the affine products read and write them; the residual and the attention sums are fp32.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LAT = 512, ROPE = 64, NOPE = 192, KEY = LAT + ROPE, QK = NOPE + ROPE, QA = 2048;
constant constexpr uint IH = 32, ID = 128, IROPE = 64, TOPK = 2048, IW = 64;

// A row: its position and where its key list starts in `keys` (the top-k writes `nkeys`; shared layers read it).
struct gx_row {
  uint slot, pos, list, nkeys;
};
struct gx_args {
  uint rows, heads, width, stride;
  uint score_stride, in_stride, pad0, pad1;
  float eps, scale, key_scale, head_scale;
};

inline float gx_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Pair i of an interleaved rope at position pos: (x[2i], x[2i+1]) -> out[i], out[half + i] (queries and keys alike).
inline float2 gx_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

// out[r] = bf16(w * x[r] / rms(x[r])) of fp32 rows (a.in_stride apart, 0: a.width), out rows a.stride apart.
kernel void gx_norm(const device float* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                    constant gx_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]],
                    uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* x = R + ulong(r) * (a.in_stride ? a.in_stride : a.width);
  float s = 0.0f;
  for (uint i = t; i < a.width; i += nt) s = fma(x[i], x[i], s);
  s = gx_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < nt / 32; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  device bfloat* y = out + ulong(r) * (a.stride ? a.stride : a.width);
  for (uint i = t; i < a.width; i += nt) y[i] = bfloat(float(w[i]) * (x[i] * rs));
}

// gx_norm of bf16 rows (a.in_stride apart, 0: a.width): the MTP head's hnorm of a normed hidden row.
kernel void gx_norm16(const device bfloat* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                      constant gx_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]],
                      uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const device bfloat* x = R + ulong(r) * (a.in_stride ? a.in_stride : a.width);
  float s = 0.0f;
  for (uint i = t; i < a.width; i += 256) s = fma(float(x[i]), float(x[i]), s);
  s = gx_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  device bfloat* y = out + ulong(r) * (a.stride ? a.stride : a.width);
  for (uint i = t; i < a.width; i += 256) y[i] = bfloat(float(w[i]) * (float(x[i]) * rs));
}

// act[r][c] = bf16(silu(gate) up) from stacked rows gu [rows, 2 * a.width] (gate columns first), in fp32.
kernel void gx_swiglu(const device bfloat* gu [[buffer(0)]], constant gx_args& a [[buffer(1)]],
                      device bfloat* act [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const float g = float(gu[ulong(p.y) * 2 * a.width + p.x]), u = float(gu[ulong(p.y) * 2 * a.width + a.width + p.x]);
  act[ulong(p.y) * a.width + p.x] = bfloat(precise::divide(g, 1.0f + precise::exp(-g)) * u);
}

// One row a threadgroup: R[r] += float(X[r]) (a sublayer's bf16 output), then out[r] = bf16(w * R[r] / rms(R[r])).
kernel void gx_add_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
                        const device bfloat* w [[buffer(2)]], constant gx_args& a [[buffer(3)]],
                        device bfloat* out [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + ulong(r) * a.width;
  const device bfloat* y = X + ulong(r) * a.width;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += nt) {
    const float v = x[i] + float(y[i]);
    x[i] = v;
    s = fma(v, v, s);
  }
  s = gx_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < nt / 32; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nt) out[ulong(r) * a.width + i] = bfloat(float(w[i]) * (x[i] * rs));
}

// One row a threadgroup: R[r] += float(bf16(sum_k wts[r, k] Y[r, k])) + float(shared[r]) (step5's s5_moe_add, the picks
// in pick order, a.heads of them), then out[r] = bf16(w * R[r] / rms(R[r])): the MoE sum and the next layer's norm.
kernel void gx_moe_add_norm(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
                            const device float* wts [[buffer(2)]], const device bfloat* shared [[buffer(3)]],
                            const device bfloat* w [[buffer(4)]], constant gx_args& a [[buffer(5)]],
                            device bfloat* out [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + ulong(r) * a.width;
  const uint k = a.heads;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += nt) {
    float sum = 0.0f;
    for (uint j = 0; j < k; ++j) sum += wts[r * k + j] * float(Y[(ulong(r) * k + j) * a.width + i]);
    const float v = x[i] + float(bfloat(sum)) + float(shared[ulong(r) * a.width + i]);
    x[i] = v;
    s = fma(v, v, s);
  }
  s = gx_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < nt / 32; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nt) out[ulong(r) * a.width + i] = bfloat(float(w[i]) * (x[i] * rs));
}

// Decode rows' per-head 4-bit products (embed_q, unembed_out): y[r][h][n] = W_h[n] . x[r][h], MLX affine 4-bit in
// groups of 64, fp32 sums, bf16 out. A simdgroup takes one output n of one head for every row (its weights read once);
// lanes take whole u32 words (8 codes of one group). Grid (n / 4, heads), 128 threads.
// a.width = K (inputs), a.stride = N (outputs a head), a.in_stride = x's head stride, a.score_stride = y's head stride.
kernel void gx_heads_gemv(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
                          const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                          constant gx_args& a [[buffer(4)]], device bfloat* Y [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.y, n = tg.x * 4 + sg, K = a.width, N = a.stride;
  if (n >= N) return;
  const uint words = K / 8, groups = K / 64;
  const ulong row = ulong(h) * N + n;
  const device uint* wr = W + row * words;
  const device bfloat* sr = S + row * groups;
  const device bfloat* br = B + row * groups;
  // this lane's words (at most 2 for K <= 512), their codes and group tables, read once for every row
  uint wv[2] = {0, 0};
  float sc[2] = {0, 0}, bi[2] = {0, 0};
  for (uint i = 0; i < 2; ++i) {
    const uint wi = lane + 32 * i;
    if (wi < words) { wv[i] = wr[wi]; sc[i] = float(sr[wi / 8]); bi[i] = float(br[wi / 8]); }
  }
  for (uint r = 0; r < a.rows; ++r) {
    const device bfloat* x = X + ulong(r) * a.heads * a.in_stride + ulong(h) * a.in_stride;
    float acc = 0.0f;
    for (uint i = 0; i < 2; ++i) {
      const uint wi = lane + 32 * i;
      if (wi >= words) continue;
      float dot = 0.0f, xs = 0.0f;
      for (uint j = 0; j < 8; ++j) {
        const float xv = float(x[wi * 8 + j]);
        dot = fma(float((wv[i] >> (4 * j)) & 15u), xv, dot);
        xs += xv;
      }
      acc = fma(sc[i], dot, fma(bi[i], xs, acc));
    }
    acc = simd_sum(acc);
    if (lane == 0) Y[ulong(r) * a.heads * a.score_stride + ulong(h) * a.score_stride + n] = bfloat(acc);
  }
}

// gx_heads_gemv for embed_q with the roped 64 fused (absorb in one launch): threadgroups x < N/4 are gx_heads_gemv's;
// the last x column ropes head tg.y's 64 dims for every row, as gx_qrope (X is q [rows, heads, QK], qlat [.., KEY]).
kernel void gx_absorb(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
                      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                      constant gx_args& a [[buffer(4)]], device bfloat* Y [[buffer(5)]],
                      const device gx_row* rows [[buffer(6)]], const device float* inv [[buffer(7)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  const uint h = tg.y, K = a.width, N = a.stride;
  if (tg.x == N / 4) {
    for (uint p = t; p < a.rows * (ROPE / 2); p += 128) {
      const uint r = p / (ROPE / 2), i = p % (ROPE / 2);
      const device bfloat* x = X + (ulong(r) * a.heads + h) * QK + NOPE;
      const float2 o = gx_rope(float(x[2 * i]), float(x[2 * i + 1]), rows[r].pos, inv, i);
      Y[(ulong(r) * a.heads + h) * KEY + LAT + i] = bfloat(o.x);
      Y[(ulong(r) * a.heads + h) * KEY + LAT + ROPE / 2 + i] = bfloat(o.y);
    }
    return;
  }
  const uint n = tg.x * 4 + sg;
  if (n >= N) return;
  const uint words = K / 8, groups = K / 64;
  const ulong row = ulong(h) * N + n;
  const device uint* wr = W + row * words;
  const device bfloat* sr = S + row * groups;
  const device bfloat* br = B + row * groups;
  uint wv[2] = {0, 0};
  float sc[2] = {0, 0}, bi[2] = {0, 0};
  for (uint i = 0; i < 2; ++i) {
    const uint wi = lane + 32 * i;
    if (wi < words) { wv[i] = wr[wi]; sc[i] = float(sr[wi / 8]); bi[i] = float(br[wi / 8]); }
  }
  for (uint r = 0; r < a.rows; ++r) {
    const device bfloat* x = X + ulong(r) * a.heads * a.in_stride + ulong(h) * a.in_stride;
    float acc = 0.0f;
    for (uint i = 0; i < 2; ++i) {
      const uint wi = lane + 32 * i;
      if (wi >= words) continue;
      float dot = 0.0f, xs = 0.0f;
      for (uint j = 0; j < 8; ++j) {
        const float xv = float(x[wi * 8 + j]);
        dot = fma(float((wv[i] >> (4 * j)) & 15u), xv, dot);
        xs += xv;
      }
      acc = fma(sc[i], dot, fma(bi[i], xs, acc));
    }
    acc = simd_sum(acc);
    if (lane == 0) Y[ulong(r) * a.heads * a.score_stride + ulong(h) * a.score_stride + n] = bfloat(acc);
  }
}

// A prompt chunk's routes, one row a threadgroup of a.width (256) threads, an expert each: the top a.heads (8) by
// sigmoid(logit) + bias, ties to the lower id, weights sigmoid(logit) normalized and times a.scale (MLX's noaux_tc
// select with one group). Logits bf16 [rows, experts], as MLX's bf16 router product leaves them.
kernel void gx_route_rows(const device bfloat* L [[buffer(0)]], const device float* bias [[buffer(1)]],
                          constant gx_args& a [[buffer(2)]], device int* pick [[buffer(3)]], device float* wts [[buffer(4)]],
                          uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float best_v[8];
  threadgroup int best_i[8];
  threadgroup int chosen[8];
  threadgroup float orig[8];
  const uint E = a.width, K = a.heads;
  const float logit = float(L[ulong(r) * E + t]);
  const float sgm = precise::divide(1.0f, 1.0f + precise::exp(-logit));
  float sel = sgm + bias[t];
  for (uint k = 0; k < K; ++k) {
    float v = sel;
    int i = int(t);
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_xor(v, d);
      const int oi = simd_shuffle_xor(i, d);
      if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
    }
    if (lane == 0) best_v[sg] = v, best_i[sg] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float bv = best_v[0];
      int bi = best_i[0];
      for (uint g = 1; g < E / 32; ++g)
        if (best_v[g] > bv || (best_v[g] == bv && best_i[g] < bi)) bv = best_v[g], bi = best_i[g];
      chosen[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) == chosen[k]) orig[k] = sgm, sel = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) sum += orig[k];
    for (uint k = 0; k < K; ++k) {
      pick[r * K + k] = chosen[k];
      wts[r * K + k] = precise::divide(orig[k], sum) * a.scale;
    }
  }
}

// R[r] += float(X[r]): a sublayer's bf16 rows into the fp32 residual.
kernel void gx_add(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
                   constant gx_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const ulong i = ulong(p.y) * a.width + p.x;
  R[i] += float(X[i]);
}

// One row a threadgroup: qr = bf16(rms(q_a) w), the cached latent bf16(rms(kv_a[:512]) w), rope key bf16(rope(rest)).
kernel void gx_prep(const device bfloat* qa [[buffer(0)]], const device bfloat* kva [[buffer(1)]],
                    const device bfloat* w_qa [[buffer(2)]], const device bfloat* w_kva [[buffer(3)]],
                    const device float* inv [[buffer(4)]], const device gx_row* rows [[buffer(5)]],
                    constant gx_args& a [[buffer(6)]], device bfloat* qr [[buffer(7)]],
                    device bfloat* cache [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2][8];
  const device bfloat* x = qa + ulong(row) * (a.in_stride ? a.in_stride : QA);
  const device bfloat* k = kva + ulong(row) * (a.in_stride ? a.in_stride : KEY);
  float sq = 0.0f, sk = 0.0f;
  for (uint i = t; i < QA; i += 256) sq = fma(float(x[i]), float(x[i]), sq);
  for (uint i = t; i < LAT; i += 256) sk = fma(float(k[i]), float(k[i]), sk);
  sq = gx_lanes(sq);
  sk = gx_lanes(sk);
  if (lane == 0) part[0][sg] = sq, part[1][sg] = sk;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vq = 0.0f, vk = 0.0f;
  for (uint g = 0; g < 8; ++g) vq += part[0][g], vk += part[1][g];
  const float rq = precise::rsqrt(vq / float(QA) + a.eps), rk = precise::rsqrt(vk / float(LAT) + a.eps);
  for (uint i = t; i < QA; i += 256) qr[ulong(row) * QA + i] = bfloat(float(w_qa[i]) * (float(x[i]) * rq));
  const gx_row r = rows[row];
  device bfloat* key = cache + ulong(r.pos) * KEY;
  for (uint i = t; i < LAT; i += 256) key[i] = bfloat(float(w_kva[i]) * (float(k[i]) * rk));
  if (t < ROPE / 2) {
    const float2 o = gx_rope(float(k[LAT + 2 * t]), float(k[LAT + 2 * t + 1]), r.pos, inv, t);
    key[LAT + t] = bfloat(o.x);
    key[LAT + ROPE / 2 + t] = bfloat(o.y);
  }
}

// Each (row, head)'s roped q_rope into its query row's last 64 dims (bf16 [rows, heads, 576]): q is [rows, heads,
// 256] bf16, a head's last 64 the rope part.
kernel void gx_qrope(const device bfloat* q [[buffer(0)]], const device gx_row* rows [[buffer(1)]],
                     const device float* inv [[buffer(2)]], constant gx_args& a [[buffer(3)]],
                     device bfloat* qlat [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  const uint h = p.x / (ROPE / 2), i = p.x % (ROPE / 2), r = p.y;
  if (h >= a.heads || r >= a.rows) return;
  const device bfloat* x = q + (ulong(r) * a.heads + h) * QK + NOPE;
  const float2 o = gx_rope(float(x[2 * i]), float(x[2 * i + 1]), rows[r].pos, inv, i);
  qlat[(ulong(r) * a.heads + h) * KEY + LAT + i] = bfloat(o.x);
  qlat[(ulong(r) * a.heads + h) * KEY + LAT + ROPE / 2 + i] = bfloat(o.y);
}

// The indexer's queries roped for the tensor-unit scores: iq [rows, 32, 128] bf16 -> iqr, each head's first 64 dims
// as rope pairs (2i, 2i + 1) -> (i, 32 + i), the rest copied. Grid (32 heads x 64, rows).
kernel void gx_index_qrope(const device bfloat* iq [[buffer(0)]], const device gx_row* rows [[buffer(1)]],
                           const device float* inv [[buffer(2)]], constant gx_args& a [[buffer(3)]],
                           device bfloat* iqr [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  const uint h = p.x / 64, i = p.x % 64, r = p.y;
  if (h >= IH || r >= a.rows) return;
  const device bfloat* x = iq + (ulong(r) * IH + h) * ID;
  device bfloat* y = iqr + (ulong(r) * IH + h) * ID;
  if (i < IROPE / 2) {
    const float2 o = gx_rope(float(x[2 * i]), float(x[2 * i + 1]), rows[r].pos, inv, i);
    y[i] = bfloat(o.x);
    y[IROPE / 2 + i] = bfloat(o.y);
  } else {
    const uint d = IROPE + 2 * (i - IROPE / 2);
    y[d] = x[d];
    y[d + 1] = x[d + 1];
  }
}

// Keys a split takes: splits of 128 list entries, widened by 128 until 16 cover the list (fixed by the count alone).
constant constexpr uint SPLITS = 16;
inline uint2 gx_split(uint n, uint s) {
  const uint span = 128 * ((n + 128 * SPLITS - 1) / (128 * SPLITS));
  return uint2(min(s * span, n), min((s + 1) * span, n));
}

// One (row, head) a simdgroup over a split of its keys: online softmax, partial (m, l, o[512]); a lane 16 + 2 dims.
kernel void gx_attend(const device bfloat* qlat [[buffer(0)]], const device float* unused [[buffer(1)]],
                      const device gx_row* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                      const device bfloat* cache [[buffer(4)]], constant gx_args& a [[buffer(5)]],
                      device float* partial [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.z, s = tg.y, h = tg.x * 4 + sg;
  if (h >= a.heads) return;
  const gx_row row = rows[r];
  const uint2 span = gx_split(row.nkeys, s);
  device float* out = partial + ((ulong(r) * a.heads + h) * SPLITS + s) * (LAT + 2);
  if (span.x >= span.y) return;
  float qv[16], qr[2], o[16], m = -INFINITY, l = 0.0f;
  const device bfloat* qh = qlat + (ulong(r) * a.heads + h) * KEY;
  for (uint i = 0; i < 16; ++i) qv[i] = float(qh[16 * lane + i]), o[i] = 0.0f;
  for (uint i = 0; i < 2; ++i) qr[i] = float(qh[LAT + 2 * lane + i]);
  for (uint j = span.x; j < span.y; ++j) {
    const device bfloat* key = cache + ulong(keys[row.list + j]) * KEY;
    float p = 0.0f;
    for (uint i = 0; i < 16; ++i) p = fma(qv[i], float(key[16 * lane + i]), p);
    for (uint i = 0; i < 2; ++i) p = fma(qr[i], float(key[LAT + 2 * lane + i]), p);
    const float sc = gx_lanes(p) * a.scale;
    const float mn = max(m, sc), corr = precise::exp(m - mn), e = precise::exp(sc - mn);
    l = fma(l, corr, e);
    for (uint i = 0; i < 16; ++i) o[i] = fma(e, float(key[16 * lane + i]), o[i] * corr);
    m = mn;
  }
  if (lane == 0) out[0] = m, out[1] = l;
  for (uint i = 0; i < 16; ++i) out[2 + 16 * lane + i] = o[i];
}

// A (row, head)'s splits merged in order: o = sum o_s exp(m_s - M) / sum l_s exp(m_s - M), bf16 latent.
kernel void gx_merge(const device float* partial [[buffer(0)]], const device gx_row* rows [[buffer(1)]],
                     constant gx_args& a [[buffer(2)]], device bfloat* olat [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, h = tg.x, n = rows[r].nkeys;
  const device float* p = partial + (ulong(r) * a.heads + h) * SPLITS * (LAT + 2);
  float M = -INFINITY;
  for (uint s = 0; s < SPLITS; ++s)
    if (gx_split(n, s).x < gx_split(n, s).y) M = max(M, p[s * (LAT + 2)]);
  float L = 0.0f, acc[2] = {0.0f, 0.0f};
  for (uint s = 0; s < SPLITS; ++s) {
    if (gx_split(n, s).x >= gx_split(n, s).y) continue;
    const float f = precise::exp(p[s * (LAT + 2)] - M);
    L = fma(p[s * (LAT + 2) + 1], f, L);
    for (uint i = 0; i < 2; ++i) acc[i] = fma(p[s * (LAT + 2) + 2 + 2 * t + i], f, acc[i]);
  }
  for (uint i = 0; i < 2; ++i) olat[(ulong(r) * a.heads + h) * LAT + 2 * t + i] = bfloat(precise::divide(acc[i], L));
}

// One row a simdgroup: the index key = bf16(rope_64(LayerNorm(wk x))) into the index cache at the row's position.
kernel void gx_index_key(const device bfloat* ik [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                         const device bfloat* b [[buffer(2)]], const device float* inv [[buffer(3)]],
                         const device gx_row* rows [[buffer(4)]], constant gx_args& a [[buffer(5)]],
                         device bfloat* cache [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const device bfloat* x = ik + ulong(row) * (a.in_stride ? a.in_stride : ID);
  float v[4], s = 0.0f;
  for (uint q = 0; q < 4; ++q) v[q] = float(x[lane + 32 * q]), s += v[q];
  const float mean = gx_lanes(s) / float(ID);
  float d = 0.0f;
  for (uint q = 0; q < 4; ++q) d = fma(v[q] - mean, v[q] - mean, d);
  const float rstd = precise::rsqrt(gx_lanes(d) / float(ID) + a.eps);
  threadgroup float y[ID];
  for (uint q = 0; q < 4; ++q) {
    const uint i = lane + 32 * q;
    y[i] = fma((v[q] - mean) * rstd, float(w[i]), float(b[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const gx_row r = rows[row];
  device bfloat* key = cache + ulong(r.pos) * ID;
  const float2 o = gx_rope(y[2 * lane], y[2 * lane + 1], r.pos, inv, lane);
  key[lane] = bfloat(o.x);
  key[IROPE / 2 + lane] = bfloat(o.y);
  key[IROPE + lane] = bfloat(y[IROPE + lane]);
  key[IROPE + 32 + lane] = bfloat(y[IROPE + 32 + lane]);
}

// The keys a thread scores at once: the query is read once for the block, not once a key.
constant constexpr uint BLOCK = 2;
constant constexpr uint THREADS = 128;

// Row r, keys 256 tg.x ..: score = sum_h (w_h / sqrt 32) relu(q_h . k / sqrt 128), heads in order, q roped here.
// A threadgroup scores THREADS * BLOCK = 256 keys: the dispatch is (last_pos / 256 + 1, rows).
kernel void gx_index_scores(const device bfloat* q [[buffer(0)]], const device bfloat* iw [[buffer(1)]],
                            const device bfloat* cache [[buffer(2)]], const device gx_row* rows [[buffer(3)]],
                            constant gx_args& a [[buffer(4)]], device float* score [[buffer(5)]],
                            const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  threadgroup float qs[IH * ID];
  threadgroup float ws[IH];
  const uint r = tg.y;
  const gx_row row = rows[r];
  if (row.pos < TOPK) return; // every visible key is kept: the top-k lists them without scores
  const device bfloat* qr = q + ulong(r) * (IH * ID);
  for (uint i = t; i < IH * ID; i += THREADS) {
    const uint h = i / ID, c = i % ID;
    if (c >= IROPE) {
      qs[i] = float(qr[i]);
    } else if (c < IROPE / 2) {
      const float2 o = gx_rope(float(qr[h * ID + 2 * c]), float(qr[h * ID + 2 * c + 1]), row.pos, inv, c);
      qs[h * ID + c] = o.x;
      qs[h * ID + IROPE / 2 + c] = o.y;
    }
  }
  if (t < IH) ws[t] = float(iw[ulong(r) * (a.in_stride ? a.in_stride : IW) + t]) * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kbase = tg.x * (THREADS * BLOCK) + t;
  float d[BLOCK][IH];
  for (uint b = 0; b < BLOCK; ++b)
    for (uint h = 0; h < IH; ++h) d[b][h] = 0.0f;
  float kv[BLOCK];
  const device bfloat* kb[BLOCK];
  uint kk[BLOCK];
  for (uint b = 0; b < BLOCK; ++b) {
    kk[b] = kbase + b * THREADS;
    kb[b] = cache + ulong(min(kk[b], row.pos)) * ID;
  }
  for (uint i = 0; i < ID; ++i) {
    for (uint b = 0; b < BLOCK; ++b) kv[b] = kk[b] <= row.pos ? float(kb[b][i]) : 0.0f;
    for (uint h = 0; h < IH; ++h) {
      const float qv = qs[h * ID + i];
      for (uint b = 0; b < BLOCK; ++b) d[b][h] = fma(qv, kv[b], d[b][h]);
    }
  }
  for (uint b = 0; b < BLOCK; ++b) {
    const uint k = kk[b];
    if (k > row.pos) continue;
    float s = 0.0f;
    for (uint h = 0; h < IH; ++h) s = fma(ws[h], max(d[b][h] * a.key_scale, 0.0f), s);
    score[ulong(r) * a.score_stride + k] = s;
  }
}

// fp32 as an unsigned key whose order is the floats' order.
inline uint gx_order(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One row a threadgroup of 1024: its top 2048 visible keys (ties to the lower position), listed by position.
kernel void gx_index_topk(const device float* score [[buffer(0)]], device gx_row* rows [[buffer(1)]],
                          constant gx_args& a [[buffer(2)]], device uint* list [[buffer(3)]],
                          uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint found[2];
  threadgroup uint sums[2][32];
  const gx_row row = rows[r];
  const uint n = row.pos + 1;
  device uint* out = list + row.list;
  const device float* sc = score + ulong(r) * a.score_stride;
  if (n <= TOPK) {
    for (uint i = t; i < n; i += 1024) out[i] = i;
    if (t == 0) rows[r].nkeys = n;
    return;
  }
  uint prefix = 0, mask = 0, need = TOPK;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 1024) {
      const uint k = gx_order(sc[i]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint above = 0, b = 255;
      for (;; --b) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (above + c >= need || b == 0) break;
        above += c;
      }
      found[0] = b;
      found[1] = need - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= found[0] << shift;
    mask |= 255u << shift;
    need = found[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // keys above the threshold all go; `need` of those equal to it, the lowest positions first
  uint gt_seen = 0, eq_seen = 0;
  for (uint base = 0; base < n; base += 1024) {
    const uint i = base + t;
    const uint k = i < n ? gx_order(sc[i]) : 0u;
    const uint gt = (i < n && k > prefix) ? 1u : 0u, eq = (i < n && k == prefix) ? 1u : 0u;
    const uint gx = simd_prefix_exclusive_sum(gt), ex = simd_prefix_exclusive_sum(eq);
    if (lane == 31) sums[0][sg] = gx + gt, sums[1][sg] = ex + eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gb = 0, eb = 0, gall = 0, eall = 0;
    for (uint g = 0; g < 32; ++g) {
      if (g < sg) gb += sums[0][g], eb += sums[1][g];
      gall += sums[0][g], eall += sums[1][g];
    }
    const uint eq_before = eq_seen + eb + ex;
    const uint take = gt | ((eq && eq_before < need) ? 1u : 0u);
    if (take) out[gt_seen + gb + gx + min(eq_before, need)] = i;
    gt_seen += gall;
    eq_seen += eall;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) rows[r].nkeys = TOPK;
}

// ---------------------------------------------------------------------------------------------------------------
// Decode MoE for 1-4 rows, all nine experts of a row (its eight picks and the shared one) per launch.
// MLX affine 4-bit, groups of 64, K = 6144 inputs for gate/up, 2048 for down; weights read in 16-byte loads.

constant constexpr uint MK = 6144, MI = 2048, MG = MK / 64, MW = MK / 32; // MW: uint4 loads a gate/up row
constant constexpr uint MOE_SG = 8; // simdgroups a threadgroup, one output each

struct gx_moe_args {
  uint rows, topk, slots, pad0; // slots = topk + 1 a row (the shared expert last)
};

// x . row for one 4-bit row (uint4 loads strided by lane), its group scales and biases, x and its group sums staged.
inline float gx_row_dot(const device uint4* w, const device bfloat* sc, const device bfloat* bi, threadgroup const float* xs,
                        threadgroup const float* gsum, uint loads, uint lane) {
  float acc = 0.0f;
  for (uint i = lane; i < loads; i += 32) {
    const uint4 q = w[i];
    const uint g = i / 2; // two 16-byte loads (32 codes each) a group of 64
    const threadgroup float* x = xs + i * 32;
    float d = 0.0f;
    const uint words[4] = {q.x, q.y, q.z, q.w};
    for (uint wd = 0; wd < 4; ++wd)
      for (uint j = 0; j < 8; ++j) d = fma(float((words[wd] >> (4 * j)) & 15u), x[wd * 8 + j], d);
    acc = fma(float(sc[g]), d, acc);
    if ((i & 1u) == 0) acc = fma(float(bi[g]), gsum[g], acc);
  }
  return simd_sum(acc);
}

// act[r * slots + k][n] = bf16(silu(bf16(gate . x_r)) * bf16(up . x_r)) for expert pick[r][k] (k = topk: the shared
// one, its gate rows then up rows stacked in sgu). Grid (MI / MOE_SG, rows * slots), MOE_SG simdgroups.
kernel void gx_moe_gu(const device bfloat* X [[buffer(0)]], const device int* pick [[buffer(1)]],
                      const device uint4* gw [[buffer(2)]], const device bfloat* gs [[buffer(3)]], const device bfloat* gb [[buffer(4)]],
                      const device uint4* uw [[buffer(5)]], const device bfloat* us [[buffer(6)]], const device bfloat* ub [[buffer(7)]],
                      const device uint4* sw [[buffer(8)]], const device bfloat* ss [[buffer(9)]], const device bfloat* sb [[buffer(10)]],
                      constant gx_moe_args& a [[buffer(11)]], device bfloat* act [[buffer(12)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float xs[MK];
  threadgroup float gsum[MG];
  const uint slot = tg.y, r = slot / a.slots, k = slot % a.slots;
  const device bfloat* x = X + ulong(r) * MK;
  for (uint i = t; i < MK; i += 32 * MOE_SG) xs[i] = float(x[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = t; g < MG; g += 32 * MOE_SG) {
    float s = 0.0f;
    for (uint j = 0; j < 64; ++j) s += xs[g * 64 + j];
    gsum[g] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = tg.x * MOE_SG + sg;
  float g, u;
  if (k < a.topk) {
    const ulong e = ulong(pick[r * a.topk + k]);
    const ulong row = e * MI + n;
    g = gx_row_dot(gw + row * MW, gs + row * MG, gb + row * MG, xs, gsum, MW, lane);
    u = gx_row_dot(uw + row * MW, us + row * MG, ub + row * MG, xs, gsum, MW, lane);
  } else {
    g = gx_row_dot(sw + ulong(n) * MW, ss + ulong(n) * MG, sb + ulong(n) * MG, xs, gsum, MW, lane);
    u = gx_row_dot(sw + ulong(MI + n) * MW, ss + ulong(MI + n) * MG, sb + ulong(MI + n) * MG, xs, gsum, MW, lane);
  }
  if (lane == 0) {
    const float gr = float(bfloat(g)), ur = float(bfloat(u));
    act[ulong(slot) * MI + n] = bfloat(precise::divide(gr, 1.0f + precise::exp(-gr)) * ur);
  }
}

constant constexpr uint DK = MI, DG = DK / 64, DW = DK / 32;

// y[r * slots + k][n] = bf16(down_e[n] . act[r * slots + k]) for each of a row's nine experts (the shared one's
// down in sd). Grid (MK / MOE_SG, rows * slots), MOE_SG simdgroups.
kernel void gx_moe_down(const device bfloat* act [[buffer(0)]], const device int* pick [[buffer(1)]],
                        const device uint4* dw [[buffer(2)]], const device bfloat* ds [[buffer(3)]], const device bfloat* db [[buffer(4)]],
                        const device uint4* sw [[buffer(5)]], const device bfloat* ss [[buffer(6)]], const device bfloat* sb [[buffer(7)]],
                        constant gx_moe_args& a [[buffer(8)]], device bfloat* Y [[buffer(9)]], device bfloat* shared_y [[buffer(10)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float xs[DK];
  threadgroup float gsum[DG];
  const uint slot = tg.y, r = slot / a.slots, k = slot % a.slots;
  const device bfloat* x = act + ulong(slot) * DK;
  for (uint i = t; i < DK; i += 32 * MOE_SG) xs[i] = float(x[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = t; g < DG; g += 32 * MOE_SG) {
    float s = 0.0f;
    for (uint j = 0; j < 64; ++j) s += xs[g * 64 + j];
    gsum[g] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = tg.x * MOE_SG + sg;
  float y;
  if (k < a.topk) {
    const ulong row = ulong(pick[r * a.topk + k]) * MK + n;
    y = gx_row_dot(dw + row * DW, ds + row * DG, db + row * DG, xs, gsum, DW, lane);
  } else {
    y = gx_row_dot(sw + ulong(n) * DW, ss + ulong(n) * DG, sb + ulong(n) * DG, xs, gsum, DW, lane);
  }
  if (lane == 0) {
    if (k < a.topk) Y[(ulong(r) * a.topk + k) * MK + n] = bfloat(y);
    else shared_y[ulong(r) * MK + n] = bfloat(y);
  }
}

// A plain streaming read for the bench's reference: threadgroup g reads its a.width uint4s (16 B each) from x, eight
// loads in flight a thread, and leaves one word so the reads stay. Grid (groups), 256 threads.
kernel void gx_stream(const device uint4* x [[buffer(0)]], constant gx_args& a [[buffer(1)]], device uint* out [[buffer(2)]],
                      uint g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const device uint4* p = x + ulong(g) * a.width;
  uint4 acc = uint4(0);
  for (uint i = t; i < a.width; i += 256 * 8) {
    uint4 v[8];
    for (uint j = 0; j < 8; ++j) v[j] = (i + 256 * j < a.width) ? p[i + 256 * j] : uint4(0);
    for (uint j = 0; j < 8; ++j) acc ^= v[j];
  }
  const uint w = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (w == 0x9e3779b9u) out[g] = w; // practically never: the compiler cannot drop the loads
}
