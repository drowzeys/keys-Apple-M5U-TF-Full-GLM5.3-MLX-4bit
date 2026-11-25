// incoai's DFlash2 drafter for GLM-5.3 (SGLang's DFlash2DraftModel) and the target's taps for it. The drafter's
// kernels are TensorFold's MiniMax-M3 ones (kernels/metal/minimax_m3/dflash2.metal: df_norm, df_norm16, df_prep,
// df_conv, df_silu, df_topk; dspark.metal: mm_tap, mm_draft_vec; glue.metal: mm_embed, mm_merge_vec), renamed gd_*;
// gd_walk takes the candidate row's stride apart from the selector's top (GLM's drafter keeps 16 of the 32).
#include <metal_stdlib>
using namespace metal;

struct gd_norm_args { uint rows, width; float eps; uint add; };

inline void gd_norm_row(device float* x, const device bfloat* w, uint width, float eps, device bfloat* out,
    uint t, uint n, threadgroup float* part) {
  float sum = 0.0f;
  for (uint i = t; i < width; i += n) sum = fma(x[i], x[i], sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(width) + eps);
  for (uint i = t; i < width; i += n) out[i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv)));
}

// Qwen3's RMSNorm: bf16(w * bf16(x * rsqrt(mean(x^2) + eps))). With a.add the residual first takes bf16(R + X).
[[kernel]] void gd_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant gd_norm_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  if (a.add)
    for (uint i = t; i < a.width; i += n) x[i] = float(bfloat(x[i] + float(X[(size_t)row * a.width + i])));
  threadgroup_barrier(mem_flags::mem_device);
  gd_norm_row(x, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part);
}

// The same norm of bf16 rows (fc's output into the context's hidden_norm).
[[kernel]] void gd_norm16(const device bfloat* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant gd_norm_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device bfloat* x = X + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) sum = fma(float(x[i]), float(x[i]), sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)));
}

struct gd_prep_args { uint rows, heads, kv_heads, pos0, stride, q_off, k_off, v_off, ring; float eps, theta; };

// Qwen3's head norm and every dim rotated (neox halves): q heads to Q, k (and v) into the ring at position % ring.
[[kernel]] void gd_prep(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant gd_prep_args& a [[buffer(3)]], device bfloat* Q [[buffer(4)]],
    device bfloat* K [[buffer(5)]], device bfloat* V [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? a.q_off : a.k_off) + (size_t)head * 128;
  const device bfloat* w = is_q ? qw : kw;
  const uint p = a.pos0 + row;
  const size_t slot = p % a.ring;
  device bfloat* y = is_q ? Q + ((size_t)row * a.heads + head) * 128 : K + (slot * a.kv_heads + head) * 128;
  float v[4];
  float sum = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / 128.0f + a.eps);
  threadgroup float nrm[128];
  for (uint j = 0; j < 4; ++j) nrm[lane + 32 * j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 4; ++j) {
    const uint d = lane + 32 * j;
    const uint i = d % 64;
    const float angle = float(p) * (1.0f / precise::pow(a.theta, float(2 * i) / 128.0f));
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float rot = d < 64 ? -nrm[d + 64] : nrm[d - 64];
    y[d] = bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s)));
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + a.v_off + (size_t)head * 128;
  device bfloat* vd = V + (slot * a.kv_heads + head) * 128;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

struct gd_conv_args { uint rows, width, block, groups, side; };

// The grouped two-tap block convolution: Y = bf16(bf16(c_0 x[r]) + bf16(c_1 x[r - 1])), c_t = base + delta by group.
[[kernel]] void gd_conv(const device bfloat* X [[buffer(0)]], const device bfloat* D [[buffer(1)]],
    const device bfloat* base [[buffer(2)]], constant gd_conv_args& a [[buffer(3)]], device bfloat* Y [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint ch = p.x, r = p.y, g = ch / (a.width / a.groups);
  const device bfloat* d = D + (size_t)r * 4 * a.groups + a.side * 2 * a.groups;
  const float c0 = float(bfloat(float(base[(a.side * 2) * a.width + ch]) + float(d[g])));
  const float c1 = float(bfloat(float(base[(a.side * 2 + 1) * a.width + ch]) + float(d[a.groups + g])));
  float y = float(bfloat(c0 * float(X[(size_t)r * a.width + ch])));
  if (r % a.block >= 1) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));
  Y[(size_t)r * a.width + ch] = bfloat(y);
}

struct gd_act_args { uint rows, width; };

// Qwen3's MLP activation on a stacked gate|up row: bf16(bf16(silu(g)) * u).
[[kernel]] void gd_silu(const device bfloat* GU [[buffer(0)]], constant gd_act_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  const float g = float(GU[at]);
  const float e = precise::exp(-fabs(g)), y = 1.0f / (1.0f + e);
  const float s = g < 0.0f ? e * y : y;
  A[(size_t)p.y * a.width + p.x] = bfloat(float(bfloat(g * s)) * float(GU[at + a.width]));
}

struct gd_topk_args { uint n, stride, chunks, from_ids; };

inline bool gd_before(float a, uint ia, float b, uint ib) { return a > b || (a == b && ia < ib); }

// The best 32 of a row's 2048-value chunk (value desc, id asc) by a bitonic sort, out at (r * chunks + c) * 32.
[[kernel]] void gd_topk(const device bfloat* L [[buffer(0)]], const device float* VIN [[buffer(1)]],
    const device uint* IIN [[buffer(2)]], constant gd_topk_args& a [[buffer(3)]], device float* VOUT [[buffer(4)]],
    device uint* IOUT [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float v[2048];
  threadgroup uint id[2048];
  const uint row = tg.y, chunk = tg.x;
  for (uint i = t; i < 2048; i += 1024) {
    const uint at = chunk * 2048 + i;
    if (at < a.n) {
      const size_t src = (size_t)row * a.stride + at;
      v[i] = a.from_ids ? VIN[src] : float(L[src]);
      id[i] = a.from_ids ? IIN[src] : at;
    } else {
      v[i] = -INFINITY;
      id[i] = 0xffffffffu;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 2048; k <<= 1)
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = t; i < 2048; i += 1024) {
        const uint l = i ^ j;
        if (l > i) {
          const bool down = (i & k) == 0;
          const bool swap = down ? gd_before(v[l], id[l], v[i], id[i]) : gd_before(v[i], id[i], v[l], id[l]);
          if (swap) {
            const float tv = v[i]; v[i] = v[l]; v[l] = tv;
            const uint ti = id[i]; id[i] = id[l]; id[l] = ti;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (t < 32) {
    VOUT[((size_t)row * a.chunks + chunk) * 32 + t] = v[t];
    IOUT[((size_t)row * a.chunks + chunk) * 32 + t] = id[t];
  }
}

struct gd_walk_args { uint slots, rank, top, vocab, stride; };

// The selector's greedy path: slot e scores c after p as unary[e][c] + sum_r bf16(A[p][r] H[e][r]) B[c][r]; seq[1 + e].
// A slot's candidates are the first `top` of its sorted row of `stride` (32).
[[kernel]] void gd_walk(const device float* UV [[buffer(0)]], const device uint* UI [[buffer(1)]],
    const device bfloat* H [[buffer(2)]], const device bfloat* A [[buffer(3)]], const device bfloat* B [[buffer(4)]],
    constant gd_walk_args& a [[buffer(5)]], device uint* seq [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][32];
  threadgroup uint prev;
  if (t == 0) prev = seq[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = 0; e < a.slots; ++e) {
    const float ah = float(bfloat(float(A[(size_t)prev * a.rank + t]) * float(H[(size_t)e * a.rank + t])));
    for (uint c = 0; c < a.top; ++c) {
      const uint cand = UI[e * a.stride + c];
      const float s = simd_sum(ah * float(B[(size_t)min(cand, a.vocab - 1) * a.rank + t]));
      if (lane == 0) part[sg][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float best = -INFINITY;
      uint bi = 0;
      for (uint c = 0; c < a.top; ++c) {
        float s = UV[e * a.stride + c];
        for (uint k = 0; k < a.rank / 32; ++k) s += part[k][c];
        if (UI[e * a.stride + c] < a.vocab && s > best) { best = s; bi = c; }
      }
      prev = UI[e * a.stride + bi];
      seq[1 + e] = prev;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct gd_embed_args { uint rows, width, bits, group; };

// The target's affine embedding rows into fp32 R (rounded to bf16, as the drafter's input).
[[kernel]] void gd_embed(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant gd_embed_args& a [[buffer(4)]],
    device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t row = ids[p.y];
  const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;
  const uint word = W[row * (a.width / per) + p.x / per];
  const float q = float((word >> ((p.x % per) * a.bits)) & mask);
  const size_t g = row * (a.width / a.group) + p.x / a.group;
  R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));
}

struct gd_tap_args { uint rows, width, taps, tap; };

// T[(r * taps + tap) * width + i] = bf16(R[r * width + i]): the residual after a tapped layer, in the drafter's order.
[[kernel]] void gd_tap(const device float* R [[buffer(0)]], constant gd_tap_args& a [[buffer(1)]],
    device bfloat* T [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  T[((size_t)p.y * a.taps + a.tap) * a.width + p.x] = bfloat(R[(size_t)p.y * a.width + p.x]);
}

#define GD_UNROLL_4 _Pragma("clang loop unroll(full)")

struct gd_dvec_args { uint rows, pos0, heads, groups, blocks, first, hi, window, ring; float scale; };

// The block's attention partials over 128-key blocks: row r sees from its window's start up to hi.
[[kernel]] void gd_draft_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant gd_dvec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, z = tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + z;
  const uint p = a.pos0 + row;
  const uint lo = p + 1 > a.window ? p + 1 - a.window : 0;
  const uint block = a.first + z;
  const uint from = max(lo, block * 128), to = min(a.hi, block * 128 + 128);
  if (from >= to) {
    if (lane == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    return;
  }
  const size_t ld = (size_t)a.groups * 128;
  const float4 qv = a.scale * float4(*(const device bfloat4*)(Q + ((size_t)row * a.heads + head) * 128 + 4 * lane));
  float4 acc = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 vr[4];
    GD_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to;
      const size_t at = (size_t)((live ? i0 + u : from) % a.ring) * ld + kv * 128 + 4 * lane;
      vr[u] = float4(*(const device bfloat4*)(V + at));
      const float sc = simd_sum(dot(qv, float4(*(const device bfloat4*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc *= rescale;
    GD_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc += w * vr[u];
    }
    top = peak;
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * 128 + 4 * lane) = acc;
}

struct gd_vec_args { uint rows, pos, heads, groups, topk; float scale; uint sel, all; };

// The partials' merge in block order: a simdgroup a (head, row), the row's max first, then its mass and outputs.
[[kernel]] void gd_merge_vec(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant gd_vec_args& a [[buffer(3)]], device bfloat* O [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const size_t at = (size_t)head * a.rows + row;
  const device float* ms = PM + at * a.topk;
  const device float* ls = PL + at * a.topk;
  float top = -INFINITY;
  for (uint b = lane; b < a.topk; b += 32) top = max(top, ms[b]);
  top = simd_max(top);
  float total = 0.0f;
  float4 acc = 0.0f;
  for (uint b = 0; b < a.topk; ++b) {
    if (ls[b] <= 0.0f) continue;
    const float f = fast::exp(ms[b] - top);
    total += ls[b] * f;
    acc += f * *(const device float4*)(PO + (at * a.topk + b) * 128 + 4 * lane);
  }
  acc /= total;
  *(device bfloat4*)(O + (row * a.heads + head) * 128 + 4 * lane) = bfloat4(acc);
}

struct gd_gemv_args { uint rows, n, k, ld; };

// Y[m, c] = bf16(sum_k X[m, k] W[c, k]) for a block's rows (1-8) against a bf16 [n, k] matrix read once: a simdgroup
// four columns, a lane 8 values of each a step, every row's sums beside each other (fp32). Grid (n / 16), 128 threads.
[[kernel]] void gd_gemv(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant gd_gemv_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c0 = (tg * 4 + sg) * 4;
  if (c0 >= a.n) return;
  float acc[8][4];
  for (uint m = 0; m < 8; ++m)
    for (uint j = 0; j < 4; ++j) acc[m][j] = 0.0f;
  const uint rows = min(a.rows, 8u);
  for (uint k0 = lane * 8; k0 < a.k; k0 += 256) {
    float w[4][8];
    for (uint j = 0; j < 4; ++j) {
      const uint c = min(c0 + j, a.n - 1);
      const device bfloat4* p = (const device bfloat4*)(W + (size_t)c * a.k + k0);
      const float4 lo = float4(p[0]), hi = float4(p[1]);
      w[j][0] = lo.x; w[j][1] = lo.y; w[j][2] = lo.z; w[j][3] = lo.w;
      w[j][4] = hi.x; w[j][5] = hi.y; w[j][6] = hi.z; w[j][7] = hi.w;
    }
    for (uint m = 0; m < 8; ++m) {
      if (m >= rows) break;
      const device bfloat4* q = (const device bfloat4*)(X + (size_t)m * a.k + k0);
      const float4 xl = float4(q[0]), xh = float4(q[1]);
      for (uint j = 0; j < 4; ++j)
        acc[m][j] += xl.x * w[j][0] + xl.y * w[j][1] + xl.z * w[j][2] + xl.w * w[j][3] +
                     xh.x * w[j][4] + xh.y * w[j][5] + xh.z * w[j][6] + xh.w * w[j][7];
    }
  }
  for (uint m = 0; m < rows; ++m)
    for (uint j = 0; j < 4; ++j) {
      const float s = simd_sum(acc[m][j]);
      if (lane == 0 && c0 + j < a.n) Y[(size_t)m * a.ld + c0 + j] = bfloat(s);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// RedHat's DSpark drafter for GLM-5.3 (speculators' DSparkDraftModel, vLLM's qwen3_dspark): Qwen3 layers of 64-wide
// heads, causal within the block, then the Markov walk with the confidence head. gd_prep64 and gd_draft_vec64 are
// gd_prep and gd_draft_vec at head dimension 64 (two values a lane); the attention takes `causal` (row r sees keys
// up to its own position).
[[kernel]] void gd_prep64(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant gd_prep_args& a [[buffer(3)]], device bfloat* Q [[buffer(4)]],
    device bfloat* K [[buffer(5)]], device bfloat* V [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? a.q_off : a.k_off) + (size_t)head * 64;
  const device bfloat* w = is_q ? qw : kw;
  const uint p = a.pos0 + row;
  const size_t slot = p % a.ring;
  device bfloat* y = is_q ? Q + ((size_t)row * a.heads + head) * 64 : K + (slot * a.kv_heads + head) * 64;
  float v[2];
  float sum = 0.0f;
  for (uint j = 0; j < 2; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / 64.0f + a.eps);
  threadgroup float nrm[64];
  for (uint j = 0; j < 2; ++j) nrm[lane + 32 * j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 2; ++j) {
    const uint d = lane + 32 * j;
    const uint i = d % 32;
    const float angle = float(p) * (1.0f / precise::pow(a.theta, float(2 * i) / 64.0f));
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float rot = d < 32 ? -nrm[d + 32] : nrm[d - 32];
    y[d] = bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s)));
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + a.v_off + (size_t)head * 64;
  device bfloat* vd = V + (slot * a.kv_heads + head) * 64;
  for (uint j = 0; j < 2; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

struct gd_dvec64_args { uint rows, pos0, heads, groups, blocks, first, hi, window, ring; float scale; uint causal; };

// gd_draft_vec at head dimension 64: a lane two values; row r's keys from its window's start to hi, or to its own
// position with `causal`.
[[kernel]] void gd_draft_vec64(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant gd_dvec64_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, z = tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + z;
  const uint p = a.pos0 + row;
  const uint lo = p + 1 > a.window ? p + 1 - a.window : 0;
  const uint block = a.first + z;
  const uint hi = a.causal ? min(a.hi, p + 1) : a.hi;
  const uint from = max(lo, block * 128), to = min(hi, block * 128 + 128);
  if (from >= to) {
    if (lane == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    *(device float2*)(PO + slot * 64 + 2 * lane) = float2(0.0f);
    return;
  }
  const size_t ld = (size_t)a.groups * 64;
  const float2 qv = a.scale * float2(*(const device bfloat2*)(Q + ((size_t)row * a.heads + head) * 64 + 2 * lane));
  float2 acc = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float2 vr[4];
    GD_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to;
      const size_t at = (size_t)((live ? i0 + u : from) % a.ring) * ld + kv * 64 + 2 * lane;
      vr[u] = float2(*(const device bfloat2*)(V + at));
      const float sc = simd_sum(dot(qv, float2(*(const device bfloat2*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc *= rescale;
    GD_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc += w * vr[u];
    }
    top = peak;
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float2*)(PO + slot * 64 + 2 * lane) = acc;
}

// gd_merge_vec at head dimension 64.
[[kernel]] void gd_merge_vec64(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant gd_vec_args& a [[buffer(3)]], device bfloat* O [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const size_t at = (size_t)head * a.rows + row;
  const device float* ms = PM + at * a.topk;
  const device float* ls = PL + at * a.topk;
  float top = -INFINITY;
  for (uint b = lane; b < a.topk; b += 32) top = max(top, ms[b]);
  top = simd_max(top);
  float total = 0.0f;
  float2 acc = 0.0f;
  for (uint b = 0; b < a.topk; ++b) {
    if (ls[b] <= 0.0f) continue;
    const float f = fast::exp(ms[b] - top);
    total += ls[b] * f;
    acc += f * *(const device float2*)(PO + (at * a.topk + b) * 64 + 2 * lane);
  }
  acc /= total;
  *(device bfloat2*)(O + (row * a.heads + head) * 64 + 2 * lane) = bfloat2(acc);
}

struct gd_mwalk_args { uint slots, rank, top, vocab, stride, hidden; };

// The DSpark Markov walk over each slot's best `top` base logits (of a sorted row `stride` wide): slot e scores
// candidate c as its logit + w2[c] . w1[prev] (prev the anchor, then the slot before's pick), the best is the draft
// (vLLM's dspark_draft_topk restriction of the full-vocabulary bias); the confidence head's sigmoid of
// [h_e, w1[prev]] . cw + cb is the slot's chance. seq[0] the anchor in, seq[1 + e] the drafts out, conf[e] out.
// One threadgroup of 256 threads.
[[kernel]] void gd_mwalk(const device float* UV [[buffer(0)]], const device uint* UI [[buffer(1)]],
    const device bfloat* H [[buffer(2)]], const device bfloat* W1 [[buffer(3)]], const device bfloat* W2 [[buffer(4)]],
    const device bfloat* CW [[buffer(5)]], const device bfloat* CB [[buffer(6)]], constant gd_mwalk_args& a [[buffer(7)]],
    device uint* seq [[buffer(8)]], device float* conf [[buffer(9)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][32];
  threadgroup float cpart[8];
  threadgroup uint prev;
  if (t == 0) prev = seq[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = 0; e < a.slots; ++e) {
    const float w1 = float(W1[(size_t)prev * a.rank + t]);
    // the confidence: [h_e (hidden), w1[prev] (rank)] . cw
    float cs = 0.0f;
    for (uint i = t; i < a.hidden; i += 256) cs = fma(float(H[(size_t)e * a.hidden + i]), float(CW[i]), cs);
    cs = fma(w1, float(CW[a.hidden + t]), cs);
    cs = simd_sum(cs);
    if (lane == 0) cpart[sg] = cs;
    for (uint c = 0; c < a.top; ++c) {
      const uint cand = UI[e * a.stride + c];
      const float s = simd_sum(w1 * float(W2[(size_t)min(cand, a.vocab - 1) * a.rank + t]));
      if (lane == 0) part[sg][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float best = -INFINITY;
      uint bi = 0;
      for (uint c = 0; c < a.top; ++c) {
        float s = 0.0f;
        for (uint k = 0; k < a.rank / 32; ++k) s += part[k][c];
        s = float(bfloat(UV[e * a.stride + c] + float(bfloat(s))));
        if (UI[e * a.stride + c] < a.vocab && s > best) { best = s; bi = c; }
      }
      float z = float(CB[0]);
      for (uint k = 0; k < 8; ++k) z += cpart[k];
      conf[e] = 1.0f / (1.0f + exp(-z));
      prev = UI[e * a.stride + bi];
      seq[1 + e] = prev;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
