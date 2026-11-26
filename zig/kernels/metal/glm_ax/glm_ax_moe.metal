
// ---------------------------------------------------------------------------------------------------------------
// GLM-5.3's fused decode MoE on core/row_projection.metal's row reduction (tf_rp_rows, appended to that source):
// each simdgroup takes 4 output rows with the input read 16 values a lane, 512 at a time, exactly as the core's
// row kernels do, so every value is the unfused path's (bit for bit), in two launches for a row's nine experts
// (its eight picks, then the shared one) instead of seven. MLX affine 4-bit, groups of 64.
// GX_MI: an expert's intermediate channels (tensor parallel builds define this rank's share; 2048 otherwise)
#ifndef GX_MI
#define GX_MI 2048
#endif
constant constexpr int GXH = 6144, GXI = GX_MI, GXGS = 64;

struct gx_moe_rows_args {
  uint rows, topk, slots, pad0; // slots = topk + 1 a row
};

// act[slot][n] = bf16(silu(bf16(gate . x)) * bf16(up . x)) as gx_swiglu rounds it; slot = row * slots + k, k = topk
// the shared expert (its gate rows then its up rows stacked in SW). Grid (GXI / 8, rows * slots), 2 simdgroups.
[[kernel]] void gx_moe_gu_rows(const device bfloat* X [[buffer(0)]], const device int* pick [[buffer(1)]],
                               const device uint32_t* GW [[buffer(2)]], const device bfloat* GSC [[buffer(3)]], const device bfloat* GBI [[buffer(4)]],
                               const device uint32_t* UW [[buffer(5)]], const device bfloat* USC [[buffer(6)]], const device bfloat* UBI [[buffer(7)]],
                               const device uint32_t* SW [[buffer(8)]], const device bfloat* SSC [[buffer(9)]], const device bfloat* SBI [[buffer(10)]],
                               constant gx_moe_rows_args& a [[buffer(11)]], device bfloat* act [[buffer(12)]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= GXI) return;
  const uint slot = tg.y, r = slot / a.slots, k = slot % a.slots;
  const device bfloat* x = X + size_t(r) * GXH;
  const size_t KB = GXH * 4 / 8, KG = GXH / GXGS;
  size_t ga, ua;
  const device uint32_t *gwb, *uwb;
  const device bfloat *gs, *gb, *us, *ub;
  if (k < a.topk) {
    ga = ua = size_t(pick[r * a.topk + k]) * GXI + size_t(row0);
    gwb = GW; gs = GSC; gb = GBI; uwb = UW; us = USC; ub = UBI;
  } else {
    ga = size_t(row0);
    ua = size_t(GXI + row0);
    gwb = uwb = SW; gs = us = SSC; gb = ub = SBI;
  }
  float g[1][4], u[1][4];
  tf_rp_rows<4, 1, 4, false>((const device uint8_t*)gwb + ga * KB, gs + ga * KG, gb + ga * KG, x, GXH, GXGS, lane, g);
  tf_rp_rows<4, 1, 4, false>((const device uint8_t*)uwb + ua * KB, us + ua * KG, ub + ua * KG, x, GXH, GXGS, lane, u);
  if (lane == 0)
    for (int j = 0; j < 4; j++) {
      const float gr = float(bfloat(g[0][j])), ur = float(bfloat(u[0][j]));
      act[size_t(slot) * GXI + row0 + j] = bfloat(precise::divide(gr, 1.0f + precise::exp(-gr)) * ur);
    }
}

// The nine experts' down: Y[row * topk + k] for the picks (the routed sum reads them in pick order), shared_y[row] for
// the shared one. Grid (GXH / 8, rows * slots), 2 simdgroups.
[[kernel]] void gx_moe_down_rows(const device bfloat* act [[buffer(0)]], const device int* pick [[buffer(1)]],
                                 const device uint32_t* DW [[buffer(2)]], const device bfloat* DSC [[buffer(3)]], const device bfloat* DBI [[buffer(4)]],
                                 const device uint32_t* SW [[buffer(5)]], const device bfloat* SSC [[buffer(6)]], const device bfloat* SBI [[buffer(7)]],
                                 constant gx_moe_rows_args& a [[buffer(8)]], device bfloat* Y [[buffer(9)]], device bfloat* shared_y [[buffer(10)]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= GXH) return;
  const uint slot = tg.y, r = slot / a.slots, k = slot % a.slots;
  const device bfloat* x = act + size_t(slot) * GXI;
  const size_t KB = GXI * 4 / 8, KG = GXI / GXGS;
  const bool routed = k < a.topk;
  const size_t at = routed ? size_t(pick[r * a.topk + k]) * GXH + size_t(row0) : size_t(row0);
  const device uint32_t* wb = routed ? DW : SW;
  const device bfloat* sc = routed ? DSC : SSC;
  const device bfloat* bi = routed ? DBI : SBI;
  float y[1][4];
  tf_rp_rows<4, 1, 4, false>((const device uint8_t*)wb + at * KB, sc + at * KG, bi + at * KG, x, GXI, GXGS, lane, y);
  if (lane == 0) {
    device bfloat* out = routed ? Y + (size_t(r) * a.topk + k) * GXH : shared_y + size_t(r) * GXH;
    for (int j = 0; j < 4; j++) out[row0 + j] = bfloat(y[0][j]);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The same two launches by expert for verify windows (1-16 rows): one threadgroup row a distinct routed expert of
// the window (the route's list: ids ascending, each with its member picks), the last the shared expert (every row).
// A simdgroup reads its 4 weight rows once for two member rows at a time (tf_rp_rows_at with MB 2), so an expert two
// rows picked, and the shared expert, are read once for both. act and Y keep the per-pick layout (slot r * topk + k;
// the shared expert's rows after the picks in act, in shared_y).
constant constexpr int GXMAXR = 16; // the route's member stride (its TF_MAXR)

struct gx_moe_set_args {
  uint rows, topk, cap, pad0; // cap: the grid's routed rows (min(rows * topk, experts)); the shared one after them
};

// A distinct expert's (or the shared expert's) member rows: their pick slots (out) and x rows; the count.
inline int gx_members(uint u, constant gx_moe_set_args& a, const device int* lcount, const device int* lmem,
                      thread int* slot, thread int* xrow, thread bool* shared) {
  *shared = u == a.cap;
  if (*shared) {
    for (uint r = 0; r < a.rows; r++) slot[r] = int(a.rows * a.topk + r), xrow[r] = int(r);
    return int(a.rows);
  }
  if (int(u) >= lcount[0]) return 0;
  int n = 0;
  for (int j = 0; j < GXMAXR; j++) {
    const int m = lmem[int(u) * GXMAXR + j];
    if (m < 0) break;
    slot[n] = m;
    xrow[n] = m / int(a.topk);
    n++;
  }
  return n;
}

[[kernel]] void gx_moe_gu_set(const device bfloat* X [[buffer(0)]], const device int* lids [[buffer(1)]],
                              const device int* lmem [[buffer(2)]], const device int* lcount [[buffer(3)]],
                              const device uint32_t* GW [[buffer(4)]], const device bfloat* GSC [[buffer(5)]], const device bfloat* GBI [[buffer(6)]],
                              const device uint32_t* UW [[buffer(7)]], const device bfloat* USC [[buffer(8)]], const device bfloat* UBI [[buffer(9)]],
                              const device uint32_t* SW [[buffer(10)]], const device bfloat* SSC [[buffer(11)]], const device bfloat* SBI [[buffer(12)]],
                              constant gx_moe_set_args& a [[buffer(13)]], device bfloat* act [[buffer(14)]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= GXI) return;
  int slot[GXMAXR], xrow[GXMAXR];
  bool shared;
  const int n = gx_members(tg.y, a, lcount, lmem, slot, xrow, &shared);
  if (n == 0) return;
  const size_t KB = GXH * 4 / 8, KG = GXH / GXGS;
  const size_t ga = shared ? size_t(row0) : size_t(lids[tg.y]) * GXI + size_t(row0);
  const size_t ua = shared ? size_t(GXI + row0) : ga;
  const device uint8_t* gw = (const device uint8_t*)(shared ? SW : GW) + ga * KB;
  const device uint8_t* uw = (const device uint8_t*)(shared ? SW : UW) + ua * KB;
  const device bfloat* gs = (shared ? SSC : GSC) + ga * KG;
  const device bfloat* gb = (shared ? SBI : GBI) + ga * KG;
  const device bfloat* us = (shared ? SSC : USC) + ua * KG;
  const device bfloat* ub = (shared ? SBI : UBI) + ua * KG;
  int m = 0;
  for (; m + 2 <= n; m += 2) {
    TfRpRows<2> x;
    x.x[0] = X + size_t(xrow[m]) * GXH;
    x.x[1] = X + size_t(xrow[m + 1]) * GXH;
    float g[2][4], u[2][4];
    tf_rp_rows_at<4, 2, 4, false>(gw, gs, gb, x, GXH, GXGS, lane, g);
    tf_rp_rows_at<4, 2, 4, false>(uw, us, ub, x, GXH, GXGS, lane, u);
    if (lane == 0)
      for (int b = 0; b < 2; b++)
        for (int j = 0; j < 4; j++) {
          const float gr = float(bfloat(g[b][j])), ur = float(bfloat(u[b][j]));
          act[size_t(slot[m + b]) * GXI + row0 + j] = bfloat(precise::divide(gr, 1.0f + precise::exp(-gr)) * ur);
        }
  }
  if (m < n) {
    TfRpRows<1> x;
    x.x[0] = X + size_t(xrow[m]) * GXH;
    float g[1][4], u[1][4];
    tf_rp_rows_at<4, 1, 4, false>(gw, gs, gb, x, GXH, GXGS, lane, g);
    tf_rp_rows_at<4, 1, 4, false>(uw, us, ub, x, GXH, GXGS, lane, u);
    if (lane == 0)
      for (int j = 0; j < 4; j++) {
        const float gr = float(bfloat(g[0][j])), ur = float(bfloat(u[0][j]));
        act[size_t(slot[m]) * GXI + row0 + j] = bfloat(precise::divide(gr, 1.0f + precise::exp(-gr)) * ur);
      }
  }
}

[[kernel]] void gx_moe_down_set(const device bfloat* act [[buffer(0)]], const device int* lids [[buffer(1)]],
                                const device int* lmem [[buffer(2)]], const device int* lcount [[buffer(3)]],
                                const device uint32_t* DW [[buffer(4)]], const device bfloat* DSC [[buffer(5)]], const device bfloat* DBI [[buffer(6)]],
                                const device uint32_t* SW [[buffer(7)]], const device bfloat* SSC [[buffer(8)]], const device bfloat* SBI [[buffer(9)]],
                                constant gx_moe_set_args& a [[buffer(10)]], device bfloat* Y [[buffer(11)]], device bfloat* shared_y [[buffer(12)]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= GXH) return;
  int slot[GXMAXR], xrow[GXMAXR];
  bool shared;
  const int n = gx_members(tg.y, a, lcount, lmem, slot, xrow, &shared);
  if (n == 0) return;
  const size_t KB = GXI * 4 / 8, KG = GXI / GXGS;
  const size_t at = shared ? size_t(row0) : size_t(lids[tg.y]) * GXH + size_t(row0);
  const device uint8_t* w = (const device uint8_t*)(shared ? SW : DW) + at * KB;
  const device bfloat* sc = (shared ? SSC : DSC) + at * KG;
  const device bfloat* bi = (shared ? SBI : DBI) + at * KG;
  int m = 0;
  for (; m < n;) {
    const int take = m + 2 <= n ? 2 : 1;
    float y[2][4];
    if (take == 2) {
      TfRpRows<2> x;
      x.x[0] = act + size_t(slot[m]) * GXI;
      x.x[1] = act + size_t(slot[m + 1]) * GXI;
      tf_rp_rows_at<4, 2, 4, false>(w, sc, bi, x, GXI, GXGS, lane, y);
    } else {
      TfRpRows<1> x;
      x.x[0] = act + size_t(slot[m]) * GXI;
      float y1[1][4];
      tf_rp_rows_at<4, 1, 4, false>(w, sc, bi, x, GXI, GXGS, lane, y1);
      for (int j = 0; j < 4; j++) y[0][j] = y1[0][j];
    }
    if (lane == 0)
      for (int b = 0; b < take; b++) {
        device bfloat* out = shared ? shared_y + size_t(xrow[m + b]) * GXH : Y + size_t(slot[m + b]) * GXH;
        for (int j = 0; j < 4; j++) out[row0 + j] = bfloat(y[b][j]);
      }
    m += take;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Verify windows' projections (2-8 rows a pass) with each weight read once a pass: a lane keeps its 4 columns' 16
// codes of a 512-wide chunk in registers and runs the window's rows over them two at a time, with the core's own
// load16 and the qdot16 arithmetic, so every row's sums are the one-row kernel's bit for bit (core/row_projection.metal
// tf_rp_rows_at with MB 2, whose pairs read the weights again for every pair). 4-bit, groups of 64.
// dims: K, N, GS, rows; X [rows, K]; W [N, K/8] words; S, B [N, K/GS]; OUT [rows, N]. Grid (N / 8), 64 threads.
inline void gx_qdot16_codes(const thread uint16_t* ws, const thread float (*xt)[16], float scale, float bias,
                            const thread float* sum, thread float* out) {
  float q[16];
  for (int i = 0; i < 4; i++) {
    const int v = ws[i];
    q[4 * i] = float(v & 0x000f); q[4 * i + 1] = float(v & 0x00f0); q[4 * i + 2] = float(v & 0x0f00); q[4 * i + 3] = float(v & 0xf000);
  }
  for (int b = 0; b < 2; b++) {
    float accum = 0.0f;
    for (int i = 0; i < 4; i++)
      accum += (tf_rp_in<4>(xt[b][4 * i], 4 * i) * q[4 * i] + tf_rp_in<4>(xt[b][4 * i + 1], 4 * i + 1) * q[4 * i + 1] +
                tf_rp_in<4>(xt[b][4 * i + 2], 4 * i + 2) * q[4 * i + 2] +
                tf_rp_in<4>(xt[b][4 * i + 3], 4 * i + 3) * q[4 * i + 3]);
    out[b] += scale * accum + sum[b] * bias;
  }
}

constant constexpr int GXMR = 8; // rows a pass over the weights

[[kernel]] void gx_rows_mr(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const int row0 = (int(tg.x) * 2 + int(sg)) * 4;
  if (row0 >= N) return;
  const int KB = K * 4 / 8, KG = K / GS, FULL = K / 512 * 512, per_lane = GS / 16;
  for (int m0 = 0; m0 < rows; m0 += GXMR) {
    const int mr = min(GXMR, rows - m0);
    float acc[GXMR][4];
    for (int b = 0; b < GXMR; b++)
      for (int j = 0; j < 4; j++) acc[b][j] = 0.0f;
    const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
    const device bfloat* sc = S + size_t(row0) * KG + lane / per_lane;
    const device bfloat* bi = B + size_t(row0) * KG + lane / per_lane;
    const int xl = int(lane) * 16;
    for (int k0 = 0; k0 <= FULL; k0 += 512) {
      if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
      uint16_t codes[4][4];
      float scale[4], bias[4];
      for (int j = 0; j < 4; j++) {
        const device uint16_t* ws = (const device uint16_t*)(w + j * KB);
        for (int i = 0; i < 4; i++) codes[j][i] = ws[i];
        scale[j] = float(sc[j * KG]);
        bias[j] = float(bi[j * KG]);
      }
      for (int p = 0; p < GXMR; p += 2) {
        if (p >= mr) break;
        const int pb = min(p + 1, mr - 1); // an odd last row runs beside itself (its second result unused)
        float xt[2][16];
        float sum[2];
        sum[0] = tf_rp_load16<4, false>(X + size_t(m0 + p) * K + xl + k0, xt[0]);
        sum[1] = tf_rp_load16<4, false>(X + size_t(m0 + pb) * K + xl + k0, xt[1]);
        for (int j = 0; j < 4; j++) {
          float two[2] = {acc[p][j], acc[p + 1 < GXMR ? p + 1 : p][j]};
          gx_qdot16_codes(codes[j], xt, scale[j], bias[j], sum, two);
          acc[p][j] = two[0];
          if (p + 1 < mr) acc[p + 1][j] = two[1];
        }
      }
      w += 512 * 4 / 8; sc += 512 / GS; bi += 512 / GS;
    }
    for (int b = 0; b < GXMR; b++) {
      if (b >= mr) break;
      for (int j = 0; j < 4; j++) {
        const float v = simd_sum(acc[b][j]);
        if (lane == 0) OUT[size_t(m0 + b) * N + row0 + j] = bfloat(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// tf_rp_rows_at over K values of rows whose weights lie `KB` bytes (and scales KG groups) apart: a slice of K.
template <int RPS, int MB, int BITS, bool BF16SUM>
inline void tf_rp_rows_at_ld(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, thread const TfRpRows<MB>& x,
                             int K, int KB, int KG, int GS, uint lane, thread float (*acc)[RPS]) {
  typedef typename metal::conditional<(MB >= 3), bfloat, float>::type X;
  const int FULL = K / 512 * 512;
  const int per_lane = GS / 16;
  w += lane * (16 * BITS / 8);
  sc += lane / per_lane;
  bi += lane / per_lane;
  const int xl = int(lane) * 16;
  float col[MB];
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = 0.0f;
  for (int k0 = 0; k0 <= FULL; k0 += 512) {
    if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
    X xt[MB][16];
    float sum[MB];
    for (int b = 0; b < MB; b++) sum[b] = tf_rp_load16<BITS, BF16SUM>(x.x[b] + xl + k0, xt[b]);
    for (int j = 0; j < RPS; j++) {
      for (int b = 0; b < MB; b++) col[b] = acc[b][j];
      tf_rp_qdot16<MB, BITS>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum, col);
      for (int b = 0; b < MB; b++) acc[b][j] = col[b];
    }
    w += 512 * BITS / 8; sc += 512 / GS; bi += 512 / GS;
  }
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = simd_sum(acc[b][j]);
}

// Decode projections with K split four ways inside a threadgroup (tf-rewind's qmv_1 shape on the core's row math):
// simdgroup s takes the s-th quarter of K for the threadgroup's 4 columns, two rows a pass (tf_rp_rows_at, MB 2 or
// 1), and simdgroup 0 adds the quarters in order. Every window width runs the same arithmetic, so a row's sums do not
// depend on its window (drafted output stays the undrafted one). K % 2048 == 0, 4-bit, groups of 64.
// dims: K, N, GS, rows. Grid (N / 4), 128 threads.
template <bool PAIRS>
inline void gx_proj_k4_body(const device bfloat* X, const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                            constant int* dims, device bfloat* OUT, uint sg, uint lane, uint3 tg,
                            threadgroup float (&part)[4][2][4]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const int row0 = int(tg.x) * 4;
  const int KQ = K / 4, KB = K * 4 / 8, KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + size_t(sg) * (KQ * 4 / 8);
  const device bfloat* sc = S + size_t(row0) * KG + size_t(sg) * (KQ / GS);
  const device bfloat* bi = B + size_t(row0) * KG + size_t(sg) * (KQ / GS);
  for (int m = 0; m < rows; m += PAIRS ? 2 : 1) {
    const int mb = min(PAIRS ? 2 : 1, rows - m);
    float acc[2][4];
    if (mb == 2) {
      TfRpRows<2> x;
      x.x[0] = X + size_t(m) * K + size_t(sg) * KQ;
      x.x[1] = X + size_t(m + 1) * K + size_t(sg) * KQ;
      tf_rp_rows_at_ld<4, 2, 4, false>(w, sc, bi, x, KQ, KB, KG, GS, lane, acc);
    } else {
      TfRpRows<1> x;
      x.x[0] = X + size_t(m) * K + size_t(sg) * KQ;
      float one[1][4];
      tf_rp_rows_at_ld<4, 1, 4, false>(w, sc, bi, x, KQ, KB, KG, GS, lane, one);
      for (int j = 0; j < 4; j++) acc[0][j] = one[0][j];
    }
    if (lane == 0)
      for (int b = 0; b < mb; b++)
        for (int j = 0; j < 4; j++) part[sg][b][j] = acc[b][j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < uint(mb * 4) && row0 + int(lane % 4) < N) {
      const int b = int(lane) / 4, j = int(lane) % 4;
      const float v = ((part[0][b][j] + part[1][b][j]) + part[2][b][j]) + part[3][b][j];
      OUT[size_t(m + b) * N + row0 + j] = bfloat(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Two rows a pass over the K quarter (the pair kernel), or one row a pass (the slice read again from cache): the same sums.
[[kernel]] void gx_proj_k4(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float part[4][2][4];
  gx_proj_k4_body<true>(X, W, S, B, dims, OUT, sg, lane, tg, part);
}
[[kernel]] void gx_proj_k4_1(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float part[4][2][4];
  gx_proj_k4_body<false>(X, W, S, B, dims, OUT, sg, lane, tg, part);
}

// ---------------------------------------------------------------------------------------------------------------
// Decode projections, one simdgroup an output column (tf-rewind's qmv shape): a lane streams 16-byte loads of its
// column's codes (32 a load, half a group of 64), and every row of the window is summed beside the others from the
// same registers, two rows a pass. A row's sums take the same operations in the same order whatever the window's
// width, so a drafted row equals the undrafted one bit for bit. 4-bit, groups of 64, K % 1024 == 0.
// dims: K, N, GS, rows. Grid (N / 4), 128 threads.
[[kernel]] void gx_proj_c1(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], rows = dims[3];
  const int col = int(tg.x) * 4 + int(sg);
  if (col >= N) return;
  const device uint4* wr = (const device uint4*)(W + size_t(col) * (K / 8));
  const device bfloat* sr = S + size_t(col) * (K / 64);
  const device bfloat* br = B + size_t(col) * (K / 64);
  const int steps = K / 1024;
  for (int m0 = 0; m0 < rows; m0 += 2) {
    const int mb = min(2, rows - m0);
    float acc[2] = {0.0f, 0.0f};
    for (int i = 0; i < steps; i++) {
      const int idx = int(lane) + 32 * i;
      const uint4 w = wr[idx];
      const float s = float(sr[idx / 2]), b = float(br[idx / 2]);
      for (int r = 0; r < 2; r++) {
        if (r >= mb) break;
        const device bfloat4* xp = (const device bfloat4*)(X + size_t(m0 + r) * K + size_t(idx) * 32);
        float dot = 0.0f, xs = 0.0f;
        for (int c = 0; c < 4; c++) {
          const uint word = w[c];
          const float4 xa = float4(xp[2 * c]), xb = float4(xp[2 * c + 1]);
          for (int j = 0; j < 4; j++) {
            dot = fma(float((word >> (4 * j)) & 15u), xa[j], dot);
            xs += xa[j];
          }
          for (int j = 0; j < 4; j++) {
            dot = fma(float((word >> (4 * (j + 4))) & 15u), xb[j], dot);
            xs += xb[j];
          }
        }
        acc[r] = fma(s, dot, fma(b, xs, acc[r]));
      }
    }
    for (int r = 0; r < mb; r++) {
      const float v = simd_sum(acc[r]);
      if (lane == 0) OUT[size_t(m0 + r) * N + col] = bfloat(v);
    }
  }
}

// gx_proj_c1 with the window's inputs staged in threadgroup memory a step at a time (1024 values a row, two rows),
// shared by the threadgroup's 8 columns instead of read from cache by each: the same operations in the same order,
// so the same bits as gx_proj_c1. dims: K, N, GS, rows. Grid (N / 8), 256 threads.
[[kernel]] void gx_proj_c8(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint t [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat4 xs4[2][256]; // two rows' 1024 inputs of this step
  const int K = dims[0], N = dims[1], rows = dims[3];
  const int col = int(tg.x) * 8 + int(sg);
  const bool live = col < N;
  const device uint4* wr = (const device uint4*)(W + size_t(min(col, N - 1)) * (K / 8));
  const device bfloat* sr = S + size_t(min(col, N - 1)) * (K / 64);
  const device bfloat* br = B + size_t(min(col, N - 1)) * (K / 64);
  const int steps = K / 1024;
  for (int m0 = 0; m0 < rows; m0 += 2) {
    const int mb = min(2, rows - m0);
    float acc[2] = {0.0f, 0.0f};
    for (int i = 0; i < steps; i++) {
      for (int r = 0; r < mb; r++)
        xs4[r][t] = ((const device bfloat4*)(X + size_t(m0 + r) * K + size_t(i) * 1024))[t];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const int idx = int(lane) + 32 * i;
      const uint4 w = wr[idx];
      const float s = float(sr[idx / 2]), b = float(br[idx / 2]);
      for (int r = 0; r < 2; r++) {
        if (r >= mb) break;
        float dot = 0.0f, xsum = 0.0f;
        for (int c = 0; c < 4; c++) {
          const uint word = w[c];
          const float4 xa = float4(xs4[r][int(lane) * 8 + 2 * c]), xb = float4(xs4[r][int(lane) * 8 + 2 * c + 1]);
          for (int j = 0; j < 4; j++) {
            dot = fma(float((word >> (4 * j)) & 15u), xa[j], dot);
            xsum += xa[j];
          }
          for (int j = 0; j < 4; j++) {
            dot = fma(float((word >> (4 * (j + 4))) & 15u), xb[j], dot);
            xsum += xb[j];
          }
        }
        acc[r] = fma(s, dot, fma(b, xsum, acc[r]));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < mb; r++) {
      const float v = simd_sum(acc[r]);
      if (lane == 0 && live) OUT[size_t(m0 + r) * N + col] = bfloat(v);
    }
  }
}
