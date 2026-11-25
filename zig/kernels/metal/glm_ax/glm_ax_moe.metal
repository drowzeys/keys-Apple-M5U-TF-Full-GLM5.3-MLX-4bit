
// ---------------------------------------------------------------------------------------------------------------
// GLM-5.3's fused decode MoE on core/row_projection.metal's row reduction (tf_rp_rows, appended to that source):
// each simdgroup takes 4 output rows with the input read 16 values a lane, 512 at a time, exactly as the core's
// row kernels do, so every value is the unfused path's (bit for bit), in two launches for a row's nine experts
// (its eight picks, then the shared one) instead of seven. MLX affine 4-bit, groups of 64.
constant constexpr int GXH = 6144, GXI = 2048, GXGS = 64;

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
