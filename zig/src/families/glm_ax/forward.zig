//! Rows through every layer in one command buffer: row kernels to 16 rows, tensor-unit matmuls for prompt chunks.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;
const Own = @import("model.zig").Own;

/// step5's glue kernels' constants (step5/glue.metal's s5_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, span: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, theta: f32 = 0, limit: f32 = 0, ring: u32 = 0 };
/// This family's kernels' constants (glm_ax.metal's gx_args).
pub const GArgs = extern struct { rows: u32 = 0, heads: u32 = 0, width: u32 = 0, stride: u32 = 0, score_stride: u32 = 0, in_stride: u32 = 0, pad0: u32 = 0, pad1: u32 = 0, eps: f32 = 0, scale: f32 = 0, key_scale: f32 = 0, head_scale: f32 = 0 };

/// SwiGLU has no clamp in GLM-5.3: the shared kernel's limit past every bf16 value.
const no_limit: f32 = 3.0e38;
/// A row's norm kernels run 1024 threads: a row is 6144 wide and one threadgroup takes it (a 256-thread group was
/// latency-bound on the MoE sum, 9 loads an element).
const norm_threads = 1024;
/// glm_ax_sparse.metal's gx_score_args.
/// glm_ax_moe.metal's gx_moe_set_args.
const SetArgs = extern struct { rows: u32, topk: u32, cap: u32, pad0: u32 = 0 };
const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32, wscale: f32, pad0: u32 = 0, pad1: u32 = 0 };
/// The MLA and indexer norms' epsilon (the layer norms take the config's).
const mla_eps: f32 = 1e-6;

pub const B = struct { buf: mtl.Buffer = undefined, off: usize = 0, args: bool = false };
fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}
fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}
fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}
const A = B{ .args = true };

/// A matrix as affine_mm takes it.
fn mat(w: wts.Linear) struct { w: B, s: B, b: B, n: u32, k: u32 } {
    return .{ .w = bo(w.w, w.w_off), .s = bo(w.scales, w.s_off), .b = bo(w.biases, w.b_off), .n = @intCast(w.n), .k = @intCast(w.k) };
}

/// Stage timing: with a profile, each stage runs in its own command buffer and its GPU time is added to its slot.
pub const Stage = enum { embed, norm, proj_x, prep, proj_q, index, absorb, attend, unembed, o_proj, route, experts, shared, dense, moe_add, head };
pub const Profile = struct {
    gpu: [@typeInfo(Stage).@"enum".field_names.len]f64 = @splat(0),
    wall: f64 = 0,
    runs: usize = 0,
    // decode rows: distinct routed experts summed over MoE layers, and the layers counted (read once a stage's
    // command buffer has finished, so profile runs only)
    distinct: usize = 0,
    moe_layers: usize = 0,
};

const Enc = struct {
    e: mtl.ComputeEncoder,
    cb: mtl.CommandBuffer,
    m: *const Model,
    s: *st.Scratch,
    prof: ?*Profile = null,
    /// The stage the next dispatches belong to; with `only`, the others are not encoded (the kernel bench).
    cur: Stage = .embed,
    only: ?Stage = null,
    dispatches: *usize,
    /// The encoder is concurrent: a dispatch waits for every earlier one (a barrier) unless `join` is set, which
    /// lets the next dispatch run beside the last (it reads and writes nothing the last writes).
    join: *bool,

    fn on(x: *const Enc) bool {
        if (x.only) |o| if (o != x.cur) return false;
        x.dispatches.* += 1;
        if (x.join.*) x.join.* = false else x.e.barrier();
        return true;
    }

    /// The next dispatch may run beside the last one.
    fn beside(x: *const Enc) void {
        if (x.only == null) x.join.* = true;
    }

    /// The dispatches from here on are stage `next`'s; with a profile, the stage before ends its own command buffer.
    fn mark(x: *Enc, next: Stage) !void {
        defer x.cur = next;
        const p = x.prof orelse return;
        x.e.end();
        x.cb.commit();
        x.cb.wait();
        if (x.cb.failure()) |msg| {
            std.log.err("glm_ax forward ({s}): {s}", .{ @tagName(x.cur), msg });
            return error.GpuFailed;
        }
        p.gpu[@intFromEnum(x.cur)] += x.cb.gpuSeconds();
        x.cb = x.m.queue.commandBuffer();
        x.e = x.cb.compute(.concurrent);
    }

    fn bindAll(x: Enc, args: anytype, bufs: []const B) void {
        var slot: usize = 0;
        for (bufs) |r| {
            if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
            slot += 1;
        }
    }

    fn glue(x: Enc, g: Glue, args: Args, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        if (!x.on()) return;
        x.e.setPipeline(x.m.pipe(g));
        x.bindAll(args, bufs);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    fn own(x: Enc, g: Own, args: GArgs, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        if (!x.on()) return;
        x.e.setPipeline(x.m.kernel(g));
        x.bindAll(args, bufs);
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    fn project(x: Enc, w: wts.Linear, in: mtl.Buffer, in_off: usize, out: mtl.Buffer, out_off: usize, n_rows: usize) !void {
        if (!x.on()) return;
        if (n_rows > st.window_rows) {
            // a prompt chunk: each group's row sums, then the tensor-unit product (each weight read once a chunk)
            affine_mm.rowSums(x.e, x.m.mm, @intCast(w.group), bo(in, in_off), b(x.s.sums), @intCast(n_rows), @intCast(w.k));
            x.e.barrier();
            affine_mm.dense(x.e, x.m.mm, bo(in, in_off), b(x.s.sums), mat(w), bo(out, out_off), @intCast(n_rows));
            return;
        }
        const call = try rows.call(&x.m.rows, w, n_rows, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, in_off, 0);
        x.e.setBuffer(w.w, w.w_off, 1);
        x.e.setBuffer(w.scales, w.s_off, 2);
        x.e.setBuffer(w.biases, w.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, out_off, 5);
        x.e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    fn experts(x: Enc, w: rows.Experts, in: mtl.Buffer, out: mtl.Buffer, ids: mtl.Buffer, slots: usize) !void {
        if (!x.on()) return;
        const call = try rows.callIndexed(&x.m.rows, w, slots, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, 0, 0);
        x.e.setBuffer(w.one.w, w.one.w_off, 1);
        x.e.setBuffer(w.one.scales, w.one.s_off, 2);
        x.e.setBuffer(w.one.biases, w.one.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, 0, 5);
        x.e.setBuffer(ids, 0, 6);
        x.e.dispatchGroups(mtl.Size.of(call.groups, slots, 1), mtl.Size.of(call.threads, 1, 1));
    }

    /// `batch` matrices [n, k] back to back in w, one a head: y[r, head] = x[r, head, :k] W_head^T on the tensor units.
    fn heads(x: Enc, w: wts.Linear, in: mtl.Buffer, in_stride: usize, out: mtl.Buffer, out_stride: usize, n_rows: usize, batch: usize) void {
        if (!x.on()) return;
        if (n_rows <= st.window_rows) {
            // decode rows: each head's weights read once for every row, one launch (gx_heads_gemv)
            x.e.setPipeline(x.m.kernel(.heads_gemv));
            x.e.setBuffer(in, 0, 0);
            x.e.setBuffer(w.w, w.w_off, 1);
            x.e.setBuffer(w.scales, w.s_off, 2);
            x.e.setBuffer(w.biases, w.b_off, 3);
            x.e.setValue(GArgs{ .rows = @intCast(n_rows), .heads = @intCast(batch), .width = @intCast(w.k), .stride = @intCast(w.n), .in_stride = @intCast(in_stride), .score_stride = @intCast(out_stride) }, 4);
            x.e.setBuffer(out, 0, 5);
            x.e.dispatchGroups(mtl.Size.of(w.n / 4, batch, 1), mtl.Size.of(128, 1, 1));
            return;
        }
        const rows_: u32 = @intCast(n_rows);
        const bt: u32 = @intCast(batch);
        affine_mm.rowSumsAt(x.e, x.m.mm, @intCast(w.group), b(in), b(x.s.sums), rows_ * bt, @intCast(w.k), @intCast(in_stride));
        x.e.barrier();
        affine_mm.denseBatch(x.e, x.m.mm, b(in), b(x.s.sums), mat(w), b(out), rows_, bt, .{
            .x_row = @intCast(batch * in_stride),
            .y_row = @intCast(batch * out_stride),
            .sums_row = @intCast(batch),
            .x_batch = @intCast(in_stride),
            .y_batch = @intCast(out_stride),
            .sums_batch = 1,
            .w_batch = @intCast(w.n),
        });
    }

    fn norm(x: Enc, resid: B, w: wts.Tensor, out: B, n: usize, in_stride: usize, out_stride: usize) void {
        const c = &x.m.config;
        x.own(.norm, .{ .rows = @intCast(n), .width = c.hidden, .stride = @intCast(out_stride), .in_stride = @intCast(in_stride), .eps = c.eps }, &.{ resid, t(w), A, out }, .{ n, 1, 1 }, .{ norm_threads, 1, 1 }, true);
    }

    fn add(x: Enc, y: mtl.Buffer, n: usize) void {
        const c = &x.m.config;
        x.own(.add, .{ .rows = @intCast(n), .width = c.hidden }, &.{ b(x.s.resid), b(y), A }, .{ c.hidden, n, 1 }, .{ 256, 1, 1 }, false);
    }
};

/// Beyond the cache: every row's logits (s.logits_all) or the last's, the rows after the final norm (`hidden`, bf16).
pub const Out = struct {
    all: bool = false,
    hidden: ?B = null,
    head: bool = true,
    profile: ?*Profile = null,
    only: ?Stage = null,
    dispatches: ?*usize = null,
    gpu: ?*f64 = null,
    /// Encode only: the step's command buffer, retained and not committed, into `pending`; its ids are written by
    /// `Pending.commit` (positions from `start`).
    pending: ?*Pending = null,
    start: ?usize = null,
};

/// A decode step encoded ahead: its command buffer (retained) and the ids buffer its embed reads.
pub const Pending = struct {
    cb: mtl.CommandBuffer = undefined,
    ids: mtl.Buffer = undefined,
    start: usize = 0,
    n: usize = 0,
    /// Encoded with every row's logits (a verify window's) rather than the last row's into the cache.
    all: bool = false,
    /// The GPU seconds once waited.
    gpu: f64 = 0,
    live: bool = false,

    /// Write the step's tokens and commit it.
    pub fn commit(p: *Pending, tokens: []const u32) void {
        std.debug.assert(p.live and tokens.len == p.n);
        @memcpy(p.ids.slice(u32, p.n), tokens);
        p.cb.commit();
    }

    /// Wait for a committed step; the buffer is released.
    pub fn wait(p: *Pending) !void {
        defer p.drop();
        p.cb.wait();
        p.gpu = p.cb.gpuSeconds();
        if (p.cb.failure()) |msg| {
            std.log.err("glm_ax forward: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    /// Give up an encoded step (never committed, or done).
    pub fn drop(p: *Pending) void {
        if (p.live) p.cb.release();
        p.live = false;
    }
};

fn rowTable(s: *st.Scratch, c: *const cfg.Config, start: usize, n: usize) void {
    for (s.rowtab.slice(st.Row, n), 0..) |*r, i| r.* = .{ .pos = @intCast(start + i), .list = @intCast(i * c.i_topk) };
}

/// `ids` at positions cache.len.. through the model: keys into the cache (len unmoved), logits out.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    const c = &m.config;
    const n = ids.len;
    const start = out.start orelse cache.len;
    if (n == 0 or n > s.rows or (out.all and n > st.window_rows)) return error.GlmRowsExceeded;
    if (start + n > cache.capacity or cache.capacity > s.capacity) return error.GlmContextFull;
    for (ids) |id| if (id >= c.vocab) return error.InvalidGlmToken;
    if (out.pending == null) @memcpy(s.ids.slice(u32, n), ids);
    rowTable(s, c, start, n);
    const h: usize = c.hidden;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    var count: usize = 0;
    var join = false;
    var x = Enc{ .e = cb.compute(.concurrent), .cb = cb, .m = m, .s = s, .prof = out.profile, .only = out.only, .dispatches = out.dispatches orelse &count, .join = &join };
    const W = m.weights;
    embed(x, n);
    for (W.layers[0..W.main], 0..) |l, i| try layer(&x, cache, l, i, start, n, i > 0, if (i + 1 < W.main) W.layers[i + 1].input_norm else null);
    try x.mark(.head);
    // the rows after the final norm: the head's input, and the draft head's (kept in `hidden` when asked for)
    const normed = out.hidden orelse b(s.hid);
    if (out.head or out.hidden != null) {
        const w = W.norm orelse return error.GlmHasNoHead;
        if (out.all or out.hidden != null) x.norm(b(s.resid), w, normed, n, 0, 0) else x.norm(bo(s.resid, (n - 1) * h * 4), w, bo(normed.buf, normed.off + (n - 1) * h * 2), 1, 0, 0);
    }
    if (out.head) {
        const head = W.head orelse return error.GlmHasNoHead;
        if (out.all) try x.project(head, normed.buf, normed.off, s.logits_all, 0, n) else try x.project(head, normed.buf, normed.off + (n - 1) * h * 2, cache.logits, 0, 1);
    }
    try x.mark(.embed);
    if (out.pending) |p| {
        x.e.end();
        p.* = .{ .cb = x.cb.retain(), .ids = s.ids, .start = start, .n = n, .all = out.all, .live = true };
        return;
    }
    try finish(x);
    if (out.gpu) |g| g.* += x.cb.gpuSeconds();
    if (out.profile) |p| p.runs += 1;
}

/// The MTP layer on `tokens` and the normed bf16 `hidden` rows before them: its rows after the shared head norm into
/// s.hid (the next draft's hidden), the last row's logits in s.logits_all.
pub fn mtp(m: *const Model, s: *st.Scratch, cache: *st.Cache, hidden: B, tokens: []const u32, start: usize) !void {
    const c = &m.config;
    const n = tokens.len;
    const head = m.weights.mtp orelse return error.GlmHasNoMtp;
    if (n == 0 or n > s.rows) return error.GlmRowsExceeded;
    if (start + n > cache.capacity) return error.GlmContextFull;
    @memcpy(s.ids.slice(u32, n), tokens);
    rowTable(s, c, start, n);
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    var count: usize = 0;
    var join = false;
    var x = Enc{ .e = cb.compute(.concurrent), .cb = cb, .m = m, .s = s, .dispatches = &count, .join = &join };
    embed(x, n);
    x.norm(b(s.resid), head.enorm, b(s.cat), n, 0, 2 * h);
    x.own(.norm16, .{ .rows = nr, .width = c.hidden, .stride = @intCast(2 * h), .eps = c.eps }, &.{ hidden, t(head.hnorm), A, bo(s.cat, h * 2) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(head.eh, s.cat, 0, s.y, 0, n);
    x.glue(.widen, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    const i = m.weights.main;
    try layer(&x, cache, m.weights.layers[i], c.layers, start, n, false, null);
    x.norm(b(s.resid), head.norm, b(s.hid), n, 0, 0);
    try x.project(m.weights.head orelse return error.GlmHasNoHead, s.hid, (n - 1) * h * 2, s.logits_all, 0, 1);
    try finish(x);
}

/// Embeddings of s.ids's first n into the fp32 residual.
fn embed(x: Enc, n: usize) void {
    const c = &x.m.config;
    const E = x.m.weights.embed;
    const args = Args{ .rows = @intCast(n), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) };
    x.glue(.embed, args, &.{ b(x.s.ids), bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(x.s.resid) }, .{ c.hidden, n, 1 }, .{ 256, 1, 1 }, false);
}

fn finish(x: Enc) !void {
    x.e.end();
    x.cb.commit();
    x.cb.wait();
    x.s.last_gpu = x.cb.gpuSeconds();
    if (x.cb.failure()) |msg| {
        std.log.err("glm_ax forward: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Decoder layer `l` with cache slot `i` over rows at positions start..: attention, then the MLP or the experts.
/// `normed`: s.x already holds the rows after this layer's input norm (the layer before fused it); `next`: the
/// following layer's input norm, fused into this layer's last residual add (null: a plain add).
fn layer(x: *Enc, cache: *st.Cache, l: wts.Layer, i: usize, start: usize, n: usize, normed: bool, next: ?wts.Tensor) !void {
    const m = x.m;
    const s = x.s;
    const c = &m.config;
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    try x.mark(.norm);
    if (!normed) x.norm(b(s.resid), l.input_norm, b(s.x), n, 0, 0);
    try attention(x, cache, l.attn, i, start, n);
    try x.mark(.norm);
    x.own(.add_norm, .{ .rows = nr, .width = c.hidden, .eps = c.eps }, &.{ b(s.resid), b(s.y), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ norm_threads, 1, 1 }, true);
    switch (l.ffn) {
        .dense => |d| {
            try x.mark(.dense);
            try mlp(x.*, d, s, s.y, n, c.dense_inter);
            if (next) |w| x.own(.add_norm, .{ .rows = nr, .width = c.hidden, .eps = c.eps }, &.{ b(s.resid), b(s.y), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ norm_threads, 1, 1 }, true) else x.add(s.y, n);
        },
        .moe => |e| {
            try x.mark(.route);
            if (n > st.window_rows and x.on()) {
                // a prompt chunk: every row's logits in one bf16 tensor-unit GEMM, then every row's top-k at once
                x.dispatches.* += 1;
                const E: usize = c.experts;
                const H: usize = c.hidden;
                const tn = (E + 127) / 128;
                const tm = (n + 63) / 64;
                x.e.setPipeline(m.gemm[@as(usize, @intFromBool(n % 64 == 0)) * 2 + @intFromBool(E % 128 == 0)]);
                x.e.setBuffer(s.x, 0, 0);
                x.e.setBuffer(e.gate_w.buffer, e.gate_w.offset, 1);
                var p: [16]i32 = @splat(0);
                for ([_]usize{ n, E, H, H, H, E, tn, tm, 2, H / 256 }, 0..) |v, j| p[j] = @intCast(v);
                x.e.setBytes(std.mem.asBytes(&p), 2);
                x.e.setBuffer(s.logits16, 0, 3);
                x.e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
                x.own(.route_rows, .{ .width = c.experts, .heads = c.topk, .scale = c.routed_scale }, &.{ b(s.logits16), b(e.bias), A, b(s.pick), b(s.wts) }, .{ n, 1, 1 }, .{ c.experts, 1, 1 }, true);
            }
            // GLM_AX_MOE_FUSED=1, decode: the nine experts' gate/up/SwiGLU in one launch and their down in another
            const fused = n <= st.window_rows and std.c.getenv("GLM_AX_MOE_FUSED") != null;
            // decode with no stage filter: the shared expert's three launches run beside the routed path's
            const interleave = n <= st.window_rows and x.only == null and !fused;
            var r0: usize = 0;
            while (n <= st.window_rows and r0 < n and x.on()) : (r0 += st.window_rows) { // decode: 16-row launches
                x.dispatches.* += 1;
                const blk: u32 = @intCast(@min(st.window_rows, n - r0));
                const at = r0 * c.topk * 4;
                route.logits(x.e, m.route[0], .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk }, bo(s.x, r0 * h * 2), b(e.router), b(s.logits32), blk);
                if (interleave) {
                    // the shared expert's gate/up beside the router's logits (both read s.x only)
                    x.beside();
                    try x.project(e.shared.gate_up, s.x, 0, s.gate, 0, n);
                }
                x.e.barrier();
                route.select(x.e, m.route[1], b(s.logits32), b(e.bias), c.routed_scale, blk, .{ 0, c.experts }, .{
                    .pick = bo(s.pick, at),
                    .wts = bo(s.wts, at),
                    .ids = b(s.route[0]),
                    .members = b(s.route[1]),
                    .count = b(s.route[2]),
                    .mine = b(s.route[3]),
                    .theirs = b(s.route[4]),
                    .counts = b(s.route[5]),
                    .word = b(s.route[6]),
                });
                if (interleave) {
                    // its SwiGLU beside the select (the barrier before the select saw gate/up done)
                    x.beside();
                    x.own(.swiglu, .{ .rows = @intCast(n), .width = c.moe_inter }, &.{ b(s.gate), A, b(s.act) }, .{ c.moe_inter, n, 1 }, .{ 256, 1, 1 }, false);
                }
            }
            try x.mark(.experts);
            const slots = n * c.topk;
            if (x.prof) |p| if (n <= st.window_rows) {
                var seen: [512]bool = @splat(false);
                var k: usize = 0;
                for (s.pick.slice(u32, slots)) |ex| if (ex < seen.len and !seen[ex]) {
                    seen[ex] = true;
                    k += 1;
                };
                p.distinct += k;
                p.moe_layers += 1;
            };
            if (fused) {
                // by expert: each distinct routed expert of the window (the route's list) and the shared one, its
                // weights read once for every member row (glm_ax_moe.metal, on core/row_projection's reduction)
                const cap: u32 = @intCast(@min(n * c.topk, c.experts));
                const sa = SetArgs{ .rows = nr, .topk = c.topk, .cap = cap };
                if (x.on()) {
                    x.e.setPipeline(m.moe_rows[2]);
                    x.e.setBuffer(s.x, 0, 0);
                    x.e.setBuffer(s.route[0], 0, 1);
                    x.e.setBuffer(s.route[1], 0, 2);
                    x.e.setBuffer(s.route[2], 0, 3);
                    inline for (.{ e.gate.one, e.up.one, e.shared.gate_up }, 0..) |w, j| {
                        x.e.setBuffer(w.w, w.w_off, 4 + 3 * j);
                        x.e.setBuffer(w.scales, w.s_off, 5 + 3 * j);
                        x.e.setBuffer(w.biases, w.b_off, 6 + 3 * j);
                    }
                    x.e.setValue(sa, 13);
                    x.e.setBuffer(s.ea, 0, 14);
                    x.e.dispatchGroups(mtl.Size.of(c.moe_inter / 8, cap + 1, 1), mtl.Size.of(64, 1, 1));
                }
                if (x.on()) {
                    x.e.setPipeline(m.moe_rows[3]);
                    x.e.setBuffer(s.ea, 0, 0);
                    x.e.setBuffer(s.route[0], 0, 1);
                    x.e.setBuffer(s.route[1], 0, 2);
                    x.e.setBuffer(s.route[2], 0, 3);
                    inline for (.{ e.down.one, e.shared.down }, 0..) |w, j| {
                        x.e.setBuffer(w.w, w.w_off, 4 + 3 * j);
                        x.e.setBuffer(w.scales, w.s_off, 5 + 3 * j);
                        x.e.setBuffer(w.biases, w.b_off, 6 + 3 * j);
                    }
                    x.e.setValue(sa, 10);
                    x.e.setBuffer(s.ey, 0, 11);
                    x.e.setBuffer(s.shared, 0, 12);
                    x.e.dispatchGroups(mtl.Size.of(c.hidden / 8, cap + 1, 1), mtl.Size.of(64, 1, 1));
                }
            } else if (n > st.window_rows) {
                try expertsByExpert(x.*, e, n);
            } else {
                var gate = e.gate;
                gate.repeat = c.topk;
                var up = e.up;
                up.repeat = c.topk;
                try x.experts(gate, s.x, s.eg, s.pick, slots);
                x.beside();
                try x.experts(up, s.x, s.eu, s.pick, slots);
                if (interleave) {
                    // the shared expert's down beside the routed gate and up
                    x.beside();
                    try x.project(e.shared.down, s.act, 0, s.shared, 0, n);
                }
                x.glue(.swiglu, Args{ .rows = @intCast(slots), .width = c.moe_inter, .limit = no_limit }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_inter, slots, 1 }, .{ 256, 1, 1 }, false);
                try x.experts(e.down, s.ea, s.ey, s.pick, slots);
            }
            try x.mark(.shared);
            if (!interleave and !fused) try mlp(x.*, e.shared, s, s.shared, n, c.moe_inter);
            try x.mark(.moe_add);
            if (next) |w| {
                x.own(.moe_add_norm, .{ .rows = nr, .width = c.hidden, .heads = c.topk, .eps = c.eps }, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ norm_threads, 1, 1 }, true);
            } else x.glue(.moe_add, Args{ .rows = nr, .width = c.hidden, .heads = c.topk }, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
    }
}

/// A chunk's routed experts by expert (each expert's weights read once): sorted, gate/up, SwiGLU, down back to s.ey.
fn expertsByExpert(x: Enc, e: wts.Moe, n: usize) !void {
    if (!x.on()) return;
    const c = &x.m.config;
    const s = x.s;
    const slots: u32 = @intCast(n * c.topk);
    const p = x.m.mm;
    const ex: u32 = c.experts;
    x.glue(.sort, Args{ .rows = slots, .width = ex }, &.{ b(s.pick), A, b(s.offsets), b(s.order) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
    x.glue(.rows_by_order, Args{ .rows = slots, .width = c.hidden, .heads = c.topk }, &.{ b(s.x), b(s.order), A, b(s.xs) }, .{ c.hidden, slots, 1 }, .{ 256, 1, 1 }, false);
    x.e.barrier();
    affine_mm.rowSums(x.e, p, @intCast(e.gate.one.group), b(s.xs), b(s.sums), slots, c.hidden);
    x.e.barrier();
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.gate.one), b(s.offsets), b(s.eg), slots, ex);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.up.one), b(s.offsets), b(s.eu), slots, ex);
    x.glue(.swiglu, Args{ .rows = slots, .width = c.moe_inter, .limit = no_limit }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_inter, slots, 1 }, .{ 256, 1, 1 }, false);
    x.e.barrier();
    affine_mm.rowSums(x.e, p, @intCast(e.down.one.group), b(s.ea), b(s.sums), slots, c.moe_inter);
    x.e.barrier();
    affine_mm.gatherTo(x.e, p, b(s.ea), b(s.sums), mat(e.down.one), b(s.offsets), b(s.ey), b(s.order), slots, ex);
}

/// gate and up from s.x, SwiGLU, down into `out`.
fn mlp(x: Enc, w: wts.Mlp, s: *st.Scratch, out: mtl.Buffer, n: usize, inter: u32) !void {
    try x.project(w.gate_up, s.x, 0, s.gate, 0, n);
    x.own(.swiglu, .{ .rows = @intCast(n), .width = inter }, &.{ b(s.gate), A, b(s.act) }, .{ inter, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(w.down, s.act, 0, out, 0, n);
}

/// MLA of the normed rows s.x into s.y: keys into the layer's cache, a full layer's key lists, attention in latent
/// space over each row's list, the per-head output maps and o_proj.
fn attention(x: *Enc, cache: *st.Cache, w: wts.Attention, i: usize, start: usize, n: usize) !void {
    const m = x.m;
    const s = x.s;
    const c = &m.config;
    const nr: u32 = @intCast(n);
    const nh: usize = c.heads;
    const qk: usize = c.qHead();
    const rt = @sizeOf(st.Row);
    const keys = bo(cache.kv, cache.keys(i));
    try x.mark(.proj_x);
    // q_a, kv_a and a full layer's wk and weights_proj in one product: a row of px columns
    try x.project(w.x, s.x, 0, s.qa, 0, n);
    const px: usize = std.mem.alignForward(usize, w.x.n, if (n > st.window_rows) 64 else 1);
    const ga = GArgs{ .rows = nr, .heads = c.heads, .in_stride = @intCast(px), .eps = mla_eps, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk))), .score_stride = @intCast(s.capacity), .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_heads))) };
    try x.mark(.prep);
    x.own(.prep, ga, &.{ b(s.qa), bo(s.qa, wts.kv_col * 2), t(w.q_norm), t(w.kv_norm), b(m.inv), b(s.rowtab), A, b(s.qr), keys }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    if (w.indexer != null) x.beside();
    if (w.indexer) |ix| x.own(.index_key, ga, &.{ bo(s.qa, wts.wk_col * 2), t(ix.k_norm), t(ix.k_bias), b(m.inv), b(s.rowtab), A, bo(cache.index, cache.indexKeys(i)) }, .{ n, 1, 1 }, .{ 32, 1, 1 }, true);
    try x.mark(.proj_q);
    try x.project(w.q_b, s.qr, 0, s.q, 0, n);
    // decode rows only: a prompt chunk's projection is row sums then a GEMM over them, one sums buffer for both
    if (w.indexer != null and n <= st.window_rows) x.beside();
    if (w.indexer) |ix| try x.project(ix.wq_b, s.qr, 0, s.iq, 0, n);
    if (w.indexer != null) {
        try x.mark(.index);
        const iq_row: usize = @as(usize, c.i_heads) * c.i_dim * 2;
        const nax = n > st.window_rows and start + n > c.i_topk;
        if (nax) x.own(.index_qrope, ga, &.{ b(s.iq), b(s.rowtab), b(m.inv), A, b(s.iqr) }, .{ c.i_heads * 64, n, 1 }, .{ 256, 1, 1 }, false);
        var r0: usize = 0;
        while (r0 < n) : (r0 += st.score_rows) {
            const blk: usize = @min(st.score_rows, n - r0);
            const last = start + r0 + blk - 1;
            var a = ga;
            a.rows = @intCast(blk);
            // a threadgroup scores 256 keys: every key to the block's last position
            if (last >= c.i_topk and nax) {
                if (x.on()) {
                    x.e.setPipeline(m.index_nax);
                    x.e.setBuffer(s.iqr, r0 * iq_row, 0);
                    x.e.setBuffer(s.qa, (r0 * px + wts.wp_col) * 2, 1);
                    x.e.setBuffer(cache.index, cache.indexKeys(i), 2);
                    x.e.setBuffer(s.score, 0, 3);
                    x.e.setValue(ScoreArgs{ .p0 = @intCast(start + r0), .q_stride = c.i_heads * c.i_dim, .w_stride = @intCast(px), .s_stride = @intCast(s.capacity), .rows = @intCast(blk), .wscale = ga.head_scale * ga.key_scale }, 4);
                    // 4 simdgroups: 64 rows a threadgroup, the block's rows in one row tile
                    x.e.dispatchGroups(mtl.Size.of(last / 64 + 1, (blk + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                }
            } else if (last >= c.i_topk) x.own(.index_scores, a, &.{ bo(s.iq, r0 * iq_row), bo(s.qa, (r0 * px + wts.wp_col) * 2), bo(cache.index, cache.indexKeys(i)), bo(s.rowtab, r0 * rt), A, b(s.score), b(m.inv) }, .{ last / 256 + 1, blk, 1 }, .{ 128, 1, 1 }, true);
            x.own(.index_topk, a, &.{ b(s.score), bo(s.rowtab, r0 * rt), A, b(s.keys) }, .{ blk, 1, 1 }, .{ 1024, 1, 1 }, true);
        }
    }
    // each head's latent query (embed_q of its 192 nope dims) and its roped 64
    try x.mark(.absorb);
    const qw: usize = c.kv_lora + c.rope; // a head's query row: its latent query, then its roped 64
    // decode rows: embed_q and the rope in one launch (gx_absorb), unless GLM_AX_UNFUSED
    const fuse = n <= st.window_rows and std.c.getenv("GLM_AX_UNFUSED") == null;
    const fuse_unembed = fuse and std.c.getenv("GLM_AX_FUSE_UNEMBED") != null;
    if (fuse) {
        const we = w.embed_q;
        if (x.on()) {
            x.e.setPipeline(m.kernel(.absorb));
            x.e.setBuffer(s.q, 0, 0);
            x.e.setBuffer(we.w, we.w_off, 1);
            x.e.setBuffer(we.scales, we.s_off, 2);
            x.e.setBuffer(we.biases, we.b_off, 3);
            x.e.setValue(GArgs{ .rows = @intCast(n), .heads = @intCast(nh), .width = @intCast(we.k), .stride = @intCast(we.n), .in_stride = @intCast(qk), .score_stride = @intCast(qw) }, 4);
            x.e.setBuffer(s.qlat, 0, 5);
            x.e.setBuffer(s.rowtab, 0, 6);
            x.e.setBuffer(m.inv, 0, 7);
            x.e.dispatchGroups(mtl.Size.of(we.n / 4 + 1, nh, 1), mtl.Size.of(128, 1, 1));
        }
    } else {
        x.heads(w.embed_q, s.q, qk, s.qlat, qw, n, nh);
        x.beside();
        x.own(.qrope, ga, &.{ b(s.q), b(s.rowtab), b(m.inv), A, b(s.qlat) }, .{ nh * c.rope / 2, n, 1 }, .{ 256, 1, 1 }, false);
    }
    try x.mark(.attend);
    if (n > st.window_rows) {
        // a prompt chunk: each row's listed keys on the tensor units, 16 heads a threadgroup (glm_ax_sparse.metal)
        if (x.on()) {
            x.e.setPipeline(m.sparse_nax);
            x.e.setBuffer(s.qlat, 0, 0);
            x.e.setBuffer(keys.buf, keys.off, 1);
            x.e.setBuffer(s.keys, 0, 2);
            x.e.setValue(ga.scale, 3);
            x.e.setValue([4]i32{ @intCast(c.i_topk), @intCast(start + n), @intCast(start), 0 }, 4);
            x.e.setBuffer(s.olat, 0, 5);
            x.e.dispatchGroups(mtl.Size.of(nh / 16, n, 1), mtl.Size.of(128, 1, 1));
        }
    } else if (std.c.getenv("GLM_AX_OLD_ATTN") == null) {
        // decode rows: each row's key list in attn_blocks blocks on the tensor units, then the blocks merged
        const width: usize = @min(c.i_topk, start + n); // the longest list a row of the window has
        const span: usize = @max(32, std.mem.alignForward(usize, (width + st.attn_blocks - 1) / st.attn_blocks, 32));
        const blocks: usize = (width + span - 1) / span;
        if (x.on()) {
            x.e.setPipeline(m.sparse_split);
            x.e.setBuffer(s.qlat, 0, 0);
            x.e.setBuffer(keys.buf, keys.off, 1);
            x.e.setBuffer(s.keys, 0, 2);
            x.e.setValue(ga.scale, 3);
            x.e.setValue([4]i32{ @intCast(c.i_topk), @intCast(start + n), @intCast(span), @intCast(start) }, 4);
            x.e.setBuffer(s.po, 0, 5);
            x.e.setBuffer(s.pm, 0, 6);
            x.e.setBuffer(s.pl, 0, 7);
            x.e.dispatchGroups(mtl.Size.of(nh / 16, blocks, n), mtl.Size.of(128, 1, 1));
        }
        if (fuse_unembed) {
            // the blocks merged and each head's unembed in one launch (gx_combine_unembed): s.o directly. Opt-in
            // (GLM_AX_FUSE_UNEMBED=1): bit-identical but ~0.5 ms a step slower, 64 threadgroups a row too few
            const wu = w.unembed;
            if (x.on()) {
                x.e.setPipeline(m.combine_unembed);
                x.e.setBuffer(s.po, 0, 0);
                x.e.setBuffer(s.pm, 0, 1);
                x.e.setBuffer(s.pl, 0, 2);
                x.e.setValue([4]i32{ @intCast(blocks), @intCast(wu.n), 0, 0 }, 3);
                x.e.setBuffer(wu.w, wu.w_off, 4);
                x.e.setBuffer(wu.scales, wu.s_off, 5);
                x.e.setBuffer(wu.biases, wu.b_off, 6);
                x.e.setBuffer(s.o, 0, 7);
                x.e.dispatchGroups(mtl.Size.of(nh, n, 1), mtl.Size.of(256, 1, 1));
            }
        } else if (x.on()) {
            x.e.setPipeline(m.sparse_combine);
            x.e.setBuffer(s.po, 0, 0);
            x.e.setBuffer(s.pm, 0, 1);
            x.e.setBuffer(s.pl, 0, 2);
            x.e.setValue(@as(i32, @intCast(blocks)), 3);
            x.e.setBuffer(s.olat, 0, 4);
            x.e.dispatchGroups(mtl.Size.of(nh, n, 1), mtl.Size.of(128, 1, 1));
        }
    } else {
        var r0: usize = 0;
        while (r0 < n) : (r0 += st.attend_rows) {
            const blk: usize = @min(st.attend_rows, n - r0);
            var a = ga;
            a.rows = @intCast(blk);
            x.own(.attend, a, &.{ bo(s.qlat, r0 * nh * qw * 2), b(s.qlat), bo(s.rowtab, r0 * rt), b(s.keys), keys, A, b(s.partial) }, .{ (nh + 3) / 4, 16, blk }, .{ 128, 1, 1 }, true);
            x.own(.merge, a, &.{ b(s.partial), bo(s.rowtab, r0 * rt), A, bo(s.olat, r0 * nh * c.kv_lora * 2) }, .{ nh, blk, 1 }, .{ 256, 1, 1 }, true);
        }
    }
    try x.mark(.unembed);
    const fused_unembed = fuse_unembed and std.c.getenv("GLM_AX_OLD_ATTN") == null;
    if (!fused_unembed) x.heads(w.unembed, s.olat, c.kv_lora, s.o, c.v_dim, n, nh);
    try x.mark(.o_proj);
    try x.project(w.o, s.o, 0, s.y, 0, n);
}
