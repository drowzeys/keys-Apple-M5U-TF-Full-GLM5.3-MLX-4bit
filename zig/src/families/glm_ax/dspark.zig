//! RedHat AI's DSpark drafter for GLM-5.3 (speculators' DSparkDraftModel; vLLM's qwen3_dspark and its dspark
//! speculator, Apache-2.0): DFlash's block over the target's taps with three Qwen3 layers of 64-wide heads, causal
//! within the block, every one of the block's 8 rows predicting (the anchor's too), a vanilla Markov head added to
//! the target head's logits as the block is walked left to right, and a confidence head (a sigmoid a slot) that the
//! lane engine's chance cut uses to choose how many drafts a window verifies. bf16 weights as released, the block's
//! products by glm_ax's DFlash2 port's row GEMV. The Markov bias is taken over each slot's best 32 base logits
//! (vLLM's dspark_draft_topk restriction): drafts change speed only, never the target's output.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ck = @import("../../core/checkpoint_metal.zig");
const rows_mod = @import("../../core/row_projection.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const K = @import("dflash2.zig").Kernel;
const Metal = @import("backend.zig").Metal;
const be = lanes.backend;

const dr = lanes.drafter;
const Stream = lanes.Stream;

pub const Config = struct {
    layers: u32 = 3,
    heads: u32 = 64,
    hidden: u32 = 6144,
    ffn: u32 = 12288,
    window: u32 = 2048,
    block: u32 = 8,
    mask: u32 = 154856,
    rank: u32 = 256,
    vocab: u32 = 154880,
    /// the target layers whose outputs the drafter reads: speculators' aux ids name a layer's input, so id - 1
    taps: [5]u32 = .{ 1, 19, 38, 57, 74 },
    theta: f32 = 8e6,
    eps: f32 = 1e-5,

    pub fn ring(c: *const Config) u32 {
        return std.mem.alignForward(u32, c.window + c.block, 128);
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDSparkConfig;
        const o = doc.value.object;
        const tl = o.get("transformer_layer_config") orelse return error.BadDSparkConfig;
        if (tl != .object) return error.BadDSparkConfig;
        const int = struct {
            fn of(x: std.json.ObjectMap, k: []const u8) !u32 {
                const v = x.get(k) orelse return error.BadDSparkConfig;
                if (v != .integer or v.integer <= 0) return error.BadDSparkConfig;
                return @intCast(v.integer);
            }
            fn f(v: ?std.json.Value) !f32 {
                return switch (v orelse return error.BadDSparkConfig) {
                    .integer => |i| @floatFromInt(i),
                    .float => |q| @floatCast(q),
                    else => error.BadDSparkConfig,
                };
            }
        };
        const tc = tl.object;
        const rope = tc.get("rope_parameters") orelse return error.BadDSparkConfig;
        if (rope != .object) return error.BadDSparkConfig;
        var c = Config{
            .layers = try int.of(tc, "num_hidden_layers"),
            .heads = try int.of(tc, "num_attention_heads"),
            .hidden = try int.of(tc, "hidden_size"),
            .ffn = try int.of(tc, "intermediate_size"),
            .window = try int.of(tc, "sliding_window"),
            .vocab = try int.of(tc, "vocab_size"),
            .block = try int.of(o, "block_size"),
            .mask = try int.of(o, "mask_token_id"),
            .rank = try int.of(o, "markov_rank"),
            .theta = try int.f(rope.object.get("rope_theta")),
            .eps = try int.f(tc.get("rms_norm_eps")),
        };
        const ids = o.get("aux_hidden_state_layer_ids") orelse return error.BadDSparkConfig;
        if (ids != .array or ids.array.items.len != c.taps.len) return error.UnsupportedDSpark;
        for (ids.array.items, &c.taps) |x, *y| {
            if (x != .integer or x.integer < 1) return error.BadDSparkConfig;
            y.* = @intCast(x.integer - 1);
        }
        if ((try int.of(tc, "head_dim")) != 64 or (try int.of(tc, "num_key_value_heads")) != c.heads or c.layers > 8 or c.block != max_block or c.rank != 256) return error.UnsupportedDSpark;
        return c;
    }
};

pub const max_block = 8;
pub const max_blocks = 8;
const absorb_rows = 2048;
/// candidates a slot's Markov walk scores: the best of the base logits
const top = 32;

const Layer = struct { in_norm: wts.Tensor, post_norm: wts.Tensor, q_norm: wts.Tensor, k_norm: wts.Tensor, q: wts.Tensor, k: wts.Tensor, v: wts.Tensor, o: wts.Tensor, gate: wts.Tensor, up: wts.Tensor, down: wts.Tensor };
const Weights = struct { fc: wts.Tensor, hidden_norm: wts.Tensor, norm: wts.Tensor, w1: wts.Tensor, w2: wts.Tensor, cw: wts.Tensor, cb: wts.Tensor, layers: [8]Layer };

fn tensor(c: *const ck.Checkpoint, name: []const u8, count: usize) !wts.Tensor {
    const x = c.get(name) catch {
        std.log.err("glm_ax DSpark: no tensor {s}", .{name});
        return error.UnexpectedDSparkTensor;
    };
    if (x.dtype != .bf16 or x.count() != count) {
        std.log.err("glm_ax DSpark: {s} is not bf16[{d}]", .{ name, count });
        return error.UnexpectedDSparkTensor;
    }
    return x;
}

fn load(ckp: *const ck.Checkpoint, c: *const Config) !Weights {
    const h: usize = c.hidden;
    const qw: usize = c.heads * 64;
    var w: Weights = .{
        .fc = try tensor(ckp, "dspark.fc.weight", h * h * c.taps.len),
        .hidden_norm = try tensor(ckp, "dspark.hidden_norm.weight", h),
        .norm = try tensor(ckp, "dspark.norm.weight", h),
        .w1 = try tensor(ckp, "dspark.markov_head.markov_w1.weight", @as(usize, c.vocab) * c.rank),
        .w2 = try tensor(ckp, "dspark.markov_head.markov_w2.weight", @as(usize, c.vocab) * c.rank),
        .cw = try tensor(ckp, "dspark.confidence_head.proj.weight", h + c.rank),
        .cb = try tensor(ckp, "dspark.confidence_head.proj.bias", 1),
        .layers = undefined,
    };
    for (w.layers[0..c.layers], 0..) |*l, i| {
        var nb: [128]u8 = undefined;
        const name = struct {
            fn of(buf: []u8, layer: usize, z: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "dspark.layers.{d}.{s}", .{ layer, z });
            }
        }.of;
        l.* = .{
            .in_norm = try tensor(ckp, try name(&nb, i, "input_layernorm.weight"), h),
            .post_norm = try tensor(ckp, try name(&nb, i, "post_attention_layernorm.weight"), h),
            .q_norm = try tensor(ckp, try name(&nb, i, "self_attn.q_norm.weight"), 64),
            .k_norm = try tensor(ckp, try name(&nb, i, "self_attn.k_norm.weight"), 64),
            .q = try tensor(ckp, try name(&nb, i, "self_attn.q_proj.weight"), qw * h),
            .k = try tensor(ckp, try name(&nb, i, "self_attn.k_proj.weight"), qw * h),
            .v = try tensor(ckp, try name(&nb, i, "self_attn.v_proj.weight"), qw * h),
            .o = try tensor(ckp, try name(&nb, i, "self_attn.o_proj.weight"), h * qw),
            .gate = try tensor(ckp, try name(&nb, i, "mlp.gate_proj.weight"), @as(usize, c.ffn) * h),
            .up = try tensor(ckp, try name(&nb, i, "mlp.up_proj.weight"), @as(usize, c.ffn) * h),
            .down = try tensor(ckp, try name(&nb, i, "mlp.down_proj.weight"), h * c.ffn),
        };
    }
    return w;
}

const Lane = struct {
    k: [8]mtl.Buffer,
    v: [8]mtl.Buffer,
    end: u64 = 0,
    from: u64 = 0,
    held: [max_block]u32 = @splat(0),
    chance: [max_block]f64 = @splat(0),
    held_len: u32 = 0,
    /// GLM_AX_DUAL: who drafts this stream's rounds (the MTP head or DSpark), chosen from measured tokens a ms
    mode: Mode = .dspark,
    rounds: u32 = 0, // rounds since the stream opened
    stat: [2]struct { tokens: f64 = 0, ms: f64 = 0 } = .{ .{}, .{} },
    last_hold: i128 = 0,
    last_mode: Mode = .dspark,
    // the last absorb: its kept rows' count, the token after each, its start, whether it was the first after a prompt
    last_n: u32 = 0,
    last_follow: [16]u32 = @splat(0),
    last_start: u64 = 0,
    last_first: bool = false,
};

const Mode = enum(u8) { dspark, mtp };

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

const fields = [_][]const u8{ "ids", "resid", "x", "qkv", "q", "attn", "y", "gu", "act", "logits", "pm", "pl", "po", "tv", "ti", "tv2", "ti2", "uv", "ui", "seq", "conf", "fcx", "hctx", "kv", "sums" };

const B = fwd.B;
fn b(x: mtl.Buffer) B {
    return .{ .buf = x };
}
fn bo(x: mtl.Buffer, off: usize) B {
    return .{ .buf = x, .off = off };
}
fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}
const A = B{ .args = true };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *Model,
    config: Config,
    ckpt: ck.Checkpoint,
    w: Weights,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    step_ms: f64 = 0,
    pending: ?mtl.CommandBuffer = null,
    s: [fields.len]mtl.Buffer = undefined,
    /// GLM_AX_DSPARK_FLOOR: the chance cut (a running product of the confidences) the lane engine verifies down to
    floor: ?f64 = null,
    /// GLM_AX_DUAL=1: the target's backend, whose MTP head drafts a stream's rounds when that measures faster
    mtp: ?*Metal = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const config = try Config.read(gpa, io, dir);
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .target = target, .config = config, .ckpt = ck.Checkpoint.init(gpa), .w = undefined };
        errdefer d.ckpt.deinit();
        if (std.c.getenv("GLM_AX_DSPARK_FLOOR")) |v| d.floor = std.fmt.parseFloat(f64, std.mem.span(v)) catch null;
        const c = &d.config;
        if (c.hidden != target.config.hidden or c.vocab != target.config.vocab) return error.DSparkForAnotherTarget;
        for (c.taps) |l| if (l >= target.weights.main) return error.DSparkForAnotherTarget;
        {
            const path = try std.fs.path.joinZ(gpa, &.{ dir, "model.safetensors" });
            defer gpa.free(path);
            try d.ckpt.addFile(target.device, path, "dspark.");
            d.ckpt.closeFiles();
        }
        d.w = try load(&d.ckpt, c);
        const h: usize = c.hidden;
        const r: usize = max_block * max_blocks;
        const qw: usize = c.heads * 64;
        const nblk: usize = c.ring() / 128 + 1;
        const vocab: usize = c.vocab;
        const chunks: usize = (vocab + 2047) / 2048;
        const sizes = [fields.len]usize{ r * 4, r * h * 4, r * h * 2, r * 3 * qw * 2, r * qw * 2, r * qw * 2, r * h * 2, r * 2 * c.ffn * 2, r * c.ffn * 2, r * vocab * 2, c.heads * max_block * nblk * 4, c.heads * max_block * nblk * 4, c.heads * max_block * nblk * 64 * 4, r * chunks * 32 * 4, r * chunks * 32 * 4, r * 64 * 4, r * 64 * 4, r * 32 * 4, r * 32 * 4, max_blocks * 16 * 4, max_blocks * 16 * 4, absorb_rows * h * 2, absorb_rows * h * 2, absorb_rows * c.layers * 2 * qw * 2, absorb_rows * h * c.taps.len / 32 * 4 + r * h / 16 * 4 };
        var made: usize = 0;
        errdefer for (d.s[0..made]) |x| x.deinit();
        for (sizes, 0..) |n, i| {
            d.s[i] = try target.device.buffer(n, mtl.ResourceOptions.shared);
            made += 1;
        }
        try d.warm();
        std.log.info("glm_ax: DSpark drafter from {s}: {d} layers, block {d}, taps {any} (layer outputs), {d:.2} GiB; a block {d:.2} ms; chance cut {?d}", .{ dir, c.layers, c.block, c.taps, @as(f64, @floatFromInt(d.ckpt.residentBytes())) / (1 << 30), d.step_ms * @as(f64, @floatFromInt(c.block)), d.floor });
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        for (d.s) |x| x.deinit();
        d.ckpt.deinit();
        d.gpa.destroy(d);
    }

    fn buf(d: *const Drafter, comptime f: []const u8) mtl.Buffer {
        inline for (fields, 0..) |n, i| if (comptime std.mem.eql(u8, n, f)) return d.s[i];
        @compileError("no drafter buffer " ++ f);
    }

    fn pipe(d: *const Drafter, k: K) mtl.Pipeline {
        return d.target.df[@intFromEnum(k)];
    }

    pub fn laneBytes(d: *const Drafter) usize {
        const c = &d.config;
        return @as(usize, c.layers) * 2 * c.ring() * c.heads * 64 * 2;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DSpark", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .chances = chancesFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).config.taps;
    }

    /// GLM_AX_DSPARK_DEPTH (1-8): the most drafts a round may verify.
    fn depthCap(d: *const Drafter) u32 {
        const most = d.config.block;
        const v = std.c.getenv("GLM_AX_DSPARK_DEPTH") orelse return most;
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return most;
        return @max(1, @min(n, most));
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = d.depthCap(), .step_ms = d.step_ms, .plain_guard = true, .batched = true, .stream_bytes = d.laneBytes(), .fixed_block = true, .window_costs = d.target.window_costs[0..d.target.window_costs_n], .chances = d.floor != null, .confidence_floor = d.floor };
    }

    /// Each held draft's chance: the running product of the confidence head's sigmoids.
    fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        if (d.floor == null) return false;
        const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
        for (out, 0..) |*o, i| o.* = if (i < l.held_len) l.chance[i] else 0;
        return true;
    }

    fn newLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .k = undefined, .v = undefined };
        const bytes = @as(usize, d.config.ring()) * d.config.heads * 64 * 2;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..d.config.layers) |i| {
            l.k[i] = try d.target.device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = d.target.device.buffer(bytes, mtl.ResourceOptions.private) catch |e| {
                l.k[i].deinit();
                return e;
            };
            made += 1;
        }
        return l;
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        for (0..d.config.layers) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        d.gpa.destroy(l);
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const l = try d.newLane();
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(s)) |e| {
            const l = e.value;
            if (d.mtp != null) std.log.info("glm_ax: dual drafting: {d} rounds; DSpark {d:.0} tokens in {d:.0} ms, MTP {d:.0} tokens in {d:.0} ms", .{ l.rounds, l.stat[0].tokens, l.stat[0].ms, l.stat[1].tokens, l.stat[1].ms });
            d.dropLane(l);
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    fn glue(e: mtl.ComputeEncoder, p: mtl.Pipeline, args: anytype, bufs: []const B, grid: [3]usize, group: [3]usize) void {
        e.setPipeline(p);
        for (bufs, 0..) |r, slot| if (r.args) e.setValue(args, slot) else e.setBuffer(r.buf, r.off, slot);
        e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    fn threads(e: mtl.ComputeEncoder, p: mtl.Pipeline, args: anytype, bufs: []const B, grid: [3]usize) void {
        e.setPipeline(p);
        for (bufs, 0..) |r, slot| if (r.args) e.setValue(args, slot) else e.setBuffer(r.buf, r.off, slot);
        e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(256, 1, 1));
    }

    /// y[m, ld] cols [0, n) = x[m, k] W[n, k]^T, W bf16: the row GEMV for a block, the tensor-unit GEMM past it.
    fn gemm(d: *const Drafter, e: mtl.ComputeEncoder, x: B, w: wts.Tensor, y: B, m: usize, n: usize, k: usize, ld: usize) void {
        if (m <= 8) {
            e.setPipeline(d.pipe(.gemv));
            e.setBuffer(x.buf, x.off, 0);
            e.setBuffer(w.buffer, w.offset, 1);
            e.setValue(extern struct { rows: u32, n: u32, k: u32, ld: u32 }{ .rows = @intCast(m), .n = @intCast(n), .k = @intCast(k), .ld = @intCast(ld) }, 2);
            e.setBuffer(y.buf, y.off, 3);
            e.dispatchGroups(mtl.Size.of((n + 15) / 16, 1, 1), mtl.Size.of(128, 1, 1));
            return;
        }
        const tn = (n + 127) / 128;
        const tm = (m + 63) / 64;
        e.setPipeline(d.target.gemm[@as(usize, @intFromBool(m % 64 == 0)) * 2 + @intFromBool(n % 128 == 0)]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buffer, w.offset, 1);
        var p: [16]i32 = @splat(0);
        for ([_]usize{ m, n, k, k, k, ld, tn, tm, 2, k / 256 }, 0..) |v, j| p[j] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    fn head(d: *const Drafter, e: mtl.ComputeEncoder, in: B, out: B, n_rows: usize) !void {
        const m = d.target;
        const w = m.weights.head orelse return error.GlmHasNoHead;
        if (n_rows > 16) {
            affine_mm.rowSums(e, m.mm, @intCast(w.group), in, b(d.buf("sums")), @intCast(n_rows), @intCast(w.k));
            affine_mm.dense(e, m.mm, in, b(d.buf("sums")), .{ .w = bo(w.w, w.w_off), .s = bo(w.scales, w.s_off), .b = bo(w.biases, w.b_off), .n = @as(u32, @intCast(w.n)), .k = @as(u32, @intCast(w.k)) }, out, @intCast(n_rows));
            return;
        }
        const call = try rows_mod.call(&m.rows, w, n_rows, false);
        e.setPipeline(call.pipeline);
        e.setBuffer(in.buf, in.off, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    const NArgs = extern struct { rows: u32, width: u32, eps: f32, add: u32 };
    const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, ring: u32, eps: f32, theta: f32 };
    const VArgs = extern struct { rows: u32, pos0: u32, heads: u32, groups: u32, blocks: u32, first: u32, hi: u32, window: u32, ring: u32, scale: f32, causal: u32 };

    /// Context rows [p0, p0 + n): fc over the taps, hidden_norm, then each layer's k (k_norm, rope) and v into its ring.
    fn absorbRows(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, n: usize) !void {
        const c = &d.config;
        const h: usize = c.hidden;
        const qw: usize = c.heads * 64;
        const ld: usize = c.layers * 2 * qw;
        d.gemm(e, src, d.w.fc, b(d.buf("fcx")), n, h, h * c.taps.len, h);
        glue(e, d.pipe(.norm16), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = 0 }, &.{ b(d.buf("fcx")), t(d.w.hidden_norm), A, b(d.buf("hctx")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        for (0..c.layers) |i| {
            d.gemm(e, b(d.buf("hctx")), d.w.layers[i].k, bo(d.buf("kv"), (i * 2 * qw) * 2), n, qw, h, ld);
            d.gemm(e, b(d.buf("hctx")), d.w.layers[i].v, bo(d.buf("kv"), (i * 2 * qw + qw) * 2), n, qw, h, ld);
        }
        for (0..c.layers) |i| {
            const pa = PArgs{ .rows = @intCast(n), .heads = 0, .kv_heads = c.heads, .pos0 = @intCast(p0), .stride = @intCast(ld), .q_off = 0, .k_off = @intCast(i * 2 * qw), .v_off = @intCast(i * 2 * qw + qw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep64), pa, &.{ b(d.buf("kv")), t(d.w.layers[i].q_norm), t(d.w.layers[i].k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 });
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        try d.settle();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        const width: usize = c.taps.len * c.hidden * 2;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width) return error.DraftFeaturesUnsupported;
            if (it.rows != null) return error.UnsupportedGlmTree;
            if (it.start > l.end) l.from = it.start;
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            const end = it.start + f.rows;
            const keep_from = if (it.prompt) @max(it.start, (it.stream.prompt_len -| (c.window + c.block))) else it.start;
            var at: u64 = keep_from;
            while (at < end) {
                const n: usize = @intCast(@min(absorb_rows, end - at));
                try d.absorbRows(e, l, bo(src, f.offset + @as(usize, @intCast(at - it.start)) * width), at, n);
                at += n;
            }
            l.end = end;
            l.held_len = 0;
            l.last_n = f.rows;
            l.last_start = it.start;
            l.last_first = it.prompt;
            const keep = @min(it.follow.len, l.last_follow.len);
            @memcpy(l.last_follow[0..keep], it.follow[0..keep]);
        }
        e.end();
        cb.commit();
        d.pending = cb.retain();
    }

    /// GLM_AX_DUAL's choice for a stream's next round: the MTP head first (12 rounds), DSpark tried for 8, then
    /// DSpark only when it measures 15% more tokens a ms; the other tried again for 8 rounds every 128.
    fn choose(l: *Lane) Mode {
        const r = l.rounds;
        if (r < 12) return .mtp;
        if (r < 20) return .dspark;
        const rate = struct {
            fn of(x: anytype) f64 {
                return if (x.ms > 0) x.tokens / x.ms else 0;
            }
        };
        const best: Mode = if (rate.of(l.stat[@intFromEnum(Mode.dspark)]) > 1.15 * rate.of(l.stat[@intFromEnum(Mode.mtp)])) .dspark else .mtp;
        if (r % 128 >= 120) return if (best == .mtp) .dspark else .mtp;
        return best;
    }

    /// Whether a DSpark round still feeds the kept rows to the MTP head: while probing or when MTP is the
    /// stream's drafter, yes; while DSpark wins (outside a probe), no (the head's cache then has gaps over those rows,
    /// which costs it acceptance after a switch back, never exactness).
    fn feedMtp(l: *Lane) bool {
        const r = l.rounds;
        if (r < 20) return true;
        return r % 128 >= 112; // the 8 rounds before a probe of MTP, and the probe: the head catches up
    }

    /// The MTP head's drafts for a stream this round (or its absorb alone, depth 0), into the lane.
    fn mtpRound(d: *Drafter, l: *Lane, it: dr.Hold, depth: u32) !void {
        const m = d.mtp orelse return error.NoMtpBackend;
        const n: usize = @max(1, l.last_n);
        var rows: [16]u32 = undefined;
        for (0..n) |i| rows[i] = @intCast(i);
        const follow_last = [1]u32{it.pending};
        const req: be.DraftRequest = if (l.last_first or l.last_n == 0)
            .{ .stream = it.stream, .follow = &follow_last, .rows = null, .start = l.last_start, .position = it.position, .depth = depth }
        else
            .{ .stream = it.stream, .follow = l.last_follow[0..@min(n, 16)], .rows = rows[0..@min(n, 16)], .start = l.last_start, .position = it.position, .depth = depth };
        try m.mtpDraft(req);
        if (depth == 0) return;
        l.held_len = @intCast(try m.mtpHeld(it.stream, l.held[0..@min(it.depth, max_block)], l.chance[0..@min(it.depth, max_block)]));
        for (l.held_len..max_block) |j| l.chance[j] = 0;
    }

    fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer cb.release();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    const Blk = struct { lane: *const Lane, pending: u32, anchor: u64 };

    /// Each block [anchor token, mask x 7] at the anchor's position, stacked; every row predicts the next position.
    fn encodeBlocks(d: *const Drafter, e: mtl.ComputeEncoder, blocks: []const Blk) !void {
        const c = &d.config;
        const m = d.target;
        const n: usize = c.block;
        const rows: usize = blocks.len * n;
        const h: usize = c.hidden;
        const qw: usize = c.heads * 64;
        const ids = d.buf("ids").slice(u32, rows);
        const seq = d.buf("seq").slice(u32, max_blocks * 16);
        for (blocks, 0..) |k, s| {
            ids[s * n] = k.pending;
            @memset(ids[s * n + 1 ..][0 .. n - 1], c.mask);
            seq[s * 16] = k.pending;
        }
        const E = m.weights.embed;
        glue(e, d.pipe(.embed), extern struct { rows: u32, width: u32, bits: u32, group: u32 }{ .rows = @intCast(rows), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) }, &.{ b(d.buf("ids")), bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(d.buf("resid")) }, .{ (h + 255) / 256, rows, 1 }, .{ 256, 1, 1 });
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) }, &.{ b(d.buf("resid")), b(d.buf("y")), t(L.in_norm), A, b(d.buf("x")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
            d.gemm(e, b(d.buf("x")), L.q, b(d.buf("qkv")), rows, qw, h, 3 * qw);
            d.gemm(e, b(d.buf("x")), L.k, bo(d.buf("qkv"), qw * 2), rows, qw, h, 3 * qw);
            d.gemm(e, b(d.buf("x")), L.v, bo(d.buf("qkv"), 2 * qw * 2), rows, qw, h, 3 * qw);
            for (blocks, 0..) |k, s| try d.attend(e, k, i, s * n);
            d.gemm(e, b(d.buf("attn")), L.o, b(d.buf("y")), rows, h, qw, h);
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("y")), t(L.post_norm), A, b(d.buf("x")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
            d.gemm(e, b(d.buf("x")), L.gate, b(d.buf("gu")), rows, c.ffn, h, 2 * @as(usize, c.ffn));
            d.gemm(e, b(d.buf("x")), L.up, bo(d.buf("gu"), @as(usize, c.ffn) * 2), rows, c.ffn, h, 2 * @as(usize, c.ffn));
            threads(e, d.pipe(.silu), extern struct { rows: u32, width: u32 }{ .rows = @intCast(rows), .width = c.ffn }, &.{ b(d.buf("gu")), A, b(d.buf("act")) }, .{ c.ffn, rows, 1 });
            d.gemm(e, b(d.buf("act")), L.down, b(d.buf("y")), rows, h, c.ffn, h);
        }
        glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("y")), t(d.w.norm), A, b(d.buf("x")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        try d.head(e, b(d.buf("x")), b(d.buf("logits")), rows);
        const vocab: u32 = c.vocab;
        const chunks: u32 = (vocab + 2047) / 2048;
        const TArgs = extern struct { n: u32, stride: u32, chunks: u32, from_ids: u32 };
        glue(e, d.pipe(.topk), TArgs{ .n = vocab, .stride = vocab, .chunks = chunks, .from_ids = 0 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv")), b(d.buf("ti")) }, .{ chunks, rows, 1 }, .{ 1024, 1, 1 });
        const second: u32 = (chunks * 32 + 2047) / 2048;
        glue(e, d.pipe(.topk), TArgs{ .n = chunks * 32, .stride = chunks * 32, .chunks = second, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv2")), b(d.buf("ti2")) }, .{ second, rows, 1 }, .{ 1024, 1, 1 });
        glue(e, d.pipe(.topk), TArgs{ .n = second * 32, .stride = second * 32, .chunks = 1, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv2")), b(d.buf("ti2")), A, b(d.buf("uv")), b(d.buf("ui")) }, .{ 1, rows, 1 }, .{ 1024, 1, 1 });
        const WArgs = extern struct { slots: u32, rank: u32, top: u32, vocab: u32, stride: u32, hidden: u32 };
        for (0..blocks.len) |s| {
            const r0 = s * n;
            glue(e, d.pipe(.mwalk), WArgs{ .slots = @intCast(n), .rank = c.rank, .top = top, .vocab = vocab, .stride = 32, .hidden = c.hidden }, &.{ bo(d.buf("uv"), r0 * 32 * 4), bo(d.buf("ui"), r0 * 32 * 4), bo(d.buf("x"), r0 * h * 2), t(d.w.w1), t(d.w.w2), t(d.w.cw), t(d.w.cb), A, bo(d.buf("seq"), s * 16 * 4), bo(d.buf("conf"), s * 16 * 4) }, .{ 1, 1, 1 }, .{ 256, 1, 1 });
        }
    }

    /// Layer `i`'s attention for one block: its q and k/v (k into the ring), then causal over the context and itself.
    fn attend(d: *const Drafter, e: mtl.ComputeEncoder, k: Blk, i: usize, row: usize) !void {
        const c = &d.config;
        const l = k.lane;
        const L = &d.w.layers[i];
        const n: usize = c.block;
        const qw: usize = c.heads * 64;
        const lo = @max((k.anchor + 1) -| c.window, l.from);
        const hi = k.anchor + n;
        const first = lo / 128;
        const blocks = (hi - 1) / 128 - first + 1;
        const pa = PArgs{ .rows = @intCast(n), .heads = c.heads, .kv_heads = c.heads, .pos0 = @intCast(k.anchor), .stride = @intCast(3 * qw), .q_off = 0, .k_off = @intCast(qw), .v_off = @intCast(2 * qw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
        const q = bo(d.buf("q"), row * qw * 2);
        glue(e, d.pipe(.prep64), pa, &.{ bo(d.buf("qkv"), row * 3 * qw * 2), t(L.q_norm), t(L.k_norm), A, q, b(l.k[i]), b(l.v[i]) }, .{ 2 * c.heads, n, 1 }, .{ 32, 1, 1 });
        const va = VArgs{ .rows = @intCast(n), .pos0 = @intCast(k.anchor), .heads = c.heads, .groups = c.heads, .blocks = @intCast(blocks), .first = @intCast(first), .hi = @intCast(hi), .window = c.window, .ring = c.ring(), .scale = 1.0 / 8.0, .causal = 1 };
        glue(e, d.pipe(.draft_vec64), va, &.{ q, b(l.k[i]), b(l.v[i]), A, b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")) }, .{ c.heads, n, blocks }, .{ 32, 1, 1 });
        const ma = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, topk: u32, scale: f32, sel: u32, all: u32 }{ .rows = @intCast(n), .pos = 0, .heads = c.heads, .groups = c.heads, .topk = @intCast(blocks), .scale = 0, .sel = 0, .all = 0 };
        glue(e, d.pipe(.merge_vec64), ma, &.{ b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")), A, bo(d.buf("attn"), row * qw * 2) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 });
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (d.mtp != null) {
            // per-request choice: each stream's last round credited to its drafter, then this round's chosen
            var ds: [max_blocks]dr.Hold = undefined;
            var nds: usize = 0;
            for (items) |it| {
                const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
                const now = nowNs();
                if (l.last_hold != 0 and l.rounds > 0) {
                    const st = &l.stat[@intFromEnum(l.last_mode)];
                    st.tokens += @floatFromInt(l.last_n);
                    st.ms += @as(f64, @floatFromInt(now - l.last_hold)) / 1e6;
                }
                l.last_hold = now;
                const mode = choose(l);
                l.last_mode = mode;
                l.rounds += 1;
                if (mode == .mtp) {
                    try d.mtpRound(l, it, @min(it.depth, 2));
                } else {
                    if (feedMtp(l)) try d.mtpRound(l, it, 0); // the head keeps up (see feedMtp)
                    if (nds == ds.len) return error.TooManyDraftStreams;
                    ds[nds] = it;
                    nds += 1;
                }
            }
            if (nds == 0) return;
            return d.holdBlocks(ds[0..nds]);
        }
        return d.holdBlocks(items);
    }

    fn holdBlocks(d: *Drafter, items: []const dr.Hold) anyerror!void {
        const c = &d.config;
        var at: usize = 0;
        while (at < items.len) {
            const count = @min(items.len - at, max_blocks);
            const pass = items[at..][0..count];
            var blocks: [max_blocks]Blk = undefined;
            for (pass, blocks[0..count]) |it, *k| {
                const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
                if (it.depth == 0 or it.depth > c.block or it.position == 0) return error.DraftDepth;
                const anchor = it.position - 1;
                if (l.end != anchor) return error.DraftContextGap;
                k.* = .{ .lane = l, .pending = it.pending, .anchor = anchor };
            }
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlocks(e, blocks[0..count]);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            const seq = d.buf("seq").slice(u32, max_blocks * 16);
            const conf = d.buf("conf").slice(f32, max_blocks * 16);
            for (pass, 0..) |it, s| {
                const l = d.lanes.get(it.stream).?;
                var run: f64 = 1;
                for (0..it.depth) |j| {
                    const x = seq[s * 16 + 1 + j];
                    if (x >= c.vocab) return error.DraftNonfinite;
                    l.held[j] = x;
                    run *= conf[s * 16 + j];
                    l.chance[j] = run;
                }
                l.held_len = it.depth;
            }
            at += count;
        }
    }

    fn warm(d: *Drafter) !void {
        const l = try d.newLane();
        defer d.dropLane(l);
        var ms: f64 = 0;
        for (0..3) |run| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlocks(e, &.{.{ .lane = l, .pending = 2, .anchor = 0 }});
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            if (run == 2) ms = 1000.0 * cb.gpuSeconds();
        }
        d.step_ms = ms / @as(f64, @floatFromInt(d.config.block));
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("glm_ax DSpark drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
};
