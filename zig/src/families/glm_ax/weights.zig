//! GLM-5.3's MLX checkpoint (mlx-community/GLM-5.3-4bit and an MTP head beside it) as affine row matrices and norms.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const mlx_affine = @import("../../core/quant/mlx_affine.zig");
const route = @import("../../core/moe_route.zig");
const cfg = @import("config.zig");
const file_cache = @import("../../core/checkpoint_file_cache.zig");

pub const Tensor = ck.Tensor;
pub const Linear = rows.Weights;

/// The index head weights' rows as the kernels read them: 32 real rows padded to a 64-row matrix.
pub const iw_rows = 64;

/// x's projections stacked into one matrix at load: q_a [0, 2048), kv_a [2048, 2624), then a full layer's indexer
/// wk [2624, 2752) and weights_proj padded to 64 rows [2752, 2816).
pub const kv_col = 2048;
pub const wk_col = 2624;
pub const wp_col = 2752;

/// A full layer's DSA indexer: wq_b [32 * 128, q_lora] and the key LayerNorm (wk and weights_proj are in Attention.x).
pub const Indexer = struct { wq_b: Linear, k_norm: Tensor, k_bias: Tensor };
pub const Attention = struct {
    /// q_a, kv_a (and wk, weights_proj) stacked: the x projections' one product, rows `x.n` wide.
    x: Linear,
    q_b: Linear,
    /// [heads] matrices [kv_lora, nope] and [v_dim, kv_lora], back to back.
    embed_q: Linear,
    unembed: Linear,
    o: Linear,
    q_norm: Tensor,
    kv_norm: Tensor,
    indexer: ?Indexer = null,
};
/// gate and up stacked into one [2 * inter, hidden] matrix at load (gate rows first), and down.
pub const Mlp = struct { gate_up: Linear, down: Linear };
/// `gate`: the router as stored (bf16 [experts, hidden]) for the prompt chunks' GEMM; `router` its 16-row repack.
pub const Moe = struct { router: mtl.Buffer, gate_w: Tensor, bias: mtl.Buffer, gate: rows.Experts, up: rows.Experts, down: rows.Experts, shared: Mlp };
pub const Mtp = struct { eh: Linear, enorm: Tensor, hnorm: Tensor, norm: Tensor };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

pub const Weights = struct {
    embed: Linear,
    head: ?Linear,
    norm: ?Tensor,
    layers: []Layer, // the main stack's first layers, then the MTP layer when loaded
    main: usize, // main-stack layers in `layers`
    mtp: ?Mtp,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.layers) |l| if (l.ffn == .moe) {
            l.ffn.moe.router.deinit();
            l.ffn.moe.bias.deinit();
        };
        gpa.free(w.layers);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

fn tensor(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, count: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    if (t.dtype != .bf16 or t.count() != count) {
        std.log.err("glm_ax: {s} is not bf16[{d}]", .{ n, count });
        return error.UnexpectedGlmTensor;
    }
    return t;
}

fn affine(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, stack: usize, n: usize, k: usize) !Linear {
    var buf: [256]u8 = undefined;
    const w = try rows.affine(try mlx_affine.read(c, try name(&buf, prefix, suffix), .{ .stack = stack, .n = n, .k = k }));
    if (w.bits != 4 or w.group != 64) {
        std.log.err("glm_ax: {s}{s} is {d}-bit in groups of {d}; this engine takes MLX 4-bit, groups of 64", .{ prefix, suffix, w.bits, w.group });
        return error.UnsupportedGlmQuantization;
    }
    return w;
}

/// An MLP's gate and up stacked, and its down; `part`: this rank's channels [part[0], part[1]) of `inter` (tensor
/// parallel: gate/up rows and down columns read from the files), null: all of them as before.
fn mlp(c: *ck.Checkpoint, device: mtl.Device, p: []const u8, hidden: usize, inter: usize, part: ?[2]usize) !Mlp {
    var gb: [128]u8 = undefined;
    var ub: [128]u8 = undefined;
    if (part) |r| {
        const local = r[1] - r[0];
        const parts = [_]Part{ .{ .module = try name(&gb, p, ".gate_proj"), .n = local, .from = r[0], .of = inter }, .{ .module = try name(&ub, p, ".up_proj"), .n = local, .from = r[0], .of = inter } };
        var db: [128]u8 = undefined;
        const gate_up = try stacked(c, device, &parts, hidden);
        return .{ .gate_up = gate_up, .down = try columns(c, device, try name(&db, p, ".down_proj"), 1, hidden, inter, r) };
    }
    const parts = [_]Part{ .{ .module = try name(&gb, p, ".gate_proj"), .n = inter }, .{ .module = try name(&ub, p, ".up_proj"), .n = inter } };
    return .{ .gate_up = try stacked(c, device, &parts, hidden), .down = try affine(c, p, ".down_proj", 1, hidden, inter) };
}

/// Tensor parallel (set by the model's load before the shards are opened): the FFN matrices this rank slices stay in
/// their files too (`rowsOf` and `columns` read its share).
pub var tp_split: bool = false;

/// The modules `stacked` reads from their files: a checkpoint keeps them out of its shard buffers.
pub fn later(n: []const u8, _: usize) bool {
    const parts = [_][]const u8{ ".self_attn.q_a_proj.", ".self_attn.kv_a_proj_with_mqa.", ".indexer.wk.", ".indexer.weights_proj.", ".shared_experts.gate_proj.", ".shared_experts.up_proj.", ".mlp.gate_proj.", ".mlp.up_proj." };
    for (parts) |x| if (std.mem.indexOf(u8, n, x) != null) return true;
    if (tp_split) {
        const split = [_][]const u8{ ".switch_mlp.gate_proj.", ".switch_mlp.up_proj.", ".switch_mlp.down_proj.", ".shared_experts.down_proj.", ".mlp.down_proj." };
        for (split) |x| if (std.mem.indexOf(u8, n, x) != null) return true;
    }
    return false;
}

/// One module of a stacked matrix: its rows (`n`), and the rows it takes (`pad`, zero past `n`; 0: n).
/// `from`, `of`: its rows are rows [from, from + n) of a stored matrix of `of` rows (0: n, the whole matrix).
const Part = struct { module: []const u8, n: usize, pad: usize = 0, from: usize = 0, of: usize = 0 };

/// MLX 4-bit modules (each [n, k], left in their files) stacked by rows into one matrix in owned buffers.
fn stacked(c: *ck.Checkpoint, device: mtl.Device, parts: []const Part, k: usize) !Linear {
    var rows_: usize = 0;
    for (parts) |p| rows_ += if (p.pad > 0) p.pad else p.n;
    const words = k / 8; // u32 words a row at 4 bits
    const groups = k / 64;
    const wn = rows_ * words * 4;
    const tn = rows_ * groups * 2;
    const wb = try device.buffer(wn, mtl.ResourceOptions.shared);
    try c.own(wb, wn);
    const sb = try device.buffer(tn, mtl.ResourceOptions.shared);
    try c.own(sb, tn);
    const bb = try device.buffer(tn, mtl.ResourceOptions.shared);
    try c.own(bb, tn);
    @memset(wb.contents()[0..wn], 0);
    @memset(sb.contents()[0..tn], 0);
    @memset(bb.contents()[0..tn], 0);
    var at: usize = 0;
    for (parts) |p| {
        var nb: [320]u8 = undefined;
        inline for (.{ ".weight", ".scales", ".biases" }, 0..) |suffix, i| {
            const full = try std.fmt.bufPrint(&nb, "{s}{s}", .{ p.module, suffix });
            const d = c.deferred.get(full) orelse {
                std.log.err("glm_ax: {s} is not in a file to stack", .{full});
                return error.MissingGlmTensor;
            };
            const per: usize = if (i == 0) words * 4 else groups * 2;
            const dt: ck.DType = if (i == 0) .u32 else .bf16;
            const of = if (p.of > 0) p.of else p.n;
            if (d.dtype != dt or d.rank != 2 or d.shape[0] != of or d.shape[1] * dt.size() != per or d.bytes != of * per or p.from + p.n > of) {
                std.log.err("glm_ax: {s} is not an MLX 4-bit [{d}, {d}] part (groups of 64)", .{ full, of, k });
                return error.UnsupportedGlmQuantization;
            }
            const buf = if (i == 0) wb else if (i == 1) sb else bb;
            try c.read(full, p.from * per, buf.contents()[at * per ..][0 .. p.n * per]);
        }
        at += if (p.pad > 0) p.pad else p.n;
    }
    return .{ .w = wb, .scales = sb, .biases = bb, .n = rows_, .k = k, .group = 64, .bits = 4 };
}

/// A deferred MLX 4-bit tensor of `stack` matrices [n, k] (rank 2 when stack is 1, else 3), checked; its per-row
/// bytes (`per`) for the weights (i 0), the scales or the biases.
fn deferredPart(c: *const ck.Checkpoint, full: []const u8, i: usize, stack: usize, n: usize, k: usize) !ck.Deferred {
    const d = c.deferred.get(full) orelse {
        std.log.err("glm_ax: {s} is not in a file to slice", .{full});
        return error.MissingGlmTensor;
    };
    const per: usize = if (i == 0) k / 8 * 4 else k / 64 * 2;
    const dt: ck.DType = if (i == 0) .u32 else .bf16;
    const ok = if (stack == 1)
        d.rank == 2 and d.shape[0] == n and d.shape[1] * dt.size() == per
    else
        d.rank == 3 and d.shape[0] == stack and d.shape[1] == n and d.shape[2] * dt.size() == per;
    if (d.dtype != dt or !ok or d.bytes != stack * n * per) {
        std.log.err("glm_ax: {s} is not an MLX 4-bit [{d}, {d}, {d}] matrix (groups of 64)", .{ full, stack, n, k });
        return error.UnsupportedGlmQuantization;
    }
    return d;
}

/// Three new shared buffers for `stack` matrices [n, k] at 4 bits, groups of 64, owned by the checkpoint.
fn ownedTriple(c: *ck.Checkpoint, device: mtl.Device, stack: usize, n: usize, k: usize) ![3]mtl.Buffer {
    const wn = stack * n * (k / 8) * 4;
    const tn = stack * n * (k / 64) * 2;
    const wb = try device.buffer(wn, mtl.ResourceOptions.shared);
    try c.own(wb, wn);
    const sb = try device.buffer(tn, mtl.ResourceOptions.shared);
    try c.own(sb, tn);
    const bb = try device.buffer(tn, mtl.ResourceOptions.shared);
    try c.own(bb, tn);
    return .{ wb, sb, bb };
}

/// Rows [part[0], part[1]) of each of `stack` stored matrices [n, k] (the gate or up projections' channels this rank
/// holds), read from the file into owned buffers as `stack` matrices [part[1] - part[0], k] back to back.
fn rowsOf(c: *ck.Checkpoint, device: mtl.Device, module: []const u8, stack: usize, n: usize, k: usize, part: [2]usize) !Linear {
    const local = part[1] - part[0];
    if (k % 64 != 0 or part[1] > n or local == 0) return error.BadGlmTpSlice;
    const bufs = try ownedTriple(c, device, stack, local, k);
    var nb: [320]u8 = undefined;
    inline for (.{ ".weight", ".scales", ".biases" }, 0..) |suffix, i| {
        const full = try std.fmt.bufPrint(&nb, "{s}{s}", .{ module, suffix });
        const d = try deferredPart(c, full, i, stack, n, k);
        const per: usize = if (i == 0) k / 8 * 4 else k / 64 * 2;
        const dest = bufs[i].contents();
        for (0..stack) |e| try c.read(full, (e * n + part[0]) * per, dest[e * local * per ..][0 .. local * per]);
        // the read pages leave the file cache (both ranks' loads read every shard)
        file_cache.release(c.files.items[d.file].fd);
    }
    return .{ .w = bufs[0], .scales = bufs[1], .biases = bufs[2], .n = local, .k = k, .group = 64, .bits = 4 };
}

/// Columns [part[0], part[1]) of each of `stack` stored matrices [n, k] (the down projections' input channels this
/// rank holds; multiples of 64, so whole u32 words and whole groups), as `stack` matrices [n, part[1] - part[0]].
/// Each stored matrix is read in pieces of rows into a staging buffer and its column window copied out.
fn columns(c: *ck.Checkpoint, device: mtl.Device, module: []const u8, stack: usize, n: usize, k: usize, part: [2]usize) !Linear {
    const local = part[1] - part[0];
    if (k % 64 != 0 or part[0] % 64 != 0 or part[1] % 64 != 0 or part[1] > k or local == 0) return error.BadGlmTpSlice;
    const bufs = try ownedTriple(c, device, stack, n, local);
    const piece_rows: usize = @min(n, 8192); // a whole expert matrix a read (6144 rows: 6 MiB of words)
    const gpa = std.heap.page_allocator;
    const stage = try gpa.alloc(u8, piece_rows * (k / 8) * 4);
    defer gpa.free(stage);
    var nb: [320]u8 = undefined;
    inline for (.{ ".weight", ".scales", ".biases" }, 0..) |suffix, i| {
        const full = try std.fmt.bufPrint(&nb, "{s}{s}", .{ module, suffix });
        const d = try deferredPart(c, full, i, stack, n, k);
        // bytes a stored row, a kept row, and where the window starts in a stored row
        const per: usize = if (i == 0) k / 8 * 4 else k / 64 * 2;
        const keep: usize = if (i == 0) local / 8 * 4 else local / 64 * 2;
        const skip: usize = if (i == 0) part[0] / 8 * 4 else part[0] / 64 * 2;
        const dest = bufs[i].contents();
        for (0..stack) |e| {
            var r0: usize = 0;
            while (r0 < n) : (r0 += piece_rows) {
                const rn = @min(piece_rows, n - r0);
                const src = stage[0 .. rn * per];
                try c.read(full, (e * n + r0) * per, src);
                for (0..rn) |r| @memcpy(dest[((e * n) + r0 + r) * keep ..][0..keep], src[r * per + skip ..][0..keep]);
            }
        }
        file_cache.release(c.files.items[d.file].fd);
    }
    return .{ .w = bufs[0], .scales = bufs[1], .biases = bufs[2], .n = n, .k = local, .group = 64, .bits = 4 };
}

/// The bf16 router [experts, hidden] repacked into a new shared buffer in moe_route's block order.
fn router(device: mtl.Device, t: Tensor, experts: usize, hidden: usize) !mtl.Buffer {
    if (t.dtype != .bf16 or t.count() != experts * hidden) return error.UnexpectedGlmTensor;
    const buf = try device.buffer(experts * hidden * 2, mtl.ResourceOptions.shared);
    route.pack(t.host(u16), buf.slice(u16, experts * hidden), experts, hidden);
    return buf;
}

/// The selection bias as fp32 in its own buffer (the pack holds fp32; an MTP head converted apart may hold bf16).
fn biasF32(device: mtl.Device, t: Tensor, experts: usize) !mtl.Buffer {
    if (t.count() != experts) return error.UnexpectedGlmTensor;
    const buf = try device.buffer(experts * 4, mtl.ResourceOptions.shared);
    const out = buf.slice(f32, experts);
    switch (t.dtype) {
        .f32 => @memcpy(out, t.host(f32)),
        .bf16 => for (out, t.host(u16)) |*o, h| {
            o.* = @bitCast(@as(u32, h) << 16);
        },
        else => {
            buf.deinit();
            return error.UnexpectedGlmTensor;
        },
    }
    return buf;
}

fn layer(c: *ck.Checkpoint, device: mtl.Device, f: *const cfg.Config, tp: cfg.Tp, p: []const u8, full: bool, moe: bool) !Layer {
    const h: usize = f.hidden;
    var ab: [96]u8 = undefined;
    const a = try name(&ab, p, "self_attn");
    var l: Layer = undefined;
    l.input_norm = try tensor(c, p, "input_layernorm.weight", h);
    l.post_norm = try tensor(c, p, "post_attention_layernorm.weight", h);
    var xb: [4][128]u8 = undefined;
    const xparts = [_]Part{
        .{ .module = try name(&xb[0], a, ".q_a_proj"), .n = f.q_lora },
        .{ .module = try name(&xb[1], a, ".kv_a_proj_with_mqa"), .n = f.kv_lora + f.rope },
        .{ .module = try name(&xb[2], a, ".indexer.wk"), .n = f.i_dim },
        .{ .module = try name(&xb[3], a, ".indexer.weights_proj"), .n = f.i_heads, .pad = iw_rows },
    };
    l.attn = .{
        .x = try stacked(c, device, xparts[0..if (full) 4 else 2], h),
        .q_b = try affine(c, a, ".q_b_proj", 1, f.heads * f.qHead(), f.q_lora),
        .embed_q = try affine(c, a, ".embed_q", f.heads, f.kv_lora, f.nope),
        .unembed = try affine(c, a, ".unembed_out", f.heads, f.v_dim, f.kv_lora),
        .o = try affine(c, a, ".o_proj", 1, h, f.heads * f.v_dim),
        .q_norm = try tensor(c, a, ".q_a_layernorm.weight", f.q_lora),
        .kv_norm = try tensor(c, a, ".kv_a_layernorm.weight", f.kv_lora),
    };
    if (full) {
        l.attn.indexer = .{
            .wq_b = try affine(c, a, ".indexer.wq_b", 1, f.i_heads * f.i_dim, f.q_lora),
            .k_norm = try tensor(c, a, ".indexer.k_norm.weight", f.i_dim),
            .k_bias = try tensor(c, a, ".indexer.k_norm.bias", f.i_dim),
        };
    }
    var mb: [96]u8 = undefined;
    const m = try name(&mb, p, "mlp");
    if (!moe) {
        l.ffn = .{ .dense = try mlp(c, device, m, h, f.dense_inter, if (tp.on()) [2]usize{ tp.d0, tp.d1 } else null) };
        return l;
    }
    const e: usize = f.experts;
    var rb: [128]u8 = undefined;
    const r = try router(device, try c.get(try name(&rb, m, ".gate.weight")), e, h);
    errdefer r.deinit();
    const bias = try biasF32(device, try c.get(try name(&rb, m, ".gate.e_score_correction_bias")), e);
    errdefer bias.deinit();
    var sb: [128]u8 = undefined;
    if (tp.on()) {
        // this rank's channels of every expert: gate/up rows [c0, c1), down columns [c0, c1)
        const part = [2]usize{ tp.c0, tp.c1 };
        var gb: [3][128]u8 = undefined;
        l.ffn = .{ .moe = .{
            .router = r,
            .gate_w = try c.get(try name(&rb, m, ".gate.weight")),
            .bias = bias,
            .gate = .{ .one = try rowsOf(c, device, try name(&gb[0], m, ".switch_mlp.gate_proj"), e, f.moe_inter, h, part), .experts = e },
            .up = .{ .one = try rowsOf(c, device, try name(&gb[1], m, ".switch_mlp.up_proj"), e, f.moe_inter, h, part), .experts = e },
            .down = .{ .one = try columns(c, device, try name(&gb[2], m, ".switch_mlp.down_proj"), e, h, f.moe_inter, part), .experts = e },
            .shared = try mlp(c, device, try name(&sb, m, ".shared_experts"), h, f.moe_inter, part),
        } };
        return l;
    }
    l.ffn = .{ .moe = .{
        .router = r,
        .gate_w = try c.get(try name(&rb, m, ".gate.weight")),
        .bias = bias,
        .gate = .{ .one = try affine(c, m, ".switch_mlp.gate_proj", e, f.moe_inter, h), .experts = e },
        .up = .{ .one = try affine(c, m, ".switch_mlp.up_proj", e, f.moe_inter, h), .experts = e },
        .down = .{ .one = try affine(c, m, ".switch_mlp.down_proj", e, h, f.moe_inter), .experts = e },
        .shared = try mlp(c, device, try name(&sb, m, ".shared_experts"), h, f.moe_inter, null),
    } };
    return l;
}

/// The first `layers` main layers (all of them: the head and final norm too) and, with `mtp`, the MTP layer.
/// `tp`: this rank's FFN channels (cfg.Tp; ranks 1 loads every matrix whole, as before).
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *ck.Checkpoint, f: *const cfg.Config, tp: cfg.Tp, layers: usize, mtp: bool) !Weights {
    const h: usize = f.hidden;
    const whole = layers == f.layers;
    var w = Weights{
        .embed = try affine(c, "model.", "embed_tokens", 1, f.vocab, h),
        .head = if (whole) try affine(c, "", "lm_head", 1, f.vocab, h) else null,
        .norm = if (whole) try tensor(c, "model.", "norm.weight", h) else null,
        .layers = try gpa.alloc(Layer, layers + @intFromBool(mtp)),
        .main = layers,
        .mtp = null,
    };
    var made: usize = 0;
    errdefer {
        for (w.layers[0..made]) |l| if (l.ffn == .moe) {
            l.ffn.moe.router.deinit();
            l.ffn.moe.bias.deinit();
        };
        gpa.free(w.layers);
    }
    for (w.layers[0..layers], 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        l.* = try layer(c, device, f, tp, p, f.full.isSet(i), f.isMoe(@intCast(i)));
        made += 1;
    }
    if (mtp) {
        w.layers[layers] = try layer(c, device, f, tp, "mtp.", true, true);
        made += 1;
        w.mtp = .{
            .eh = try affine(c, "mtp.", "eh_proj", 1, h, 2 * h),
            .enorm = try tensor(c, "mtp.", "enorm.weight", h),
            .hnorm = try tensor(c, "mtp.", "hnorm.weight", h),
            .norm = try tensor(c, "mtp.", "shared_head_norm.weight", h),
        };
    }
    return w;
}
