//! A native GLM-5.3 model on MLX affine weights: config, the checkpoint in Metal buffers, and every pipeline.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");
const file_cache = @import("../../core/checkpoint_file_cache.zig");
const weights = @import("weights.zig");
const dflash2 = @import("dflash2.zig");
const lanes = @import("lanes");

pub const own_names = [_][:0]const u8{ "gx_norm", "gx_norm16", "gx_add", "gx_prep", "gx_qrope", "gx_attend", "gx_merge", "gx_index_key", "gx_index_scores", "gx_index_topk", "gx_index_qrope", "gx_swiglu", "gx_add_norm", "gx_moe_add_norm", "gx_heads_gemv", "gx_route_rows", "gx_moe_gu", "gx_moe_down", "gx_absorb", "gx_stream" };
pub const Own = enum { norm, norm16, add, prep, qrope, attend, merge, index_key, index_scores, index_topk, index_qrope, swiglu, add_norm, moe_add_norm, heads_gemv, route_rows, moe_gu, moe_down, absorb, stream };
pub const glue_names = [_][:0]const u8{ "s5_embed", "s5_swiglu", "s5_moe_add", "s5_sort", "s5_rows_by_order", "s5_copy32", "s5_widen" };
pub const Glue = enum { embed, swiglu, moe_add, sort, rows_by_order, copy32, widen };

/// What to load: the first `layers` main layers (null: all, with the head), the MTP layer, and at most `shards`
/// main shard files in name order (null: every one the index names).
/// `require_mtp`: drafts were asked for, so a checkpoint without the MTP head is refused rather than served undrafted.
pub const Options = struct { layers: ?u32 = null, mtp: bool = true, shards: ?usize = null, require_mtp: bool = false };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    own: [own_names.len]mtl.Pipeline,
    glue: [glue_names.len]mtl.Pipeline,
    rows: rows.Pipelines,
    route: [2]mtl.Pipeline,
    mm: affine_mm.Pipes,
    /// A prompt chunk's sparse MLA on the tensor units (glm_ax_sparse.metal).
    sparse_nax: mtl.Pipeline,
    /// A prompt block's index scores on the tensor units (glm_ax_sparse.metal).
    index_nax: mtl.Pipeline,
    /// Decode rows' attention on the tensor units: a key list's blocks, then their merge (glm_ax_sparse.metal).
    sparse_split: mtl.Pipeline,
    sparse_combine: mtl.Pipeline,
    combine_unembed: mtl.Pipeline,
    /// Prompt chunks' attention by MetalPerformancePrimitives (glm_ax_mpp.metal; oMLX's and Splash's method: measured
    /// attention 49 -> 76 TFLOPS, prefill +10-12%); null with GLM_AX_NO_MPP=1 (gx_sparse_nax).
    sparse_mpp: ?mtl.Pipeline = null,
    /// core/qwen_vision.metal's bf16 tensor-unit GEMM, y = x w^T (fp32 sums, bf16 out), by (M % 64 == 0, N % 128 == 0).
    gemm: [4]mtl.Pipeline,
    /// The fused decode MoE on core/row_projection.metal's row reduction (glm_ax_moe.metal): gate/up/SwiGLU, down.
    moe_rows: [9]mtl.Pipeline,
    /// The rope's 32 inverse frequencies, fp32.
    inv: mtl.Buffer,
    /// The DFlash2 drafter's kernels and the target's taps for it (glm_ax_dflash2.metal), built with the model.
    df: [dflash2.names.len]mtl.Pipeline = undefined,
    /// Each layer's place in a drafter's taps, -1 when untapped; `taps` tapped layers (0: no drafter).
    tap_at: [cfg.max_layers]i8 = @splat(-1),
    taps: u32 = 0,
    /// Decode windows' forward ms by rows (1..), timed at load by the backend for a drafter's depth rule.
    window_costs: [16]lanes.config.Cost = undefined,
    window_costs_n: usize = 0,
    load_seconds: f64 = 0,
    /// The keepalive's queue when `queue` is Metal 4 (null: it shares `queue`).
    tick_queue: ?mtl.Queue = null,

    pub fn tickQueue(m: *const Model) mtl.Queue {
        return m.tick_queue orelse m.queue;
    }

    fn lessThan(_: void, a: [:0]u8, b: [:0]u8) bool {
        return std.mem.lessThan(u8, a, b);
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = std.Io.Timestamp.now(io, .awake);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.load_seconds = 0;
        // the fields with defaults, set (create leaves them undefined)
        m.tick_queue = null;
        m.sparse_mpp = null;
        m.taps = 0;
        m.tap_at = @splat(-1);
        m.window_costs_n = 0;
        m.config = try cfg.read(gpa, io, dir);
        const c = &m.config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        // GLM_AX_METAL4=1: the queue through Metal 4 (command allocators, argument tables, a constants ring, stage
        // barriers) where the binary compiles it in; Metal 3 otherwise
        m.queue = if (std.c.getenv("GLM_AX_METAL4") != null) (m.device.queue4() catch try m.device.queue()) else try m.device.queue();
        if (m.queue.isMetal4()) {
            // the idle keepalive's probes come from the lane host's ticker thread: a Metal 3 queue of their own
            m.tick_queue = try m.device.queue();
            std.log.info("glm_ax: Metal 4 queue (the keepalive on a Metal 3 queue)", .{});
        }
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        std.mem.sort([:0]u8, files, {}, lessThan);
        var main: usize = 0;
        var has_mtp = false;
        for (files) |path| {
            const is_mtp = std.mem.indexOf(u8, std.fs.path.basename(path), "mtp") != null;
            if (is_mtp) {
                has_mtp = true;
                if (!o.mtp) continue;
            } else {
                if (o.shards) |most| if (main >= most) continue;
                main += 1;
            }
            const before = m.checkpoint.files.items.len;
            try m.checkpoint.addFileSplit(m.device, path, "", null, weights.later);
            // a shard with stacked parts stays open for them: its copied pages leave the file cache now, not at
            // the end of the load (else ~390 GB of cache stand beside the buffers and macOS compresses the process)
            if (m.checkpoint.files.items.len > before) file_cache.release(m.checkpoint.files.items[before].fd);
        }
        const mtp = o.mtp and c.mtp > 0 and has_mtp and m.checkpoint.has("mtp.eh_proj.weight");
        if (o.require_mtp and !mtp) {
            std.log.err("glm_ax: drafts are on but {s} has no MTP head (num_nextn_predict_layers {d}, model-mtp shards {s}, mtp.eh_proj.weight {s}): serve with --no-drafts or a folder holding the head under mtp.* names", .{ dir, c.mtp, if (has_mtp) "present" else "absent", if (m.checkpoint.has("mtp.eh_proj.weight")) "present" else "absent" });
            return error.GlmMtpHeadMissing;
        }
        if (!mtp) c.mtp = 0;
        m.weights = try weights.load(gpa, m.device, &m.checkpoint, c, o.layers orelse c.layers, mtp);
        errdefer m.weights.deinit(gpa);
        m.checkpoint.closeFiles();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.glm_ax, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.own[0..made]) |p| p.deinit();
            for (own_names, 0..) |n, i| {
                m.own[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.own) |p| p.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.step5_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.glue[0..made]) |p| p.deinit();
            for (glue_names, 0..) |n, i| {
                m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.glue) |p| p.deinit();
        m.rows = try rows.Pipelines.load(m.device);
        // verify windows on the narrow row kernels (two rows a pass) at every width: measured here, the wide 8-row
        // kernel is slower at every width it takes (4 rows 102 vs 73 ms a step, 8 rows 166 vs ~131); both give the
        // one-row kernel's bits. GLM_AX_WIDE=<rows> takes the wide kernel from that width.
        if (m.rows.wide) {
            const from: usize = if (std.c.getenv("GLM_AX_WIDE")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch std.math.maxInt(usize)) else std.math.maxInt(usize);
            m.rows.wide_from = @splat(from);
        }
        errdefer m.rows.deinit();
        {
            const text = try route.source(gpa, .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk, .round_logits = true });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            m.route[0] = try mtl.Pipeline.init(m.device, lib, route.names[0], false);
            errdefer m.route[0].deinit();
            m.route[1] = try mtl.Pipeline.init(m.device, lib, route.names[1], false);
        }
        errdefer for (m.route) |p| p.deinit();
        {
            const text = try affine_mm.source(m.device, gpa, .{ .bits = 4, .group = 64 });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.mm[0..made]) |p| p.deinit();
            for (affine_mm.names, 0..) |n, i| {
                m.mm[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.mm) |p| p.deinit();
        {
            const text = try frags.source(m.device, gpa, ks.glm_ax_sparse);
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sparse_nax = try mtl.Pipeline.init(m.device, lib, "gx_sparse_nax", false);
            errdefer m.sparse_nax.deinit();
            // prompt chunks' attention with the probabilities in two bf16 parts, not three (one value product a tile
            // fewer; about 2^-16 of a probability dropped; measured +5% prefill); GLM_AX_P3=1 keeps three
            if (std.c.getenv("GLM_AX_P3") == null) {
                const text2 = try std.mem.concat(gpa, u8, &.{ "#define GX_P2 1\n", text });
                defer gpa.free(text2);
                const lib2 = try mtl.Library.fromSource(m.device, text2, mtl.CompileOptions.mlx());
                defer lib2.deinit();
                m.sparse_nax.deinit();
                m.sparse_nax = try mtl.Pipeline.init(m.device, lib2, "gx_sparse_nax", false);
            }
            m.index_nax = try mtl.Pipeline.init(m.device, lib, "gx_index_nax", false);
            m.sparse_split = try mtl.Pipeline.init(m.device, lib, "gx_sparse_split", false);
            m.sparse_combine = try mtl.Pipeline.init(m.device, lib, "gx_sparse_combine", false);
            m.combine_unembed = try mtl.Pipeline.init(m.device, lib, "gx_combine_unembed", false);
        }
        errdefer m.sparse_nax.deinit();
        errdefer m.index_nax.deinit();
        m.sparse_mpp = null;
        if (std.c.getenv("GLM_AX_NO_MPP") == null) {
            const lib = try mtl.Library.fromSource(m.device, ks.glm_ax_mpp, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sparse_mpp = try mtl.Pipeline.init(m.device, lib, "gx_sparse_mpp", false);
        }
        errdefer if (m.sparse_mpp) |p| p.deinit();
        {
            const text = try frags.source(m.device, gpa, ks.core_qwen_vision);
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (&m.gemm, [_][:0]const u8{ "qwv_gemm_bf16_nt_00", "qwv_gemm_bf16_nt_01", "qwv_gemm_bf16_nt_10", "qwv_gemm_bf16_nt_11" }) |*p, name| p.* = try mtl.Pipeline.init(m.device, lib, name, false);
        }
        errdefer for (m.gemm) |p| p.deinit();
        {
            const text = try std.mem.concat(gpa, u8, &.{ ks.core_row_projection, ks.glm_ax_moe });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.moe_rows[0] = try mtl.Pipeline.init(m.device, lib, "gx_moe_gu_rows", false);
            errdefer m.moe_rows[0].deinit();
            for (m.moe_rows[1..], [_][:0]const u8{ "gx_moe_down_rows", "gx_moe_gu_set", "gx_moe_down_set", "gx_rows_mr", "gx_proj_k4", "gx_proj_k4_1", "gx_proj_c1", "gx_proj_c8" }) |*p, name| p.* = try mtl.Pipeline.init(m.device, lib, name, false);
        }
        errdefer for (m.moe_rows) |p| p.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.glm_ax_dflash2, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.df[0..made]) |p| p.deinit();
            for (dflash2.names, 0..) |n, i| {
                m.df[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.df) |p| p.deinit();
        m.inv = try m.device.buffer(c.rope / 2 * 4, mtl.ResourceOptions.shared);
        for (m.inv.slice(f32, c.rope / 2), 0..) |*v, i| {
            v.* = @floatCast(1.0 / std.math.pow(f64, c.rope_theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(c.rope))));
        }
        m.load_seconds = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn kernel(m: *const Model, g: Own) mtl.Pipeline {
        return m.own[@backingInt(g)];
    }

    /// The drafter's layers, in its features' order: each one's tap slot.
    pub fn setTaps(m: *Model, taps: []const u32) !void {
        if (taps.len > 8) return error.TooManyTaps;
        m.tap_at = @splat(-1);
        for (taps, 0..) |l, i| {
            if (l >= m.weights.main) return error.TapPastTheStack;
            m.tap_at[l] = @intCast(i);
        }
        m.taps = @intCast(taps.len);
    }

    pub fn deinit(m: *Model) void {
        for (m.df) |p| p.deinit();
        m.inv.deinit();
        m.sparse_nax.deinit();
        if (m.sparse_mpp) |p| p.deinit();
        m.index_nax.deinit();
        m.sparse_split.deinit();
        m.sparse_combine.deinit();
        m.combine_unembed.deinit();
        for (m.gemm) |p| p.deinit();
        for (m.moe_rows) |p| p.deinit();
        for (m.mm) |p| p.deinit();
        for (m.route) |p| p.deinit();
        m.rows.deinit();
        for (m.glue) |p| p.deinit();
        for (m.own) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        if (m.tick_queue) |q| q.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
