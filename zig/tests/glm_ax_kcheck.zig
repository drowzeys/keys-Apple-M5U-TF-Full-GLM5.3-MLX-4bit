//! glm_ax's attention kernels on synthetic keys, no checkpoint: the prompt chunk's tensor-unit sparse MLA
//! (gx_sparse_nax) against the decode kernels (gx_attend + gx_merge) on the same key lists, then its speed.
//! GLM_ROWS (rows, 64), GLM_KEYS (cache positions, 3000), GLM_TIME=N (time N launches at 2048 rows, 8192 keys).
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ks = tf.glm_ax.kernel_sources;
const frags = tf.frags;
const a = std.heap.page_allocator;

const H = 64;
const KEY = 576;
const LAT = 512;
const TOPK = 2048;

const Row = extern struct { slot: u32 = 0, pos: u32, list: u32, nkeys: u32 };
const GArgs = extern struct { rows: u32 = 0, heads: u32 = 0, width: u32 = 0, stride: u32 = 0, score_stride: u32 = 0, in_stride: u32 = 0, pad0: u32 = 0, pad1: u32 = 0, eps: f32 = 0, scale: f32 = 0, key_scale: f32 = 0, head_scale: f32 = 0 };

fn env(name: [:0]const u8, default: u64) u64 {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch default;
}

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const own = try mtl.Library.fromSource(device, ks.glm_ax, mtl.CompileOptions.mlx());
    const attend = try mtl.Pipeline.init(device, own, "gx_attend", false);
    const merge = try mtl.Pipeline.init(device, own, "gx_merge", false);
    const text = try frags.source(device, a, ks.glm_ax_sparse);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    const nax = try mtl.Pipeline.init(device, lib, "gx_sparse_nax", false);
    std.debug.print("compiled: gx_attend, gx_merge, gx_sparse_nax (tensor units: {})\n", .{device.tensorUnits()});
    if (env("GLM_INDEX", 0) > 0) return indexCheck(device, queue, own, lib, env("GLM_TIME", 0));
    const timing = env("GLM_TIME", 0);
    const R: usize = if (timing > 0) 2048 else @intCast(env("GLM_ROWS", 64));
    const K: usize = if (timing > 0) 8192 else @intCast(env("GLM_KEYS", 3000));
    const start = K - R;
    const opts = mtl.ResourceOptions.shared;
    const keys = try device.buffer((K + 64) * KEY * 2, opts); // 64 rows past the window: NaN, as an unwritten cache
    const q = try device.buffer(R * H * KEY * 2, opts);
    const lists = try device.buffer(R * TOPK * 4, opts);
    const rows = try device.buffer(R * @sizeOf(Row), opts);
    const out_ref = try device.buffer(R * H * LAT * 2, opts);
    const out_nax = try device.buffer(R * H * LAT * 2, opts);
    const partial = try device.buffer(16 * H * 16 * (LAT + 2) * 4, opts);
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    for (keys.slice(u16, K * KEY)) |*v| v.* = bf16((rnd.float(f32) * 2 - 1) * 1.5);
    for (keys.slice(u16, (K + 64) * KEY)[K * KEY ..]) |*v| v.* = 0x7fc0;
    for (q.slice(u16, R * H * KEY)) |*v| v.* = bf16((rnd.float(f32) * 2 - 1) * 0.4);
    const rt = rows.slice(Row, R);
    const ls = lists.slice(u32, R * TOPK);
    var pick = try a.alloc(bool, K);
    for (0..R) |r| {
        const pos = start + r;
        const n = @min(pos + 1, TOPK);
        rt[r] = .{ .pos = @intCast(pos), .list = @intCast(r * TOPK), .nkeys = @intCast(n) };
        if (pos + 1 <= TOPK) {
            for (0..n) |i| ls[r * TOPK + i] = @intCast(i);
            continue;
        }
        @memset(pick[0 .. pos + 1], false);
        var got: usize = 0;
        while (got < TOPK) {
            const k = rnd.uintLessThan(usize, pos + 1);
            if (!pick[k]) {
                pick[k] = true;
                got += 1;
            }
        }
        var at: usize = 0;
        for (0..pos + 1) |k| if (pick[k]) {
            ls[r * TOPK + at] = @intCast(k);
            at += 1;
        };
    }
    const scale: f32 = 1.0 / @sqrt(256.0);
    const reps: usize = if (timing > 0) @intCast(timing) else 1;
    // the reference: the decode kernels, 16 rows a launch
    if (timing == 0) {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        var r0: usize = 0;
        while (r0 < R) : (r0 += 16) {
            const blk: u32 = @intCast(@min(16, R - r0));
            const ga = GArgs{ .rows = blk, .heads = H, .scale = scale };
            e.setPipeline(attend);
            e.setBuffer(q, r0 * H * KEY * 2, 0);
            e.setBuffer(q, 0, 1);
            e.setBuffer(rows, r0 * @sizeOf(Row), 2);
            e.setBuffer(lists, 0, 3);
            e.setBuffer(keys, 0, 4);
            e.setValue(ga, 5);
            e.setBuffer(partial, 0, 6);
            e.dispatchGroups(mtl.Size.of(H / 4, 16, blk), mtl.Size.of(128, 1, 1));
            e.barrier();
            e.setPipeline(merge);
            e.setBuffer(partial, 0, 0);
            e.setBuffer(rows, r0 * @sizeOf(Row), 1);
            e.setValue(ga, 2);
            e.setBuffer(out_ref, r0 * H * LAT * 2, 3);
            e.dispatchGroups(mtl.Size.of(H, blk, 1), mtl.Size.of(256, 1, 1));
            e.barrier();
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("reference failed: {s}\n", .{msg});
    }
    var gpu: f64 = 0;
    if (env("GLM_SPLIT", 0) > 0) {
        // the decode rows' split + combine against the decode reference, rows R (<= 16)
        const split = try mtl.Pipeline.init(device, lib, "gx_sparse_split", false);
        const combine = try mtl.Pipeline.init(device, lib, "gx_sparse_combine", false);
        const blocks_n: usize = 16;
        const width: usize = @min(TOPK, K);
        const span: usize = @max(32, std.mem.alignForward(usize, (width + blocks_n - 1) / blocks_n, 32));
        const blocks: usize = (width + span - 1) / span;
        const po = try device.buffer(R * blocks * H * LAT * 4, opts);
        const pm = try device.buffer(R * blocks * H * 4, opts);
        const pl = try device.buffer(R * blocks * H * 4, opts);
        const cb = queue.commandBuffer();
        const e = cb.compute(if (env("GLM_CONC", 0) > 0) .concurrent else .serial);
        e.setPipeline(split);
        e.setBuffer(q, 0, 0);
        e.setBuffer(keys, 0, 1);
        e.setBuffer(lists, 0, 2);
        e.setValue(scale, 3);
        e.setValue([4]i32{ TOPK, @intCast(K), @intCast(span), @intCast(start) }, 4);
        e.setBuffer(po, 0, 5);
        e.setBuffer(pm, 0, 6);
        e.setBuffer(pl, 0, 7);
        e.dispatchGroups(mtl.Size.of(H / 16, blocks, R), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(combine);
        e.setBuffer(po, 0, 0);
        e.setBuffer(pm, 0, 1);
        e.setBuffer(pl, 0, 2);
        e.setValue(@as(i32, @intCast(blocks)), 3);
        e.setBuffer(out_nax, 0, 4);
        e.dispatchGroups(mtl.Size.of(H, R, 1), mtl.Size.of(128, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("split failed: {s}\n", .{msg});
        var nans: usize = 0;
        for (out_nax.slice(u16, R * H * LAT)) |v| nans += @intFromBool(std.math.isNan(f32of(v)));
        var bad: usize = 0;
        for (pm.slice(f32, R * blocks * H), pl.slice(f32, R * blocks * H)) |m_, l_| bad += @intFromBool(std.math.isNan(m_) or std.math.isNan(l_));
        std.debug.print("split: {d} blocks of {d}, NaN outputs {d}, NaN block max/sum {d}\n", .{ blocks, span, nans, bad });
        var zero_heads: usize = 0;
        for (0..R * blocks * H) |hh| {
            var any = false;
            for (po.slice(f32, R * blocks * H * LAT)[hh * LAT ..][0..LAT]) |v| any = any or v != 0;
            zero_heads += @intFromBool(!any);
        }
        std.debug.print("split: (row, block, head) partials left all zero {d}/{d}\n", .{ zero_heads, R * blocks * H });
    } else {
    for (0..reps) |_| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(nax);
        e.setBuffer(q, 0, 0);
        e.setBuffer(keys, 0, 1);
        e.setBuffer(lists, 0, 2);
        e.setValue(scale, 3);
        e.setValue([4]i32{ TOPK, @intCast(K), @intCast(start), 0 }, 4);
        e.setBuffer(out_nax, 0, 5);
        e.dispatchGroups(mtl.Size.of(H / 16, R, 1), mtl.Size.of(128, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("gx_sparse_nax failed: {s}\n", .{msg});
        gpu += cb.gpuSeconds();
    }
    }
    if (timing > 0) {
        const ms = 1000.0 * gpu / @as(f64, @floatFromInt(reps));
        var listed: f64 = 0;
        for (rt) |x| listed += @floatFromInt(x.nkeys);
        const flop = listed * H * (KEY + LAT) * 2.0;
        std.debug.print("gx_sparse_nax: {d} rows x {d} heads, {d:.0} listed keys a row, {d:.3} ms = {d:.1} TFLOPS\n", .{ R, H, listed / @as(f64, @floatFromInt(R)), ms, flop / (ms / 1000.0) / 1e12 });
        return;
    }
    const x = out_ref.slice(u16, R * H * LAT);
    const y = out_nax.slice(u16, R * H * LAT);
    var worst: f64 = 1;
    var maxabs: f64 = 0;
    for (0..R * H) |rh| {
        var dot: f64 = 0;
        var nx: f64 = 0;
        var ny: f64 = 0;
        for (0..LAT) |d| {
            const u: f64 = f32of(x[rh * LAT + d]);
            const v: f64 = f32of(y[rh * LAT + d]);
            dot += u * v;
            nx += u * u;
            ny += v * v;
            maxabs = @max(maxabs, @abs(u - v));
        }
        worst = @min(worst, dot / @sqrt(nx * ny + 1e-30));
    }
    std.debug.print("{s} against gx_attend+gx_merge, {d} rows at positions {d}..{d}: worst (row, head) cosine {d:.6}, max |diff| {d:.5}\n", .{ if (env("GLM_SPLIT", 0) > 0) "gx_sparse_split+combine" else "gx_sparse_nax", R, start, K - 1, worst, maxabs });
}

const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32, wscale: f32, pad0: u32 = 0, pad1: u32 = 0 };

/// The tensor-unit index scores (gx_index_qrope + gx_index_nax) against gx_index_scores on 64 rows.
fn indexCheck(device: mtl.Device, queue: mtl.Queue, own: mtl.Library, lib: mtl.Library, timing: u64) !void {
    const scalar = try mtl.Pipeline.init(device, own, "gx_index_scores", false);
    const qrope = try mtl.Pipeline.init(device, own, "gx_index_qrope", false);
    const nax = try mtl.Pipeline.init(device, lib, "gx_index_nax", false);
    const R: usize = 64;
    const K: usize = if (timing > 0) 65536 else 5000;
    const start = K - R;
    const opts = mtl.ResourceOptions.shared;
    const keys = try device.buffer(K * 128 * 2, opts);
    const iq = try device.buffer(R * 4096 * 2, opts);
    const iqr = try device.buffer(R * 4096 * 2, opts);
    const iw = try device.buffer(R * 64 * 2, opts);
    const rows = try device.buffer(R * @sizeOf(Row), opts);
    const s1 = try device.buffer(R * K * 4, opts);
    const s2 = try device.buffer(R * K * 4, opts);
    const inv = try device.buffer(32 * 4, opts);
    for (inv.slice(f32, 32), 0..) |*v, i| v.* = @floatCast(1.0 / std.math.pow(f64, 8e6, @as(f64, @floatFromInt(2 * i)) / 64.0));
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (keys.slice(u16, K * 128)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (iq.slice(u16, R * 4096)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (iw.slice(u16, R * 64)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (rows.slice(Row, R), 0..) |*r, i| r.* = .{ .pos = @intCast(start + i), .list = @intCast(i * TOPK), .nkeys = 0 };
    const ga = GArgs{ .rows = R, .heads = H, .score_stride = @intCast(K), .key_scale = 1.0 / @sqrt(128.0), .head_scale = 1.0 / @sqrt(32.0) };
    if (timing == 0) {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(scalar);
        e.setBuffer(iq, 0, 0);
        e.setBuffer(iw, 0, 1);
        e.setBuffer(keys, 0, 2);
        e.setBuffer(rows, 0, 3);
        e.setValue(ga, 4);
        e.setBuffer(s1, 0, 5);
        e.setBuffer(inv, 0, 6);
        e.dispatchGroups(mtl.Size.of((K - 1) / 256 + 1, R, 1), mtl.Size.of(128, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
    }
    var gpu: f64 = 0;
    const reps: usize = if (timing > 0) @intCast(timing) else 1;
    for (0..reps) |_| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(qrope);
        e.setBuffer(iq, 0, 0);
        e.setBuffer(rows, 0, 1);
        e.setBuffer(inv, 0, 2);
        e.setValue(ga, 3);
        e.setBuffer(iqr, 0, 4);
        e.dispatchThreads(mtl.Size.of(32 * 64, R, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(nax);
        e.setBuffer(iqr, 0, 0);
        e.setBuffer(iw, 0, 1);
        e.setBuffer(keys, 0, 2);
        e.setBuffer(s2, 0, 3);
        e.setValue(ScoreArgs{ .p0 = @intCast(start), .q_stride = 4096, .w_stride = 64, .s_stride = @intCast(K), .rows = R, .wscale = ga.head_scale * ga.key_scale }, 4);
        e.dispatchGroups(mtl.Size.of((K - 1) / 64 + 1, 1, 1), mtl.Size.of(128, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("gx_index_nax failed: {s}\n", .{msg});
        gpu += cb.gpuSeconds();
    }
    if (timing > 0) {
        const ms = 1000.0 * gpu / @as(f64, @floatFromInt(reps));
        const flop: f64 = @as(f64, @floatFromInt(R)) * @as(f64, @floatFromInt(K)) * 32.0 * 128.0 * 2.0;
        std.debug.print("gx_index_nax: {d} rows x {d} keys, {d:.3} ms = {d:.1} TFLOPS\n", .{ R, K, ms, flop / (ms / 1000.0) / 1e12 });
        return;
    }
    const x = s1.slice(f32, R * K);
    const y = s2.slice(f32, R * K);
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (0..R) |r| for (0..start + r + 1) |k| {
        worst = @max(worst, @abs(@as(f64, x[r * K + k]) - y[r * K + k]));
        scale = @max(scale, @abs(@as(f64, x[r * K + k])));
    };
    // the top-k each way: how many of the 2048 picked keys agree
    var agree: usize = 0;
    for (0..R) |r| {
        const n = start + r + 1;
        const a1 = try top(x[r * K ..][0..n]);
        const a2 = try top(y[r * K ..][0..n]);
        for (a1) |k| agree += @intFromBool(std.mem.indexOfScalar(u32, a2, k) != null);
    }
    std.debug.print("gx_index_nax against gx_index_scores, {d} rows x {d} keys: max |diff| {d:.6} of scores up to {d:.3}; top-2048 lists share {d:.4} of their keys\n", .{ R, K, worst, scale, @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(R * TOPK)) });
}

fn top(v: []const f32) ![]u32 {
    const idx = try a.alloc(u32, v.len);
    for (idx, 0..) |*i, k| i.* = @intCast(k);
    const Ctx = struct {
        v: []const f32,
        fn more(c: @This(), p: u32, q: u32) bool {
            return c.v[p] > c.v[q] or (c.v[p] == c.v[q] and p < q);
        }
    };
    std.mem.sort(u32, idx, Ctx{ .v = v }, Ctx.more);
    return idx[0..TOPK];
}
