//! glm_ax's bounded draw against the general row draw on synthetic GLM-sized rows: tokens drawn and time, CPU only.
const std = @import("std");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const fast = tf.glm_ax.draw;
const a = std.heap.page_allocator;
const V = 154880;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const words = try a.alloc(u16, V);
    var prng = std.Random.DefaultPrng.init(53);
    const r = prng.random();
    var same: usize = 0;
    var total: usize = 0;
    var t_fast: f64 = 0;
    var t_full: f64 = 0;
    var fell: usize = 0;
    for (0..40) |row_i| {
        // a peaked row (a few strong tokens) or a flat one, like prose and code steps
        const spread: f32 = if (row_i % 4 == 3) 1.0 else 3.5;
        for (words) |*w| w.* = bf16(@as(f32, @floatCast(r.floatNorm(f64))) * spread - 4.0);
        for (0..(row_i % 5) + 1) |_| words[r.uintLessThan(usize, V)] = bf16(14.0 + r.float(f32) * 6.0);
        if (row_i % 7 == 0) for (0..50) |_| { // ties at the top
            words[r.uintLessThan(usize, V)] = bf16(12.0);
        };
        for ([_]lanes.Sampling{
            .{ .seed = 7, .temperature = 0.6, .top_k = 0, .top_p = 0.95 },
            .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.95 },
            .{ .seed = 11, .temperature = 0.6, .top_k = 500, .top_p = 0.95, .min_p = 0.05 },
            .{ .seed = 13, .temperature = 0.8, .top_k = 0, .top_p = 0.8, .min_p = 0.02 },
        }) |s| for (0..8) |pos| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            const want = try lanes.sampling_row.draw(a, words, s, pos * 101 + row_i, &.{});
            const t1 = std.Io.Timestamp.now(io, .awake);
            const got = try fast.draw(a, words, s, pos * 101 + row_i, &.{});
            const t2 = std.Io.Timestamp.now(io, .awake);
            if ((try fast.bounded(a, words, s, pos * 101 + row_i)) == null) fell += 1;
            t_full += @as(f64, @floatFromInt(t0.durationTo(t1).nanoseconds)) / 1e6;
            t_fast += @as(f64, @floatFromInt(t1.durationTo(t2).nanoseconds)) / 1e6;
            same += @intFromBool(want == got);
            total += 1;
        };
    }
    std.debug.print("bounded draw vs general draw: {d} of {d} draws the same token ({d} took the general path); mean {d:.3} ms against {d:.3} ms\n", .{ same, total, fell, t_fast / @as(f64, @floatFromInt(total)), t_full / @as(f64, @floatFromInt(total)) });
}
