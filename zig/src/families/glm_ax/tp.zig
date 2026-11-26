//! Tensor parallel exchange for glm_ax (milestone A: the FFN split over two ranks, attention whole on both).
//! At each FFN's exchange point the forward ends its command buffer, waits for it, and hands this rank's fp32
//! partial (rows x hidden, in `Pair.part`, shared storage) to `exchange`, which returns once the ranks' partials
//! are summed in rank order (rank 0 + rank 1, fp32) into it: both ranks then hold the same bits.
//! `FakePair` is the in-process pair (two threads, one Model each, host memory and a barrier); an MCDMA-backed pair
//! (core/pair.zig style) can stand behind the same `Pair` later.
const std = @import("std");
const mtl = @import("metal");

pub const Pair = struct {
    rank: u32,
    ranks: u32 = 2,
    /// This rank's partial: fp32 [rows, hidden], shared storage (the forward's gx_tp_partial writes it).
    part: mtl.Buffer,
    rows: usize,
    ctx: *anyopaque,
    exchangeFn: *const fn (ctx: *anyopaque, rank: u32, part: []f32) anyerror!void,
    abortFn: *const fn (ctx: *anyopaque) void,

    /// Sum the ranks' `part` (the first rows x hidden floats of `part`) into it, every rank alike.
    pub fn exchange(p: *const Pair, part: []f32) !void {
        return p.exchangeFn(p.ctx, p.rank, part);
    }

    /// This rank failed: a peer waiting at an exchange returns error.GlmTpPeerFailed instead of waiting forever.
    pub fn abort(p: *const Pair) void {
        p.abortFn(p.ctx);
    }
};

/// Two ranks in one process: each publishes its partial, both meet at a barrier, each sums half of the elements
/// (rank 0's value + rank 1's, fp32) into a shared host buffer, both meet again, each copies the sum back.
pub const FakePair = struct {
    gpa: std.mem.Allocator,
    count: std.atomic.Value(u32) = .init(0),
    generation: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    parts: [2][]f32 = .{ &.{}, &.{} },
    sum: []f32,
    /// Exchanges done (rank 0 counts them) and each rank's seconds spent waiting at the barriers.
    exchanges: u64 = 0,
    wait_ns: [2]u64 = .{ 0, 0 },

    /// Room for partials of up to `floats` values.
    pub fn init(gpa: std.mem.Allocator, floats: usize) !FakePair {
        return .{ .gpa = gpa, .sum = try gpa.alloc(f32, floats) };
    }

    pub fn deinit(f: *FakePair) void {
        f.gpa.free(f.sum);
    }

    /// Rank `rank`'s handle, its partial in `part` (rows x hidden fp32, shared storage).
    pub fn pair(f: *FakePair, rank: u32, part: mtl.Buffer, rows: usize) Pair {
        return .{ .rank = rank, .part = part, .rows = rows, .ctx = f, .exchangeFn = exchangeImpl, .abortFn = abortImpl };
    }

    fn abortImpl(ctx: *anyopaque) void {
        const f: *FakePair = @ptrCast(@alignCast(ctx));
        f.failed.store(true, .release);
    }

    fn nowNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }

    fn nap() void {
        const ts = std.c.timespec{ .sec = 0, .nsec = 20_000 };
        _ = std.c.nanosleep(&ts, null);
    }

    /// Both ranks here: the second to arrive moves the generation on; a failed peer ends the wait.
    fn barrier(f: *FakePair, rank: u32) !void {
        const t0 = nowNs();
        defer f.wait_ns[rank] += nowNs() - t0;
        const gen = f.generation.load(.acquire);
        if (f.count.fetchAdd(1, .acq_rel) == 1) {
            f.count.store(0, .monotonic);
            _ = f.generation.fetchAdd(1, .acq_rel);
            return;
        }
        var spins: u32 = 0;
        while (f.generation.load(.acquire) == gen) {
            if (f.failed.load(.acquire)) return error.GlmTpPeerFailed;
            spins += 1;
            if (spins < 4096) std.atomic.spinLoopHint() else nap();
        }
    }

    /// A bare meeting of the two ranks (the harness's token agreement), failing when the peer failed.
    pub fn sync(f: *FakePair, rank: u32) !void {
        return f.barrier(rank);
    }

    fn exchangeImpl(ctx: *anyopaque, rank: u32, part: []f32) anyerror!void {
        const f: *FakePair = @ptrCast(@alignCast(ctx));
        if (rank > 1) return error.GlmTpBadRank;
        f.parts[rank] = part;
        try f.barrier(rank);
        const a = f.parts[0];
        const b = f.parts[1];
        // both ranks see the same lengths, so both refuse alike
        if (a.len != b.len or a.len > f.sum.len) return error.GlmTpPartialMismatch;
        const half = a.len / 2;
        const lo = if (rank == 0) 0 else half;
        const hi = if (rank == 0) half else a.len;
        for (f.sum[lo..hi], a[lo..hi], b[lo..hi]) |*s, x, y| s.* = x + y;
        try f.barrier(rank);
        // every read of the published partials is done (before the barrier): write this rank's own back. The next
        // exchange's first barrier holds the shared sum until both ranks have copied it.
        @memcpy(part, f.sum[0..part.len]);
        if (rank == 0) f.exchanges += 1;
    }
};

test "the fake pair sums two ranks' partials in rank order on both" {
    var f = try FakePair.init(std.testing.allocator, 8);
    defer f.deinit();
    var a = [_]f32{ 1, 2, 3, 4, 5 };
    var b = [_]f32{ 10, 20, 30, 40, 50 };
    const T = struct {
        fn run(fp: *FakePair, x: []f32) void {
            FakePair.exchangeImpl(fp, 1, x) catch unreachable;
        }
    };
    const t = try std.Thread.spawn(.{}, T.run, .{ &f, @as([]f32, &b) });
    try FakePair.exchangeImpl(&f, 0, &a);
    t.join();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44, 55 }, &a);
    try std.testing.expectEqualSlices(f32, &a, &b);
}
