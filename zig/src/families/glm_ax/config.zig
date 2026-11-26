//! GLM-5.3's shape for the affine engine: glm_dsa's config.json reader, one Mac.
const std = @import("std");
const dsa = @import("../glm_dsa/config.zig");

pub const Config = dsa.Config;
pub const max_layers = dsa.max_layers;
/// The longest context the engine serves.
pub const max_context = 1048576; // the checkpoint's max_position_embeddings (was 262144)

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return dsa.parse(gpa, bytes);
}

/// Tensor parallel over two ranks, milestone A: only the FFN splits (attention, router and head whole on both).
/// Each rank holds channels [c0, c1) of every routed and shared expert's moe_inter and [d0, d1) of the dense
/// MLPs' dense_inter; ranks 1 (the default) holds them all and changes nothing.
pub const Tp = struct {
    rank: u32 = 0,
    ranks: u32 = 1,
    c0: u32 = 0,
    c1: u32 = 0,
    d0: u32 = 0,
    d1: u32 = 0,

    pub fn on(t: Tp) bool {
        return t.ranks > 1;
    }
    pub fn moeLocal(t: Tp) u32 {
        return t.c1 - t.c0;
    }
    pub fn denseLocal(t: Tp) u32 {
        return t.d1 - t.d0;
    }
};

fn envU32(name: [*:0]const u8) !?u32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.BadGlmTpEnv;
}

/// The split from GLM_AX_TP_RANKS (1 | 2), GLM_AX_TP_RANK (0 | 1; `rank` overrides it: the one-process harness) and
/// GLM_AX_TP_CHANNELS (rank 0's share of moe_inter, a multiple of 64, default half; rank 1 takes the rest; the
/// dense MLPs split in the same proportion, rounded to a multiple of 64).
pub fn tpFromEnv(c: *const Config, rank: ?u32) !Tp {
    const ranks = (try envU32("GLM_AX_TP_RANKS")) orelse 1;
    if (ranks == 1) return .{ .c1 = c.moe_inter, .d1 = c.dense_inter };
    if (ranks != 2) {
        std.log.err("glm_ax: GLM_AX_TP_RANKS={d}: only 1 or 2", .{ranks});
        return error.BadGlmTpEnv;
    }
    const r = rank orelse (try envU32("GLM_AX_TP_RANK")) orelse 0;
    const mi = c.moe_inter;
    const di = c.dense_inter;
    const ch = (try envU32("GLM_AX_TP_CHANNELS")) orelse mi / 2;
    if (r > 1 or ch % 64 != 0 or ch < 64 or ch + 64 > mi or di % 64 != 0 or di < 128) {
        std.log.err("glm_ax: GLM_AX_TP_RANK={d}, GLM_AX_TP_CHANNELS={d}: the rank is 0 or 1, the channels a multiple of 64 in [64, {d}]", .{ r, ch, mi - 64 });
        return error.BadGlmTpEnv;
    }
    // the dense share in the same proportion, to the nearest multiple of 64, at least 64 on each side
    const groups: u64 = (@as(u64, ch) * di + @as(u64, mi) * 32) / (@as(u64, mi) * 64);
    const dch: u32 = @intCast(std.math.clamp(groups * 64, 64, @as(u64, di) - 64));
    return if (r == 0) .{ .rank = 0, .ranks = 2, .c0 = 0, .c1 = ch, .d0 = 0, .d1 = dch } else .{ .rank = 1, .ranks = 2, .c0 = ch, .c1 = mi, .d0 = dch, .d1 = di };
}

/// True when `dir` holds MLX shards (model*.safetensors) and no EXLR8 manifest.
pub fn isMlx(io: std.Io, dir: []const u8) bool {
    var buf: [1024]u8 = undefined;
    const m = std.fmt.bufPrint(&buf, "{s}/manifests", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, m, .{})) |_| return false else |_| {}
    const idx = std.fmt.bufPrint(&buf, "{s}/model.safetensors.index.json", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, idx, .{})) |_| return true else |_| {}
    const one = std.fmt.bufPrint(&buf, "{s}/model.safetensors", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, one, .{})) |_| return true else |_| {}
    return false;
}
